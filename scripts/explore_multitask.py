"""
DravidianCodeMix 2026 -- SubTask 2: Multi-Task Hope Speech Detection in Code-Mixed Tulu
Week-1 diagnostics + baselines.

Usage:
    python explore_multitask.py --train train_multitask.csv --dev dev_multitask.csv

Requires: pandas, scikit-learn
    pip install pandas scikit-learn
"""

import argparse
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

COARSE = "Coarse-Grained"
FINE = "Fine-Grained Hope"

KANNADA = re.compile(r"[\u0C80-\u0CFF]")
LATIN = re.compile(r"[A-Za-z]")
# Rows that look like a stringified Python list got dumped into the text field
MANGLED = re.compile(r"(\"\"|', '|', \")")


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load(path):
    """Load with proper CSV quoting. Never split on ',' by hand -- the text
    field contains commas, quotes and (in the ACL-era files) embedded newlines."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in ("Id", "text", COARSE, FINE):
        if col not in df.columns:
            sys.exit(f"ERROR: expected column {col!r}, found {list(df.columns)}")
    for col in ("text", COARSE, FINE):
        df[col] = df[col].str.strip()
    return df


def sanity(df, name):
    rule(f"1. SANITY -- {name}")
    print(f"rows: {len(df)}   columns: {list(df.columns)}")
    print(f"\nempty text fields: {(df['text'] == '').sum()}")
    print(f"duplicate Ids:     {df['Id'].duplicated().sum()}")

    print(f"\n{COARSE} value counts (must be a small, clean set of labels):")
    print(df[COARSE].value_counts().to_string())
    print(f"\n{FINE} value counts:")
    print(df[FINE].value_counts().to_string())

    # If either list contains something like '#KooInTulu' or an empty string,
    # the parse is broken -- stop and fix it before going further.
    n_coarse, n_fine = df[COARSE].nunique(), df[FINE].nunique()
    print(f"\ndistinct labels: coarse={n_coarse}, fine={n_fine}")
    if n_coarse > 6 or n_fine > 8:
        print("!! WARNING: too many distinct labels -- CSV parse is probably broken")

    print("\nclass balance (%):")
    print((df[COARSE].value_counts(normalize=True) * 100).round(1).to_string())
    print((df[FINE].value_counts(normalize=True) * 100).round(1).to_string())


def crosstab(df, name):
    rule(f"2. COARSE x FINE STRUCTURE -- {name}")
    ct = pd.crosstab(df[COARSE], df[FINE])
    print(ct.to_string())

    total_cells = ct.size
    valid = int((ct > 0).sum().sum())
    print(f"\nvalid (coarse, fine) pairs: {valid} of {total_cells} possible")
    if valid < total_cells:
        print("Empty cells (never co-occur) -- candidates for a decoding constraint:")
        for c in ct.index:
            for f in ct.columns:
                if ct.loc[c, f] == 0:
                    print(f"    {c:<20} x {f}")
    print("\nHow predictable is coarse from fine?")
    print("  (if a fine label maps to one coarse label almost always, a flat")
    print("   joint classifier over valid pairs may beat two separate heads)")
    print((ct.div(ct.sum(axis=0), axis=1) * 100).round(1).to_string())


def duplicates(train, dev):
    rule("3. DUPLICATES AND LEAKAGE")
    for name, df in (("train", train), ("dev", dev)):
        dup_text = df["text"].duplicated().sum()
        dup_full = df.duplicated(subset=["text", COARSE, FINE]).sum()
        print(f"{name}: duplicate texts={dup_text}, fully identical rows={dup_full}")
        # Same text, different labels = annotation inconsistency, and it caps
        # the accuracy any model can reach.
        conflict = (
            df.groupby("text")[[COARSE, FINE]].nunique().max(axis=1).gt(1).sum()
        )
        print(f"{name}: texts carrying CONFLICTING labels = {conflict}")

    overlap = set(train["text"]) & set(dev["text"])
    print(f"\ntrain/dev text overlap: {len(overlap)} texts")
    if overlap:
        print("!! Dev scores are inflated by this much leakage. Report it.")
        for t in list(overlap)[:5]:
            print(f"    {t[:70]!r}")


def text_profile(df, name):
    rule(f"4. TEXT PROFILE -- {name}")
    n_words = df["text"].str.split().str.len()
    print(f"words per comment: median={n_words.median():.0f}  "
          f"mean={n_words.mean():.1f}  p95={n_words.quantile(0.95):.0f}  "
          f"max={n_words.max()}")
    print("-> use max_length=64 (or 96) when tokenising; 512 wastes compute")

    has_kn = df["text"].str.contains(KANNADA)
    has_la = df["text"].str.contains(LATIN)
    print(f"\nscript mix:")
    print(f"  Roman only    : {(has_la & ~has_kn).sum()}")
    print(f"  Kannada only  : {(has_kn & ~has_la).sum()}")
    print(f"  both scripts  : {(has_kn & has_la).sum()}")
    print(f"  neither       : {(~has_kn & ~has_la).sum()}")
    print("-> if the Kannada-script share is non-trivial, transliteration")
    print("   normalisation is worth an ablation")

    mangled = df["text"].str.contains(MANGLED)
    print(f"\nrows that look like stringified lists / broken quoting: {mangled.sum()}")
    for t in df.loc[mangled, "text"].head(5):
        print(f"    {t[:90]!r}")


def make_m1_model(weighted=True):
    """M1: character TF-IDF + logistic regression (one model per task).
    Char n-grams beat word n-grams here: Romanised Tulu has no standard
    spelling, so the same word appears in many forms.
    Also used by m1_probs.py, so both scripts train exactly the same model."""
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                        min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0,
                           class_weight="balanced" if weighted else None),
    )


def baselines(train, dev):
    rule("5. BASELINES (macro F1 on the official dev set)")
    results = {}
    for task in (COARSE, FINE):
        y_tr, y_dev = train[task], dev[task]

        # Floor: always predict the most frequent class.
        majority = y_tr.value_counts().idxmax()
        mf1 = f1_score(y_dev, [majority] * len(y_dev), average="macro", zero_division=0)
        results[(task, "majority")] = mf1

        for weighted in (False, True):
            clf = make_m1_model(weighted)
            clf.fit(train["text"], y_tr)
            pred = clf.predict(dev["text"])
            score = f1_score(y_dev, pred, average="macro", zero_division=0)
            results[(task, "tfidf+lr" + ("+balanced" if weighted else ""))] = score

            if weighted:
                print(f"\n--- {task}: char TF-IDF + LR (class_weight=balanced) ---")
                print(classification_report(y_dev, pred, zero_division=0, digits=3))

    print("\nSUMMARY (macro F1):")
    for (task, model), score in results.items():
        print(f"  {task:<20} {model:<22} {score:.3f}")
    print("\nAny neural model must clear the +balanced rows to be worth its cost.")
    print("Watch for classes with 0.000 F1 -- that class was never predicted,")
    print("which is exactly the failure that sank the published ACL systems.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_multitask.csv")
    ap.add_argument("--dev", default="dev_multitask.csv")
    args = ap.parse_args()

    train, dev = load(args.train), load(args.dev)

    sanity(train, "train")
    sanity(dev, "dev")
    crosstab(train, "train")
    crosstab(dev, "dev")
    duplicates(train, dev)
    text_profile(train, "train")
    baselines(train, dev)

    rule("DONE")
    print("Record these numbers. They are your paper's baseline table.")


if __name__ == "__main__":
    main()
