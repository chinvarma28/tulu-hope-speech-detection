"""
clean_train.py -- build a clean training set. Dev and test are never modified.

Steps, in order:
  1. drop texts whose copies carry conflicting labels (no way to know which is right)
  2. drop exact duplicate rows, keeping one copy
  3. drop texts that also appear in dev or test (leakage)

Run:
    python clean_train.py
    python clean_train.py --train train_multitask.csv --dev dev_multitask.csv \
                          --test test_multitask_clean.csv --out train_clean.csv

Requires: pandas
"""

import argparse
import re
import sys

import pandas as pd

C = "Coarse-Grained"
F = "Fine-Grained Hope"


def norm_text(t):
    """Match texts the same way the diagnostics did: trimmed, exact otherwise."""
    return str(t).strip()


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    text_col = next((c for c in df.columns if re.sub(r"[^a-z]", "", c.lower()) == "text"), None)
    if text_col is None:
        sys.exit(f"ERROR: no text column in {path}; found {list(df.columns)}")
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].map(norm_text)
    return df[df["text"] != ""].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_multitask.csv")
    ap.add_argument("--dev", default="dev_multitask.csv")
    ap.add_argument("--test", default="test_multitask_clean.csv")
    ap.add_argument("--out", default="train_clean.csv")
    args = ap.parse_args()

    train = load(args.train)
    for col in (C, F):
        if col not in train.columns:
            sys.exit(f"ERROR: training file lacks {col!r}; found {list(train.columns)}")
    dev, test = load(args.dev), load(args.test)

    n0 = len(train)
    print(f"training rows at start               : {n0}")

    # 1. conflicting labels: same text, more than one distinct label pair
    pairs = train.groupby("text")[[C, F]].nunique()
    conflicting = set(pairs[(pairs[C] > 1) | (pairs[F] > 1)].index)
    train = train[~train["text"].isin(conflicting)]
    n1 = len(train)
    print(f"  - conflicting-label texts removed  : {len(conflicting)} texts, {n0 - n1} rows")

    # 2. exact duplicates: keep the first copy
    train = train.drop_duplicates(subset=["text", C, F], keep="first")
    n2 = len(train)
    print(f"  - extra duplicate copies removed   : {n1 - n2} rows")

    # 3. leakage: anything that also appears in dev or test
    held_out = set(dev["text"]) | set(test["text"])
    leaked_dev = train["text"].isin(set(dev["text"])).sum()
    leaked_test = train["text"].isin(set(test["text"])).sum()
    train = train[~train["text"].isin(held_out)]
    n3 = len(train)
    print(f"  - leaked into dev or test removed  : {n2 - n3} rows "
          f"(dev {leaked_dev}, test {leaked_test}; some may overlap both)")

    print(f"training rows after cleaning          : {n3}  "
          f"({n0 - n3} removed, {100 * (n0 - n3) / n0:.1f}%)")

    # Verification: every property we cleaned for must now hold.
    assert not (set(train["text"]) & held_out), "leakage remains"
    assert not train.duplicated(subset=["text", C, F]).any(), "duplicates remain"
    assert train.groupby("text")[[C, F]].nunique().max().max() <= 1, "conflicts remain"
    print("checks passed: no leakage, no duplicates, no conflicting labels.")

    print("\nclass counts after cleaning:")
    for col in (C, F):
        print(f"  {col}")
        for k, v in train[col].value_counts().items():
            print(f"    {k:<20}{v:>6}")

    train.to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}   (dev and test untouched)")


if __name__ == "__main__":
    main()
