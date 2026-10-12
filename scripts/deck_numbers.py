"""
deck_numbers.py -- reproduce every figure used in the project proposal deck.

Run:
    python deck_numbers.py --train train_multitask.csv --dev dev_multitask.csv

Either file may be omitted; the script reports whatever it is given.
Writes deck_numbers.txt alongside the console output so you can paste
the tables straight into slides.

Requires: pandas, scikit-learn
    pip install pandas scikit-learn
"""

import argparse
import io
import re
import sys

import pandas as pd
from sklearn.metrics import f1_score

C = "Coarse-Grained"
F = "Fine-Grained Hope"

# The "Semantic Dependency Between Outputs" table from the task page.
OFFICIAL = {
    "Optimistic Hope": {"Encouraging Hope"},
    "Realistic Hope":  {"Encouraging Hope"},
    "Inspiring Hope":  {"Encouraging Hope", "Blended Hope"},
    "Fading Hope":     {"Blended Hope", "Discouraging Hope"},
    "Hopelessness":    {"Discouraging Hope"},
}

KANNADA = re.compile(r"[\u0C80-\u0CFF]")
LATIN = re.compile(r"[A-Za-z]")
# Rows where a Python list looks to have been stringified into the text field.
MANGLED = re.compile(r"(?:\"\"|', '|', \")")

OUT = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.write(line + "\n")


def rule(title):
    say("\n" + "=" * 74)
    say(title)
    say("=" * 74)


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Header variants seen across the released files and Google Sheets exports.
ALIASES = {
    "text": {"text", "comment", "comments", "sentence", "sentences", "post"},
    C: {"coarsegrained", "coarsegrainedtone", "coarse", "coarsetone",
        "coarselabel", "l1", "tone"},
    F: {"finegrainedhope", "finegrainedhopetype", "fine", "finehope",
        "finelabel", "l2", "hopetype"},
    "Id": {"id", "ids", "index", "sno", "slno"},
}


def load(path, require_labels=True):
    """Proper CSV parsing, tolerant of header naming differences.

    The text field contains commas and quotes, so never split on ',' by hand.
    Set require_labels=False for the unlabelled test file."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Map whatever headers this file uses onto the canonical names.
    rename = {}
    for col in df.columns:
        n = _norm(col)
        for canon, alts in ALIASES.items():
            if n in alts:
                rename[col] = canon
                break
    df = df.rename(columns=rename)

    if "text" not in df.columns:
        sys.exit(f"ERROR in {path}: no text column; found {list(df.columns)}")
    need = [C, F] if require_labels else []
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"ERROR in {path}: missing label columns {missing}; "
                 f"found {list(df.columns)}. If this is the unlabelled test "
                 f"file, pass it with --test-unlabelled instead.")

    for col in ["text"] + [c for c in (C, F) if c in df.columns]:
        df[col] = df[col].str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def distribution(df, split):
    rule(f"CLASS DISTRIBUTION -- {split}  (n = {len(df)})")
    for level, col in (("L1  Coarse-Grained Tone", C), ("L2  Fine-Grained Hope Type", F)):
        vc = df[col].value_counts()
        say(f"\n{level}   ({vc.size} classes)")
        say(f"  {'class':<22}{'n':>7}{'% of split':>13}")
        for k, v in vc.items():
            say(f"  {k:<22}{v:>7}{100*v/len(df):>12.1f}%")
        # Rarest class carries the same macro-F1 weight as the largest one.
        rare = vc.idxmin()
        say(f"  -> macro F1 weight per class = {100/vc.size:.1f}%; "
            f"'{rare}' is {100*vc.min()/len(df):.1f}% of the data "
            f"but {100/vc.size:.1f}% of the score "
            f"({(vc.max()/vc.min()):.1f}x the per-sample value of '{vc.idxmax()}')")


def dependency(df, split):
    rule(f"COARSE x FINE DEPENDENCY -- {split}")
    ct = pd.crosstab(df[C], df[F])
    say("")
    say(ct.to_string())

    occupied = int((ct > 0).sum().sum())
    say(f"\npairs occurring in the data: {occupied} of {ct.size} possible")

    bad = [(c, f) for c, f in zip(df[C], df[F]) if c not in OFFICIAL.get(f, {c})]
    say(f"rows breaking the published rule table: {len(bad)} / {len(df)} "
        f"({100*len(bad)/len(df):.1f}%)")
    if bad:
        for (c, f), n in pd.Series(bad).value_counts().items():
            say(f"    {c:<20} + {f:<18} {n:>5} rows")

    # Ceiling of replacing the coarse head with a lookup from the fine label.
    maj = df.groupby(F)[C].agg(lambda s: s.value_counts().idxmax())
    pred = df[F].map(maj)
    say(f"\nbest possible fine -> coarse lookup:")
    for f, c in maj.items():
        say(f"    {f:<18} -> {c}")
    say(f"  accuracy  {(pred == df[C]).mean():.4f}")
    say(f"  macro F1  {f1_score(df[C], pred, average='macro', zero_division=0):.4f}")
    say("  -> a rule cannot replace the coarse head; both outputs need their own.")


def floors(df, split):
    rule(f"BASELINE FLOORS -- {split}")
    for level, col in (("L1 coarse", C), ("L2 fine", F)):
        m = df[col].value_counts().idxmax()
        s = f1_score(df[col], [m] * len(df), average="macro", zero_division=0)
        say(f"  majority-class macro F1, {level:<10}: {s:.4f}   (always predict '{m}')")
    say("  -> any model scoring near these has learned nothing.")


def text_profile(df, split):
    rule(f"TEXT PROFILE -- {split}")
    n = df["text"].str.split().str.len()
    say(f"  words per comment: median {n.median():.0f}, mean {n.mean():.1f}, "
        f"p95 {n.quantile(0.95):.0f}, max {n.max()}")
    say(f"  -> max_length = 64 tokens is ample")

    k, l = df["text"].str.contains(KANNADA), df["text"].str.contains(LATIN)
    tot = len(df)
    say(f"\n  Roman script only : {(l & ~k).sum():>5} ({100*(l & ~k).sum()/tot:.0f}%)")
    say(f"  Kannada only      : {(k & ~l).sum():>5} ({100*(k & ~l).sum()/tot:.0f}%)")
    say(f"  both scripts      : {(k & l).sum():>5} ({100*(k & l).sum()/tot:.0f}%)")
    say(f"  neither           : {(~k & ~l).sum():>5}")

    mang = df["text"].str.contains(MANGLED)
    say(f"\n  rows with broken quoting / stringified lists: {mang.sum()}")
    for t in df.loc[mang, "text"].head(4):
        say(f"      {t[:80]!r}")


def integrity(parts):
    rule("DATA INTEGRITY")
    for split, df in parts:
        dup = df["text"].duplicated().sum()
        if C in df.columns:
            exact = df.duplicated(subset=["text", C, F]).sum()
            conflict = df.groupby("text")[[C, F]].nunique().max(axis=1).gt(1).sum()
            say(f"  {split:<16} duplicate texts {dup:>4} | identical rows {exact:>4} | "
                f"texts with conflicting labels {conflict:>4}")
        else:
            say(f"  {split:<16} duplicate texts {dup:>4} | (no labels in this file)")

    say("")
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            (na, da), (nb, db) = parts[i], parts[j]
            overlap = set(da["text"]) & set(db["text"])
            say(f"  {na} / {nb} text overlap: {len(overlap)} texts")
            if overlap:
                say(f"    !! any {nb} score is inflated by this much leakage -- report it")
                for t in list(overlap)[:5]:
                    say(f"      {t[:70]!r}")


def slide_block(parts):
    rule("PASTE-READY BLOCK FOR THE DATASET SLIDE")
    for level, col in (("Coarse-Grained Tone  (L1)", C), ("Fine-Grained Hope Type  (L2)", F)):
        say(f"\n{level}")
        order = parts[0][1][col].value_counts()
        classes = sorted({c for _, d in parts for c in d[col].unique()},
                         key=lambda c: -order.get(c, 0))
        say(f"  {'class':<22}" + "".join(f"{n.capitalize():>10}{'%':>8}" for n, _ in parts))
        for c in classes:
            row = f"  {c:<22}"
            for _, d in parts:
                v = int((d[col] == c).sum())
                row += f"{v:>10}{100*v/len(d):>7.1f}%"
            say(row)
        say(f"  {'TOTAL':<22}" + "".join(f"{len(d):>10}{'':>8}" for _, d in parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_multitask.csv")
    ap.add_argument("--dev", default="dev_multitask.csv")
    ap.add_argument("--test", default="test_with_label_Sheet1.csv",
                    help="test file WITH gold labels")
    ap.add_argument("--test-unlabelled", dest="test_u",
                    default="multitask_test_withoutlabel.csv",
                    help="the released test file without labels; checked for "
                         "row alignment and leakage only")
    args = ap.parse_args()

    labelled = []
    for name, path in (("train", args.train), ("dev", args.dev), ("test", args.test)):
        try:
            labelled.append((name, load(path)))
        except FileNotFoundError:
            say(f"(no {name} file at {path!r} -- skipping)")

    unlabelled = None
    try:
        unlabelled = load(args.test_u, require_labels=False)
    except FileNotFoundError:
        say(f"(no unlabelled test file at {args.test_u!r} -- skipping)")

    if not labelled:
        sys.exit("Nothing to report.")

    for split, df in labelled:
        distribution(df, split)
        dependency(df, split)
        floors(df, split)
        text_profile(df, split)

    # Confirm the labelled test file really corresponds to the released one.
    if unlabelled is not None:
        rule("TEST FILE CROSS-CHECK")
        say(f"  unlabelled test rows : {len(unlabelled)}")
        gold = dict(labelled).get("test")
        if gold is None:
            say("  (no labelled test file loaded, nothing to compare against)")
        else:
            say(f"  labelled test rows   : {len(gold)}")
            if len(gold) != len(unlabelled):
                say("  !! row counts differ -- these are not the same release")
            same_order = (len(gold) == len(unlabelled)
                          and (gold["text"].values == unlabelled["text"].values).all())
            say(f"  identical text, same order: {same_order}")
            if not same_order:
                missing = set(unlabelled["text"]) - set(gold["text"])
                say(f"  texts in unlabelled but not in labelled: {len(missing)}")
                say("  -> align on Id before scoring, never on row position")
        text_profile(unlabelled, "test (unlabelled)")

    parts = labelled + ([("test (unlab.)", unlabelled)] if unlabelled is not None else [])
    integrity(parts)
    slide_block(labelled)

    with open("deck_numbers.txt", "w") as fh:
        fh.write(OUT.getvalue())
    print("\nSaved to deck_numbers.txt")


if __name__ == "__main__":
    main()
