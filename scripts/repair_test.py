"""
repair_test.py -- fix the label columns in the released test file.

The labelled test file is a merge of two differently formatted sources.
On roughly 42% of rows the coarse and fine label columns are swapped, and
that block also drops the word "Hope" from the tone names. Scoring against
the file as-is grades those rows on the wrong column.

This script detects the swap per row from the label vocabulary (not from row
position), normalises the label strings, verifies the result, and writes a
clean file.

Run:
    python repair_test.py --in "test_with_label_Sheet1.csv" \
                          --out test_multitask_clean.csv

Requires: pandas
"""

import argparse
import re
import sys

import pandas as pd

C = "Coarse-Grained"
F = "Fine-Grained Hope"

COARSE_SET = {"Encouraging Hope", "Discouraging Hope", "Blended Hope"}
FINE_SET = {"Inspiring Hope", "Optimistic Hope", "Realistic Hope",
            "Fading Hope", "Hopelessness"}

# Canonical form for every spelling seen across the released files.
CANON = {
    "encouraging": "Encouraging Hope", "encouraginghope": "Encouraging Hope",
    "discouraging": "Discouraging Hope", "discouraginghope": "Discouraging Hope",
    "blended": "Blended Hope", "blendedhope": "Blended Hope",
    "inspiring": "Inspiring Hope", "inspiringhope": "Inspiring Hope",
    "optimistic": "Optimistic Hope", "optimistichope": "Optimistic Hope",
    "realistic": "Realistic Hope", "realistichope": "Realistic Hope",
    "fading": "Fading Hope", "fadinghope": "Fading Hope",
    "hopeless": "Hopelessness", "hopelessness": "Hopelessness",
}

# The dependency table published on the task page, for reporting only.
OFFICIAL = {
    "Optimistic Hope": {"Encouraging Hope"},
    "Realistic Hope":  {"Encouraging Hope"},
    "Inspiring Hope":  {"Encouraging Hope", "Blended Hope"},
    "Fading Hope":     {"Blended Hope", "Discouraging Hope"},
    "Hopelessness":    {"Discouraging Hope"},
}

ALIASES = {
    "text": {"text", "comment", "comments", "sentence", "post"},
    C: {"coarsegrained", "coarsegrainedtone", "coarse", "coarsetone",
        "coarselabel", "l1", "tone"},
    F: {"finegrainedhope", "finegrainedhopetype", "fine", "finehope",
        "finelabel", "l2", "hopetype"},
    "Id": {"id", "ids", "index", "sno", "slno"},
}


def canon(v):
    """Normalise one label string. Ambiguity is impossible here because the
    coarse and fine vocabularies are disjoint once 'Hope' is stripped."""
    key = re.sub(r"[^a-z]", "", str(v).lower())
    return CANON.get(key, str(v).strip())


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rename = {}
    for col in df.columns:
        n = re.sub(r"[^a-z0-9]", "", str(col).lower())
        for target, alts in ALIASES.items():
            if n in alts:
                rename[col] = target
                break
    df = df.rename(columns=rename)
    for need in ("text", C, F):
        if need not in df.columns:
            sys.exit(f"ERROR: no {need!r} column; found {list(df.columns)}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="test with label - Sheet1.csv")
    ap.add_argument("--out", dest="dst", default="test_multitask_clean.csv")
    args = ap.parse_args()

    df = load(args.src)
    n = len(df)
    print(f"loaded {n} rows from {args.src!r}\n")

    a = df[C].map(canon)
    b = df[F].map(canon)

    ok      = a.isin(COARSE_SET) & b.isin(FINE_SET)    # already correct
    swapped = a.isin(FINE_SET) & b.isin(COARSE_SET)    # columns reversed
    broken  = ~(ok | swapped)                          # neither -- inspect by hand

    print(f"  rows already correct : {ok.sum():>5} ({100*ok.sum()/n:.1f}%)")
    print(f"  rows with columns swapped: {swapped.sum():>5} ({100*swapped.sum()/n:.1f}%)")
    print(f"  rows matching neither pattern: {broken.sum():>5}")
    if broken.any():
        print("  !! inspect these before trusting the output:")
        for _, r in df[broken].head(10).iterrows():
            print(f"       {r[C]!r} | {r[F]!r}")

    # Build the repaired columns. Swapped rows get their two labels exchanged.
    coarse = a.where(~swapped, b)
    fine = b.where(~swapped, a)

    out = df.copy()
    out[C], out[F] = coarse, fine
    out = out[~broken].reset_index(drop=True)

    print(f"\nrepaired label vocabulary:")
    print(f"  {C}: {sorted(out[C].unique())}")
    print(f"  {F}: {sorted(out[F].unique())}")
    if set(out[C].unique()) - COARSE_SET or set(out[F].unique()) - FINE_SET:
        print("  !! unexpected labels remain -- do not score against this file yet")
    else:
        print("  -> 3 coarse and 5 fine classes, as the task defines.")

    print(f"\nrepaired distribution ({len(out)} rows):")
    for level, col in (("L1 coarse", C), ("L2 fine", F)):
        print(f"  {level}")
        for k, v in out[col].value_counts().items():
            print(f"    {k:<20}{v:>6}{100*v/len(out):>8.1f}%")

    print("\ncoarse x fine:")
    print(pd.crosstab(out[C], out[F]).to_string())

    bad = [(c, f) for c, f in zip(out[C], out[F]) if c not in OFFICIAL[f]]
    blended = int((out[C] == "Blended Hope").sum())
    print(f"\nrows breaking the published rule table: {len(bad)} / {len(out)} "
          f"({100*len(bad)/len(out):.1f}%)")
    if blended:
        nb = sum(1 for c, _ in bad if c == "Blended Hope")
        print(f"  of which involve Blended Hope: {nb} "
              f"({100*nb/blended:.1f}% of all Blended rows)")

    out.to_csv(args.dst, index=False)
    print(f"\nwritten: {args.dst}")
    print("Score against this file, never against the raw one.")


if __name__ == "__main__":
    main()
