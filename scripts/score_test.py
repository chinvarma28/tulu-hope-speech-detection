"""
score_test.py -- score predictions against the repaired test labels.

Aligns on Id, never on row position. Row-order alignment is the failure mode
that produces a plausible-looking but meaningless score, and the released
test files contain duplicate texts, so text-based joins are unsafe too.

Run:
    # sanity-check the two test files line up (no predictions needed)
    python score_test.py --check-only

    # score a predictions file
    python score_test.py --pred my_predictions.csv

The predictions file needs an Id column plus the two label columns, under
any of the usual header spellings.

Requires: pandas, scikit-learn
"""

import argparse
import re
import sys

import pandas as pd
from sklearn.metrics import classification_report, f1_score

C = "Coarse-Grained"
F = "Fine-Grained Hope"

COARSE_SET = {"Encouraging Hope", "Discouraging Hope", "Blended Hope"}
FINE_SET = {"Inspiring Hope", "Optimistic Hope", "Realistic Hope",
            "Fading Hope", "Hopelessness"}

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

ALIASES = {
    "text": {"text", "comment", "comments", "sentence", "post"},
    C: {"coarsegrained", "coarsegrainedtone", "coarse", "coarsetone",
        "coarselabel", "l1", "tone", "predcoarse", "coarsepred"},
    F: {"finegrainedhope", "finegrainedhopetype", "fine", "finehope",
        "finelabel", "l2", "hopetype", "predfine", "finepred"},
    "Id": {"id", "ids", "index", "sno", "slno"},
}


def canon(v):
    return CANON.get(re.sub(r"[^a-z]", "", str(v).lower()), str(v).strip())


def load(path, need):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rename = {}
    for col in df.columns:
        n = re.sub(r"[^a-z0-9]", "", str(col).lower())
        for target, alts in ALIASES.items():
            if n in alts:
                rename[col] = target
                break
    df = df.rename(columns=rename)
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"ERROR in {path}: missing {missing}; found {list(df.columns)}")
    df["Id"] = df["Id"].astype(str).str.strip()
    return df


def check(unlab, gold):
    print("TEST FILE ALIGNMENT")
    print(f"  unlabelled rows : {len(unlab)}")
    print(f"  gold rows       : {len(gold)}")
    if len(unlab) != len(gold):
        print("  !! row counts differ -- these are not the same release")

    dup_u = unlab["Id"].duplicated().sum()
    dup_g = gold["Id"].duplicated().sum()
    print(f"  duplicate Ids   : {dup_u} unlabelled, {dup_g} gold")
    if dup_u or dup_g:
        print("  !! Ids are not unique -- an Id join will silently multiply rows")

    only_u = set(unlab["Id"]) - set(gold["Id"])
    only_g = set(gold["Id"]) - set(unlab["Id"])
    print(f"  Ids only in unlabelled : {len(only_u)}")
    print(f"  Ids only in gold       : {len(only_g)}")

    if "text" in gold.columns:
        m = unlab.merge(gold[["Id", "text"]], on="Id", suffixes=("_u", "_g"))
        same = (m["text_u"].str.strip() == m["text_g"].str.strip()).sum()
        print(f"  matching text on Id    : {same} / {len(m)}")
        if same != len(m):
            print("  !! same Id, different text -- do not trust this pairing")

    ok = (len(unlab) == len(gold) and not only_u and not only_g
          and not dup_u and not dup_g)
    print("  ->", "aligned; safe to score on Id."
          if ok else "NOT aligned; resolve the above before scoring.")
    return ok


def score(pred, gold):
    for col in (C, F):
        pred[col] = pred[col].map(canon)
        gold[col] = gold[col].map(canon)

    bad = (set(pred[C]) - COARSE_SET) | (set(pred[F]) - FINE_SET)
    if bad:
        print(f"\n!! predictions contain unrecognised labels: {sorted(bad)}")

    m = gold.merge(pred[["Id", C, F]], on="Id", suffixes=("_true", "_pred"))
    print(f"\nscored rows: {len(m)} of {len(gold)}")
    if len(m) != len(gold):
        print("!! not every gold row got a prediction -- the score is not comparable")

    f1c = f1_score(m[C + "_true"], m[C + "_pred"], average="macro", zero_division=0)
    f1f = f1_score(m[F + "_true"], m[F + "_pred"], average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print(f"L1 coarse macro F1 : {f1c:.4f}")
    print(f"L2 fine   macro F1 : {f1f:.4f}")
    print(f"JOINT SCORE        : {(f1c + f1f) / 2:.4f}   <- official ranking metric")
    print("=" * 60)

    both = ((m[C + "_true"] == m[C + "_pred"]) & (m[F + "_true"] == m[F + "_pred"])).mean()
    print(f"both labels correct on the same row: {both:.4f}")

    for name, t, p in (("COARSE", C + "_true", C + "_pred"),
                       ("FINE", F + "_true", F + "_pred")):
        print(f"\n--- {name} ---")
        print(classification_report(m[t], m[p], zero_division=0, digits=3))
    print("A class at 0.000 was never predicted. Fix that before a bigger model.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="test_multitask_clean.csv")
    ap.add_argument("--unlabelled", default="multitask_test_withoutlabel.csv")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    gold = load(args.gold, ["Id", C, F])
    try:
        unlab = load(args.unlabelled, ["Id"])
        check(unlab, gold)
    except FileNotFoundError:
        print(f"(no unlabelled file at {args.unlabelled!r} -- skipping alignment check)")

    if args.check_only or not args.pred:
        return
    score(load(args.pred, ["Id", C, F]), gold)


if __name__ == "__main__":
    main()
