"""
score_final.py -- score every predictions file of the final systems with score_test.py.

The scoring itself is done by score_test.py (functions load, check and score), so every
system is scored in exactly the same way: matched on Id, macro F1 for tone and hope type.
This script only runs it once per file and collects the numbers in one table.

Run from the main project folder (tulu-hope-speech), with .venv active.

    # 1. Check on dev first (dev labels, allowed any number of times):
    python scripts/score_final.py --split dev --gold data/dev_multitask.csv --out_dir <scratch folder>

    # 2. The test set, ONCE, at the very end:
    python scripts/score_final.py --split test

Output (for the test run):
    results/test_scores.csv   one row per system: tone, type, joint (+/- std over seeds)
    results/test_report.txt   the full score_test.py report for every file

Requires: pandas, scikit-learn
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

import score_test

# The systems, in the order of the final table. "seeds" = scored per seed, then mean +/- std.
SYSTEMS = [
    ("M3 multi-task (main system)", "M3"),
    ("All four combined (best overall)", "ALL4"),
    ("M1 TF-IDF + LR", "M1"),
    ("CNN multi-task", "CNN"),
    ("M4b MuRIL pair head", "M4b"),
]
SEEDS = [1, 2, 3, 4, 5]


def score_file(pred_path, gold_path, report):
    """Score one predictions file with score_test.py. Its printed report is captured
    (redirect_stdout sends everything print() writes into `buffer` instead of the screen)
    and added to `report`. Returns (tone F1, type F1)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        gold = score_test.load(gold_path, ["Id", score_test.C, score_test.F])
        pred = score_test.load(pred_path, ["Id", score_test.C, score_test.F])
        tone, typ = score_test.score(pred, gold)
    report.append(f"\n{'#' * 70}\n# {pred_path}\n{'#' * 70}\n{buffer.getvalue()}")
    return tone, typ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--gold", default="data/test_multitask_clean.csv")
    ap.add_argument("--unlabelled", default="data/multitask_test_withoutlabel.csv")
    ap.add_argument("--pred_dir", default="results/predictions")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    if args.split == "test":
        # The same alignment check as "score_test.py --check-only": the unlabelled file
        # (whose Ids the models predicted) and the gold file must contain the same Ids.
        gold = score_test.load(args.gold, ["Id", score_test.C, score_test.F])
        unlab = score_test.load(args.unlabelled, ["Id"])
        if not score_test.check(unlab, gold):
            raise SystemExit("Test files are not aligned; nothing was scored.")

    pred_dir = Path(args.pred_dir) / args.split
    report = []
    rows = []
    for label, short in SYSTEMS:
        if short == "M1":   # one run only
            tone, typ = score_file(pred_dir / "M1.csv", args.gold, report)
            rows.append({"system": label, "kind": "single run", "tone": tone, "tone_std": 0.0,
                         "type": typ, "type_std": 0.0, "joint": (tone + typ) / 2, "joint_std": 0.0})
            continue

        # Per seed -> mean and std (np.std divides by n, like the training scripts).
        per_seed = [score_file(pred_dir / f"{short}_seed{s}.csv", args.gold, report) for s in SEEDS]
        tones = np.array([t for t, _ in per_seed])
        types = np.array([f for _, f in per_seed])
        joints = (tones + types) / 2
        rows.append({"system": label, "kind": "5 seeds, mean +/- std",
                     "tone": tones.mean(), "tone_std": tones.std(),
                     "type": types.mean(), "type_std": types.std(),
                     "joint": joints.mean(), "joint_std": joints.std()})

        # The 5 seeds' probabilities averaged into one prediction.
        tone, typ = score_file(pred_dir / f"{short}_avg.csv", args.gold, report)
        rows.append({"system": label, "kind": "5 seeds averaged", "tone": tone,
                     "type": typ, "joint": (tone + typ) / 2})

    table = pd.DataFrame(rows)
    print(f"\nFINAL SCORES on {args.split} (macro F1)")
    print(f"{'system':<34} {'kind':<22} {'tone':>15} {'type':>15} {'joint':>15}")
    for _, r in table.iterrows():
        def cell(col):
            std = r.get(col + "_std")
            if pd.isna(std) or r["kind"] != "5 seeds, mean +/- std":
                return f"{r[col]:.3f}"
            return f"{r[col]:.3f} +/- {std:.3f}"
        print(f"{r['system']:<34} {r['kind']:<22} {cell('tone'):>15} {cell('type'):>15} "
              f"{cell('joint'):>15}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / f"{args.split}_scores.csv", index=False)
    (out_dir / f"{args.split}_report.txt").write_text("".join(report))
    print(f"\nsaved: {out_dir / f'{args.split}_scores.csv'}  and  {out_dir / f'{args.split}_report.txt'}")


if __name__ == "__main__":
    main()
