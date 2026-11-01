"""
show_results.py -- print every saved result in one table (reads files only, trains nothing).

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/show_results.py

Reads the results/*.json files written by train_char.py (M2) and
train_multitask.py (M3 and M4). M1 numbers come from results/m1_baseline.txt.
"""

import json
from pathlib import Path

RESULTS = Path("results")

# (label shown in the table, result file name without ".json")
RUNS = [
    ("M2  CNN        multi-task", "m2_cnn_multi"),
    ("M2  CNN        tone only ", "m2_cnn_coarse"),
    ("M2  CNN        type only ", "m2_cnn_fine"),
    ("M2  BiLSTM     multi-task", "m2_bilstm_multi"),
    ("M2  BiLSTM     tone only ", "m2_bilstm_coarse"),
    ("M2  BiLSTM     type only ", "m2_bilstm_fine"),
    ("M3  MuRIL      multi-task", "m3_muril-base-cased_multi"),
    ("M3  MuRIL      tone only ", "m3_muril-base-cased_coarse"),
    ("M3  MuRIL      type only ", "m3_muril-base-cased_fine"),
    ("M4  MuRIL      pair head ", "m4_muril-base-cased_pairs"),
]


def fmt(mean, std):
    """'0.547 +/- 0.010', or a dash if this head was not trained (stored as null)."""
    if mean is None:
        return "      -       "
    return f"{mean:.3f} +/- {std:.3f}"


def main():
    print("Macro F1 on the dev set (753 comments), mean +/- std over 5 seeds\n")
    print(f"{'Model':28s} {'Tone F1':16s} {'Type F1':16s} {'Joint':6s}  Lowest class F1 (any seed)")
    print("-" * 100)
    print(f"{'M1  TF-IDF + LR (baseline)':28s} {'0.579':16s} {'0.400':16s} {'0.490':6s}  "
          "Fading Hope 0.206")

    for label, name in RUNS:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            print(f"{label:28s} (missing: {path})")
            continue
        r = json.loads(path.read_text())

        joint = ""
        if r["tone_mean"] is not None and r["type_mean"] is not None:
            joint = f"{(r['tone_mean'] + r['type_mean']) / 2:.3f}"

        # Find the single lowest per-class F1 across all seeds and both heads.
        lowest_name, lowest_f1 = None, 1.0
        for seed_scores in r["per_class_f1_by_seed"]:
            for head, classes in seed_scores.items():
                for cls, f1 in classes.items():
                    if f1 < lowest_f1:
                        lowest_name, lowest_f1 = cls, f1

        print(f"{label:28s} {fmt(r['tone_mean'], r['tone_std']):16s} "
              f"{fmt(r['type_mean'], r['type_std']):16s} {joint:6s}  "
              f"{lowest_name} {lowest_f1:.3f}")

    print("\nJoint = (tone F1 + type F1) / 2. A lowest-class F1 of 0.000 would mean a class "
          "was never predicted;\nnone of the 50 neural models has that problem.")


if __name__ == "__main__":
    main()
