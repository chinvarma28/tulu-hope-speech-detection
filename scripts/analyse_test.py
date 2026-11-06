"""
analyse_test.py -- explain the drop from dev to test. Diagnosis only:
nothing here changes a model, a prediction or a score.

The released labelled test file is a merge of two sources (see repair_test.py):
rows 1-428 had the label columns in the normal order, rows 429-744 had them swapped.
This script compares the two parts:
  1. their label mix, against train and dev
  2. the test score of each final system on each part (same metric as score_test.py)
  3. a "shift" check: do the labels agree better with the predictions if we compare
     row i's prediction with row i+k's label? A clear peak at k != 0 would mean the
     labels were moved by k rows. (Uses the M3 seed-averaged predictions.)
  4. writes example comments from the swapped part where the true tone and the
     predicted tone are opposite (Encouraging vs Discouraging), for reading by hand.

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/analyse_test.py

Output: printed tables, results/test_parts.csv, results/test_swapped_examples.csv

Requires: pandas, scikit-learn
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from repair_test import C, F, FINE_SET, canon, load as load_raw

RAW = "data/raw/test_with_label_Sheet1.csv"
GOLD = "data/test_multitask_clean.csv"
PRED_DIR = Path("results/predictions/test")
SYSTEMS = ["M3_avg", "ALL4_avg", "M1", "CNN_avg", "M4b_avg"]


def percent(series):
    return (series.value_counts(normalize=True) * 100).round(1)


def main():
    gold = pd.read_csv(GOLD, dtype=str, keep_default_na=False)

    # Which part each row came from: in the raw file, the swapped rows have a
    # hope-type name in the tone column.
    raw = load_raw(RAW)
    raw["Id"] = raw["Id"].str.strip()
    raw["part"] = raw[C].map(canon).isin(FINE_SET).map({False: "normal", True: "swapped"})
    gold = gold.merge(raw[["Id", "part"]], on="Id")

    # 1. label mix
    train = pd.read_csv("data/train_clean.csv", dtype=str, keep_default_na=False)
    dev = pd.read_csv("data/dev_multitask.csv", dtype=str, keep_default_na=False)
    for col in (C, F):
        table = pd.DataFrame({
            "train": percent(train[col]), "dev": percent(dev[col]),
            "test, normal part": percent(gold.loc[gold.part == "normal", col]),
            "test, swapped part": percent(gold.loc[gold.part == "swapped", col]),
        }).fillna(0)
        print(f"\n{col} (% of rows)\n{table.to_string()}")

    # 2. score per part
    rows = []
    for name in SYSTEMS:
        pred = pd.read_csv(PRED_DIR / f"{name}.csv", dtype=str).set_index("Id")
        for part in ("normal", "swapped"):
            sub = gold[gold.part == part]
            p = pred.loc[sub["Id"]]
            tone = f1_score(sub[C], p[C], average="macro", zero_division=0)
            typ = f1_score(sub[F], p[F], average="macro", zero_division=0)
            acc = (sub[C].values == p[C].values).mean()
            rows.append({"system": name, "part": part, "rows": len(sub), "tone": tone,
                         "type": typ, "joint": (tone + typ) / 2, "tone_accuracy": acc})
    parts = pd.DataFrame(rows).round(3)
    print("\nTEST SCORE PER PART (macro F1)\n" + parts.to_string(index=False))
    parts.to_csv("results/test_parts.csv", index=False)

    # 3. shift check, inside each part
    pred = pd.read_csv(PRED_DIR / "M3_avg.csv", dtype=str).set_index("Id").loc[gold["Id"]]
    print("\nSHIFT CHECK: tone agreement of prediction(row i) with label(row i+k), M3_avg")
    for part in ("normal", "swapped"):
        mask = (gold.part == part).values
        labels, preds = gold[C].values[mask], pred[C].values[mask]
        cells = []
        for k in range(-5, 6):
            if k >= 0:
                a, b = preds[:len(preds) - k], labels[k:]
            else:
                a, b = preds[-k:], labels[:len(labels) + k]
            cells.append(f"{k:+d}:{(a == b).mean():.2f}")
        print(f"  {part:<8} " + "  ".join(cells))

    # 4. examples to read by hand
    all4 = pd.read_csv(PRED_DIR / "ALL4_avg.csv", dtype=str).set_index("Id").loc[gold["Id"]]
    m3 = pred
    opposite = {("Discouraging Hope", "Encouraging Hope"), ("Encouraging Hope", "Discouraging Hope")}
    keep = [(t, p1) in opposite and p1 == p2
            for t, p1, p2 in zip(gold[C], m3[C].values, all4[C].values)]
    ex = gold.loc[keep & (gold.part == "swapped").values, ["Id", "text", C, F]].copy()
    ex["predicted tone (M3 and all four agree)"] = m3.loc[ex["Id"], C].values
    ex.to_csv("results/test_swapped_examples.csv", index=False)
    print(f"\n{len(ex)} swapped-part comments where the label and both systems give opposite "
          f"tones -> results/test_swapped_examples.csv")


if __name__ == "__main__":
    main()
