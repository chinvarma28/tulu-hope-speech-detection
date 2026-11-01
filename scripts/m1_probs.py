"""
m1_probs.py -- save M1's class probabilities for the dev set and the UNLABELLED test set.

M1 = character TF-IDF + logistic regression, one model for tone and one for hope type
(the model itself is defined in explore_multitask.py, make_m1_model).

The saved probabilities are used later to combine models (ensembles). They are chosen
on the dev set; the test file here has no labels, so nothing is scored on test.

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/m1_probs.py

Output (one row per comment):
    results/probs/m1_tfidf_lr/dev_seed0.csv
    results/probs/m1_tfidf_lr/test_seed0.csv
Columns: Id, then "tone_<label>" for the 3 tones and "type_<label>" for the 5 hope types.
M1 has no randomness, so there is one file per split, called seed 0.

Requires: pandas, scikit-learn
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

# explore_multitask.py is in the same folder, so Python can import from it.
from explore_multitask import COARSE, FINE, load, make_m1_model

# Short name used in the column headers -> column in the CSV files.
TASKS = {"tone": COARSE, "type": FINE}


def load_unlabelled(path):
    """The unlabelled test file has only Id and text. The file on disk is never changed."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["text"] = df["text"].str.strip()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_clean.csv")
    ap.add_argument("--dev", default="data/dev_multitask.csv")
    ap.add_argument("--test", default="data/multitask_test_withoutlabel.csv",
                    help="the UNLABELLED test file (Id, text)")
    ap.add_argument("--out_dir", default="results/probs/m1_tfidf_lr")
    args = ap.parse_args()

    train, dev = load(args.train), load(args.dev)
    test = load_unlabelled(args.test)
    print(f"train={len(train)}  dev={len(dev)}  test (unlabelled)={len(test)}")

    # One table per split. [["Id"]] (double brackets) keeps a one-column table.
    out = {"dev": dev[["Id"]].copy(), "test": test[["Id"]].copy()}

    for short, col in TASKS.items():
        model = make_m1_model(weighted=True)   # the "balanced" M1 from the baseline table
        model.fit(train["text"], train[col])   # trained on train only

        # predict_proba gives one column per class, in the order of model.classes_.
        for split, df in (("dev", dev), ("test", test)):
            probs = model.predict_proba(df["text"])
            for k, label in enumerate(model.classes_):
                out[split][f"{short}_{label}"] = probs[:, k]

        # Check: the dev score must match the M1 baseline (tone 0.579, type 0.400).
        f1 = f1_score(dev[col], model.predict(dev["text"]), average="macro", zero_division=0)
        print(f"dev macro F1, {short}: {f1:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, table in out.items():
        path = out_dir / f"{split}_seed0.csv"
        table.to_csv(path, index=False)
        print(f"saved: {path}  ({len(table)} rows, {table.shape[1] - 1} probability columns)")


if __name__ == "__main__":
    main()
