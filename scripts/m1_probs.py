"""
m1_probs.py -- save M1's class probabilities for the dev set and the UNLABELLED test set.

M1 = character TF-IDF + logistic regression, one model for tone and one for hope type
(the model itself is defined in explore_multitask.py, make_m1_model).

The saved probabilities are used later to combine models (ensembles). They are chosen
on the dev set; the test file here has no labels, so nothing is scored on test.

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/m1_probs.py

Output (format: see probs_io.py):
    results/probs/m1_tfidf_lr/dev_seed0.csv
    results/probs/m1_tfidf_lr/test_seed0.csv
M1 has no randomness, so there is one file per split, called seed 0.

Requires: pandas, scikit-learn
"""

import argparse
from pathlib import Path

from sklearn.metrics import f1_score

# These files are in the same folder, so Python can import from them.
from explore_multitask import COARSE, FINE, load, make_m1_model
from probs_io import load_unlabelled, write_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_clean.csv")
    ap.add_argument("--dev", default="data/dev_multitask.csv")
    ap.add_argument("--test", default="data/multitask_test_withoutlabel.csv",
                    help="the UNLABELLED test file (Id, text)")
    ap.add_argument("--out_dir", default="results/probs/m1_tfidf_lr")
    args = ap.parse_args()

    train, dev = load(args.train), load(args.dev)
    test = load_unlabelled(args.test, "none", "none")   # placeholder labels, never used
    print(f"train={len(train)}  dev={len(dev)}  test (unlabelled)={len(test)}")

    # results[split][column] = (predicted class numbers, probabilities, label names)
    results = {"dev": {}, "test": {}}
    for col in (COARSE, FINE):
        model = make_m1_model(weighted=True)   # the "balanced" M1 from the baseline table
        model.fit(train["text"], train[col])   # trained on train only
        labels = list(model.classes_)          # label names, in the order of the columns

        for split, df in (("dev", dev), ("test", test)):
            # predict_proba gives one column per class, in the order of model.classes_.
            probs = model.predict_proba(df["text"])
            preds = probs.argmax(axis=1)       # the class with the highest probability
            results[split][col] = (preds, probs, labels)

        # Check: the dev score must match the M1 baseline (tone 0.579, type 0.400).
        dev_preds, _, _ = results["dev"][col]
        dev_true = [labels.index(x) for x in dev[col]]
        f1 = f1_score(dev_true, dev_preds, average="macro", zero_division=0)
        print(f"dev macro F1, {col}: {f1:.3f}")

    for split, df in (("dev", dev), ("test", test)):
        pred_c, prob_c, tone_labels = results[split][COARSE]
        pred_f, prob_f, type_labels = results[split][FINE]
        path = write_probs(Path(args.out_dir) / f"{split}_seed0.csv", df["Id"],
                           pred_c, pred_f, prob_c, prob_f, tone_labels, type_labels)
        print(f"saved: {path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
