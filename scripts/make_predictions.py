"""
make_predictions.py -- turn saved probabilities into one predictions file per final system.

The list of systems below was FIXED ON DEV (see combine_dev.py and results/combine_dev.csv)
before the test labels were looked at:
    Main system  : M3 (MuRIL, two heads, one forward pass), 5 seeds + the 5 seeds averaged
    Best overall : all four models combined (M1 + CNN + M3 + M4b, equal weights)
    For comparison: M1, CNN multi-task and M4b on their own

This script never reads the test labels. It only reads the probability files in
results/probs/ and the file that lists the comment Ids.

Each predictions file has the columns score_test.py needs:
    Id, Coarse-Grained, Fine-Grained Hope

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/make_predictions.py --split dev    # check: must reproduce the dev numbers
    python scripts/make_predictions.py --split test   # the real test predictions

Output: results/predictions/<split>/<system>.csv
    e.g. M3_seed1.csv ... M3_seed5.csv, M3_avg.csv, ALL4_avg.csv, ...

Requires: pandas
"""

import argparse
from pathlib import Path

import pandas as pd

from combine_dev import TASKS, average_tables, read_model, seed_average
from explore_multitask import COARSE, FINE

# The models, as folders inside results/probs/.
MODELS = {
    "M1": "m1_tfidf_lr",
    "CNN": "m2_cnn_multi",
    "M3": "m3_muril-base-cased_multi",
    "M4b": "m4_muril-base-cased_pairs_w-product_ep8",
}
ALL4 = ["M1", "CNN", "M3", "M4b"]   # the "best overall" combination
SEEDS = [1, 2, 3, 4, 5]


def labels_from_probs(tables):
    """Pick the most likely class for every comment, for tone and for type."""
    return {task: tables[task].idxmax(axis=1).tolist() for task in TASKS}


def save(out_dir, name, ids, labels):
    """Write one predictions file in the format score_test.py reads."""
    table = pd.DataFrame({"Id": ids, COARSE: labels["tone"], FINE: labels["type"]})
    path = out_dir / f"{name}.csv"
    table.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--dev", default="data/dev_multitask.csv")
    ap.add_argument("--test", default="data/multitask_test_withoutlabel.csv",
                    help="the UNLABELLED test file: only its Id column is used")
    ap.add_argument("--probs_dir", default="results/probs")
    ap.add_argument("--out_dir", default="results/predictions")
    args = ap.parse_args()

    # Only the Id column is read, so the order of the rows is known.
    id_file = args.dev if args.split == "dev" else args.test
    ids = pd.read_csv(id_file, dtype=str, keep_default_na=False)["Id"].str.strip().tolist()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for name, folder in MODELS.items():
        models[name] = read_model(Path(args.probs_dir) / folder, ids, split=args.split)
        print(f"{name:<4} seeds found: {sorted(models[name])}")

    written = []

    # M1 has one run (seed 0): its own predicted labels.
    m1_run = models["M1"][0]
    written.append(save(out_dir, "M1", ids,
                        {"tone": m1_run["pred_tone"], "type": m1_run["pred_type"]}))

    # CNN, M3, M4b: each seed's own predictions, plus the seed-averaged probabilities.
    # (For M4b the per-seed labels come from its best pair, as during training.)
    for name in ["CNN", "M3", "M4b"]:
        for seed in SEEDS:
            run = models[name][seed]
            written.append(save(out_dir, f"{name}_seed{seed}", ids,
                                {"tone": run["pred_tone"], "type": run["pred_type"]}))
        written.append(save(out_dir, f"{name}_avg", ids,
                            labels_from_probs(seed_average(models[name]))))

    # All four combined, per seed: seed k of CNN, M3 and M4b, together with M1.
    for seed in SEEDS:
        picked = [m1_run] + [models[name][seed] for name in ["CNN", "M3", "M4b"]]
        tables = {task: average_tables([run[task] for run in picked]) for task in TASKS}
        written.append(save(out_dir, f"ALL4_seed{seed}", ids, labels_from_probs(tables)))

    # All four combined, every model's seeds averaged first.
    averaged = [seed_average(models[name]) for name in ALL4]
    tables = {task: average_tables([a[task] for a in averaged]) for task in TASKS}
    written.append(save(out_dir, "ALL4_avg", ids, labels_from_probs(tables)))

    print(f"\nwrote {len(written)} prediction files ({len(ids)} rows each) to {out_dir}/")


if __name__ == "__main__":
    main()
