"""
combine_dev.py -- try seed-averaging and model combinations, on the DEV set only.

It reads the probability files written by m1_probs.py, train_char.py and
train_multitask.py (format: see probs_io.py), from results/probs/<run name>/dev_seed<N>.csv.
The test set is NOT used here. We pick the final system from these dev numbers,
and only then score the test set once with score_test.py.

Three ways of predicting are compared:

  1. Each model alone, per seed: the labels the model itself predicted, averaged over
     its seeds (mean +/- std). These should match the numbers printed during training.
  2. Each model, seed-averaged: average the probabilities of all its seeds, then take
     the class with the highest probability.
  3. Combinations of models: average the probabilities of several models (equal weight
     for every model), then take the class with the highest probability. Reported two ways:
       - "all seeds": every model first averages its own seeds, then models are averaged.
       - "per seed" : seed k of every neural model (+ M1, which has only one run) ->
                      one score per seed, reported as mean +/- std, like the single models.

Only equal weights are tried, on purpose: tuning a weight per model on 753 dev comments
would fit the dev set itself, and the dev score would then look better than it really is.

Run from the main project folder (tulu-hope-speech), with .venv active:
    python scripts/combine_dev.py
    python scripts/combine_dev.py --models m1_tfidf_lr m2_cnn_multi     (only some models)

Output: printed tables, and results/combine_dev.csv (one row per model or combination).

Requires: numpy, pandas, scikit-learn
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from explore_multitask import COARSE, FINE, load

# Short names for the tables. A folder not listed here is shown under its folder name.
SHORT_NAMES = {
    "m1_tfidf_lr": "M1",
    "m2_cnn_multi": "CNN",
    "m2_bilstm_multi": "BiLSTM",
    "m3_muril-base-cased_multi": "M3",
    "m4_muril-base-cased_pairs_w-product_ep8": "M4b",
}

# Our two tasks: short name -> (column in the data file, prefix of the probability columns)
TASKS = {"tone": (COARSE, "tone_"), "type": (FINE, "type_")}


def read_model(folder, dev_ids):
    """Read all dev probability files of one model.

    Returns a dictionary:  seed number -> {"tone": table, "type": table}
    where each table has one row per dev comment (in the same order as the dev file)
    and one column per class, plus the model's own predicted labels under "pred_tone"/"pred_type".
    """
    runs = {}
    for path in sorted(folder.glob("dev_seed*.csv")):
        seed = int(path.stem.replace("dev_seed", ""))   # "dev_seed3" -> 3
        df = pd.read_csv(path, dtype={"Id": str})

        # Put the rows in the same order as the dev file, matched on Id.
        # .loc[list of Ids] picks rows in exactly that order; it fails if an Id is missing.
        df = df.set_index("Id").loc[dev_ids]

        run = {}
        for task, (_, prefix) in TASKS.items():
            prob_cols = sorted(c for c in df.columns if c.startswith(prefix))
            probs = df[prob_cols]
            # Column "tone_Blended Hope" -> class name "Blended Hope"
            probs.columns = [c[len(prefix):] for c in prob_cols]
            run[task] = probs
            run["pred_" + task] = df["pred_" + task].tolist()
        runs[seed] = run
    return runs


def macro_f1(true_labels, pred_labels, classes):
    return f1_score(true_labels, pred_labels, labels=classes, average="macro", zero_division=0)


def score_probs(prob_tables, gold):
    """prob_tables: {"tone": table, "type": table}. Picks the most likely class per row
    and returns (tone F1, type F1, joint score)."""
    scores = []
    for task, (col, _) in TASKS.items():
        table = prob_tables[task]
        classes = list(table.columns)
        # .idxmax(axis=1) gives, for each row, the name of the column with the largest value.
        pred = table.idxmax(axis=1).tolist()
        scores.append(macro_f1(gold[col].tolist(), pred, classes))
    tone, typ = scores
    return tone, typ, (tone + typ) / 2


def average_tables(tables):
    """Average several probability tables (same rows, same columns) cell by cell."""
    total = tables[0].copy()
    for t in tables[1:]:
        total = total + t
    return total / len(tables)


def seed_average(runs):
    """Average one model's probabilities over all its seeds."""
    return {task: average_tables([run[task] for run in runs.values()]) for task in TASKS}


def mean_std(values):
    """Mean and standard deviation. np.std divides by n (not n-1), exactly like the
    training scripts, so the +/- numbers here can be compared with theirs."""
    values = np.array(values)
    return values.mean(), values.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="data/dev_multitask.csv")
    ap.add_argument("--probs_dir", default="results/probs")
    ap.add_argument("--models", nargs="*", default=None,
                    help="folder names inside probs_dir (default: every folder with dev files)")
    ap.add_argument("--out", default="results/combine_dev.csv")
    args = ap.parse_args()

    gold = load(args.dev)
    dev_ids = gold["Id"].tolist()

    # Which models to use.
    probs_dir = Path(args.probs_dir)
    if args.models:
        folders = [probs_dir / m for m in args.models]
    else:
        folders = sorted(f for f in probs_dir.iterdir() if any(f.glob("dev_seed*.csv")))

    models = {}   # short name -> {seed: run}
    for folder in folders:
        name = SHORT_NAMES.get(folder.name, folder.name)
        models[name] = read_model(folder, dev_ids)
        print(f"{name:<8} {folder}   seeds: {sorted(models[name])}")
    print(f"dev comments: {len(gold)}\n")

    rows = []   # one dictionary per line of the final table

    # ---- 1 and 2: each model alone ----
    print("EACH MODEL ALONE (dev macro F1)")
    print(f"{'model':<26} {'tone':>15} {'type':>15} {'joint':>7}")
    for name, runs in models.items():
        # 1. per seed, the model's own predicted labels
        per_seed = []
        for run in runs.values():
            tone = macro_f1(gold[COARSE].tolist(), run["pred_tone"], list(run["tone"].columns))
            typ = macro_f1(gold[FINE].tolist(), run["pred_type"], list(run["type"].columns))
            per_seed.append((tone, typ, (tone + typ) / 2))
        tone_m, tone_s = mean_std([s[0] for s in per_seed])
        type_m, type_s = mean_std([s[1] for s in per_seed])
        joint_m, joint_s = mean_std([s[2] for s in per_seed])
        print(f"{name + ', per seed':<26} {tone_m:.3f} +/- {tone_s:.3f}  "
              f"{type_m:.3f} +/- {type_s:.3f}  {joint_m:.3f}")
        rows.append({"system": name, "kind": "single, per seed", "n_seeds": len(runs),
                     "tone": tone_m, "tone_std": tone_s, "type": type_m, "type_std": type_s,
                     "joint": joint_m, "joint_std": joint_s})

        # 2. seed-averaged probabilities (only meaningful with more than one seed)
        if len(runs) > 1:
            tone, typ, joint = score_probs(seed_average(runs), gold)
            print(f"{name + ', seed-averaged':<26} {tone:>15.3f} {typ:>15.3f} {joint:>7.3f}")
            rows.append({"system": name, "kind": "single, seed-averaged", "n_seeds": len(runs),
                         "tone": tone, "type": typ, "joint": joint})

    # ---- 3: combinations of 2 or more models ----
    names = list(models)
    combos = []
    for size in range(2, len(names) + 1):
        # itertools.combinations(["A","B","C"], 2) gives ("A","B"), ("A","C"), ("B","C")
        combos.extend(itertools.combinations(names, size))

    if combos:
        print("\nCOMBINATIONS (equal-weight average of probabilities, dev macro F1)")
        print(f"{'models':<22} {'all seeds: tone   type  joint':>29}   "
              f"{'per seed: joint':>15}")
    for combo in combos:
        label = " + ".join(combo)

        # "all seeds": each model averages its own seeds first, then the models are averaged.
        averaged = [seed_average(models[m]) for m in combo]
        combined = {task: average_tables([a[task] for a in averaged]) for task in TASKS}
        tone, typ, joint = score_probs(combined, gold)
        rows.append({"system": label, "kind": "combination, all seeds",
                     "tone": tone, "type": typ, "joint": joint})

        # "per seed": seed k of every model that has it; a model with one run (M1) is
        # used as it is. Only seeds that every multi-seed model in the combination has.
        multi_seed = [m for m in combo if len(models[m]) > 1]
        per_seed_text = "-"
        if multi_seed:
            shared = set.intersection(*[set(models[m]) for m in multi_seed])
            per_seed = []
            for k in sorted(shared):
                picked = []
                for m in combo:
                    runs = models[m]
                    run = runs[k] if len(runs) > 1 else next(iter(runs.values()))
                    picked.append(run)
                tables = {task: average_tables([r[task] for r in picked]) for task in TASKS}
                per_seed.append(score_probs(tables, gold))
            tone_m, tone_s = mean_std([s[0] for s in per_seed])
            type_m, type_s = mean_std([s[1] for s in per_seed])
            joint_m, joint_s = mean_std([s[2] for s in per_seed])
            per_seed_text = f"{joint_m:.3f} +/- {joint_s:.3f}"
            rows.append({"system": label, "kind": "combination, per seed",
                         "n_seeds": len(per_seed),
                         "tone": tone_m, "tone_std": tone_s, "type": type_m, "type_std": type_s,
                         "joint": joint_m, "joint_std": joint_s})

        print(f"{label:<22} {tone:>11.3f} {typ:>6.3f} {joint:>6.3f}   {per_seed_text:>15}")

    table = pd.DataFrame(rows).round(4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}")
    print("Reminder: these are dev scores. Pick the final system here; score test once at the end.")


if __name__ == "__main__":
    main()
