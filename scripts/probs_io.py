"""
probs_io.py -- shared helpers for saving class probabilities (used by m1_probs.py,
train_char.py and train_multitask.py), so every model writes the same file format.

One CSV per model, split and seed:  results/probs/<run name>/<split>_seed<N>.csv
One row per comment, with these columns:
    Id
    pred_tone, pred_type        the labels the model itself predicts
    tone_<label>  (3 columns)   probability of each tone
    type_<label>  (5 columns)   probability of each hope type
The probabilities are used later to average seeds and combine models, chosen on dev.

This file is not run directly; other scripts import from it.
"""

from pathlib import Path

import pandas as pd

COARSE = "Coarse-Grained"
FINE = "Fine-Grained Hope"


def load_unlabelled(path, placeholder_tone, placeholder_type):
    """Read the UNLABELLED test file (Id, text). The file on disk is never changed.

    The training scripts' datasets expect label columns, so we add placeholder
    labels. They are never used: nothing is scored on this file."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["text"] = df["text"].str.strip()
    df[COARSE] = placeholder_tone
    df[FINE] = placeholder_type
    return df


def write_probs(path, ids, pred_c, pred_f, prob_c, prob_f, tone_labels, type_labels):
    """Write one probability file.

    ids            : the comment Ids, in the same order as the other lists
    pred_c, pred_f : predicted class numbers (0, 1, 2, ...) for tone and type
    prob_c, prob_f : one row of probabilities per comment (3 numbers / 5 numbers)
    tone_labels, type_labels : label names, in class-number order
    """
    table = pd.DataFrame({"Id": list(ids)})
    table["pred_tone"] = [tone_labels[k] for k in pred_c]   # class number -> name
    table["pred_type"] = [type_labels[k] for k in pred_f]
    for k, label in enumerate(tone_labels):
        table[f"tone_{label}"] = [row[k] for row in prob_c]
    for k, label in enumerate(type_labels):
        table[f"type_{label}"] = [row[k] for row in prob_f]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path
