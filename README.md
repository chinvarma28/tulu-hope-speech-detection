# Multi-task hope speech detection in code-mixed Tulu

Course project for IT702 Deep Learning (MTech), NITK Surathkal.
The task is DravidianCodeMix 2026, SubTask 2: Multi-Task Hope Speech Detection in
Code-Mixed Tulu.

Each comment (short, romanised Tulu mixed with English and Kannada) gets two labels,
predicted together:

- **Tone** (3 classes): Encouraging Hope, Discouraging Hope, Blended Hope
- **Hope type** (5 classes): Inspiring Hope, Optimistic Hope, Realistic Hope,
  Fading Hope, Hopelessness

**Metric:** macro F1 for tone, macro F1 for hope type, and their average (the Joint Score).

## Models

| Name | What it is | Script |
|---|---|---|
| M1 | Character TF-IDF + logistic regression (baseline) | `scripts/explore_multitask.py`, `scripts/m1_probs.py` |
| M2 | Character-level CNN and BiLSTM with attention, trained from scratch, single-task and multi-task | `scripts/train_char.py` |
| M3 | MuRIL encoder shared by two heads (tone, hope type), one forward pass | `scripts/train_multitask.py --task multi` |
| M4 | MuRIL encoder with one head over the 14 (tone, type) pairs seen in training | `scripts/train_multitask.py --task pairs` |

M4b, used in the results below, is a second M4 run in which each pair's loss weight is
the tone weight × the type weight, trained for 8 epochs.

All neural models use class-weighted loss and keep the checkpoint with the best dev
macro F1. Results are the mean ± standard deviation over 5 seeds.

## Results

Models and settings were chosen on the dev set only. The test set was scored once,
at the end.

| System | Dev Joint | Test tone | Test type | Test Joint |
|---|---|---|---|---|
| **M3 multi-task (main system), per seed** | 0.482 | 0.389 ± 0.015 | 0.311 ± 0.010 | **0.350 ± 0.003** |
| M3 multi-task, 5 seeds averaged | 0.493 | 0.393 | 0.336 | **0.364** |
| All four models combined, per seed | 0.503 | 0.375 ± 0.007 | 0.318 ± 0.010 | 0.346 ± 0.008 |
| CNN multi-task, per seed | 0.480 | 0.392 ± 0.020 | 0.306 ± 0.009 | 0.349 ± 0.008 |
| M4b pair head, per seed | 0.478 | 0.378 ± 0.007 | 0.303 ± 0.010 | 0.341 ± 0.006 |
| M1 TF-IDF + LR | 0.489 | 0.363 | 0.289 | 0.326 |

Every system drops by roughly 0.13–0.16 Joint Score from dev to test. Most of the drop
comes from the second part of the released test file (rows 429–744), where every system
gets the tone right only about a third of the time, close to random guessing over three
classes. The labels in that part look unreliable. See
[`results/phase5_summary.md`](results/phase5_summary.md) for the full analysis.

Write-ups for each phase are in `results/phase1_summary.md` to `results/phase5_summary.md`.

## The data is not included

The dataset belongs to the shared task organisers and is not part of this repository.
To rerun anything, get the files from the shared task and put them here:

```
data/
  raw/
    train_multitask.csv
    dev_multitask.csv
    multitask_test_withoutlabel.csv
    test_with_label_Sheet1.csv
  dev_multitask.csv                 (copy of the raw dev file)
  multitask_test_withoutlabel.csv   (copy of the raw unlabelled test file)
```

Every file has the columns `Id`, `text`, `Coarse-Grained`, `Fine-Grained Hope`.

Then build the two cleaned files:

```bash
# Fix the labelled test file (on part of the rows the two label columns are swapped)
python scripts/repair_test.py --in data/raw/test_with_label_Sheet1.csv \
                              --out data/test_multitask_clean.csv

# Remove conflicting labels, duplicates and dev/test overlap from the training set
python scripts/clean_train.py --train data/raw/train_multitask.csv \
                              --dev data/dev_multitask.csv \
                              --test data/test_multitask_clean.csv \
                              --out data/train_clean.csv
```

## Setup

Tested with Python 3.11 on a MacBook with Apple Silicon (PyTorch `mps` device). The
MuRIL runs (M3, M4) were done on a Kaggle T4 GPU with the notebooks in `notebooks/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers scikit-learn pandas numpy
```

## Running

Run everything from the project folder with `.venv` active.

```bash
# M1 baseline
python scripts/explore_multitask.py --train data/train_clean.csv --dev data/dev_multitask.csv

# M2: character CNN / BiLSTM (task: multi, coarse or fine)
python scripts/train_char.py --model cnn --task multi

# M3: MuRIL with two heads, and M4: MuRIL with one pair head
python scripts/train_multitask.py --task multi
python scripts/train_multitask.py --task pairs

# All saved results in one table
python scripts/show_results.py
```

The final systems are built and scored with `scripts/combine_dev.py` (dev only),
`scripts/make_predictions.py` and `scripts/score_final.py`. Each script's docstring
at the top of the file explains its options.

## Repository layout

```
scripts/        all code (training, cleaning, scoring, analysis)
notebooks/      Kaggle notebooks for the GPU runs
results/        scores, logs, saved probabilities, predictions and phase summaries
presentation/   mid-semester and final slide decks
```
