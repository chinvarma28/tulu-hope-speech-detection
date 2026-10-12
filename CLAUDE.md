# Project: Multi-task hope speech detection in code-mixed Tulu
MTech Deep Learning course project (IT702), NITK Surathkal.

## How to work with us
- We are learning. We are not Python experts.
- Work in small steps. After each step, stop so we can run it ourselves.
- For every change, explain what it does and why.
- Explain any Python syntax that a beginner might not know.
- Prefer simple, readable code over clever or compact code.
- Never delete or overwrite data files.

## Data rules (never break these)
- Train only on `data/train_clean.csv`.
- Never modify `data/dev_multitask.csv`, `data/test_multitask_clean.csv`, or anything in `data/raw/`.
- Choose models and settings using the dev set only.
- Score the test set once, at the very end, using `scripts/score_test.py`.
- Columns in every file: `Id`, `text`, `Coarse-Grained`, `Fine-Grained Hope`.

## The task
Each comment gets two labels, predicted together in one forward pass:
- Tone (3 classes): Encouraging Hope, Discouraging Hope, Blended Hope
- Hope type (5 classes): Inspiring Hope, Optimistic Hope, Realistic Hope, Fading Hope, Hopelessness

Metric: macro F1 for tone, macro F1 for hope type, and their average (Joint Score).

## Models we plan to build
- M1: character TF-IDF + logistic regression (in `scripts/explore_multitask.py`)
- M2: character-level CNN and BiLSTM with attention, trained from scratch, single-task and multi-task
- M3: MuRIL shared encoder + two heads (in `scripts/train_multitask.py`)
- M4: MuRIL encoder + one head over the 14 observed (tone, type) pairs

## Conventions
- Machine: MacBook with Apple Silicon. Use the PyTorch `mps` device when available.
- Python environment lives in `.venv/`. Activate it with `source .venv/bin/activate`.
- Use class-weighted loss for the imbalanced classes.
- Pick the best checkpoint by dev macro F1, not by loss.
- Fix random seeds. Report mean and standard deviation over 5 seeds.
- Save every run's results to the `results/` folder, together with the settings used.
