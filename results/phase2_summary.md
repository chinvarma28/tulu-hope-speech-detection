# Phase 2 (M2) summary: character CNN and BiLSTM, single-task vs multi-task

All scores are **macro F1 on the official dev set** (753 comments), as mean ± standard
deviation over 5 seeds (1–5). For each seed, the epoch with the best dev score was kept.
M1 is a single deterministic run, so it has no standard deviation.

## Main table

| Model | Task | Tone F1 | Hope type F1 | Joint Score |
|---|---|---|---|---|
| M1: char TF-IDF + LR (balanced) | two separate models | **0.579** | 0.400 | **0.490** |
| M2: char CNN | multi-task | 0.547 ± 0.010 | **0.411 ± 0.009** | 0.479 |
| M2: char CNN | tone only | 0.558 ± 0.013 | — | — |
| M2: char CNN | type only | — | 0.409 ± 0.012 | — |
| M2: BiLSTM + attention | multi-task | 0.513 ± 0.011 | 0.382 ± 0.012 | 0.448 |
| M2: BiLSTM + attention | tone only | 0.523 ± 0.023 | — | — |
| M2: BiLSTM + attention | type only | — | 0.375 ± 0.012 | — |

Joint Score = (tone F1 + type F1) / 2. For comparison, pairing the two single-task models
gives a joint score of 0.484 for the CNN and 0.449 for the BiLSTM.

## Weakest classes (per-class F1 over the 5 seeds: min / mean)

No class reached 0.000 F1 in any of the 30 trained models.

| Run | Blended (tone) | Discouraging (tone) | Fading Hope (type) | Realistic Hope (type) |
|---|---|---|---|---|
| M1 | 0.508 | 0.434 | 0.206 | 0.235 |
| CNN multi | 0.384 / 0.429 | 0.391 / 0.412 | 0.186 / 0.214 | 0.191 / 0.252 |
| CNN single | 0.390 / 0.456 | 0.335 / 0.417 | 0.182 / 0.225 | 0.155 / 0.232 |
| BiLSTM multi | 0.319 / 0.389 | 0.348 / 0.395 | **0.086** / 0.175 | 0.112 / 0.178 |
| BiLSTM single | 0.387 / 0.422 | 0.304 / 0.382 | 0.159 / 0.182 | 0.172 / 0.200 |

## Findings

1. **Neither M2 model beats M1 on tone.** The CNN comes closest (0.558 single-task vs 0.579).
2. **On hope type, the CNN ties M1** (0.411 vs 0.400; the gap is about one standard
   deviation). The BiLSTM is below M1 on hope type (0.382 vs 0.400).
3. **The CNN is better than the BiLSTM on every measure**, by about 0.03 F1.
4. **Multi-task does not help tone.** Multi-task is slightly worse than single-task for both
   models (CNN 0.547 vs 0.558; BiLSTM 0.513 vs 0.523).
5. **Multi-task is about equal for hope type.** It is slightly better (CNN 0.411 vs 0.409;
   BiLSTM 0.382 vs 0.375), but the gaps are smaller than one standard deviation.
6. **No class collapses to 0.000 F1.** Fading Hope and Realistic Hope are the weakest
   classes everywhere. One BiLSTM multi-task seed got only 0.086 on Fading Hope.

**Takeaway:** with 3,365 training comments, character models trained from scratch do no better
than TF-IDF + logistic regression. This motivates M3/M4, which start from a pretrained
encoder (MuRIL).

## Settings (same for all six runs)

- Script: `scripts/train_char.py`; full settings are in each `results/m2_<model>_<task>.json`
- Train: `data/train_clean.csv` (3,365 rows); dev: `data/dev_multitask.csv` (753 rows)
- Lowercased text, at most 150 characters, character vocabulary of 141 (train only)
- Adam, lr 0.001, batch 64, up to 30 epochs, early stopping after 6 epochs without improvement
- Loss: cross-entropy with sqrt-balanced class weights and label smoothing 0.1;
  multi-task loss = 0.5 × tone + 0.5 × type
- Best epoch chosen by dev macro F1 (tone for tone-only, type for type-only, Joint Score for multi-task)
- CNN: embeddings of size 64, 128 filters for each width 2/3/4/5, dropout 0.5
- BiLSTM: embeddings of size 64, 128 hidden units per direction, additive attention, dropout 0.5
- Hardware: CNN on Apple M-series GPU (mps); BiLSTM on Kaggle Tesla T4 (cuda), using
  `notebooks/m2_bilstm_kaggle.ipynb`. Different hardware can shift scores slightly even
  with the same seed; such shifts are well inside the seed-to-seed spread.
- Logs: `results/logs/`. Model checkpoints: `checkpoints/` (not in git, because the files are large).
