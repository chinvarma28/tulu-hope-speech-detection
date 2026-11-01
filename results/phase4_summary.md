# Phase 4 (M4) summary: MuRIL encoder + one head over the (tone, type) pairs

All scores are **macro F1 on the official dev set** (753 comments), as mean ± standard
deviation over 5 seeds (1–5). For each seed, the epoch with the best dev Joint Score was
kept. M1 is a single deterministic run, so it has no standard deviation.

## Idea

M3 predicts tone and hope type with two separate heads, so it can produce a pair that
never occurs, such as Discouraging + Optimistic. M4 has **one head with one output per
(tone, type) pair seen in training**. The predicted pair is split back into a tone and a
type, so M4 is scored exactly like M1–M3.

Of the 15 possible pairs, 14 occur in `train_clean.csv` (Discouraging + Optimistic never
does). Four of them are tiny:

| Pair | Training comments | Dev comments |
|---|---|---|
| Encouraging + Fading Hope | 1 | 2 |
| Encouraging + Hopelessness | 1 | 0 |
| Discouraging + Inspiring Hope | 3 | 2 |
| Discouraging + Realistic Hope | 3 | 0 |

## Main table

| Model | Task | Tone F1 | Hope type F1 | Joint Score |
|---|---|---|---|---|
| M1: char TF-IDF + LR (balanced) | two separate models | **0.579** | 0.400 | **0.490** |
| M2: char CNN | multi-task | 0.547 ± 0.010 | 0.411 ± 0.009 | 0.479 |
| M3: MuRIL + two heads | multi-task | 0.540 ± 0.015 | 0.426 ± 0.013 | 0.483 |
| M3: MuRIL | tone only / type only | 0.551 ± 0.015 | **0.434 ± 0.015** | (0.493) |
| **M4: MuRIL + one pair head** | joint (14 pairs) | 0.521 ± 0.012 | 0.418 ± 0.007 | 0.469 |

Joint Score = (tone F1 + type F1) / 2. The joint score in brackets pairs the two separate
single-task models.

## Weakest classes (per-class F1 over the 5 seeds: min / mean)

No class reached 0.000 F1 in any of the 5 M4 models.

| Run | Blended (tone) | Discouraging (tone) | Fading Hope (type) | Realistic Hope (type) |
|---|---|---|---|---|
| M1 | 0.508 | 0.434 | 0.206 | 0.235 |
| M3 multi | 0.339 / 0.419 | 0.357 / 0.398 | 0.247 / 0.278 | 0.213 / 0.238 |
| M4 pairs | 0.370 / 0.403 | 0.339 / **0.361** | 0.188 / 0.267 | 0.183 / 0.212 |

## Findings

1. **M4 is below M3 on both tasks.** Tone 0.521 vs 0.540 (about 1.5 standard deviations
   lower) and hope type 0.418 vs 0.426 (under one standard deviation). The Joint Score
   drops from 0.483 to 0.469, below M1 (0.490).
2. **The tiny pairs are over-predicted, and those predictions are always wrong.** In every
   seed, 25–40 dev comments (3–5%) were predicted as one of the four tiny pairs, and not
   one of them was correct. Most of them were Encouraging + Fading Hope (13–17 per seed)
   and Encouraging + Hopelessness (10–22 per seed), each of which has only 1 training
   comment. The likely cause is the class-weighted loss: with sqrt-balanced weights over
   14 pairs, a pair with 1 comment gets about 33 times the weight of the largest pair
   (Encouraging + Inspiring, 1,105 comments).
3. **These wrong predictions mostly hurt hope type.** Over the 5 seeds, 151 dev comments
   were predicted as Encouraging + Fading Hope or Encouraging + Hopelessness. 85 of them were
   really Encouraging comments with a hopeful type (mostly Inspiring or Optimistic), so the
   tone was right but the type was turned into Fading Hope or Hopelessness. The other 66
   were Blended (42) or Discouraging (24), so the tone was wrong too.
   Discouraging Hope has the lowest tone F1 of any MuRIL run (mean 0.361); these predictions
   explain only a small part of that.
4. **The model may be under-trained.** The best epoch was the last one (5) in 4 of the 5
   seeds, and the dev score was still rising. A 14-way head has fewer examples per output
   than a 3-way or 5-way head, so it may need more epochs.
5. **The pair head does what it was built for:** it never predicts Discouraging + Optimistic,
   and its seed-to-seed spread on hope type is the smallest of all models (± 0.007).

## Possible next steps (choose with the dev set only)

- Soften the pair weights (for example, cap them, or use the product of the tone and type
  weights instead of the pair counts), so the 1-comment pairs stop attracting predictions.
- Train for more epochs (e.g. 8), since most seeds peaked at the last epoch.

## Settings

- Script: `scripts/train_multitask.py --task pairs`; full settings are in
  `results/m4_muril-base-cased_pairs.json`
- Same as M3: `google/muril-base-cased`, mean pooling → BatchNorm → dropout 0.2; AdamW
  (weight decay 0.01), MuRIL lr 2e-5, BatchNorm + head lr 1e-3, 10% linear warmup then
  linear decay; 5 epochs, batch 16, max 64 word pieces
- Head: one linear layer, 768 → 14 pairs
- Loss: cross-entropy over the 14 pairs, sqrt-balanced pair weights, label smoothing 0.1
- Best epoch chosen by the dev Joint Score
- Hardware: Kaggle Tesla T4, about 4.7 minutes per seed (23 minutes in all), using
  `notebooks/m4_muril_pairs_kaggle.ipynb`
- Log: `results/logs/m4_muril-base-cased_pairs.log`. Checkpoints (scores and predictions
  only): `checkpoints/` (not in git)
