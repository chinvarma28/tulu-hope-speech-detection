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

## Two M4 runs

| Run | Pair weights in the loss | Epochs |
|---|---|---|
| **M4** (first run) | sqrt-balanced over the 14 pair counts (same formula as M2/M3) | 5 |
| **M4b** (second run) | tone weight × type weight ("product") | 8 |

M4b was set up after looking at M4's dev results (see "Why a second run" below). Both
changes were chosen on the dev set only.

## Main table

| Model | Task | Tone F1 | Hope type F1 | Joint Score |
|---|---|---|---|---|
| M1: char TF-IDF + LR (balanced) | two separate models | **0.579** | 0.400 | **0.490** |
| M2: char CNN | multi-task | 0.547 ± 0.010 | 0.411 ± 0.009 | 0.479 |
| M3: MuRIL + two heads | multi-task | 0.540 ± 0.015 | 0.426 ± 0.013 | 0.483 |
| M3: MuRIL | tone only / type only | 0.551 ± 0.015 | 0.434 ± 0.015 | (0.493) |
| M4: MuRIL + pair head | joint (14 pairs) | 0.521 ± 0.012 | 0.418 ± 0.007 | 0.469 |
| **M4b: MuRIL + pair head, product weights, 8 epochs** | joint (14 pairs) | 0.523 ± 0.011 | **0.437 ± 0.009** | 0.480 |

Joint Score = (tone F1 + type F1) / 2. The joint score in brackets pairs the two separate
single-task models.

## Weakest classes (per-class F1 over the 5 seeds: min / mean)

No class reached 0.000 F1 in any of the 10 M4 models.

| Run | Blended (tone) | Discouraging (tone) | Fading Hope (type) | Realistic Hope (type) |
|---|---|---|---|---|
| M1 | 0.508 | 0.434 | 0.206 | 0.235 |
| M3 multi | 0.339 / 0.419 | 0.357 / 0.398 | 0.247 / 0.278 | 0.213 / 0.238 |
| M4 | 0.370 / 0.403 | 0.339 / 0.361 | 0.188 / 0.267 | 0.183 / 0.212 |
| M4b | 0.314 / **0.365** | 0.367 / 0.400 | **0.288 / 0.302** | 0.181 / 0.219 |

## Findings: first run (M4)

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
4. **The model looked under-trained.** The best epoch was the last one (5) in 4 of the 5
   seeds.

## Why a second run (M4b)

- **Product weights.** Macro F1 is computed over the 3 tones and the 5 types, not over the
  14 pairs. So M4b weights each pair by (tone weight × type weight), using the same sqrt
  formula as before. The 1-comment pairs drop from about 33× to about 1–2× the weight of
  the largest pair. The highest weights (about 3.5× the largest pair) now go to the pairs
  with Fading Hope, the rarest hope type.
- **8 epochs**, because M4 was still improving at epoch 5.

## Findings: second run (M4b)

5. **The tiny-pair problem is gone.** In all 5 seeds, not one dev comment was predicted as
   one of the four tiny pairs (M4: 25–40 per seed).
6. **Hope type improves clearly; tone does not.** Type F1 rises from 0.418 to **0.437**
   (about 2 standard deviations), the highest hope-type score of any model so far. It is
   level with single-task M3 (0.434) and well above M1 (0.400). Fading Hope gains most: its
   average (0.302) ties single-task M3, and its worst seed (0.288) is better than the worst
   seed of every other model (next best: single-task M3, 0.264). Tone stays at
   0.523, below M3 (0.540) and M1 (0.579).
7. **The Joint Score (0.480) catches up with M3 multi-task (0.483) but not with M1 (0.490).**
8. **The weights, not the extra epochs, probably made the difference.** Averaged over the
   5 seeds, the dev Joint Score reaches 0.474 at epoch 3 and then stays flat
   (0.463–0.473 for epochs 4–8). The best epochs were 3, 4, 5, 6 and 7. Because both
   settings changed at once, this cannot be proven from one run.
9. **Blended tone is now the weak spot** (mean F1 0.365, the lowest of any MuRIL run). In
   the best seed, Blended + Fading Hope is predicted 80 times (43 true), while the three
   Blended pairs with a hopeful type (Blended + Inspiring / Optimistic / Realistic, 57 true
   comments together) are predicted only 7 times. Discouraging tone improves
   (0.361 → 0.400).
10. **A small caveat on M4b's numbers.** Each seed keeps its best epoch on dev. Picking the
    best of 8 epochs gives slightly more room for a lucky epoch than picking the best of 5.
    The mean of the per-seed best scores (0.480) is above the best per-epoch average
    (0.474). The same kind of selection is used for every model, but it is slightly larger
    here.

## Settings

- Script: `scripts/train_multitask.py --task pairs` (M4) and
  `--task pairs --pair_weights product --epochs 8` (M4b); full settings are in
  `results/m4_muril-base-cased_pairs.json` and
  `results/m4_muril-base-cased_pairs_w-product_ep8.json`
- Same as M3: `google/muril-base-cased`, mean pooling → BatchNorm → dropout 0.2; AdamW
  (weight decay 0.01), MuRIL lr 2e-5, BatchNorm + head lr 1e-3, 10% linear warmup then
  linear decay; batch 16, max 64 word pieces
- Head: one linear layer, 768 → 14 pairs
- Loss: cross-entropy over the 14 pairs with label smoothing 0.1 and pair weights as above
- Best epoch chosen by the dev Joint Score
- Hardware: Kaggle Tesla T4, using `notebooks/m4_muril_pairs_kaggle.ipynb`.
  M4: about 4.7 minutes per seed (23 minutes in all). M4b: about 7 minutes per seed
  (36 minutes in all).
- Logs: `results/logs/m4_muril-base-cased_pairs.log` and
  `results/logs/m4_muril-base-cased_pairs_w-product_ep8.log`. Checkpoints (scores and
  predictions only): `checkpoints/` (not in git)
