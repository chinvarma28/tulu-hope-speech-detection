# Phase 3 (M3) summary: MuRIL shared encoder + two heads

All scores are **macro F1 on the official dev set** (753 comments), as mean ± standard
deviation over 5 seeds (1–5). For each seed, the epoch with the best dev score was kept.
M1 is a single deterministic run, so it has no standard deviation.

## Main table

| Model | Task | Tone F1 | Hope type F1 | Joint Score |
|---|---|---|---|---|
| M1: char TF-IDF + LR (balanced) | two separate models | **0.579** | 0.400 | **0.490** |
| M2: char CNN | multi-task | 0.547 ± 0.010 | 0.411 ± 0.009 | 0.479 |
| M2: char CNN | tone only / type only | 0.558 ± 0.013 | 0.409 ± 0.012 | (0.484) |
| **M3: MuRIL + two heads** | multi-task | 0.540 ± 0.015 | 0.426 ± 0.013 | 0.483 |
| **M3: MuRIL** | tone only | 0.551 ± 0.015 | — | — |
| **M3: MuRIL** | type only | — | **0.434 ± 0.015** | — |

Joint Score = (tone F1 + type F1) / 2. The joint score in brackets pairs the two separate
single-task models; for M3 that pairing gives (0.551 + 0.434) / 2 = 0.493.

## Weakest classes (per-class F1 over the 5 seeds: min / mean)

No class reached 0.000 F1 in any of the 15 M3 models.

| Run | Blended (tone) | Discouraging (tone) | Fading Hope (type) | Realistic Hope (type) |
|---|---|---|---|---|
| M1 | 0.508 | 0.434 | 0.206 | 0.235 |
| CNN multi | 0.384 / 0.429 | 0.391 / 0.412 | 0.186 / 0.214 | 0.191 / 0.252 |
| M3 multi | 0.339 / 0.419 | 0.357 / 0.398 | 0.247 / 0.278 | 0.213 / 0.238 |
| M3 single | 0.403 / 0.435 | 0.355 / 0.414 | 0.264 / **0.302** | 0.187 / 0.233 |

## Findings

1. **M3 is the best model so far for hope type.** Type-only MuRIL reaches 0.434, compared
   with 0.400 for M1 and 0.411 for the CNN. The gap to M1 is about two standard deviations.
   Most of the gain comes from the rarest class, Fading Hope (0.302 vs 0.206 for M1).
2. **M3 does not beat M1 on tone.** At 0.540–0.551 it is about 0.03–0.04 below M1 (0.579),
   at the same level as the CNN.
3. **The Joint Score is level with M1.** It is 0.483 for one multi-task model, and 0.493
   for the two single-task models together, compared with 0.490 for M1.
4. **Multi-task does not help.** Single-task is slightly better on both tone (0.551 vs
   0.540) and hope type (0.434 vs 0.426). Both gaps are under one standard deviation. This
   is the same pattern as in M2.
5. **No class collapses to 0.000.**

## How the M3 settings were chosen (dev set, seed 1 only)

The first MuRIL runs failed to learn: the models predicted only the most common class.
These diagnostics and settings were tried before the final run:

| Try | Settings | Result (seed 1, dev) |
|---|---|---|
| 1 | lr 2e-5 for everything, 5 epochs | multi: tone 0.439, type 0.318; type-only: stuck on the majority class (type 0.103) |
| 2 | lr 5e-5, heads 1e-3, 10 epochs | multi: stuck on the majority class from epoch 1 (tone 0.241, type 0.103) |
| 3 | **lr 2e-5, heads 1e-3, BatchNorm on the pooled vector, 5 epochs** | multi: tone 0.554, type 0.408 → used for the final run |

**Cause.** A check of the untrained MuRIL showed that its mean-pooled vectors are nearly
identical for every comment (average cosine similarity 0.992): one or two of the 768
numbers are large, and the rest are close to 0. The heads therefore received nearly the
same input for every comment and fell back on the majority class. A BatchNorm layer
between MuRIL and the heads rescales each of the 768 numbers, which fixed this. Logistic
regression on standardised, *frozen* MuRIL vectors already reached tone 0.454 / type 0.327,
so the information was present before any fine-tuning.

MuRIL itself loads correctly: it completes "The capital of India is [MASK]." with *Delhi*.

## Final settings (same for all three M3 runs)

- Script: `scripts/train_multitask.py`; full settings are in each `results/m3_muril-base-cased_<task>.json`
- Encoder: `google/muril-base-cased` (237M weights), fine-tuned end to end
- Comment vector: mean of MuRIL's word-piece vectors → BatchNorm → dropout 0.2 → two linear heads
- AdamW (weight decay 0.01): MuRIL lr 2e-5, BatchNorm + heads lr 1e-3; linear warmup for 10% of steps, then linear decay
- 5 epochs, batch 16, max 64 word pieces (only 8 of 3,365 training comments are longer)
- Loss: cross-entropy with sqrt-balanced class weights and label smoothing 0.1;
  multi-task loss = 0.5 × tone + 0.5 × type
- Best epoch chosen by dev macro F1 (tone for tone-only, type for type-only, Joint Score for multi-task)
- No pair mask (`--mask none`)
- Hardware: Kaggle Tesla T4, about 5 minutes per seed, using `notebooks/m3_muril_kaggle.ipynb`
- Logs: `results/logs/`. Checkpoints (scores and predictions only): `checkpoints/` (not in git)
