# Phase 1 (M1) summary: data checks and the character TF-IDF + logistic regression baseline

Source: `results/m1_baseline.txt` (produced by `scripts/explore_multitask.py`).
All scores are **macro F1 on the official dev set** (753 comments). M1 is a single
deterministic run, so there is no standard deviation.

## Baseline results

| Model | Tone F1 | Hope type F1 | Joint Score |
|---|---|---|---|
| Majority class | 0.241 | 0.103 | 0.172 |
| char TF-IDF + LR | 0.574 | 0.380 | 0.477 |
| char TF-IDF + LR (class_weight=balanced) | **0.579** | **0.400** | **0.490** |

The balanced row is the number every later model has to beat.

Weakest classes for M1 (balanced), F1: Fading Hope 0.206, Realistic Hope 0.235,
Discouraging Hope 0.434, Optimistic Hope 0.431. No class had 0.000 F1.

## Data findings

1. **Sizes:** train 3,365 rows, dev 753 rows. No empty texts and no duplicate Ids in either.
2. **Class imbalance (train):**
   - Tone: Encouraging 56.6%, Blended 22.2%, Discouraging 21.2%.
   - Hope type: Inspiring 34.8%, Hopelessness 30.5%, Realistic 15.2%, Optimistic 12.0%, Fading 7.5%.
   - Dev has almost the same proportions, so dev is a fair stand-in for the train distribution.
3. **Tone and hope type are strongly linked.** Only 14 of the 15 possible (tone, type) pairs occur in train;
   Discouraging x Optimistic never does. Inspiring, Optimistic and Realistic are 85–94% Encouraging;
   Fading is 92% Blended; Hopelessness is 67% Discouraging and 33% Blended.
   This is the reason for trying M4 (one head over the observed pairs).
4. **Dev has a few more empty pairs (12 of 15)**, simply because it is smaller.
5. **No leakage:** 0 texts appear in both train and dev. Train has no duplicate texts.
   Dev has 5 duplicate texts, 4 of them carrying conflicting labels, so the labels are somewhat noisy.
6. **Text is short:** median 5 words, 95th percentile 13, max 37. A max length of 64 tokens is plenty.
7. **Script mix (train):** 3,049 Roman only, 212 Kannada only, 104 both.
   Kannada-script text is about 9% of rows, so transliteration normalisation might be worth an ablation.
8. **Formatting problem:** 32 train rows look like stringified Python lists with broken quoting
   (for example `'Nice movie', "Nice film Shaine....& Mamu's team", 'good movie'`).

## Takeaway

TF-IDF + logistic regression already reaches 0.490 Joint Score, far above the majority baseline (0.172).
Hope type is much harder than tone (0.400 vs 0.579), mainly because of Fading Hope and Realistic Hope.
Later phases (M2, M3) are compared against these numbers.

## Settings

- Script: `scripts/explore_multitask.py`
- Train: `data/train_clean.csv`; dev: `data/dev_multitask.csv`
- Features: TF-IDF on character n-grams (`char_wb`, lengths 2–5); classifier: logistic regression
  (C = 4.0, max_iter 2000), with and without `class_weight="balanced"`;
  one separate model per label (tone, hope type).
