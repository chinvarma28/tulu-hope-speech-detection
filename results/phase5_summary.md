# Phase 5 summary: combining models, final systems and the test set

Dev scores are **macro F1 on the official dev set** (753 comments). Test scores are
**macro F1 on the repaired test set** (744 comments, `data/test_multitask_clean.csv`),
scored **once**, with `scripts/score_test.py`. Where a ± is given, it is the standard
deviation over 5 seeds (1–5). Joint Score = (tone F1 + type F1) / 2.

## What was done

1. **Probabilities saved for every model.** For dev and for the *unlabelled* test file,
   each model saved the probability of every tone and every hope type, per seed
   (`results/probs/`, format in `scripts/probs_io.py`). This needed new runs:
   - M1: `scripts/m1_probs.py` (same model as the baseline; dev scores identical).
   - CNN multi-task: rerun on the Mac (`train_char.py --test`).
   - M3 multi-task and M4b: rerun on Kaggle (`notebooks/final_probs_kaggle.ipynb`).
   The reruns use the same settings as Phases 2–4. Their scores differ a little from the
   earlier runs, because GPU training is not exactly repeatable even with fixed seeds
   (see "Caveats"). **All numbers below, dev and test, come from these reruns.**
2. **Combinations tried on dev only** (`scripts/combine_dev.py`): averaging each model's
   5 seeds, and averaging the probabilities of 2, 3 or 4 models with equal weights.
3. **Final systems fixed on dev**, and test predictions written and committed to git
   (commit `733f2ae`) **before** the test labels were used
   (`scripts/make_predictions.py`).
4. **Test set scored once** (`scripts/score_final.py`, which calls `score_test.py` for each
   predictions file).
5. **The dev-to-test drop analysed** (`scripts/analyse_test.py`). Diagnosis only: nothing
   was changed or re-chosen after scoring.

## Reruns compared with the earlier runs (dev)

| Model | Earlier run: tone / type | Rerun: tone / type |
|---|---|---|
| CNN multi-task | 0.547 ± 0.010 / 0.411 ± 0.009 | 0.548 ± 0.019 / 0.412 ± 0.011 |
| M3 multi-task | 0.540 ± 0.015 / 0.426 ± 0.013 | 0.543 ± 0.007 / 0.421 ± 0.010 |
| M4b | 0.523 ± 0.011 / 0.437 ± 0.009 | 0.530 ± 0.014 / 0.427 ± 0.013 |

All differences are within about one standard deviation.

## Combinations on dev

Full table: `results/combine_dev.csv`. "All seeds averaged" = each model averages its 5
seeds first, then the models are averaged. "Per seed" = seed k of every neural model
(plus M1, which has one run), giving 5 scores.

| System | Tone | Type | Joint, all seeds averaged | Joint, per seed |
|---|---|---|---|---|
| M1 alone | 0.579 | 0.400 | 0.489 | 0.489 |
| CNN alone | 0.557 | 0.426 | 0.492 | 0.480 ± 0.009 |
| M3 alone | 0.553 | 0.432 | 0.493 | 0.482 ± 0.006 |
| M4b alone | 0.524 | 0.434 | 0.479 | 0.478 ± 0.013 |
| M1 + CNN | 0.579 | 0.421 | 0.500 | 0.497 ± 0.007 |
| M1 + M3 | 0.551 | 0.439 | 0.495 | 0.498 ± 0.007 |
| M1 + M4b | 0.569 | 0.456 | **0.512** | 0.501 ± 0.012 |
| M1 + CNN + M4b | 0.561 | 0.452 | 0.506 | **0.505 ± 0.007** |
| M1 + M3 + M4b | 0.561 | 0.452 | 0.507 | 0.502 ± 0.012 |
| All four | 0.559 | 0.449 | 0.504 | 0.503 ± 0.007 |

(Tone and type columns are for "all seeds averaged".)

Only equal weights were tried. Tuning one weight per model on 753 dev comments would fit
the dev set itself.

## Choosing the final systems (on dev)

- **The top combinations are within chance of each other.** A one-off bootstrap check
  (1,000 resamples of the 753 dev comments, random seed 0) gave 95% ranges for the Joint
  Score of 0.482–0.542 (M1 + M4b), 0.479–0.535 (M1 + CNN + M4b), 0.475–0.532 (all four)
  and 0.463–0.519 (M3 alone). M1 + M4b was also the best of 11 combinations tried on the
  same dev set, so part of its lead is luck.
- **Declared before scoring test:**
  - **Main system: M3 multi-task.** One MuRIL model predicts both labels in one forward
    pass, as the task asks. Reported per seed (mean ± std) and with the 5 seeds averaged.
  - **Best overall: all four combined** (M1 + CNN + M3 + M4b, equal weights). Chosen
    because it needs no picking between combinations. It is not one forward pass:
    M1 and the CNN are separate models.
  - **For comparison:** M1, CNN multi-task and M4b on their own.

## Test results (scored once)

Source: `results/test_scores.csv`; full per-class reports: `results/test_report.txt`.

| System | Tone | Type | Joint | Dev Joint |
|---|---|---|---|---|
| **M3 multi-task (main), per seed** | 0.389 ± 0.015 | 0.311 ± 0.010 | **0.350 ± 0.003** | 0.482 |
| **M3 multi-task, 5 seeds averaged** | 0.393 | **0.336** | **0.364** | 0.493 |
| All four combined, per seed | 0.375 ± 0.007 | 0.318 ± 0.010 | 0.346 ± 0.008 | 0.503 |
| All four combined, all seeds averaged | 0.366 | 0.316 | 0.341 | 0.504 |
| CNN multi-task, per seed | 0.392 ± 0.020 | 0.306 ± 0.009 | 0.349 ± 0.008 | 0.480 |
| CNN multi-task, 5 seeds averaged | 0.392 | 0.312 | 0.352 | 0.492 |
| M4b pair head, per seed | 0.378 ± 0.007 | 0.303 ± 0.010 | 0.341 ± 0.006 | 0.478 |
| M4b pair head, 5 seeds averaged | 0.370 | 0.316 | 0.343 | 0.479 |
| M1 TF-IDF + LR | 0.363 | 0.289 | 0.326 | 0.489 |

Per-class test F1 (5 seeds averaged; M1 single run):

| Class | M3 | All four | M1 |
|---|---|---|---|
| Encouraging Hope | 0.635 | 0.640 | 0.615 |
| Discouraging Hope | 0.325 | 0.275 | 0.298 |
| Blended Hope | 0.218 | 0.184 | 0.177 |
| Inspiring Hope | 0.562 | 0.564 | 0.506 |
| Hopelessness | 0.472 | 0.461 | 0.413 |
| Realistic Hope | 0.234 | 0.209 | 0.231 |
| Optimistic Hope | 0.231 | 0.190 | 0.191 |
| Fading Hope | 0.180 | 0.155 | 0.106 |
| Both labels right on the same comment | 0.284 | 0.262 | 0.226 |

No class has 0.000 F1 in any test predictions file.

## Findings

1. **M3 multi-task, the declared main system, is the best system on test.** With its
   5 seeds averaged it reaches Joint 0.364; per seed 0.350 ± 0.003, the smallest spread of
   any system. In the per-class table it is best on 6 of the 8 classes; on the other two
   (Encouraging and Inspiring Hope) all four combined is ahead by 0.005 or less.
2. **Combining models helped on dev but not on test.** All four combined was +0.011
   above M3 on dev (all seeds averaged), but −0.023 below it on test. M1 dropped the most
   from dev to test (0.489 → 0.326), and it is part of every strong combination.
3. **Every system loses about 0.13–0.16 Joint Score from dev to test.** The ranking of
   the neural models changes little (M3 ≈ CNN > M4b), but M1 falls from the best single
   model on dev (per seed: 0.489 vs M3 0.482) to the last on test.
4. **Most of the drop comes from one part of the test file.** The labelled test file was
   released as a merge of two parts. In rows 429–744 the two label columns were swapped
   (found and fixed in Phase 1, `scripts/repair_test.py`). The two parts behave very
   differently (`results/test_parts.csv`):

   | Test part | Rows | Tone mix (Enc / Disc / Blended) | M3 Joint | M3 tone accuracy |
   |---|---|---|---|---|
   | Rows 1–428 | 428 | 57 / 22 / 21% (same as train and dev) | 0.410 | 0.55 |
   | Rows 429–744 | 316 | 42 / 56 / 2% | 0.298 | **0.33** |

   On rows 429–744 the tone accuracy of every system is 0.30–0.34, about what random
   guessing gives with 3 classes. The label mix there is also unlike train and dev:
   Blended Hope almost disappears (2% vs 22%) and Fading Hope triples (25% vs 8%).
5. **The labels in rows 429–744 look unreliable, not shifted.** If the labels were moved
   by a few rows, the predictions would agree best with a neighbouring row's label. They
   do not: agreement is 0.31–0.39 at every shift from −5 to +5 (rows 1–428: clear peak
   of 0.55 at shift 0). The repair script copies the raw labels correctly; the raw file
   itself has, for example, "edde aathundu...god bless" and "concept superb & comedy
   superb" as Discouraging Hope, and "Movie name dada?" as Hopelessness.
   `results/test_swapped_examples.csv` lists 52 comments from this part where the label
   and both final systems give opposite tones (Encouraging vs Discouraging), for checking
   by hand. (This file contains dataset comments, so it is not in the public repo; run
   `scripts/analyse_test.py` to rebuild it.)
6. **Even rows 1–428 score below dev** (M3: 0.410 vs 0.493 on dev), so part of the drop
   is an ordinary dev-to-test difference.

## Caveats

- **Test numbers are the full 744 comments.** The split into two parts was made *after*
  scoring, to explain the result. It is not a second official score and was not used to
  choose anything.
- **GPU training is not exactly repeatable.** Even with fixed seeds, the Mac GPU (`mps`)
  and the Kaggle GPU give slightly different models from run to run. For example, the old
  CNN script, run twice on seed 1 on the Mac, gave tone 0.524 and 0.556; on the CPU two
  runs match exactly. The 5-seed means are stable (see the rerun table above).
- **The bootstrap check above was a one-off** (not part of a saved script).
- **Combinations are not one forward pass.** Only single models (one seed of M3, M4b or
  the CNN) predict both labels in one forward pass. Averaging 5 seeds means 5 passes.

## Files

- Probabilities: `results/probs/<run>/{dev,test}_seed<N>.csv`
- Dev combinations: `results/combine_dev.csv` (`scripts/combine_dev.py`)
- Test predictions: `results/predictions/test/*.csv` (`scripts/make_predictions.py`)
- Test scores: `results/test_scores.csv`, `results/test_report.txt` (`scripts/score_final.py`)
- Test analysis: `results/test_parts.csv`, `results/test_swapped_examples.csv`
  (`scripts/analyse_test.py`)
- Kaggle rerun of M3 and M4b: `results/final_run/` (results and logs), using
  `notebooks/final_probs_kaggle.ipynb` on a Tesla T4
