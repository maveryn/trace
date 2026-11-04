# TRACE–VERO merging study: findings and data

Study `trace_vero_qwen25_v1`, complete as of 2026-11-03. It compares the released TRACE and VERO Qwen2.5-VL-7B checkpoints with their shared backbone, then tests three training-free 50:50 weight merges. Nothing was trained.

This file collects the study's key findings and its data in one place.
- **Interpretation, caveats and the decision.** These are in `analysis_report.md` and `decision_memo.md`, in this directory. The frozen design is `../../DESIGN.md`.
- **Where the numbers come from.** Numbers are read programmatically from the aggregate tables in `../tables/`, and each section names its source. §6.2 also uses the runner summaries. Two values are quoted from the A8 record instead: the pre-repair M050 − T estimate and the untimed A40 attempt.
- **Rounding.** Two decimals for percentage-point values and one for rates, rounded half-up.

## How to read the numbers

**Models.**
- B: Qwen2.5-VL-7B-Instruct, the base.
- T: the released TRACE model.
- V: the released VERO model. V@P0 is VERO with its own think/answer chat template (the archived canonical runs); V@P1 is VERO with the backbone's template.
- M050 = ½V + ½T, CV050 = ½V + ½B, CT050 = ½T + ½B. Unsuffixed merge labels are served with the backbone's interface (P1).
- Labels ending @P2 (merges, B and T) are served with VERO's interface. That is a development-only diagnostic.

**Metric.** The macro is the unweighted mean of the 24 official benchmark scores, averaged over seeds.

**Intervals and tests.** Every interval is a 95% paired cluster bootstrap: 10,000 replicates over the bootstrap units (image/problem groups, with near-duplicate twins joined; DESIGN.md A7), with all models and seeds resampled together. p-values are two-sided bootstrap p-values; with 10,000 replicates the smallest attainable p is 0.0002, shown as < 0.001.
- Seeds 42–44 are decoding seeds. The intervals cover item and decoding variability for these fixed checkpoints, not training-seed variability: each recipe is one released checkpoint.
- Only the four comparisons of the prespecified primary family are confirmatory and Holm-adjusted.
- Everything else is descriptive or exploratory. That includes categories, benchmarks, capture/damage, sensitivity, the development screen and P2.

**Tiers.**
- Stage B: the archived runs on the full split (32,805 rows, seeds 42–44).
- Development screen: 2,910 rows, seed 42.
- Confirmation: 26,228 rows, seeds 42–44.

## 1. Key findings

1. **TRACE and VERO gain similar amounts over the base** (Stage B, native interfaces).
   - T − B = +4.06 [+3.66, +4.45] and V@P0 − B = +4.30 [+3.78, +4.82].
   - Their difference, T − V@P0 = −0.24 [−0.75, +0.28], is not resolved.
   - Exploratory category contrasts put TRACE ahead in Charts & Tables (+2.42 [+1.40, +3.43]) and VERO ahead in Visual Math (−2.02 [−3.15, −0.86]).
2. **The parents' unique successes are partly reproducible.**
   - On a single seed they disagree on 21.8% of binary-scored rows.
   - A T-only success set recurs on another decoding seed 47.1% of the time (label-swap null 26.9%). A V-only set recurs 50.0% of the time (null 26.9%).
3. **VERO's gain depends on its interface.**
   - With the backbone template, V@P1 − V@P0 = −2.94 [−3.33, −2.56] on confirmation.
   - 66% of that macro drop is in PuzzleVQA (−33.10) and VisualPuzzles (−13.29). There, many V@P1 answers end in a LaTeX-wrapped option letter that the pinned scorers do not read: 32.0% of its PuzzleVQA rows are scored 0 with the correct final letter.
   - TableVQABench (−15.97) is another 23% and is not diagnosed.
4. **The merge beats both controls and VERO-on-the-backbone-template, but is not shown to beat TRACE**
   (confirmation, Holm-adjusted):
   - M050 − T = +0.23 [−0.19, +0.65], Holm p = 0.29.
   - M050 − V@P1 = +3.07 [+2.62, +3.51].
   - M050 − CV050 = +1.29 [+0.93, +1.64].
   - M050 − CT050 = +1.88 [+1.45, +2.31].
   - The last three have Holm p < 0.001. No noninferiority margin was set.
5. **M050 scores 52.36**, against T 52.13 and native VERO 52.23. M050 − V@P0 is a difference of point estimates outside the primary family; no paired interval was computed.
6. **Halving each update costs differently.**
   - CT050 keeps 59% of TRACE's gain over the base.
   - CV050 scores 1.77 pp above VERO itself under P1, consistent with half of VERO's update softening its answer-format shift.
7. **Category pattern (descriptive).**
   - M050 is below T only in Charts & Tables with an interval excluding 0 (−0.91 [−1.76, −0.07]).
   - It is above T in Science & General (+1.15 [+0.24, +2.04]) and Spatial Reasoning (+1.13 [+0.14, +2.15]).
   - VERO's half-update on top of the TRACE blend gains most in Visual Math (+3.73 [+2.69, +4.78]).
   - TRACE's half-update on top of the VERO blend gains most in Science & General (+2.33 [+1.52, +3.15]).
8. **Row-level capture** (independent repeat; the base reference is post hoc, with no paired intervals):
   - M050 keeps VERO-only successes at 52.8%, against the base model's 36.7% on the same rows.
   - It keeps TRACE-only successes at 49.0%, close to the base model's 47.6%.
9. **Frozen hypotheses.**
   - H1 (integration beyond shrinkage) holds in direction.
   - H2 (the category trade-off transfers) is half supported.
   - H3 (interface dependence) is supported for the macro and for Puzzles & Logic. It is not supported for Visual Math being one of the two largest drops: Charts & Tables drops more (−3.89 [−4.73, −3.04], through TableVQABench), although Visual Math also drops (−1.77 [−2.68, −0.90]).
10. **One technical repair** (DESIGN amendments A8, A9). One confirmation run's judge (Qwen3-32B at tensor-parallel 4 on A800 cards) produced degraded extractions. The run was rescored at tensor-parallel 1, and every table here uses the repaired scores (§6).
11. **Decision** (`decision_memo.md`).
    - Do not adopt a merge as an improvement.
    - Stop the merge search.
    - Do not train.
    - One exploratory, evaluation-only follow-up is proposed. It needs a DESIGN amendment, researcher approval and GPU authorization.

## 2. Runs

Every new generation run, with its final attempt's hardware. Source: `../tables/evaluation_costs.csv` and each run's `timing.seed<S>.json`.
- B, T and V@P0 are not re-run; they come from the archive.
- GPU-hours and the attempts column count the 41 attempts with a timed phase. The timing files record 47 attempts; 6 stopped right after starting, with no timed phase (smoke M050, smoke V@P1, V@P1 confirmation seed 44).
- Generated tokens are counted from the ingested responses.

| tier | model | seed | rows | attempts | final attempt: host, GPUs | GPU-hours | generated tokens |
|---|---|---|---|---|---|---|---|
| smoke | CT050 | 42 | 135 | 1 | notch426, 3x NVIDIA A40 44GiB | 0.80 | 25,976 |
| smoke | CV050 | 42 | 135 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.25 | 29,439 |
| smoke | M050 | 42 | 135 | 4 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.38 | 26,398 |
| smoke | V@P1 | 42 | 135 | 2 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.17 | 27,209 |
| development screen | CT050 | 42 | 2,910 | 1 | notch426, 3x NVIDIA A40 44GiB | 1.39 | 608,038 |
| development screen | CT050@P2 | 42 | 2,910 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.44 | 761,700 |
| development screen | CV050 | 42 | 2,910 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.44 | 599,284 |
| development screen | CV050@P2 | 42 | 2,910 | 1 | notch426, 2x NVIDIA A40 44GiB | 1.09 | 722,222 |
| development screen | M050 | 42 | 2,910 | 1 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.24 | 651,122 |
| development screen | M050@P2 | 42 | 2,910 | 1 | notch369, 2x NVIDIA A100 80GB PCIe 80GiB | 0.90 | 793,755 |
| development screen | B@P2 | 42 | 2,910 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 1.29 | 718,888 |
| development screen | T@P2 | 42 | 2,910 | 1 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 0.44 | 812,149 |
| development screen | V@P1 | 42 | 2,910 | 1 | grn067, 1x NVIDIA H200 NVL 140GiB | 0.24 | 601,447 |
| confirmation | CT050 | 42 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 6.78 | 5,475,899 |
| confirmation | CT050 | 43 | 26,228 | 2 | notch369, 2x NVIDIA A100 80GB PCIe 80GiB | 3.15 | 5,480,405 |
| confirmation | CT050 | 44 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 3.80 | 5,388,013 |
| confirmation | CV050 | 42 | 26,228 | 2 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 2.42 | 5,417,796 |
| confirmation | CV050 | 43 | 26,228 | 2 | notch426, 3x NVIDIA A40 44GiB | 7.18 | 5,475,198 |
| confirmation | CV050 | 44 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 5.07 | 5,298,910 |
| confirmation | M050 | 42 | 26,228 | 1 | grn073, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 2.42 | 5,911,585 |
| confirmation | M050 | 43 | 26,228 | 2 | grn067, 1x NVIDIA H200 NVL 140GiB | 3.97 | 5,994,008 |
| confirmation | M050 | 44 | 26,228 | 2 | grn072, 2x NVIDIA RTX PRO 6000 Blackwell 95GiB (scoring redone, A8) | 4.88 | 5,904,482 |
| confirmation | V@P1 | 42 | 26,228 | 2 | grn013, 2x NVIDIA RTX 6000 Ada Generation 47GiB | 5.29 | 5,370,273 |
| confirmation | V@P1 | 43 | 26,228 | 2 | grn074, 1x NVIDIA RTX PRO 6000 Blackwell 95GiB | 2.45 | 5,428,293 |
| confirmation | V@P1 | 44 | 26,228 | 2 | grn067, 1x NVIDIA H200 NVL 140GiB | 2.09 | 5,345,857 |

Totals: 57.57 recorded GPU-hours over 41 attempts (15 failed or incomplete, all resumed) and 72,868,346 generated tokens. Merged weights take 46.34 GiB; run directories take 5.33 GiB.

One attempt is not in the timing files: M050 confirmation seed 44 first generated all its rows on 2× A40 for about 2.3 hours (about 4.6 GPU-hours) before its allocation expired. Its rows were kept.

## 3. Stage B: TRACE and VERO against the base (archived runs, full split, seeds 42–44)

### 3.1 Official scores (`stage_b/official_scores.all.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 47.93 [47.41, 48.45] | 54.45 | 43.03 | 51.00 | 54.95 | 48.26 | 35.87 |
| T | 51.99 [51.43, 52.52] | 59.60 | 48.68 | 55.55 | 57.20 | 51.03 | 39.85 |
| V@P0 | 52.22 [51.62, 52.83] | 57.18 | 50.70 | 55.48 | 57.59 | 51.28 | 41.11 |

### 3.2 Contrasts (`stage_b/differences.all.csv`; categories exploratory, unadjusted)

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| T − B | +4.06 [+3.66, +4.45] | < 0.001 | +5.15 [+4.27, +5.99] | +5.65 [+4.64, +6.66] | +4.55 [+3.64, +5.49] | +2.25 [+1.32, +3.17] | +2.77 [+1.92, +3.65] | +3.99 [+2.81, +5.14] |
| V@P0 − B | +4.30 [+3.78, +4.82] | < 0.001 | +2.73 [+1.73, +3.73] | +7.67 [+6.54, +8.77] | +4.48 [+3.31, +5.64] | +2.64 [+1.25, +4.07] | +3.02 [+1.64, +4.38] | +5.24 [+3.79, +6.71] |
| T − V@P0 | −0.24 [−0.75, +0.28] | 0.36 | +2.42 [+1.40, +3.43] | −2.02 [−3.15, −0.86] | +0.07 [−0.95, +1.13] | −0.39 [−1.82, +0.98] | −0.25 [−1.64, +1.16] | −1.25 [−2.73, +0.22] |

### 3.3 Per-benchmark scores and T − V@P0 (`stage_b/official_scores.all.csv`, `stage_b/differences.all.csv`; exploratory, unadjusted)

| category | benchmark | B | T | V@P0 | T − V@P0 [95% CI] |
|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 45.81 | 48.05 | 40.87 | +7.18 [+5.07, +9.33] |
| Charts & Tables | CharXivReason | 39.73 | 47.13 | 45.83 | +1.30 [−0.97, +3.60] |
| Charts & Tables | TableVQABench | 75.20 | 78.31 | 78.47 | −0.16 [−1.80, +1.48] |
| Charts & Tables | EvoChart | 57.07 | 64.91 | 63.55 | +1.36 [−0.65, +3.32] |
| Visual Math | MathVision | 24.81 | 27.42 | 28.15 | −0.72 [−1.94, +0.45] |
| Visual Math | MathVista | 68.67 | 73.37 | 76.80 | −3.43 [−5.38, −1.48] |
| Visual Math | MathVerse | 43.44 | 47.76 | 50.93 | −3.17 [−5.61, −0.71] |
| Visual Math | WeMath | 35.20 | 46.16 | 46.92 | −0.76 [−3.84, +2.41] |
| Science & General | PhyX mini MC | 40.97 | 48.70 | 46.63 | +2.07 [−0.26, +4.49] |
| Science & General | MMMU-ProVis | 35.70 | 39.36 | 39.67 | −0.31 [−1.96, +1.37] |
| Science & General | RealWorldQA | 65.45 | 68.50 | 69.85 | −1.35 [−3.83, +1.16] |
| Science & General | MMStar | 61.89 | 65.64 | 65.78 | −0.13 [−1.82, +1.52] |
| Spatial Reasoning | EmbSpatial | 69.42 | 70.95 | 70.60 | +0.35 [−0.81, +1.46] |
| Spatial Reasoning | SpatialVizBench COT | 35.14 | 35.68 | 32.06 | +3.62 [+0.86, +6.36] |
| Spatial Reasoning | CV-Bench 3D | 76.25 | 81.00 | 83.11 | −2.11 [−4.10, −0.17] |
| Spatial Reasoning | ERQA | 39.00 | 41.17 | 44.58 | −3.42 [−7.80, +0.96] |
| Perception & Counting | BLINK | 53.31 | 56.57 | 57.51 | −0.95 [−2.87, +1.02] |
| Perception & Counting | CountBenchQA | 82.14 | 84.80 | 82.96 | +1.85 [−0.91, +4.68] |
| Perception & Counting | CountQA | 19.59 | 22.51 | 23.17 | −0.65 [−2.72, +1.32] |
| Perception & Counting | TreeBench | 38.02 | 40.25 | 41.48 | −1.23 [−5.18, +2.63] |
| Puzzles & Logic | PuzzleVQA | 44.57 | 50.02 | 49.22 | +0.80 [−1.17, +2.78] |
| Puzzles & Logic | VisualPuzzles | 31.08 | 34.02 | 36.96 | −2.94 [−5.37, −0.50] |
| Puzzles & Logic | LogicVista | 42.65 | 47.35 | 47.65 | −0.30 [−4.71, +4.13] |
| Puzzles & Logic | MME-Reasoning | 25.17 | 28.03 | 30.61 | −2.58 [−4.75, −0.40] |

### 3.4 Row-level behaviour (21 binary-eligible benchmarks, 27,617 rows per seed; mean over seeds, rows pooled)

Corrections and regressions against B (`stage_b/corrections_regressions.csv`):

| model | correction rate (% of rows) | regression rate | net [95% CI] | corrections / B errors | regressions / B successes |
|---|---|---|---|---|---|
| T | 12.31 | 8.66 | +3.65 [+3.28, +4.01] | 23.2% | 18.4% |
| V@P0 | 14.52 | 10.49 | +4.03 [+3.57, +4.51] | 27.4% | 22.3% |

Eight outcome cells, % of rows (`stage_b/joint_outcomes.csv`):

|  | B = 0 | B = 1 |
|---|---|---|
| T = 0, V@P0 = 0 | 33.08 | 5.21 |
| T = 0, V@P0 = 1 | 7.63 | 3.45 |
| T = 1, V@P0 = 0 | 5.42 | 5.28 |
| T = 1, V@P0 = 1 | 6.89 | 33.03 |

Parent overlap (`stage_b/parent_overlap.csv`):
- T-only 10.7%, V-only 11.1%, both correct 39.9%, both wrong 38.3%; agreement 78.2%.
- The hindsight "either parent correct" rate is 61.7%. It uses two responses and correctness labels, so it is neither a router nor a bound on a merge.

Persistence of outcome sets across seeds (`stage_b/persistence.csv`; defined on one seed, checked on another, all ordered seed pairs):

| set | persists (%) | label-swap null (%) | excess (pp) [95% CI] |
|---|---|---|---|
| T-only (T=1, V@P0=0) | 47.1 | 26.9 | +20.2 [+19.2, +21.3] |
| V-only (T=0, V@P0=1) | 50.0 | 26.9 | +23.1 [+22.1, +24.1] |
| T corrections (B=0, T=1) | 32.9 | 20.5 | +12.4 [+11.6, +13.2] |
| V corrections (B=0, V@P0=1) | 49.3 | 25.2 | +24.0 [+23.2, +24.8] |
| T regressions (B=1, T=0) | 23.3 | 20.5 | +2.8 [+1.8, +3.8] |
| V regressions (B=1, V@P0=0) | 39.7 | 25.2 | +14.4 [+13.4, +15.4] |

## 4. Development screen (seed 42, 2,910 rows; descriptive)

### 4.1 Scores under P1 (`screen/scores.P1.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 47.97 [46.18, 49.76] | 53.13 | 42.54 | 49.61 | 60.30 | 47.62 | 34.64 |
| T | 51.40 [49.56, 53.24] | 60.09 | 45.90 | 54.88 | 60.48 | 49.11 | 37.97 |
| V@P0 | 53.11 [51.19, 55.08] | 57.61 | 49.06 | 56.45 | 59.41 | 54.60 | 41.55 |
| V@P1 | 48.44 [46.58, 50.39] | 51.92 | 47.50 | 54.69 | 58.30 | 51.03 | 27.23 |
| M050 | 50.38 [48.50, 52.24] | 57.13 | 47.54 | 53.71 | 58.24 | 49.15 | 36.49 |
| CV050 | 49.53 [47.66, 51.42] | 53.38 | 44.96 | 54.10 | 58.52 | 48.51 | 37.69 |
| CT050 | 48.51 [46.64, 50.37] | 56.71 | 42.30 | 52.93 | 58.74 | 46.63 | 33.77 |

### 4.2 Contrasts under P1 (`screen/differences.P1.csv`)

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| M050 − T | −1.03 [−2.57, +0.54] | 0.20 | −2.95 [−6.52, +0.57] | +1.64 [−3.13, +6.35] | −1.17 [−4.38, +1.93] | −2.23 [−6.15, +1.58] | +0.04 [−3.01, +3.34] | −1.48 [−5.62, +2.76] |
| M050 − V@P1 | +1.93 [+0.34, +3.45] | 0.02 | +5.22 [+2.05, +8.43] | +0.04 [−4.53, +4.52] | −0.98 [−4.17, +2.23] | −0.06 [−3.93, +3.87] | −1.88 [−5.72, +1.62] | +9.27 [+4.71, +13.66] |
| M050 − CV050 | +0.85 [−0.67, +2.35] | 0.26 | +3.75 [+1.03, +6.61] | +2.58 [−2.53, +7.73] | −0.39 [−3.40, +2.68] | −0.28 [−3.31, +2.79] | +0.64 [−2.18, +3.57] | −1.20 [−5.83, +3.30] |
| M050 − CT050 | +1.86 [+0.20, +3.54] | 0.03 | +0.43 [−3.34, +4.21] | +5.23 [+0.45, +9.95] | +0.78 [−2.30, +3.88] | −0.50 [−4.67, +3.51] | +2.52 [−0.71, +6.18] | +2.72 [−2.28, +7.81] |
| CV050 − B | +1.55 [−0.15, +3.31] | 0.07 | +0.25 [−3.43, +4.01] | +2.42 [−2.50, +7.35] | +4.49 [+0.69, +8.38] | −1.78 [−5.89, +2.29] | +0.89 [−2.54, +4.52] | +3.05 [−2.06, +8.37] |
| CT050 − B | +0.54 [−1.08, +2.16] | 0.50 | +3.58 [+0.37, +6.78] | −0.23 [−5.09, +4.59] | +3.32 [−0.33, +7.00] | −1.56 [−5.07, +1.98] | −1.00 [−3.74, +1.77] | −0.87 [−6.22, +4.55] |
| T − B | +3.43 [+1.70, +5.18] | < 0.001 | +6.95 [+3.03, +10.90] | +3.36 [−1.37, +8.18] | +5.27 [+1.21, +9.53] | +0.17 [−3.63, +3.97] | +1.48 [−1.83, +4.99] | +3.33 [−1.95, +8.44] |
| V@P1 − B | +0.47 [−1.29, +2.37] | 0.58 | −1.22 [−5.28, +2.97] | +4.96 [+0.24, +9.79] | +5.08 [+1.19, +8.91] | −2.00 [−6.94, +2.73] | +3.41 [−0.91, +8.32] | −7.41 [−12.16, −2.53] |
| V@P1 − V@P0 | −4.67 [−6.09, −3.21] | < 0.001 | −5.69 [−9.23, −2.29] | −1.56 [−6.02, +2.95] | −1.76 [−5.07, +1.55] | −1.11 [−3.76, +1.41] | −3.57 [−6.59, −0.43] | −14.32 [−18.33, −10.18] |

### 4.3 Scores under P2, VERO's interface (`screen/scores.P2.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B@P2 | 42.71 [40.94, 44.44] | 39.64 | 33.01 | 46.09 | 57.55 | 47.93 | 32.02 |
| T@P2 | 47.21 [45.44, 49.05] | 43.78 | 39.77 | 52.93 | 59.34 | 49.29 | 38.16 |
| V@P0 | 53.11 [51.19, 55.08] | 57.61 | 49.06 | 56.45 | 59.41 | 54.60 | 41.55 |
| M050@P2 | 52.34 [50.46, 54.29] | 59.77 | 47.89 | 54.30 | 60.83 | 52.10 | 39.14 |
| CV050@P2 | 52.98 [51.05, 54.90] | 60.54 | 47.27 | 58.40 | 58.55 | 50.87 | 42.24 |
| CT050@P2 | 46.80 [44.96, 48.63] | 44.49 | 41.17 | 53.13 | 61.47 | 46.77 | 33.77 |

### 4.4 Contrasts under P2 (`screen/differences.P2.csv`)

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| M050@P2 − V@P0 | −0.77 [−2.40, +0.86] | 0.36 | +2.16 [−2.45, +6.66] | −1.17 [−5.94, +3.41] | −2.15 [−5.52, +1.20] | +1.41 [−2.49, +5.31] | −2.50 [−5.99, +1.12] | −2.41 [−6.39, +1.46] |
| M050@P2 − CV050@P2 | −0.64 [−2.13, +0.87] | 0.43 | −0.77 [−4.80, +3.29] | +0.63 [−3.35, +4.75] | −4.10 [−7.16, −0.96] | +2.28 [−1.13, +5.70] | +1.23 [−1.99, +4.64] | −3.10 [−7.41, +1.20] |
| M050@P2 − CT050@P2 | +5.54 [+3.61, +7.52] | < 0.001 | +15.28 [+10.73, +19.80] | +6.72 [+1.72, +11.95] | +1.17 [−2.68, +5.10] | −0.64 [−5.81, +4.47] | +5.33 [+0.68, +10.28] | +5.37 [+0.35, +10.36] |
| M050@P2 − T@P2 | +5.13 [+3.36, +6.93] | < 0.001 | +16.00 [+11.54, +20.39] | +8.13 [+3.34, +12.94] | +1.37 [−2.35, +5.12] | +1.49 [−2.83, +5.77] | +2.81 [−1.71, +7.61] | +0.98 [−3.34, +5.38] |
| T@P2 − B@P2 | +4.50 [+2.67, +6.42] | < 0.001 | +4.13 [+0.23, +8.07] | +6.76 [+1.37, +12.22] | +6.84 [+2.63, +11.09] | +1.79 [−2.73, +6.28] | +1.37 [−3.35, +6.10] | +6.14 [+1.30, +11.10] |
| V@P0 − B@P2 | +10.41 [+8.40, +12.44] | < 0.001 | +17.96 [+13.26, +22.59] | +16.05 [+10.68, +21.56] | +10.35 [+5.79, +14.91] | +1.86 [−3.47, +7.15] | +6.68 [+2.14, +11.46] | +9.52 [+4.42, +14.86] |

### 4.5 Capture and damage, same-seed parents (`screen/merge_capture_damage.csv`)

| merge | TRACE-only capture (n = 313) | VERO-only capture (n = 256) | shared-success damage (n = 974) | joint-failure rescue (n = 982) |
|---|---|---|---|---|
| M050 | 51.4 [45.7, 57.3] | 52.7 [46.5, 59.0] | 7.3 [5.7, 9.0] | 6.8 [5.3, 8.5] |
| CV050 | 44.4 [38.5, 50.2] | 53.5 [47.4, 59.6] | 10.4 [8.5, 12.4] | 10.6 [8.6, 12.7] |
| CT050 | 56.9 [51.2, 62.6] | 27.0 [21.7, 32.6] | 12.0 [10.0, 14.1] | 11.8 [9.8, 14.0] |

### 4.6 Answer format (`screen/answer_format_sensitivity.csv`; % of rows)

| model | PuzzleVQA accuracy | PuzzleVQA: ends in \( X \) | PuzzleVQA: scored 0, final letter correct | VisualPuzzles accuracy | VisualPuzzles: ends in \( X \) | VisualPuzzles: scored 0, final letter correct |
|---|---|---|---|---|---|---|
| B | 43.8 | 0.0 | 3.9 | 25.8 | 1.6 | 0.0 |
| T | 54.7 | 0.0 | 3.1 | 28.9 | 0.8 | 0.8 |
| V@P0 | 52.3 | 0.0 | 0.0 | 33.6 | 0.0 | 0.0 |
| V@P1 | 15.6 | 30.5 | 37.5 | 18.8 | 21.1 | 9.4 |
| M050 | 46.9 | 1.6 | 7.0 | 24.2 | 7.0 | 3.1 |
| CV050 | 50.8 | 0.0 | 2.3 | 28.1 | 3.1 | 1.6 |
| CT050 | 50.8 | 0.0 | 0.8 | 24.2 | 0.0 | 1.6 |
| B@P2 | 46.9 | 0.0 | 0.0 | 18.0 | 0.0 | 0.0 |
| T@P2 | 52.3 | 0.0 | 0.0 | 25.0 | 0.0 | 0.0 |
| M050@P2 | 59.4 | 0.0 | 0.0 | 25.0 | 0.0 | 0.0 |
| CV050@P2 | 57.0 | 0.0 | 0.0 | 33.6 | 0.0 | 0.0 |
| CT050@P2 | 47.7 | 0.0 | 0.0 | 26.6 | 0.0 | 0.0 |

## 5. Confirmation (P1, 26,228 rows, seeds 42–44)

The confirmed set, the primary family and every input hash were locked at 2026-11-02T09:04:27+00:00 (`selection_lock.json`) before any confirmation run had scores.

### 5.1 Official scores (`confirmation/scores.csv`)

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 48.10 [47.53, 48.68] | 54.27 | 42.92 | 51.30 | 54.94 | 48.79 | 36.38 |
| T | 52.13 [51.51, 52.74] | 59.38 | 48.69 | 55.78 | 56.89 | 51.60 | 40.41 |
| V@P0 | 52.23 [51.56, 52.90] | 57.36 | 50.41 | 55.70 | 57.59 | 51.34 | 41.00 |
| V@P1 | 49.29 [48.64, 49.94] | 53.48 | 48.64 | 55.89 | 57.44 | 50.50 | 29.80 |
| M050 | 52.36 [51.73, 52.97] | 58.47 | 49.08 | 56.94 | 58.02 | 51.93 | 39.69 |
| CV050 | 51.06 [50.45, 51.68] | 56.89 | 47.55 | 54.60 | 56.78 | 51.38 | 39.17 |
| CT050 | 50.48 [49.88, 51.07] | 57.32 | 45.35 | 54.75 | 56.01 | 50.63 | 38.82 |

### 5.2 Macro per seed (`confirmation/seed_scores.csv`; unweighted mean of the 24 benchmarks)

| model | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|
| B | 48.17 | 48.31 | 47.83 | 48.10 |
| T | 52.33 | 51.89 | 52.16 | 52.13 |
| V@P0 | 52.23 | 52.17 | 52.30 | 52.23 |
| V@P1 | 49.22 | 49.33 | 49.32 | 49.29 |
| M050 | 52.48 | 52.32 | 52.27 | 52.36 |
| CV050 | 50.83 | 51.16 | 51.19 | 51.06 |
| CT050 | 50.54 | 50.44 | 50.45 | 50.48 |

### 5.3 Per-benchmark scores, mean over seeds (`confirmation/scores.csv`)

| category | benchmark | B | T | V@P0 | V@P1 | M050 | CV050 | CT050 |
|---|---|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | 45.70 | 48.27 | 40.91 | 41.34 | 49.99 | 47.43 | 48.15 |
| Charts & Tables | CharXivReason | 39.50 | 46.33 | 45.83 | 45.79 | 47.54 | 45.00 | 43.75 |
| Charts & Tables | TableVQABench | 74.99 | 78.37 | 78.67 | 62.70 | 70.74 | 71.29 | 75.75 |
| Charts & Tables | EvoChart | 56.89 | 64.56 | 64.03 | 64.06 | 65.63 | 63.83 | 61.63 |
| Visual Math | MathVision | 24.23 | 26.67 | 27.66 | 26.64 | 28.17 | 27.70 | 26.58 |
| Visual Math | MathVista | 69.33 | 74.50 | 77.96 | 76.21 | 77.46 | 75.58 | 71.54 |
| Visual Math | MathVerse | 42.80 | 47.25 | 49.58 | 47.35 | 44.44 | 43.12 | 42.01 |
| Visual Math | WeMath | 35.33 | 46.33 | 46.45 | 44.34 | 46.25 | 43.78 | 41.27 |
| Science & General | PhyX mini MC | 40.42 | 48.71 | 45.96 | 45.75 | 48.50 | 44.96 | 47.88 |
| Science & General | MMMU-ProVis | 35.89 | 39.31 | 39.74 | 40.58 | 41.04 | 39.09 | 37.98 |
| Science & General | RealWorldQA | 66.12 | 68.95 | 70.21 | 70.26 | 70.10 | 67.97 | 67.48 |
| Science & General | MMStar | 62.78 | 66.17 | 66.89 | 66.97 | 68.11 | 66.39 | 65.67 |
| Spatial Reasoning | EmbSpatial | 69.15 | 70.62 | 70.42 | 71.14 | 71.84 | 71.36 | 69.97 |
| Spatial Reasoning | SpatialVizBench COT | 35.06 | 35.31 | 31.92 | 33.05 | 37.50 | 34.75 | 36.16 |
| Spatial Reasoning | CV-Bench 3D | 76.56 | 80.97 | 83.37 | 84.48 | 82.29 | 80.76 | 77.95 |
| Spatial Reasoning | ERQA | 38.99 | 40.67 | 44.65 | 41.09 | 40.46 | 40.25 | 39.94 |
| Perception & Counting | BLINK | 52.95 | 56.39 | 56.89 | 56.30 | 58.01 | 55.64 | 54.70 |
| Perception & Counting | CountBenchQA | 83.68 | 86.41 | 83.59 | 83.93 | 87.35 | 87.52 | 85.47 |
| Perception & Counting | CountQA | 20.28 | 23.15 | 23.91 | 24.11 | 24.19 | 24.08 | 22.93 |
| Perception & Counting | TreeBench | 38.27 | 40.43 | 40.95 | 37.65 | 38.17 | 38.27 | 39.40 |
| Puzzles & Logic | PuzzleVQA | 43.91 | 49.74 | 48.76 | 15.65 | 45.12 | 45.85 | 47.46 |
| Puzzles & Logic | VisualPuzzles | 32.12 | 34.66 | 37.69 | 24.40 | 33.23 | 31.87 | 32.90 |
| Puzzles & Logic | LogicVista | 43.58 | 47.95 | 46.46 | 48.98 | 50.65 | 49.72 | 47.58 |
| Puzzles & Logic | MME-Reasoning | 25.93 | 29.31 | 31.08 | 30.16 | 29.77 | 29.24 | 27.34 |

### 5.4 Primary family (Holm-adjusted) and secondary contrasts (`confirmation/differences.csv`)

| comparison | macro [95% CI] | p | Holm p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|---|
| M050 − T | +0.23 [−0.19, +0.65] | 0.29 | 0.29 | −0.91 [−1.76, −0.07] | +0.39 [−0.70, +1.48] | +1.15 [+0.24, +2.04] | +1.13 [+0.14, +2.15] | +0.33 [−0.75, +1.38] | −0.72 [−1.97, +0.48] |
| M050 − V@P1 | +3.07 [+2.62, +3.51] | < 0.001 | < 0.001 | +5.00 [+4.02, +5.95] | +0.44 [−0.59, +1.47] | +1.05 [+0.12, +1.97] | +0.58 [−0.54, +1.65] | +1.43 [+0.26, +2.62] | +9.89 [+8.59, +11.19] |
| M050 − CV050 | +1.29 [+0.93, +1.64] | < 0.001 | < 0.001 | +1.59 [+0.83, +2.34] | +1.53 [+0.65, +2.45] | +2.33 [+1.52, +3.15] | +1.24 [+0.46, +2.05] | +0.55 [−0.21, +1.30] | +0.52 [−0.58, +1.59] |
| M050 − CT050 | +1.88 [+1.45, +2.31] | < 0.001 | < 0.001 | +1.15 [+0.26, +2.04] | +3.73 [+2.69, +4.78] | +2.19 [+1.25, +3.14] | +2.02 [+1.01, +3.08] | +1.30 [+0.29, +2.32] | +0.87 [−0.38, +2.11] |

| comparison | macro [95% CI] | p | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|---|
| T − B | +4.02 [+3.58, +4.45] | < 0.001 | +5.11 [+4.17, +6.05] | +5.76 [+4.66, +6.91] | +4.48 [+3.44, +5.50] | +1.95 [+0.93, +2.94] | +2.80 [+1.83, +3.81] | +4.03 [+2.75, +5.32] |
| V@P1 − B | +1.19 [+0.65, +1.72] | < 0.001 | −0.80 [−1.92, +0.32] | +5.71 [+4.49, +6.97] | +4.59 [+3.32, +5.89] | +2.50 [+1.11, +3.87] | +1.70 [+0.31, +3.09] | −6.58 [−8.05, −5.13] |
| V@P1 − V@P0 | −2.94 [−3.33, −2.56] | < 0.001 | −3.89 [−4.73, −3.04] | −1.77 [−2.68, −0.90] | +0.19 [−0.63, +1.02] | −0.15 [−1.13, +0.79] | −0.84 [−1.87, +0.19] | −11.20 [−12.38, −10.03] |
| M050 − B | +4.25 [+3.78, +4.72] | < 0.001 | +4.20 [+3.20, +5.23] | +6.16 [+5.01, +7.30] | +5.64 [+4.50, +6.78] | +3.08 [+1.98, +4.18] | +3.13 [+2.03, +4.26] | +3.31 [+1.92, +4.69] |
| CV050 − B | +2.96 [+2.52, +3.38] | < 0.001 | +2.62 [+1.77, +3.49] | +4.62 [+3.56, +5.68] | +3.30 [+2.28, +4.30] | +1.84 [+0.82, +2.82] | +2.58 [+1.56, +3.65] | +2.79 [+1.57, +4.06] |
| CT050 − B | +2.38 [+2.00, +2.75] | < 0.001 | +3.05 [+2.32, +3.79] | +2.42 [+1.41, +3.45] | +3.45 [+2.52, +4.38] | +1.06 [+0.19, +1.91] | +1.83 [+1.13, +2.54] | +2.44 [+1.26, +3.60] |

### 5.5 Per-benchmark contrasts (`confirmation/differences.csv`; exploratory, unadjusted)

| category | benchmark | M050 − T | M050 − V@P1 | M050 − CV050 | M050 − CT050 | V@P1 − V@P0 |
|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | +1.71 [+0.31, +3.11] | +8.64 [+6.46, +10.88] | +2.55 [+1.11, +4.03] | +1.83 [+0.42, +3.21] | +0.43 [−0.84, +1.71] |
| Charts & Tables | CharXivReason | +1.21 [−0.64, +3.07] | +1.75 [−0.25, +3.79] | +2.54 [+0.70, +4.44] | +3.79 [+1.73, +5.88] | −0.04 [−1.89, +1.79] |
| Charts & Tables | TableVQABench | −7.63 [−9.52, −5.66] | +8.03 [+6.14, +9.88] | −0.55 [−1.75, +0.68] | −5.02 [−6.86, −3.10] | −15.97 [−18.07, −13.89] |
| Charts & Tables | EvoChart | +1.07 [−0.52, +2.67] | +1.57 [−0.13, +3.25] | +1.80 [+0.38, +3.25] | +4.00 [+2.25, +5.77] | +0.03 [−1.34, +1.38] |
| Visual Math | MathVision | +1.49 [+0.38, +2.63] | +1.52 [+0.38, +2.69] | +0.47 [−0.57, +1.51] | +1.59 [+0.45, +2.72] | −1.01 [−2.00, −0.05] |
| Visual Math | MathVista | +2.96 [+1.23, +4.73] | +1.25 [−0.30, +2.92] | +1.88 [+0.27, +3.47] | +5.92 [+4.09, +7.75] | −1.75 [−3.10, −0.46] |
| Visual Math | MathVerse | −2.80 [−5.36, −0.35] | −2.91 [−5.25, −0.54] | +1.32 [−0.79, +3.51] | +2.43 [−0.11, +4.95] | −2.22 [−4.27, −0.16] |
| Visual Math | WeMath | −0.08 [−2.99, +2.78] | +1.91 [−0.87, +4.69] | +2.47 [+0.24, +4.69] | +4.98 [+2.33, +7.58] | −2.11 [−4.52, +0.32] |
| Science & General | PhyX mini MC | −0.21 [−2.46, +2.03] | +2.75 [+0.66, +4.91] | +3.54 [+1.24, +5.81] | +0.63 [−1.70, +2.93] | −0.21 [−1.98, +1.55] |
| Science & General | MMMU-ProVis | +1.73 [+0.20, +3.25] | +0.46 [−1.06, +2.01] | +1.95 [+0.56, +3.32] | +3.06 [+1.51, +4.61] | +0.84 [−0.40, +2.11] |
| Science & General | RealWorldQA | +1.14 [−0.82, +3.06] | −0.16 [−2.23, +1.95] | +2.12 [+0.54, +3.72] | +2.61 [+0.40, +4.77] | +0.05 [−2.04, +2.16] |
| Science & General | MMStar | +1.94 [+0.59, +3.34] | +1.14 [−0.35, +2.58] | +1.72 [+0.62, +2.85] | +2.44 [+1.07, +3.79] | +0.08 [−1.26, +1.41] |
| Spatial Reasoning | EmbSpatial | +1.22 [+0.46, +1.99] | +0.70 [−0.02, +1.44] | +0.48 [+0.09, +0.88] | +1.87 [+1.09, +2.65] | +0.72 [−0.01, +1.46] |
| Spatial Reasoning | SpatialVizBench COT | +2.19 [−0.27, +4.60] | +4.45 [+2.05, +6.90] | +2.75 [+0.54, +5.07] | +1.34 [−1.19, +3.83] | +1.13 [−0.46, +2.74] |
| Spatial Reasoning | CV-Bench 3D | +1.32 [−0.38, +3.01] | −2.19 [−3.97, −0.48] | +1.53 [+0.18, +2.89] | +4.34 [+2.58, +6.16] | +1.11 [−0.13, +2.37] |
| Spatial Reasoning | ERQA | −0.21 [−2.80, +2.37] | −0.63 [−3.79, +2.42] | +0.21 [−1.51, +1.91] | +0.52 [−2.08, +3.15] | −3.56 [−6.86, −0.49] |
| Perception & Counting | BLINK | +1.62 [+0.13, +3.13] | +1.71 [+0.02, +3.40] | +2.37 [+1.12, +3.61] | +3.31 [+1.63, +4.99] | −0.59 [−2.12, +0.92] |
| Perception & Counting | CountBenchQA | +0.94 [−1.31, +3.23] | +3.42 [+0.97, +5.94] | −0.17 [−1.65, +1.28] | +1.88 [−0.17, +3.96] | +0.34 [−1.46, +2.11] |
| Perception & Counting | CountQA | +1.04 [−0.46, +2.56] | +0.08 [−1.79, +1.95] | +0.11 [−1.55, +1.74] | +1.26 [−0.44, +2.93] | +0.19 [−1.19, +1.54] |
| Perception & Counting | TreeBench | −2.26 [−5.29, +0.64] | +0.51 [−2.64, +3.69] | −0.10 [−1.79, +1.52] | −1.23 [−3.84, +1.38] | −3.29 [−6.54, −0.11] |
| Puzzles & Logic | PuzzleVQA | −4.62 [−6.57, −2.70] | +29.47 [+27.25, +31.75] | −0.73 [−2.36, +0.88] | −2.34 [−4.44, −0.21] | −33.10 [−35.63, −30.61] |
| Puzzles & Logic | VisualPuzzles | −1.43 [−3.69, +0.84] | +8.82 [+6.41, +11.27] | +1.36 [−0.75, +3.41] | +0.32 [−1.98, +2.59] | −13.29 [−15.74, −10.84] |
| Puzzles & Logic | LogicVista | +2.70 [−0.71, +5.97] | +1.68 [−1.79, +5.09] | +0.93 [−2.03, +3.82] | +3.07 [−0.27, +6.51] | +2.51 [−0.27, +5.26] |
| Puzzles & Logic | MME-Reasoning | +0.46 [−1.44, +2.34] | −0.39 [−2.39, +1.68] | +0.53 [−1.26, +2.32] | +2.43 [+0.54, +4.33] | −0.92 [−2.41, +0.48] |

| category | benchmark | T − B | V@P1 − B | M050 − B | CV050 − B | CT050 − B |
|---|---|---|---|---|---|---|
| Charts & Tables | ChartQAPro | +2.57 [+0.92, +4.22] | −4.36 [−6.48, −2.32] | +4.29 [+2.60, +5.92] | +1.73 [+0.33, +3.12] | +2.45 [+1.05, +3.90] |
| Charts & Tables | CharXivReason | +6.83 [+4.54, +9.21] | +6.29 [+3.72, +8.92] | +8.04 [+5.66, +10.49] | +5.50 [+3.50, +7.47] | +4.25 [+2.62, +5.95] |
| Charts & Tables | TableVQABench | +3.37 [+2.00, +4.78] | −12.29 [−14.55, −10.01] | −4.26 [−6.13, −2.36] | −3.70 [−5.33, −2.04] | +0.76 [−0.06, +1.61] |
| Charts & Tables | EvoChart | +7.67 [+5.62, +9.76] | +7.17 [+5.06, +9.30] | +8.74 [+6.67, +10.86] | +6.94 [+5.09, +8.84] | +4.74 [+2.98, +6.51] |
| Visual Math | MathVision | +2.44 [+1.27, +3.62] | +2.41 [+1.08, +3.75] | +3.93 [+2.72, +5.18] | +3.47 [+2.29, +4.64] | +2.34 [+1.23, +3.48] |
| Visual Math | MathVista | +5.17 [+3.17, +7.21] | +6.88 [+4.59, +9.12] | +8.13 [+6.14, +10.18] | +6.25 [+4.42, +8.05] | +2.21 [+0.44, +3.96] |
| Visual Math | MathVerse | +4.44 [+2.03, +6.93] | +4.55 [+1.74, +7.47] | +1.64 [−0.93, +4.18] | +0.32 [−2.23, +2.92] | −0.79 [−3.17, +1.61] |
| Visual Math | WeMath | +11.00 [+7.98, +14.04] | +9.01 [+5.99, +12.04] | +10.93 [+8.04, +13.83] | +8.45 [+5.86, +11.05] | +5.94 [+3.43, +8.50] |
| Science & General | PhyX mini MC | +8.29 [+5.35, +11.22] | +5.33 [+2.07, +8.69] | +8.08 [+4.89, +11.34] | +4.54 [+1.71, +7.39] | +7.46 [+4.49, +10.36] |
| Science & General | MMMU-ProVis | +3.42 [+1.78, +5.07] | +4.70 [+2.91, +6.54] | +5.15 [+3.53, +6.83] | +3.20 [+1.64, +4.85] | +2.10 [+0.57, +3.59] |
| Science & General | RealWorldQA | +2.83 [+0.97, +4.75] | +4.14 [+1.36, +6.89] | +3.98 [+1.55, +6.38] | +1.85 [−0.06, +3.73] | +1.36 [+0.11, +2.63] |
| Science & General | MMStar | +3.39 [+1.91, +4.93] | +4.19 [+2.26, +6.11] | +5.33 [+3.73, +6.97] | +3.61 [+2.17, +5.05] | +2.89 [+1.73, +4.08] |
| Spatial Reasoning | EmbSpatial | +1.47 [+0.85, +2.10] | +1.99 [+0.92, +3.07] | +2.69 [+1.86, +3.53] | +2.21 [+1.46, +2.97] | +0.82 [+0.41, +1.24] |
| Spatial Reasoning | SpatialVizBench COT | +0.25 [−2.27, +2.77] | −2.01 [−5.00, +1.06] | +2.44 [−0.11, +5.16] | −0.32 [−2.75, +2.13] | +1.09 [−1.24, +3.41] |
| Spatial Reasoning | CV-Bench 3D | +4.41 [+2.56, +6.24] | +7.92 [+5.74, +10.09] | +5.73 [+3.80, +7.62] | +4.20 [+2.34, +6.09] | +1.39 [−0.33, +3.08] |
| Spatial Reasoning | ERQA | +1.68 [−0.95, +4.25] | +2.10 [−1.61, +5.93] | +1.47 [−1.42, +4.31] | +1.26 [−1.20, +3.66] | +0.94 [−0.96, +2.82] |
| Perception & Counting | BLINK | +3.44 [+1.99, +4.85] | +3.35 [+1.16, +5.50] | +5.06 [+3.17, +6.90] | +2.70 [+1.15, +4.24] | +1.75 [+0.69, +2.79] |
| Perception & Counting | CountBenchQA | +2.74 [+0.97, +4.58] | +0.26 [−2.73, +3.27] | +3.68 [+1.47, +5.96] | +3.85 [+1.76, +6.01] | +1.79 [+0.55, +3.08] |
| Perception & Counting | CountQA | +2.87 [+1.08, +4.66] | +3.82 [+1.82, +5.79] | +3.90 [+2.06, +5.75] | +3.79 [+2.06, +5.62] | +2.65 [+1.15, +4.15] |
| Perception & Counting | TreeBench | +2.16 [−0.38, +4.81] | −0.62 [−4.40, +3.04] | −0.10 [−2.97, +2.80] | +0.00 [−2.68, +2.77] | +1.13 [−0.66, +2.93] |
| Puzzles & Logic | PuzzleVQA | +5.83 [+3.81, +7.77] | −28.25 [−30.50, −25.97] | +1.21 [−1.24, +3.67] | +1.94 [−0.30, +4.20] | +3.55 [+1.55, +5.52] |
| Puzzles & Logic | VisualPuzzles | +2.54 [+0.18, +4.83] | −7.72 [−10.34, −5.06] | +1.11 [−1.24, +3.48] | −0.25 [−2.40, +1.94] | +0.79 [−1.40, +2.95] |
| Puzzles & Logic | LogicVista | +4.38 [+0.80, +8.06] | +5.40 [+1.46, +9.43] | +7.08 [+3.30, +10.91] | +6.15 [+2.79, +9.54] | +4.00 [+0.87, +7.23] |
| Puzzles & Logic | MME-Reasoning | +3.39 [+1.33, +5.48] | +4.23 [+1.69, +6.70] | +3.84 [+1.80, +5.90] | +3.32 [+1.30, +5.24] | +1.41 [−0.42, +3.20] |

### 5.6 Capture, retention and damage (`confirmation/merge_capture_damage.base_reference.csv`)

Setup:
- 21 binary-eligible benchmarks, 22,078 rows per seed.
- Groups are defined by T and V@P1 on the confirmation partition.
- Rates are pooled over conditions, and group sizes are mean rows per condition.
- Intervals are for each rate on its own, not for differences.
- The B (reference) rows score the base model as if it were a merge. They were added post hoc.

**all 21 benchmarks**

| model (variant, group sizes) | TRACE-only capture | VERO-only capture | shared-success damage | joint-failure rescue |
|---|---|---|---|---|
| **independent repeat** (2,732 / 2,141 / 8,446 / 8,759) |  |  |  |  |
| M050 | 49.0 [47.7, 50.4] | 52.8 [51.4, 54.1] | 7.0 [6.6, 7.3] | 11.4 [10.9, 11.8] |
| CV050 | 46.1 [44.8, 47.3] | 49.7 [48.3, 51.1] | 9.6 [9.1, 10.0] | 12.5 [12.1, 13.0] |
| CT050 | 54.6 [53.4, 55.8] | 36.4 [35.2, 37.6] | 12.8 [12.3, 13.3] | 14.2 [13.7, 14.6] |
| B (reference, post hoc) | 47.6 [46.4, 48.8] | 36.7 [35.4, 37.9] | 17.7 [17.1, 18.3] | 15.4 [14.9, 15.9] |
| **contemporaneous** (2,732 / 2,141 / 8,446 / 8,759) |  |  |  |  |
| M050 | 50.6 [49.3, 52.0] | 51.5 [50.0, 52.9] | 5.8 [5.5, 6.2] | 10.1 [9.7, 10.6] |
| CV050 | 47.5 [46.2, 48.8] | 47.5 [46.0, 48.9] | 8.3 [7.8, 8.7] | 11.4 [11.0, 11.9] |
| CT050 | 58.5 [57.2, 59.7] | 30.3 [29.0, 31.6] | 10.7 [10.2, 11.1] | 12.4 [12.0, 12.9] |
| B (reference, post hoc) | 50.2 [49.0, 51.5] | 31.5 [30.2, 32.8] | 15.9 [15.3, 16.4] | 14.1 [13.6, 14.6] |
| **stable other seeds** (1,316 / 978 / 7,304 / 7,229) |  |  |  |  |
| M050 | 55.5 [53.5, 57.6] | 50.9 [48.6, 53.1] | 3.6 [3.4, 3.9] | 7.0 [6.6, 7.4] |
| CV050 | 52.1 [50.1, 54.1] | 47.5 [45.3, 49.8] | 5.7 [5.3, 6.1] | 8.6 [8.2, 9.0] |
| CT050 | 63.9 [62.1, 65.6] | 29.9 [28.0, 31.8] | 8.4 [8.0, 8.8] | 10.0 [9.6, 10.5] |
| B (reference, post hoc) | 54.9 [53.1, 56.8] | 32.8 [30.7, 34.7] | 13.0 [12.4, 13.6] | 12.0 [11.5, 12.5] |

**without PuzzleVQA and VisualPuzzles**

| model (variant, group sizes) | TRACE-only capture | VERO-only capture | shared-success damage | joint-failure rescue |
|---|---|---|---|---|
| **independent repeat** (1,922 / 1,971 / 8,139 / 7,518) |  |  |  |  |
| M050 | 45.3 [43.9, 46.7] | 53.0 [51.5, 54.4] | 6.6 [6.3, 6.9] | 10.3 [9.9, 10.8] |
| CV050 | 42.3 [40.9, 43.7] | 50.1 [48.7, 51.4] | 9.1 [8.7, 9.6] | 11.1 [10.6, 11.6] |
| CT050 | 52.5 [51.1, 53.9] | 36.8 [35.5, 38.1] | 12.2 [11.7, 12.7] | 12.4 [12.0, 12.9] |
| B (reference, post hoc) | 45.4 [44.0, 46.7] | 36.6 [35.3, 37.8] | 16.9 [16.3, 17.5] | 13.7 [13.2, 14.2] |
| **contemporaneous** (1,922 / 1,971 / 8,139 / 7,518) |  |  |  |  |
| M050 | 46.9 [45.4, 48.5] | 51.5 [50.0, 53.0] | 5.4 [5.1, 5.7] | 9.0 [8.6, 9.4] |
| CV050 | 44.2 [42.7, 45.7] | 47.7 [46.2, 49.2] | 7.7 [7.3, 8.1] | 9.7 [9.3, 10.2] |
| CT050 | 57.8 [56.3, 59.3] | 30.0 [28.7, 31.4] | 10.1 [9.6, 10.5] | 10.5 [10.1, 11.0] |
| B (reference, post hoc) | 49.2 [47.7, 50.7] | 31.1 [29.8, 32.5] | 15.1 [14.5, 15.6] | 12.1 [11.6, 12.6] |
| **stable other seeds** (863 / 915 / 7,138 / 6,318) |  |  |  |  |
| M050 | 49.4 [47.0, 51.9] | 50.9 [48.5, 53.2] | 3.5 [3.2, 3.8] | 6.4 [6.0, 6.8] |
| CV050 | 46.2 [43.8, 48.6] | 47.8 [45.5, 50.1] | 5.5 [5.1, 5.9] | 7.5 [7.0, 7.9] |
| CT050 | 62.0 [59.8, 64.2] | 30.3 [28.4, 32.3] | 8.1 [7.7, 8.5] | 8.7 [8.3, 9.2] |
| B (reference, post hoc) | 52.2 [49.9, 54.5] | 32.5 [30.5, 34.6] | 12.6 [12.0, 13.2] | 10.6 [10.1, 11.1] |

### 5.7 Answer format (`confirmation/answer_format_sensitivity.csv`; mean over seeds, % of rows)

| model | PuzzleVQA accuracy | PuzzleVQA: ends in \( X \) | PuzzleVQA: scored 0, final letter correct | VisualPuzzles accuracy | VisualPuzzles: ends in \( X \) | VisualPuzzles: scored 0, final letter correct |
|---|---|---|---|---|---|---|
| B | 43.9 | 0.0 | 2.7 | 32.1 | 0.4 | 1.1 |
| T | 49.7 | 0.0 | 1.9 | 34.7 | 0.5 | 0.5 |
| V@P0 | 48.8 | 0.0 | 0.0 | 37.7 | 0.0 | 0.0 |
| V@P1 | 15.7 | 27.6 | 32.0 | 24.4 | 24.6 | 12.6 |
| M050 | 45.1 | 0.9 | 5.7 | 33.2 | 7.7 | 3.6 |
| CV050 | 45.9 | 0.5 | 5.5 | 31.9 | 4.3 | 4.0 |
| CT050 | 47.5 | 0.0 | 1.7 | 32.9 | 0.2 | 0.5 |

### 5.8 Length, finish reasons and extraction (`length_format_judge_audit.csv`)

B, T and V@P0 come from the archive (full split, 98,415 responses each). The new models come from the confirmation runs (78,684 responses each). Rates are row-weighted.

| model | responses | length stops | median of per-benchmark median completion tokens | with </think> | with \boxed{} | extraction not resolved |
|---|---|---|---|---|---|---|
| B | 98,415 | 313 | 96 | 0.9% | 19.4% | 0.006% |
| T | 98,415 | 14 | 179 | 2.4% | 25.5% | 0.005% |
| V@P0 | 98,415 | 8 | 228 | 100.0% | 67.8% | 0.000% |
| V@P1 | 78,684 | 8 | 127 | 3.6% | 39.0% | 0.000% |
| M050 | 78,684 | 9 | 170 | 3.6% | 30.1% | 0.003% |
| CV050 | 78,684 | 15 | 125 | 3.6% | 26.3% | 0.003% |
| CT050 | 78,684 | 8 | 149 | 1.3% | 23.1% | 0.004% |

### 5.9 Sensitivity analyses

Without the 130 confirmation rows that have a near-duplicate development twin (`confirmation/sensitivity_without_dev_twins.csv`):

| comparison | all rows | without twins |
|---|---|---|
| M050 − T | +0.23 [−0.19, +0.65] | +0.23 [−0.19, +0.66] |
| M050 − V@P1 | +3.07 [+2.62, +3.51] | +3.03 [+2.60, +3.48] |
| M050 − CV050 | +1.29 [+0.93, +1.64] | +1.28 [+0.93, +1.62] |
| M050 − CT050 | +1.88 [+1.45, +2.31] | +1.85 [+1.44, +2.28] |
| T − B | +4.02 [+3.58, +4.45] | +3.97 [+3.54, +4.41] |
| V@P1 − B | +1.19 [+0.65, +1.72] | +1.17 [+0.63, +1.70] |
| V@P1 − V@P0 | −2.94 [−3.33, −2.56] | −2.91 [−3.30, −2.52] |
| M050 − B | +4.25 [+3.78, +4.72] | +4.20 [+3.74, +4.67] |
| CV050 − B | +2.96 [+2.52, +3.38] | +2.92 [+2.51, +3.34] |
| CT050 − B | +2.38 [+2.00, +2.75] | +2.35 [+1.99, +2.72] |

Seed 42 only, with the archived B/T or with local reruns of B/T made in the new environment (`confirmation/sensitivity_seed42.*.csv`):

| comparison | archived B/T | local-rerun B/T |
|---|---|---|
| M050 − T | +0.15 [−0.46, +0.77] | +0.27 [−0.34, +0.88] |
| M050 − V@P1 | +3.26 [+2.67, +3.86] | +3.26 [+2.67, +3.86] |
| M050 − CV050 | +1.65 [+1.11, +2.18] | +1.65 [+1.11, +2.18] |
| M050 − CT050 | +1.94 [+1.32, +2.56] | +1.94 [+1.32, +2.56] |
| CV050 − B | +2.67 [+2.02, +3.30] | +2.92 [+2.29, +3.57] |
| CT050 − B | +2.38 [+1.76, +3.00] | +2.63 [+2.04, +3.24] |
| V@P1 − B | +1.05 [+0.34, +1.77] | +1.31 [+0.60, +2.02] |
| T − B | +4.16 [+3.50, +4.82] | +4.31 [+3.66, +4.97] |
| M050 − B | +4.32 [+3.64, +4.99] | +4.57 [+3.91, +5.26] |

## 6. Technical record

### 6.1 Judge health (`judge_extraction_agreement.csv`)

For each run's MathVerse slice: the share of rows whose judge-extracted answer equals the response's own final `\boxed{}` answer. The floor is 30% (DESIGN.md A8).

| run | tier | seed | quarantined | rows with a boxed answer | extracted = boxed (%) | below floor |
|---|---|---|---|---|---|---|
| CT050 | confirmation | 42 |  | 561 | 63.3 |  |
| CT050 | confirmation | 43 |  | 573 | 67.4 |  |
| CT050 | confirmation | 44 |  | 566 | 71.7 |  |
| CV050 | confirmation | 42 |  | 583 | 66.6 |  |
| CV050 | confirmation | 43 |  | 574 | 69.3 |  |
| CV050 | confirmation | 44 |  | 586 | 69.8 |  |
| M050 | confirmation | 42 |  | 586 | 66.0 |  |
| M050 | confirmation | 43 |  | 587 | 79.7 |  |
| M050 | confirmation | 44 |  | 582 | 69.2 |  |
| M050 | confirmation | 44 | yes | 582 | 14.8 | yes |
| V@P1 | confirmation | 42 |  | 589 | 69.1 |  |
| V@P1 | confirmation | 43 |  | 589 | 72.0 |  |
| V@P1 | confirmation | 44 |  | 587 | 82.5 |  |
| CT050 | development screen | 42 |  | 120 | 60.0 |  |
| CT050@P2 | development screen | 42 |  | 123 | 57.7 |  |
| CV050 | development screen | 42 |  | 122 | 61.5 |  |
| CV050@P2 | development screen | 42 |  | 121 | 81.8 |  |
| M050 | development screen | 42 |  | 124 | 70.2 |  |
| M050@P2 | development screen | 42 |  | 121 | 59.5 |  |
| B@P2 | development screen | 42 |  | 114 | 43.0 |  |
| T@P2 | development screen | 42 |  | 116 | 56.9 |  |
| V@P1 | development screen | 42 |  | 120 | 87.5 |  |
| CT050 | smoke | 42 |  | 3 | 66.7 |  |
| CV050 | smoke | 42 |  | 4 | 75.0 |  |
| M050 | smoke | 42 |  | 4 | 50.0 |  |
| V@P1 | smoke | 42 |  | 4 | 75.0 |  |

### 6.2 The A8 repair: M050 confirmation seed 44 before and after rescoring

The quarantined first scoring was done with the judge at tensor-parallel 4 on A800 cards; the rescore used tensor-parallel 1 on RTX PRO 6000 cards. The values are the score campaign's own per-benchmark aggregates from the runner summaries. WeMath is the campaign's fixed-525 value; the study's WeMath metric differs (DESIGN.md A2).

| benchmark | seed 44 before | seed 44 after | change | seed 42 (study metric) | seed 43 (study metric) |
|---|---|---|---|---|---|
| MathVision | 27.18 | 27.92 | +0.74 | 28.78 | 27.80 |
| MathVista | 67.25 | 76.63 | +9.38 | 78.00 | 77.75 |
| MathVerse | 16.98 | 42.38 | +25.40 | 43.65 | 47.30 |
| MMStar | 68.08 | 68.17 | +0.08 | 67.67 | 68.50 |
| SpatialVizBench COT | 37.50 | 37.08 | −0.42 | 38.77 | 36.65 |
| LogicVista | 47.21 | 49.44 | +2.23 | 50.84 | 51.68 |

Only benchmarks whose campaign value changed are listed. Every other benchmark is identical before and after.

Effect on the primary contrast:
- Before the repair, M050 − T was −0.29: the point estimate from the superseded tables, which the second verification pass recomputed from the quarantined scores.
- After it, M050 − T is +0.23 [−0.19, +0.65].
- The old confirmation tables are superseded. `analysis_report.md` §4 and `decision_memo.md` quote only this pre-repair estimate and the Visual Math shortfall it showed, to describe the fault.

## 7. Where the data lives (`experiments/trace_merge/results/`)

| path | contents |
|---|---|
| `reports/analysis_report.md`, `reports/decision_memo.md` | interpretation, caveats, decision |
| `reports/behavior_report.md`, `reports/merge_screen.md` | frozen Stage B report; development-screen report |
| `reports/selection_lock.json` | hashes frozen before confirmation |
| `tables/stage_b/` | Stage B scores, contrasts, corrections/regressions, joint cells, overlap, persistence |
| `tables/screen/` | development screen: P1/P2 scores, contrasts, seed scores, capture/damage, answer format |
| `tables/confirmation/` | confirmation: scores, seed scores, contrasts, capture/damage (+ base reference), answer format, sensitivities |
| `tables/length_format_judge_audit.csv` | lengths, finish reasons, extraction and judge use per model, seed and benchmark |
| `tables/judge_extraction_agreement.csv` | judge-health check per run |
| `tables/evaluation_costs.csv` | GPU-hours, attempts, tokens per run |
| `audit/` | asset, archive, metric-reproduction, protocol and environment audits |
| `figures/` | Stage B, screen and confirmation figures |
| `merges/*/merge_manifest.json` | merge recipes and hashes |
| `tests/test_report.txt` | test-suite run |
