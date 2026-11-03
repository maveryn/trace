# Stage B: behaviour of TRACE and VERO against their shared backbone

Study `trace_vero_qwen25_v1`. Inputs: the archived canonical runs (`maveryn/trace-eval-runs@4178a839`, run
`qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1`), seeds 42, 43, 44, each model with its native interface (P0). B is
Qwen2.5-VL-7B-Instruct, T is the released TRACE model, and V@P0 is the released VERO model with its own chat template.
No merged-model output existed when this report and its hypotheses were written; a sha256 of this file is recorded in
`reports/behavior_report.freeze.json`.

Uncertainty: 95% paired cluster-bootstrap intervals (10,000 replicates, seed 314159) over 26,422 connected
image/problem groups, with every model and seed resampled together. They are conditional on these checkpoints and this
suite, and they do not reflect training-seed variability.

## 1. Official scores reproduce the release

The 24-benchmark macro (unweighted mean of official benchmark scores, averaged over the three seeds) is B 47.93
[47.40, 48.45], T 51.99 [51.45, 52.54] and V@P0 52.22 [51.62, 52.83]. These equal `results.json` (47.9285, 51.9859,
52.2238). All 216 model × seed × benchmark aggregates are reproduced from row-level data to within 4.5e-11 pp, including
WeMath's strict group metric and the seven benchmarks whose archive holds only an aggregate.

| comparison | 24-benchmark macro (pp) | 95% CI | p |
|---|---|---|---|
| T − B | +4.06 | [3.67, 4.46] | < 0.001 |
| V@P0 − B | +4.30 | [3.79, 4.82] | < 0.001 |
| T − V@P0 | −0.24 | [−0.74, 0.26] | 0.36 |

The two recipes give the same overall gain. The confirmation partition alone gives the same picture: T − V@P0 is
−0.11 [−0.67, 0.47].

## 2. Where the gains differ (category and benchmark contrasts, full split)

| category | T − B | V@P0 − B | T − V@P0 [95% CI] |
|---|---|---|---|
| Charts & Tables | +5.15 | +2.73 | **+2.42 [1.40, 3.45]** |
| Visual Math | +5.65 | +7.67 | **−2.02 [−3.18, −0.89]** |
| Science & General | +4.55 | +4.48 | +0.07 [−0.99, 1.10] |
| Spatial Reasoning | +2.25 | +2.64 | −0.39 [−1.77, 1.02] |
| Perception & Counting | +2.77 | +3.02 | −0.25 [−1.65, 1.17] |
| Puzzles & Logic | +3.99 | +5.24 | −1.26 [−2.66, 0.15] |

Benchmarks with a T − V@P0 interval excluding zero:
- In T's favour: ChartQAPro +7.18 [5.17, 9.32] and SpatialVizBench +3.62 [0.90, 6.29].
- In V's favour: MathVista −3.43, MathVerse −3.17, VisualPuzzles −2.94, MME-Reasoning −2.58 and CV-Bench 3D −2.11.

These are 24 exploratory comparisons without multiplicity adjustment.

Counterexamples to a "TRACE = reasoning, VERO = perception" reading:
- T is ahead on SpatialVizBench, a spatial benchmark, while V is ahead on CV-Bench 3D, also spatial.
- V's largest gains are on mathematical reasoning, not perception.

Category names do not identify skills.

## 3. Row-level transitions (21 binary-eligible benchmarks, 27,617 rows, seed-averaged)

Corrections, regressions and the eight-cell table use the 16 benchmarks with archived 0/1 rows plus MathVision, MathVista,
CountBenchQA, PuzzleVQA and VisualPuzzles (0/1 rows re-derived with the pinned scorers). ChartQAPro, TableVQABench and
WeMath are excluded (fractional, split-macro and group metrics; DESIGN amendment A3).

| | correction rate (% of all rows) | regression rate | net | corrections / B errors | regressions / B successes |
|---|---|---|---|---|---|
| T | 12.31 | 8.66 | +3.65 [3.28, 4.01] | 23.2% | 18.4% |
| V@P0 | 14.52 | 10.49 | +4.03 [3.56, 4.51] | 27.4% | 22.3% |

Both models churn far more rows than their net gain: roughly three corrections for every two regressions. V changes
more rows in both directions.

**Eight cells** (pooled rows, % of all rows, mean over seeds):

| | B=0 | B=1 |
|---|---|---|
| T=0, V=0 | 33.08 | 5.21 |
| T=0, V=1 | 7.63 | 3.45 |
| T=1, V=0 | 5.42 | 5.28 |
| T=1, V=1 | 6.89 | 33.03 |

On a single seed the parents disagree on 21.8% of rows. 10.7% are T-only successes and 11.1% V-only. Of the T-only
successes, 5.4 points are rows B also failed; of the V-only successes, 7.6. The hindsight "either parent correct"
diagnostic is 61.7% against 51.0% for the better parent. It uses two responses and correctness labels, so it is not an
achievable router and not a bound on a merge.

**Persistence across seeds.** Each set is defined on one seed and checked on another, against a label-swap null:
- T-only sets persist 47.1% of the time (null 26.9%; excess +20.3 pp [19.2, 21.3]).
- V-only sets persist 50.0% (excess +23.1 pp [22.1, 24.1]).

The parents' unique successes are therefore mostly reproducible, not decoding noise. For the base-relative changes:
- V's corrections persist far beyond the null (+24.0 pp), and so do its regressions (+14.4 pp).
- T's corrections persist less (+12.4 pp), and its regressions barely persist (+2.8 pp [1.8, 3.8]).

VERO changes the base's behaviour more systematically, in both directions. Most of TRACE's regressions look like
resampling noise. This matches the weight-space diagnostic: V's update is 4–5 times larger than T's in every parameter
group, and the two update directions are nearly orthogonal (cosine below 0.01). That diagnostic says nothing about skills.

## 4. Response format and length (archived seeds 42–44)

- V@P0 wraps 100% of responses in `<think>…</think><answer>…</answer>`, a direct effect of its native system prompt.
  Its prompts are 486 tokens longer at the median.
- Length stops are rare: B 313, T 14, V 8 over 98,415 responses each. Of the length-stopped responses that have a row
  score, 129 of 133 score 0.
- The largest T − V gap, ChartQAPro, coincides with very different response lengths: median 226 tokens for T against 25
  for V. V's `<answer>` block nearly always carries the mandated "The answer is" sentence, but it reasons very little
  first. On MathVista and MathVerse, where V leads, V writes longer responses than T (median 304 vs 205 and 544 vs 365
  tokens) and always boxes its answer.

These are observable correlates, not causes.

## 5. Limitations

- Whole released checkpoints and recipes are compared. Data, RL algorithm, prompts and training length are all
  confounded, so nothing here isolates synthetic vs natural data.
- V@P0 is evaluated with its own system prompt and T and B are not, so T − V@P0 mixes weights and interface. The merge
  study evaluates V under the backbone interface (V@P1) for that reason.
- Categories are the suite's six reporting groups. No new slice labels were constructed.
- The suite was examined in earlier work. The confirmation partition is reserved from merge tuning, but it is not
  pristine.

## 6. Frozen hypotheses for the merge study (at most three; written before any merge output)

**H1: integration beyond shrinkage.** The parents' unique successes are reproducible (Section 3), so a weight average
that integrates them should keep more of each parent's unique successes than the matching shrinkage control does.
Prediction on the confirmation partition, independent-repeat variant (parent sets defined on a different seed from the
merge outcome):
- P(M050 = 1 | T = 1, V@P1 = 0) > P(CV050 = 1 | T = 1, V@P1 = 0), and
- P(M050 = 1 | V@P1 = 1, T = 0) > P(CT050 = 1 | V@P1 = 1, T = 0).

The confirmatory test of whether the merge beats both parents and both controls is the prespecified primary family in
DESIGN.md; H1 only describes the mechanism.

**H2: the category trade-off transfers.** T's advantage concentrates in Charts & Tables, and V's in Visual Math and
Puzzles & Logic. Prediction (descriptive, confirmation partition):
- Adding TRACE's half-update to the VERO blend (M050 − CV050) helps most in Charts & Tables.
- Adding VERO's half-update to the TRACE blend (M050 − CT050) helps most in Visual Math.

**H3: part of VERO's gain is interface-dependent.** V's measured advantage was obtained with its own think/answer system
prompt. Prediction:
- V@P1 (backbone interface) scores below V@P0 on the macro.
- The drop is largest where V's long, boxed reasoning style earned its lead (Visual Math, Puzzles & Logic).

If so, every P1 merge inherits the handicap, and VERO-specific capture under P1 understates what the merge could do
under VERO's interface. The development-screen P2 diagnostic is the check.
