# TRACE–VERO analysis-first merging study: analysis report

Study `trace_vero_qwen25_v1`. B is Qwen2.5-VL-7B-Instruct, T the released TRACE model, V the released VERO model. Three
training-free 50:50 weight interpolations were built and evaluated: M050 = ½V + ½T, CV050 = ½V + ½B and CT050 = ½T + ½B.
Nothing was trained. The decision that follows from this report is in `reports/decision_memo.md`.

## Summary

- **Q1, behaviour.**
  - TRACE and VERO raise the 24-benchmark macro by similar amounts over the base (+4.06 and +4.30 pp, native
    interfaces). Their difference, −0.24 [−0.75, +0.28], is not resolved.
  - Exploratory category contrasts place TRACE's lead in Charts & Tables and VERO's in Visual Math.
  - About half of each parent's unique successes recur on another decoding seed, well above chance.
  - Under the backbone's own interface (P1), most of VERO's gain disappears (V@P1 − V@P0 = −2.94 pp on confirmation).
    Two-thirds of that drop sits in PuzzleVQA and VisualPuzzles, where about a quarter of VERO's P1 answers end in a
    LaTeX-wrapped option letter that the pinned scorers do not read.
- **Q2, integration.** Confirmation partition: 26,228 rows, seeds 42–44, Holm-adjusted over the four prespecified
  comparisons.
  - M050 beats V@P1 (+3.07), the VERO blend CV050 (+1.29) and the TRACE blend CT050 (+1.88).
  - It is **not shown to be better than TRACE**: M050 − T = +0.23 [−0.19, +0.65], Holm p = 0.29. The sign is not
    resolved, and no noninferiority margin was set.
  - Under P1, each parent's half-update therefore adds measurably on top of the other parent's blend.
  - At the row level, M050 keeps VERO-only successes well above the base model's own rate, but TRACE-only successes
    only at about the base model's rate (post-hoc base reference, no paired interval; §4.4).
- **Q3, component explanation.** Not run. The Stage D trigger did not appear, and the dominant effect is the interface
  and answer format, which a weight partition would not separate (§5).
- **One technical repair** (DESIGN amendments A8, A9). One confirmation run's judge returned degraded, run-on
  extractions that copied text from the judge prompt's few-shot examples. Its scoring half was redone, and every table
  here uses the repaired scores (§4).

## 1. What was available, what was run, what was not verified

**Assets** (all identities verified; `audit/audit.md`):

- **Models.** Three pinned checkpoints:
  - B: `Qwen/Qwen2.5-VL-7B-Instruct@cc594898`.
  - T: TRACE, `maveryn/trace-qwen2.5-vl-7b@4d0f1ae8`.
  - V: VERO, `zlab-princeton/Vero-Qwen25-7B@180e84be`.

  Every weight file's sha256 equals its LFS object id. The three are architecture-, tensor- and tokenizer-compatible,
  with no hard stops.
- **Archive.** The archived canonical runs, `maveryn/trace-eval-runs@4178a839`, run
  `qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1`:
  - All 1,303 files match their sha256 and size.
  - 295,245 rows: 3 models × 3 seeds × 24 benchmarks, 32,805 rows per model and seed.
  - Every row joins one-to-one to the prepared benchmark data by content key. CountQA joins through the release's own
    documented answer-normalization alias (DESIGN amendment A1).
- **Protocol.** The pinned `trace_eval_v1` protocol: rlvr `af32fe0`, VLMEvalKit `a8b12bf1`, Qwen3-32B judge
  `9216db57`, vLLM 0.10.2, temperature 0.6 and a 4,096-token cap.

**Reused, not repeated.** All B, T and V@P0 numbers come from the archive (three seeds each). Existing local seed-42
full-suite reruns of B and T, made earlier in the same pinned environment, were reused for an environment-drift check.
They are not new experiments.

**New generations.** Every new generation is a model the archive does not contain:

| tier | rows per run | models | seeds | runs |
|---|---|---|---|---|
| smoke | 135 | V@P1, M050, CV050, CT050 (P1) | 42 | 4 |
| development screen | 2,910 | the same four (P1); B, T, M050, CV050, CT050 under P2 | 42 | 9 |
| confirmation | 26,228 | V@P1, M050, CV050, CT050 (P1) | 42, 43, 44 | 12 |

One confirmation run's scoring half was redone after a judge fault (DESIGN amendment A8, §4). All 25 new runs, and
the two reused local B/T reruns, render byte-identical requests to their archived reference rows
(prompt digest, model-neutral request digest and prompt-token count on every row; `audit/protocol_comparison.json`).

**Interfaces.**
- P1, the primary, gives every model the backbone's tokenizer, chat template and processor. B's and T's archived
  outputs are P1 outputs.
- P0 is each model's native interface. For VERO this adds a ~490-token think/answer system prompt and a `<think>`
  prefill.
- P2 gives every model VERO's interface. It is a development-only diagnostic.

**Merges.** Three CPU-built checkpoints:
- Interpolation runs in float32 from the bf16 parents, with one round-to-nearest-even cast to bf16, over all 729
  tensors.
- Parents are hashed before and after the build, and every merge carries a manifest (`merges/<id>/merge_manifest.json`).
- Nothing was trained, fine-tuned or uploaded, and no Qwen3 model was used.

**Not verified, or not done:**
- Bitwise determinism across GPU types. Drift between environments was measured instead (`audit/audit.md` §5, and the
  seed-42 sensitivity in §4.5).
- V's base-model revision. Its model card names the backbone without a revision, so every reading of V − B as a task
  vector assumes it.
- The two recipes' trainable-module histories. Both changed every parameter group, the vision tower included.
- Benchmark exposure. The suite was examined in earlier work, so neither partition is pristine. The confirmation
  partition was only reserved from merge tuning, and there was no tuning to reserve it from.
- The optional full-suite evaluation (handoff Tier 4) was not run. No component (Stage D) experiment was run (§5).

## 2. TRACE and VERO against their shared backbone (Stage B, archived runs)

Full split, seeds 42–44. Each model is served with its native interface (P0), so V@P0 carries its own system prompt.
Intervals are 95% paired cluster bootstraps over 26,070 bootstrap units (A7). Details are in
`reports/behavior_report.md`, which was frozen before any merge output existed, and in `tables/stage_b/`. The frozen
report's intervals were computed before A7 joined near-duplicate twins into one resampling unit. The regenerated
tables keep every point estimate. Of the intervals quoted in the frozen report, the largest move is 0.10 pp (the lower
end of ChartQAPro's T − V@P0 interval); the next largest are 0.07 pp (Puzzles & Logic and SpatialVizBench). Values
below are the regenerated ones, so a few differ in the last digit from the frozen report.

**Official scores reproduce the release.** All 216 aggregates are rebuilt from row-level data to within 4.5e-11 pp.

| model | 24-benchmark macro [95% CI] |
|---|---|
| B | 47.93 [47.41, 48.45] |
| T | 51.99 [51.43, 52.52] |
| V@P0 | 52.22 [51.62, 52.83] |

| comparison | macro difference (pp) [95% CI] |
|---|---|
| T − B | +4.06 [+3.66, +4.45] |
| V@P0 − B | +4.30 [+3.78, +4.82] |
| T − V@P0 | −0.24 [−0.75, +0.28] |

**Similar gains, different places.** The overall difference T − V@P0 is not resolved. The category contrasts below are
exploratory and unadjusted, and they compare T under the backbone interface with V under its own:
- T leads in Charts & Tables: T − V@P0 = +2.42 [+1.40, +3.43].
- V@P0 leads in Visual Math: −2.02 [−3.15, −0.86]. Under the shared backbone interface the point estimates are level
  on the confirmation partition (V@P1 48.64 against T 48.69, §4.1); no paired interval was computed for that.
- Every other category's interval includes 0.

**Unique successes are partly reproducible.** Over 21 binary-eligible benchmarks (27,617 rows per seed):
- **Disagreement.** The parents disagree on 21.8% of rows on a single seed: 10.7% T-only successes, 11.1% V-only.
- **Persistence.** About half of a unique-success set defined on one seed recurs on another (T-only 47.1%, V-only
  50.0%). That exceeds a label-swap null by T-only +20.2 pp [19.2, 21.3] and V-only +23.1 pp [22.1, 24.1], so each set
  is partly reproducible across decoding seeds. Recurrence on one other seed does not separate reproducible rows from
  noise rows, and decoding seeds say nothing about training-seed variability.
- **Corrections and regressions.** Both models make far more corrections and regressions than their net gain:
  - T corrects 12.31% of rows and regresses 8.66%, net +3.65 [3.28, 4.01].
  - V@P0 corrects 14.52% and regresses 10.49%, net +4.03 [3.57, 4.51].
  - V's regressions persist across seeds well above the null (+14.4 pp). T's barely exceed it (+2.8 pp [1.8, 3.8]).

**The interface differs.**
- V@P0 wraps every response in `<think>`/`<answer>`, and its prompts are 486 tokens longer at the median.
- The largest T − V gap, ChartQAPro, coincides with very different response lengths: T's per-seed medians are
  225–228 tokens, V's 25.

These are correlates, not causes.

**Three hypotheses were frozen before any merge output:**
- **H1, integration beyond shrinkage.** M050 keeps more TRACE-only successes than CV050 and more VERO-only successes
  than CT050.
- **H2, the category trade-off transfers.** M050 − CV050 is largest in Charts & Tables, and M050 − CT050 in Visual Math.
- **H3, part of VERO's gain is interface-dependent.** V@P1 < V@P0, with the largest drops in Visual Math and Puzzles &
  Logic.

## 3. Development screen (seed 42, descriptive)

The frozen development screen holds 2,910 rows, sampled in whole image/problem groups (81–129 per benchmark). Nothing
was selected from it: the confirmed set and the primary family were fixed in advance and locked at 09:04:27Z on
2026-11-02 (`reports/selection_lock.json`). The full write-up is `reports/merge_screen.md`.

**Technical checks.**
- All four P1 models passed the 135-row smoke tier. All nine development runs passed three checks: byte-identical
  rendered requests, metric adapters that reproduce the score campaign's own aggregates (maximum difference < 1e-14),
  and served weights that hash to the audited parent or merge.
- Two environment failures were repaired before any confirmation run had scores (DESIGN amendments A5, A6): a
  pinned-pandas segfault on hash strings shaped like large exponents, and generation-server OOMs on 44–48 GB cards.

**P1 macros [95% CI].** B 47.97 [46.18, 49.76], T 51.40 [49.56, 53.24], V@P0 53.11 [51.19, 55.08], V@P1 48.44
[46.58, 50.39], M050 50.38 [48.50, 52.24], CV050 49.53 [47.66, 51.42], CT050 48.51 [46.64, 50.37].

**What the screen showed.**
- **VERO depends on its interface.** V@P1 − V@P0 = −4.67 [−6.09, −3.21], concentrated in Puzzles & Logic (−14.32).
- **M050 beat V@P1 and CT050, but not T.** M050 − V@P1 = +1.93 [+0.34, +3.45]; M050 − CT050 = +1.86 [+0.20, +3.54];
  M050 − CV050 = +0.85 [−0.67, +2.35]; M050 − T = −1.03 [−2.57, +0.54].
- **H2's direction appeared.** M050 − CV050 was largest in Charts & Tables (+3.75 [+1.03, +6.61]), and M050 − CT050
  in Visual Math (+5.23 [+0.45, +9.95]). The confirmation partition reproduces the second but not the first (§4.2,
  §4.7).
- **P2 diagnostic.** Under VERO's interface, the VERO-containing merges score near VERO's own level (M050@P2 52.34,
  CV050@P2 52.98, V@P0 53.11), while B and T lose (B@P2 42.71, T@P2 47.21). Under P2, no gain from TRACE's
  half-update on top of the VERO blend was detected: M050@P2 − CV050@P2 = −0.64 [−2.13, +0.87]. Its upper bound lies
  below the +1.29 later confirmed under P1, but this is a single-seed screen whose P1 and P2 estimates are not paired
  (the same screen gave +0.85 [−0.67, +2.35] under P1), so it does not show the gain is absent under P2.

## 4. Confirmation (P1, 26,228 rows, seeds 42–44)

The four prespecified models were confirmed on the full confirmation partition with three decoding seeds each, against
the archived B, T and V@P0 on the same rows. Before any table was computed, `confirm` re-verified the selection lock:
- Config, split, merge and view hashes all match.
- DESIGN.md has only grown by the appended amendments A5–A9.
- Every run was served at its locked view revision.
- Every confirmed model's smoke and development runs are complete.

Tables are in `tables/confirmation/`; figures `figures/confirmation_*.png`.

**Technical repair (DESIGN amendments A8, A9).**
- **What was found.** Independent verification of a first version of these tables found M050 seed 44 scoring 16.98 on
  MathVerse against 43.65 and 47.30 on seeds 42 and 43. MathVista and LogicVista were also low.
- **What went wrong.** The run's judge, Qwen3-32B served at tensor-parallel degree 4 on A800 cards, returned degraded
  outputs: they often began with the right answer and then ran on, copying text from the judge prompt's few-shot
  examples (133 of 630 MathVerse extractions, against none after the rescore), so the extracted answers were wrong. It
  was the only run with that judge configuration. The cause inside the serving stack was not isolated (A8, A9).
- **Repair.** The scoring half was redone with the judge at degree 1, and the faulty scoring outputs were moved aside,
  not deleted. Generation outputs were not touched.
- **Check on every run.** `tables/judge_extraction_agreement.csv` checks every run's MathVerse slice: how often the
  judge's extracted answer equals the response's own final `\boxed{}` answer. The faulty slice scores 14.8%. Every
  other slice, including the repaired one (69.2%), scores 43–88%. The faulty judge also changed MathVision and,
  slightly, other judge-routed benchmarks (A9); the rescore replaced all of them.
- **Status of the old tables.** The tables computed with the faulty scores are superseded. They had put M050 − T at
  −0.29 and shown a Visual Math shortfall for M050; both came from the fault.

### 4.1 Official scores and the primary family

| model | 24-benchmark macro [95% CI] | Charts & Tables | Visual Math | Science & General | Spatial Reasoning | Perception & Counting | Puzzles & Logic |
|---|---|---|---|---|---|---|---|
| B | 48.10 [47.53, 48.68] | 54.27 | 42.92 | 51.30 | 54.94 | 48.79 | 36.38 |
| T | 52.13 [51.51, 52.74] | 59.38 | 48.69 | 55.78 | 56.89 | 51.60 | 40.41 |
| V@P0 (own template) | 52.23 [51.56, 52.90] | 57.36 | 50.41 | 55.70 | 57.59 | 51.34 | 41.00 |
| V@P1 (backbone template) | 49.29 [48.64, 49.94] | 53.48 | 48.64 | 55.89 | 57.44 | 50.50 | 29.80 |
| M050 = ½V + ½T | 52.36 [51.73, 52.97] | 58.47 | 49.08 | 56.94 | 58.02 | 51.93 | 39.69 |
| CV050 = ½V + ½B | 51.06 [50.45, 51.68] | 56.89 | 47.55 | 54.60 | 56.78 | 51.38 | 39.17 |
| CT050 = ½T + ½B | 50.48 [49.88, 51.07] | 57.32 | 45.35 | 54.75 | 56.01 | 50.63 | 38.82 |

**Primary confirmatory family.** Macro, Holm-adjusted over the four comparisons. With 10,000 bootstrap replicates, the
smallest attainable p is 0.0002 and the smallest Holm-adjusted p is 0.0008.

| comparison | difference (pp) [95% CI] | p | Holm p |
|---|---|---|---|
| M050 − T | +0.23 [−0.19, +0.65] | 0.29 | 0.29 |
| M050 − V@P1 | +3.07 [+2.62, +3.51] | 0.0002 | 0.0008 |
| M050 − CV050 | +1.29 [+0.93, +1.64] | 0.0002 | 0.0008 |
| M050 − CT050 | +1.88 [+1.45, +2.31] | 0.0002 | 0.0008 |

**Reading the family.**
- **The merge is not shown to be better than TRACE.** The interval includes 0, so the sign of M050 − T is not
  resolved. No noninferiority margin was set, so this is also not a claim that the merge is as good as TRACE.
- **The merge beats both shrinkage controls.**
  - M050 − CV050 = ½(T − B) at the weight level: adding TRACE's half-update to the VERO blend raises the macro by
    +1.29.
  - M050 − CT050 = ½(V − B): adding VERO's half-update to the TRACE blend raises it by +1.88.
  - So, under P1, each parent's half-update raises the macro on top of the other parent's half-update. This alone does
    not show that the two updates carry different content; §4.4 has the row-level evidence.
- **The win over V@P1 is partly an answer-format effect.** M050 − V@P1 is largest in Puzzles & Logic (+9.89
  [+8.59, +11.19]), where V@P1's own scores are depressed by format (§4.3).
- **M050 against native VERO.** VERO's native score (V@P0, 52.23) is outside the prespecified P1 family. M050 is 0.12 pp
  above it as a difference of point estimates. No paired interval was computed for that comparison.

**Shrinkage controls against the base** (secondary, unadjusted):

| comparison | difference (pp) [95% CI] |
|---|---|
| T − B | +4.02 [+3.58, +4.45] |
| V@P1 − B | +1.19 [+0.65, +1.72] |
| V@P1 − V@P0 | −2.94 [−3.33, −2.56] |
| M050 − B | +4.25 [+3.78, +4.72] |
| CV050 − B | +2.96 [+2.52, +3.38] |
| CT050 − B | +2.38 [+2.00, +2.75] |

- **Halving TRACE's update keeps 59% of its gain** (CT050 − B = +2.38 against T − B = +4.02).
- **Halving VERO's update scores 1.77 pp higher than VERO itself under P1.** CV050 − B = +2.96 against V@P1 − B = +1.19
  (the difference of the two paired estimates). This fits the screen: the half-update softens VERO's P1 answer-format
  shift (§4.3).
- **The screen could not estimate CT050 − B.** It put CT050 − B at only +0.54 [−1.08, +2.16]; on this partition it is
  +2.38. The screen's 81–129 rows per benchmark gave an interval 3.2 pp wide. The two controls were ordered the same way
  on both tiers.

### 4.2 Category and benchmark contrasts (descriptive, unadjusted)

| category | M050 − T | M050 − V@P1 | M050 − CV050 | M050 − CT050 |
|---|---|---|---|---|
| Charts & Tables | −0.91 [−1.76, −0.07] | +5.00 [+4.02, +5.95] | +1.59 [+0.83, +2.34] | +1.15 [+0.26, +2.04] |
| Visual Math | +0.39 [−0.70, +1.48] | +0.44 [−0.59, +1.47] | +1.53 [+0.65, +2.45] | **+3.73 [+2.69, +4.78]** |
| Science & General | +1.15 [+0.24, +2.04] | +1.05 [+0.12, +1.97] | **+2.33 [+1.52, +3.15]** | +2.19 [+1.25, +3.14] |
| Spatial Reasoning | +1.13 [+0.14, +2.15] | +0.58 [−0.54, +1.65] | +1.24 [+0.46, +2.05] | +2.02 [+1.01, +3.08] |
| Perception & Counting | +0.33 [−0.75, +1.38] | +1.43 [+0.26, +2.62] | +0.55 [−0.21, +1.30] | +1.30 [+0.29, +2.32] |
| Puzzles & Logic | −0.72 [−1.97, +0.48] | +9.89 [+8.59, +11.19] | +0.52 [−0.58, +1.59] | +0.87 [−0.38, +2.11] |

- **Against TRACE.** M050's only category contrast against T with an interval entirely below 0 is Charts & Tables
  (−0.91), TRACE's lead category; Puzzles & Logic is also below in point estimate (−0.72 [−1.97, +0.48]). M050 is
  above T in Science & General (+1.15) and Spatial Reasoning (+1.13). By benchmark, its largest shortfalls against T are TableVQABench
  (−7.63 [−9.52, −5.66]) and PuzzleVQA (−4.62 [−6.57, −2.70]); these are two of the three benchmarks where VERO's own
  P1 deficit is largest (§4.3).
- **Each half-update gains in most categories.**
  - VERO's half-update added to the TRACE blend (M050 − CT050) gains most in Visual Math (+3.73).
  - TRACE's half-update added to the VERO blend (M050 − CV050) gains most in Science & General (+2.33), then Charts &
    Tables (+1.59) and Visual Math (+1.53).
  - No category interval of either contrast lies entirely below 0. No category regression margin was set, so this is
    not a claim that neither half-update harms any category.
- These are 24 category contrasts (and 96 benchmark contrasts) without multiplicity adjustment. They describe where the
  macro differences sit; they are not separate findings.

### 4.3 Interface and answer format

VERO's P1 deficit against its own template, by category (the other three categories' intervals include 0):

| contrast | Charts & Tables | Visual Math | Puzzles & Logic | macro |
|---|---|---|---|---|
| V@P1 − V@P0 | −3.89 [−4.73, −3.04] | −1.77 [−2.68, −0.90] | −11.20 [−12.38, −10.03] | −2.94 [−3.33, −2.56] |

By benchmark, the deficit is concentrated in three benchmarks:
- PuzzleVQA −33.10 [−35.63, −30.61] and VisualPuzzles −13.29 [−15.74, −10.84]. Together, about 66% of the macro drop.
- TableVQABench −15.97 [−18.07, −13.89], about 23%. It is the whole Charts & Tables drop; the other three Charts
  & Tables benchmarks are within ±0.5.
- The Visual Math drop is spread over four benchmarks, each between −1.0 and −2.2.

Answer-format diagnostic for the two puzzle benchmarks (`tables/confirmation/answer_format_sensitivity.csv`). Means
over seeds 42–44, % of all rows. This is a diagnostic, not a metric: the final option letter is read leniently (bare,
`\( X \)`, `$X$`, `\boxed{X}`).

| model | PuzzleVQA official accuracy | PuzzleVQA: ends in `\( X \)` | PuzzleVQA: scored 0, final letter correct | VisualPuzzles official accuracy | VisualPuzzles: ends in `\( X \)` | VisualPuzzles: scored 0, final letter correct |
|---|---|---|---|---|---|---|
| B | 43.9 | 0.0 | 2.7 | 32.1 | 0.4 | 1.1 |
| T | 49.7 | 0.0 | 1.9 | 34.7 | 0.5 | 0.5 |
| V@P0 | 48.8 | 0.0 | 0.0 | 37.7 | 0.0 | 0.0 |
| **V@P1** | **15.7** | **27.6** | **32.0** | **24.4** | **24.6** | **12.6** |
| M050 | 45.1 | 0.9 | 5.7 | 33.2 | 7.7 | 3.6 |
| CV050 | 45.9 | 0.5 | 5.5 | 31.9 | 4.3 | 4.0 |
| CT050 | 47.5 | 0.0 | 1.7 | 32.9 | 0.2 | 0.5 |

- **On PuzzleVQA and VisualPuzzles the loss is an observable answer-format effect.** 32.0% of V@P1's PuzzleVQA rows
  are scored 0 although their final letter is the correct option, against 0.0% for V@P0. On these two benchmarks this
  is not evidence that VERO lost the ability.
- **The TableVQABench and Visual Math drops are not diagnosed.** Nothing here says whether they are format or ability.
- **The merges carry a little of the format.** M050's VisualPuzzles responses end in a LaTeX-wrapped letter 7.7% of the
  time, against 0.5% for T. 5.7% of its PuzzleVQA rows are scored 0 although the final letter is correct, against 1.9%
  for T.
- The official scores stay the primary metric, as prespecified.

### 4.4 Capture, retention and damage (21 binary-eligible benchmarks, 22,078 rows per seed)

The groups are defined by T and V@P1 on the confirmation partition, so "VERO-only" here means VERO under the backbone
interface. These are not the V@P0-based sets of §2.
- Group sizes are the mean number of rows per condition, and rates are pooled over conditions.
- Intervals are 95% cluster bootstraps of each rate on its own, not of differences between models.
- **Independent repeat** (the variant H1 names): the parent sets come from a different seed than the outcome.
- **Contemporaneous**: the same seed.
- **Stable other seeds**: parent sets that hold on both other seeds.

The **B (reference)** rows are post hoc. They were added after the first confirmation tables had been read, at the
verification's suggestion. They score the base model as if it were a merge, which gives the rate a model reaches with
no post-training update at all (`tables/confirmation/merge_capture_damage.base_reference.csv`).

| model | TRACE-only capture P(m=1 \| T=1, V@P1=0) | VERO-only capture P(m=1 \| V@P1=1, T=0) | shared-success damage P(m=0 \| both=1) | joint-failure rescue P(m=1 \| both=0) |
|---|---|---|---|---|
| **independent repeat** (group sizes 2,732 / 2,141 / 8,446 / 8,759) | | | | |
| M050 | 49.0 [47.7, 50.4] | 52.8 [51.4, 54.1] | 7.0 [6.6, 7.3] | 11.4 [10.9, 11.8] |
| CV050 | 46.1 [44.8, 47.3] | 49.7 [48.3, 51.1] | 9.6 [9.1, 10.0] | 12.5 [12.1, 13.0] |
| CT050 | 54.6 [53.4, 55.8] | 36.4 [35.2, 37.6] | 12.8 [12.3, 13.3] | 14.2 [13.7, 14.6] |
| B (reference, post hoc) | 47.6 [46.4, 48.8] | 36.7 [35.4, 37.9] | 17.7 [17.1, 18.3] | 15.4 [14.9, 15.9] |
| **contemporaneous** (group sizes 2,732 / 2,141 / 8,446 / 8,759) | | | | |
| M050 | 50.6 [49.3, 52.0] | 51.5 [50.0, 52.9] | 5.8 [5.5, 6.2] | 10.1 [9.7, 10.6] |
| CV050 | 47.5 [46.2, 48.8] | 47.5 [46.0, 48.9] | 8.3 [7.8, 8.7] | 11.4 [11.0, 11.9] |
| CT050 | 58.5 [57.2, 59.7] | 30.3 [29.0, 31.6] | 10.7 [10.2, 11.1] | 12.4 [12.0, 12.9] |
| B (reference, post hoc) | 50.2 [49.0, 51.5] | 31.5 [30.2, 32.8] | 15.9 [15.3, 16.4] | 14.1 [13.6, 14.6] |
| **stable other seeds** (group sizes 1,316 / 978 / 7,304 / 7,229) | | | | |
| M050 | 55.5 [53.5, 57.6] | 50.9 [48.6, 53.1] | 3.6 [3.4, 3.9] | 7.0 [6.6, 7.4] |
| CV050 | 52.1 [50.1, 54.1] | 47.5 [45.3, 49.8] | 5.7 [5.3, 6.1] | 8.6 [8.2, 9.0] |
| CT050 | 63.9 [62.1, 65.6] | 29.9 [28.0, 31.8] | 8.4 [8.0, 8.8] | 10.0 [9.6, 10.5] |
| B (reference, post hoc) | 54.9 [53.1, 56.8] | 32.8 [30.7, 34.7] | 13.0 [12.4, 13.6] | 12.0 [11.5, 12.5] |

- **H1 holds in direction, as frozen, in every variant.** M050's VERO-only capture is far above CT050's (52.8 against
  36.4), and its TRACE-only capture is above CV050's (49.0 against 46.1). No interval for the paired differences was
  computed.
- **On the VERO side, M050 retains successes well beyond the base.** It is 16 pp above B (52.8 against 36.7).
- **On the TRACE side, M050's capture is close to the base model's post-hoc reference rate.** B itself is correct on
  47.6% of TRACE-only rows on an independent repeat, against M050's 49.0% (no paired interval).
  - Without PuzzleVQA and VisualPuzzles, where V@P1's format failures fill the TRACE-only group (810 of its 2,732 rows),
    M050 is at 45.3 [43.9, 46.7] against B's 45.4 [44.0, 46.7].
  - The TRACE blend CT050 does retain TRACE-only successes above the base (54.6, or 52.5 without the two benchmarks).
  - On all 21 benchmarks, M050 is 1.5 pp above the base reference and CV050 1.5 pp below it (point estimates). Without
    the two puzzle benchmarks, M050 and B are level, so there the margin over CV050 (42.3) comes from CV050 being below
    the base.
- **M050 moves least from its parents in both directions.** It damages the fewest shared successes (7.0%; the base
  "damages" 17.7%) and rescues the fewest joint failures (11.4%).
- M050's two captures are worth 6.1 and 5.1 pp of all rows on the independent-repeat variant; across the four rows of
  the table, captures range from 3.5 to 6.8 pp.
- The rates are not compared with the hindsight "either parent correct" diagnostic of `behavior_report.md` §3. That
  diagnostic is neither an achievable router nor a bound on a merge.

### 4.5 Sensitivity analyses (prespecified)

**Without development twins** (A7: the 130 confirmation rows with a near-duplicate development image dropped). The
primary family is unchanged:
- M050 − T +0.23 [−0.19, +0.66].
- M050 − V@P1 +3.03 [+2.60, +3.48].
- M050 − CV050 +1.28 [+0.93, +1.62].
- M050 − CT050 +1.85 [+1.44, +2.28].

**Environment, seed 42 only.** The archived B/T are substituted with the existing local reruns made in the new
environment:

| comparison | archived B/T, seed 42 | local-rerun B/T, seed 42 |
|---|---|---|
| M050 − T | +0.15 [−0.46, +0.77] | +0.27 [−0.34, +0.88] |
| T − B | +4.16 [+3.50, +4.82] | +4.31 [+3.66, +4.97] |
| V@P1 − B | +1.05 [+0.34, +1.77] | +1.31 [+0.60, +2.02] |

- On seed 42, the point estimate of M050 − T is slightly positive with either source (+0.15, +0.27; both intervals
  include 0), in line with the three-seed +0.23.
- Substituting the local reruns moves each comparison against B or T by 0.1–0.3 pp. This single seed-42 comparison
  combines environment drift with run-to-run variation.
- The shift is of the same order as M050 − T, the only family comparison that sets new runs against archived ones.

### 4.6 Length, finish reasons, extraction and judge health

Confirmation runs, seeds 42–44, 78,684 responses per new model, against the archive's 98,415 per parent
(`tables/length_format_judge_audit.csv`):

| model | length stops | median of per-benchmark median completion tokens | responses with `</think>` | responses with `\boxed{}` |
|---|---|---|---|---|
| B | 313 | 96 | 0.9% | 19.4% |
| T | 14 | 179 | 2.4% | 25.5% |
| V@P0 | 8 | 228 | 100.0% | 67.8% |
| V@P1 | 8 | 127 | 3.6% | 39.0% |
| M050 | 9 | 170 | 3.6% | 30.1% |
| CV050 | 15 | 125 | 3.6% | 26.3% |
| CT050 | 8 | 149 | 1.3% | 23.1% |

- **Length stops are negligible** for every new model. No model was given a longer output budget than its parents.
- **Unresolved extraction status** is at most 0.01% for every model.
- **VERO under P1** writes much shorter responses than under its own template, and stops writing `<think>` blocks.
- **Judge health.** Every locally generated run's MathVerse slice is above the 30% agreement floor of the A8 check
  (range 43–88%; the quarantined slice was 14.8%; `tables/judge_extraction_agreement.csv`). This screens one
  judge-routed benchmark for the A8 failure signature. It does not verify judge outputs on other benchmarks or on the
  archived runs.

### 4.7 Frozen hypotheses: outcome on confirmation

| hypothesis | outcome |
|---|---|
| H1, integration beyond shrinkage | **Holds in direction, as frozen.** M050's VERO-only capture far exceeds CT050's, and its TRACE-only capture exceeds CV050's (§4.4). The post-hoc base reference qualifies the TRACE side: there M050's capture is close to the base model's rate (49.0 against 47.6, no paired interval; 45.3 against 45.4 without PuzzleVQA and VisualPuzzles). |
| H2, the category trade-off transfers | **Half supported.** M050 − CT050 is largest in Visual Math (+3.73), as predicted. M050 − CV050 is largest in Science & General (+2.33), not Charts & Tables (+1.59). |
| H3, VERO's gain depends on its interface | **Supported for the macro** (V@P1 − V@P0 = −2.94) and **for Puzzles & Logic** (−11.20), where the format diagnostic explains it. **Not supported for Visual Math** being one of the two largest drops: Charts & Tables, through TableVQABench, drops more. |

## 5. Component intervention (Stage D): not run

The handoff allows at most one targeted component experiment, and only after a written development finding and the
researcher's approval. None was proposed, so none was run.

**Why none was proposed.** The handoff's example trigger is a merge that helps some categories while regressing on
others. On the development screen, no category contrast of M050 against T, V@P1, CV050 or CT050 was negative with an
interval excluding 0. A category pattern would not, by itself, show which component holds a skill.

**The confirmation tier is not a trigger either.**
- M050's only negative category contrast with an interval excluding 0 is Charts & Tables against T (−0.91,
  unadjusted).
- It was not available when Stage D was decided.
- Confirmation results cannot trigger new tuning on the same partition.

**What dominates instead.** The large effect in this study is the interface and answer format:
- On the confirmation tier, VERO itself loses 2.94 pp under P1, two-thirds of it in two benchmarks with a documented
  format failure.
- On the development screen's P2 diagnostic (seed 42), the VERO-containing merges' point estimates are near VERO's own
  (M050@P2 52.34, CV050@P2 52.98, V@P0 53.11; M050@P2 − V@P0 = −0.78 [−2.40, +0.86]).

A partition of weights into frozen and merged components would not separate the interface from the weights.

## 6. Uncertainty, scorer coverage, prior exposure, recipe confounds and cost

**Uncertainty.**
- **What the intervals cover.** All intervals are paired cluster bootstraps: 10,000 replicates, seed 314159, bootstrap
  units resampled with every model and seed together. They reflect item and decoding variability conditional on these
  checkpoints, this suite and this judge. They do not reflect training-seed variability: each recipe is one released
  checkpoint.
- **Multiplicity.** Only the four-comparison primary family is Holm-adjusted. Category, benchmark, capture/damage,
  sensitivity and P2 numbers are descriptive. The base-reference capture rows are post hoc.
- **Twins.** The partition is by exact image bytes. The 130 confirmation rows with a near-duplicate development twin
  are handled by joining the twins into one bootstrap unit and by a sensitivity analysis without them (A7, §4.5).

**Scorer coverage.**
- Every macro uses the 24 official adapters, each reproducing the archived aggregate.
- Binary row diagnostics cover 21 benchmarks. ChartQAPro (fractional scores), TableVQABench (a five-split mean) and
  WeMath (a problem-group metric) are excluded (A3).
- WeMath on subsets uses the problem groups present (A2). The pinned fixed-525 formula would score absent groups as
  mastered.
- Seven benchmarks' archived scores exist only as aggregates; their row scores were re-derived with the pinned scorer
  functions.
- The Qwen3-32B judge decides answer extraction on judge-routed benchmarks for every model alike. A response style that
  the judge or a deterministic parser reads poorly is scored as wrong; §4.3 shows this happening to VERO under P1. A
  judge fault on one run was found and repaired (A8, §4).

**Prior exposure.** The suite and the archived runs were studied before this work, by the release and by earlier
analyses. The 20/80 split guards only against selection inside this study. Nothing was selected, because there was no
coefficient search.

**Recipe confounds.** T and V are whole released recipes: procedural-data post-training (TRACE) against heterogeneous
curated multimodal post-training (VERO). VERO's inventory includes photographs, diagrams, interfaces and synthetic
sources. The recipes also differ in RL algorithm, prompts, reward routes, trainable modules and training length.
- Nothing here isolates the effect of procedural data or of image provenance.
- A merge result does not predict how joint RL training would behave.

**Evaluation cost** (`tables/evaluation_costs.csv`).
- **Recorded:** 57.57 GPU-hours over 41 run attempts (15 failed or incomplete, all resumed) and 72.9 M generated
  tokens. This includes the A8 rescore. Merged weights take 46.3 GiB and run directories 5.3 GiB.
- **Not in the timing files:** one attempt. M050's confirmation seed-44 run generated all its rows on 2× A40 for 2.3
  hours before its allocation expired, about 4.6 GPU-hours. That brings the total to about 62 GPU-hours. Its rows were
  kept and finalized by the next attempt.
- **Training cost:** zero.
