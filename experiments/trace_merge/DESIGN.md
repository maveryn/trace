# TRACE–VERO analysis-first merging study: frozen design

Study id `trace_vero_qwen25_v1`. This file records every decision that must be fixed before any merged-model result
is read. Later edits are appended under "Amendments" with a reason and a timestamp; nothing above that heading changes
after the development screen starts.

Phase: **analysis only**. No training, no RL continuation, no new training corpus, no uploads, no Qwen3. Coefficient
expansion (alpha 0.25/0.75), component (frontend-preserving) experiments, and other merge algorithms are disabled
unless a written development finding and explicit supervisor approval enable them (Stage D).

## 1. Assets

| Role | Meaning | Source | Pinned revision |
|---|---|---|---|
| `B` | original instruction-tuned backbone | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `T` | released TRACE model | `maveryn/trace-qwen2.5-vl-7b` | `4d0f1ae8ee25022058090dbdbff61957ece7331d` |
| `V` | released VERO model (not a new VERO-64K model) | `zlab-princeton/Vero-Qwen25-7B` | `180e84be5acb2aa887cf51015b84b6a6e453ee90` |

Archived evaluations: `maveryn/trace-eval-runs@4178a839b689babe16f8ac36f0de7b1b2c5ef36c`, run
`qwen2.5-vl-7b-comparison-temp06-seeds42-44-v1` (manifest sha256 `0334afa7…`, result sha256 `578e574d…`, both pinned in
`rlvr/evaluation/trace_eval/release_receipts.v1.json`). Archived model ids: `qwen2.5-vl-7b-base` (B),
`trace-qwen2.5-vl-7b` (T, archived runtime view `sha256set:8396b0be…`, a content-bound view of the same weights as the
HF commit), `vero-qwen2.5-vl-7b` (V). The archive holds responses, extractions and scores for 3 models x 3 seeds x 24
benchmarks; it does not redistribute benchmark inputs. The RL-baselines archive root (`4ca25af7…`) holds no B/T/V run
and is not used.

New generations use the published protocol code (`rlvr` branch `af32fe0`) from a prepared clone with the pinned
VLMEvalKit `a8b12bf1` and the Qwen3-32B judge `9216db57`, plus three scorer-side fail-closed fixes (CharXiv judge
parser that the public release omits, MME-Reasoning official-function exceptions scored incorrect, judge output still
unusable after the official five attempts scored incorrect). These fixes only change rows on which the canonical
producer would have raised; the protocol comparison records how many rows of each new run take those paths.

## 2. Code location (deviation from the handoff)

The handoff proposed `rlvr/evaluation/trace_merge/`. The release validator
(`rlvr/evaluation/scripts/validate_release_inputs.py`, source-map check) and `rlvr/train.py` both reject any file under
`rlvr/` that is not in the frozen source map, so adding files there would break canonical validation. The study lives
in `experiments/trace_merge/` instead; `rlvr/` is untouched and the validator is run before and after integration.

## 3. Evaluation protocols

The protocol's generation endpoint renders each request with the served model directory's own chat template. The three
parents do not share one:

* B and T render identically (same Qwen2.5-VL template text, identical token-to-id maps, identical median prompt length
  674 tokens in the archive). T additionally lacks `<|endoftext|>` (151643) as an extra stop id; B and V list it.
* V's native template replaces any system message with a fixed ~490-token "Reasoning in `<think>`, final answer in
  `<answer>` with `\boxed{}`" system prompt and pre-fills `<think>\n` after the assistant header (archived median prompt
  length 1160 tokens; 32,804 of 32,805 archived V responses contain `</think>` and `<answer>`).
* All three resolve to the fast Qwen2-VL image processor in the pinned environment (min 3136, max 12,845,056 pixels) and
  are served in bf16 (V's config says float32; vLLM downcasts float32 configs to bf16).

A merged checkpoint has no native template, so one **inference view** (the non-weight files served with the weights) is
fixed in advance. Templates are never averaged and never chosen by score.

| Protocol id | Inference view | Role |
|---|---|---|
| `P0_native` | each parent's own files | the archived canonical runs |
| `P1_backbone` | B's non-weight files (`config.json`, `generation_config.json`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `chat_template.json`) | **primary** protocol for every merge and control |
| `P2_vero` | V's non-weight files | protocol-sensitivity diagnostic, development screen only |

Why P1 is primary: it is the interface of the original backbone that both post-training recipes started from; B's and
T's archived outputs are already valid under it (identical rendered inputs), and choosing V's view would inject one
parent's recipe-specific system prompt into every model. Consequences, fixed now:

* B and T are not regenerated under P1 (their P0 outputs are P1 outputs; T's one extra stop id is recorded as a known
  difference).
* V is regenerated under P1 on every evaluated subset (`V@P1`). The archived `V@P0` stays the canonical VERO score and
  is reported separately; differences between `V@P0` and `V@P1` measure VERO's prompt dependence, not merging.
* P2 diagnostic (development screen, seed 42 only, never used for selection): `B@P2`, `T@P2`, `M050@P2`, `CV050@P2`,
  `CT050@P2`; `V@P2` is the archived `V@P0`.

Environment drift: archived runs and new runs use the same pinned code, data and decoding but different hardware. The
existing local full-suite seed-42 reruns of B and T (made earlier for another study with this same environment, not
repeated here) are compared row by row with the archived seed-42 runs before any merge comparison is read. Primary
comparisons use archived B/T for all seeds; a sensitivity table substitutes the local seed-42 B/T.

## 4. Content identity and splits

Content key: `(benchmark_id, source_ordinal, source_index, source_row_sha256, source_record_sha256)`; the archive's
`sample_id` is not used as a join key (it differs across models and seeds by construction). Every archived row must
match exactly one prepared-dataset manifest row on all five fields plus `media_set_sha256`, and every model x seed slice
must cover the same key set with no duplicates; otherwise ingestion fails.

Groups: rows are connected when they share any image (per-image sha256, across all benchmarks) or a known problem
family (WeMath `ID`). Connected components are the independent units for splitting and resampling.

Partition (seed 271828): a component is ranked by `blake2b("271828|" + component_key)`. Components that span several
benchmarks go to development when their rank fraction is < 0.20. Then, per benchmark, single-benchmark components are
added to development in rank order until that benchmark's development rows reach `round(0.20 * rows)`. Everything else
is confirmation. All seeds of a row and all question variants of a component stay in one partition.

Development screen: per benchmark, development components in rank order until at least 128 rows (all development rows
if fewer), then the best-ranked development component carrying each missing value of the per-stratum columns the
official scorer divides by (TableVQABench `split`; PuzzleVQA `category`; VisualPuzzles `category`, `difficulty`).

The historical suite was examined in earlier work; the confirmation partition is reserved from merge tuning, not
pristine.

## 5. Metrics

Each benchmark has one official primary metric (suite.v1.json). The metric registry records its kind:

* `binary_row`: archived row scores are 0/1 and their mean equals the official aggregate (16 benchmarks).
* `grouped_row`: WeMath; archived rows are 0/1 but the official "Score (Strict)" is a problem-group statistic computed
  with the licensed `ID`/step metadata.
* `aggregate_only`: ChartQAPro, TableVQABench, MathVision, MathVista, CountBenchQA, PuzzleVQA, VisualPuzzles; the archive
  has only the aggregate. Row-level official scores are re-derived from the archived extractions and the licensed
  answers with the pinned scorer functions (no new judge calls).

An adapter is accepted only if it reproduces the archived aggregate of all 9 model x seed slices of its benchmark to
1e-6 (percent units) on the full split, and the score campaign's own aggregate on every new subset run. A benchmark
whose adapter fails stays in full-suite reporting and is excluded from subset macros, which are then renamed (e.g.
"23-benchmark subset macro"). Fractional or grouped scores are never thresholded into binary diagnostics.

Primary subset metric: equal-weight macro over the benchmarks with accepted adapters of each benchmark's official
metric recomputed on the subset's rows, averaged over the evaluated seeds. Category scores: the suite's six categories.

## 6. Model matrix (initial: three constructed checkpoints, zero trained)

| id | weights | role |
|---|---|---|
| `M050` | 0.5 V + 0.5 T | complementarity test |
| `CV050` | 0.5 V + 0.5 B | VERO update-shrinkage control |
| `CT050` | 0.5 T + 0.5 B | TRACE update-shrinkage control |

`M050 - CV050 = 0.5 (T - B)` at the weight level. Construction: streaming over safetensors in bounded slices, every
float tensor interpolated in float32 from bf16 parents and cast once to bf16 (round-to-nearest-even), no optimizer, no
trainer state; all 729 tensors must match by name, shape and dtype in all parents; the output keeps B's tensor-to-shard
layout; parents are hashed before and after.

Tiers: smoke (4 rows per benchmark, all P1 models) -> development screen (seed 42; P1 models plus the P2 diagnostic) ->
lock -> confirmation (P1: `V@P1`, `M050`, `CV050`, `CT050` on the full confirmation partition, seeds 42, 43, 44).
Tier 4 (full suite) is not planned.

## 7. Statistics and selection

Uncertainty: paired bootstrap over connected components (10,000 replicates, seed 314159), all models and seeds resampled
together, equal benchmark weights preserved, grouped metrics recomputed per replicate. Intervals are conditional on these
checkpoints and this suite.

Primary confirmatory family (confirmation partition, primary subset macro, Holm-adjusted two-sided bootstrap p-values):
`M050 - T`, `M050 - V@P1`, `M050 - CV050`, `M050 - CT050`. Everything else (categories, benchmarks, slices,
capture/damage, P2) is descriptive or exploratory and labelled so.

Selection: there is no coefficient search, so there is nothing to select; all three midpoints and both controls go to
confirmation as prespecified. No category regression margin is set, so no noninferiority or "no harm" claim is made.

## 8. Behaviour analysis (Stage B)

On `binary_row` benchmarks (and re-derived binary rows of `aggregate_only` benchmarks whose official metric is a row
mean): corrections/regressions of T and V against B with both unconditional and conditional rates, all eight
`(B, T, V)` cells, parent overlap, per seed and averaged, plus cross-repeat persistence of T-only/V-only sets. Length,
finish reason, extraction status and judge usage per benchmark. A local casebook (never committed) with balanced
examples. At most three hypotheses are frozen in `behavior_report.md` before any merge result is read.

## 9. Outputs

Committed: code, configuration, tests, this design, and sanitized aggregate tables/reports (no benchmark inputs, no raw
responses, no machine paths). Local only (work root): archives, aligned predictions, merged weights, run directories,
casebook.

## Amendments

All amendments below were made before any merged-model generation existed (the development wave had not started).

**A1 (2026-11-02, identity): `countqa-cached-answer-normalization`.** Every archived CountQA row has the prepared
manifest's ordinal, index, media and record hash but a different `source_row_sha256`. The canonical producer hashed the
cached integer answer; the public code (and the prepared data) hashes it as a string, which the release's own post-run
patch ledger records as `countqa-cached-answer-normalization`. Recomputing both hashes from the licensed TSV reproduces
the archived and the manifest value on all 1,528 rows. Archive CountQA rows therefore take the manifest identity
(config `identity.producer_row_hash_aliases: [countqa]`), keep their archived hashes in
`archived_source_row_sha256`/`archived_source_record_sha256`, and are marked in `row_identity_status`. The alias
applies to archive rows only; new runs must match the manifest directly. No other benchmark needs an alias.

**A2 (2026-11-02, metrics): WeMath on subsets.** The pinned "Score (Strict)" is `(525 - 0.5 IG - RM - IK) / 525`, which
scores every problem group absent from a subset as fully mastered (on 105 groups the pinned value is 90.10 where the
groups present give 50.48). The study's WeMath metric is `100 (G - 0.5 IG - RM - IK) / G` over the G complete problem
groups present, with the pinned two-decimal rounding only at G = 525. It equals the pinned value on the full split for
all 9 archived slices. New subset runs are checked against the score campaign's own fixed-525 value
(`campaign_value`), and the study value is what enters subset macros and bootstraps.

**A3 (2026-11-02, metrics): row diagnostics.** Binary row diagnostics (corrections, regressions, eight-cell tables,
overlap, capture/damage) use the 21 benchmarks whose official row score is 0/1 and whose official value is the pooled
mean of those rows: the 16 `binary_row` benchmarks plus MathVision, MathVista, CountBenchQA, PuzzleVQA and
VisualPuzzles (re-derived rows). Excluded from binary diagnostics, but kept in every macro through their adapters:
ChartQAPro (fractional row scores), TableVQABench (mean of five split scores; the pinned row score is a component mean)
and WeMath (the official unit is the problem group).

**A4 (2026-11-02 02:05 MST, schedule): confirmation runs concurrently with the development screen.** At the
supervisor's request to use idle GPUs, the confirmation wave starts before the development screen has finished rather
than after it. This changes no decision: Section 7 fixes the confirmed set (all three midpoints and both controls, plus
V@P1) and the primary family in advance, there is no coefficient search, and the selection lock
(`reports/selection_lock.json`, written before any confirmation run) freezes the config, design, split, merge and view
hashes. Every model's own smoke tier is still a precondition for its confirmation runs counting (a confirmation run of a
model whose smoke or development run fails technically is discarded and re-run after the repair). Development results
are still reported first and are the only input to any Stage D proposal.

**A5 (2026-11-02 02:15 MST, environment): two serving/scoring robustness fixes, no effect on scores.**
(a) pandas 2.3.3, as pinned, segfaults in its numeric type inference on strings that begin like a float with an
exponent of 11 or more digits. The scorers load prediction workbooks whose `request_hash` column holds SHA-256 hex
strings, and about 1 in 2,000 such hashes has that shape (`7649e84210128867284d…`), so a score worker crashes
deterministically on the corresponding benchmark (observed on M050@P1 LogicVista; which rows hit depends on the served
model name, which enters the request hash). `jobs/pandas_guard/sitecustomize.py` is placed on the runner's PYTHONPATH:
in a private copy passed to the parser it rewrites an exponent of 10 or more digits as `9999`, which leaves every
value's classification unchanged (non-numeric strings stay non-numeric with their original values; numeric strings
convert to the same overflowed float). (b) Generation servers on 44-48 GB cards use gpu-memory-utilization 0.80 instead
of 0.90 (two A40 servers hit CUDA OOM in the vision tower with VERO-interface prompts); a serving parameter only. Runs
started before these fixes that completed are kept; incomplete ones resume under the fixed runner. The runner also
resets score-queue entries left "running" by a dead worker (held run lease), which the protocol would otherwise
reclaim only after 12 h.

**A6 (2026-11-02 02:37 MST, record): amendment timing, the lock, and the smoke size.** (a) The sentence under
"Amendments" saying every amendment predates merged-model generation holds only for A1-A3; it stays as written because
the lock covers it. A4 was written after merged-model generation had started (smoke tier: the first CV050@P1 smoke
payloads date from 01:47 MST) and before the lock; its text is covered by the lock written at 02:04:27 MST, so its
02:05 stamp is rounded. A5 was written after the confirmation runs had started generating (first confirmation run files
02:11 MST; A5 cites a crash observed while scoring M050@P1) but before any confirmation run had scores. (b) The
selection lock (`reports/selection_lock.json`) hashed this file as it stood through A4; A5 and A6 were appended after
it. The lock check (`evaluation.verify_lock`, run by `plan-eval --tier confirmation` and by `confirm`) recomputes every
locked hash and accepts this file only if the locked text is an exact prefix of the current one ending at an amendment
boundary; `confirm` records the appended amendments (A5, A6) in its summary, and any edit to locked text, Sections 1-9
included, fails the check. `confirm` also takes the confirmed labels and seeds from the lock, checks each confirmation
run's served view revision against the lock, requires each confirmed model's smoke and development runs to be complete
(A4), and refuses ingested rows that no longer match their run directory. Neither A5 nor A6 changes the confirmed set,
the primary family, the split, the merges or the views. (c) Section 6's "4 rows per benchmark" is the smoke target. The
frozen smoke list takes whole components and covers the scorer's per-stratum values, so it holds 135 rows: 4 for each
of 19 benchmarks, PuzzleVQA 32, VisualPuzzles 9, TableVQABench 7, WeMath 6 and ChartQAPro 5.

**A7 (2026-11-02 03:35 MST, before any confirmation run had scores): near-duplicate image twins, the pandas guard's
wording, and serving parameters.** (a) The partition joins rows on byte-identical image files. A review found
cross-benchmark re-encodings of the same picture with the same question and answer that are therefore in different
components: MathVista/MMStar (285 image pairs, 294 row pairs) and MME-Reasoning/PuzzleVQA (67 image pairs, 171 row
pairs), verified at full resolution (`splits/near_duplicate_twins.csv`, 465 row pairs; a lower bound: resized copies and
within-benchmark duplicates were not searched). The frozen partition is kept, because development and confirmation
runs were already in flight and there is no selection on development (Section 7) that a development twin could leak
into. For every bootstrap (Stage B, screen, confirmation) the twins' components are merged into one resampling unit
(`splits/bootstrap_units.parquet`: 26,422 components into 26,070 units), so twins are never treated as independent
groups. A prespecified sensitivity analysis repeats the primary family without the 130 confirmation rows that have a
development twin (62 MMStar, 45 MathVista, 15 MME-Reasoning, 8 PuzzleVQA). Stage B tables are regenerated with the
units; their point estimates are unchanged. (b) A5(a)'s description of the pandas guard was imprecise: pinned pandas
reports an exponent overflow as unparsable (ValueError) and an underflow as 0.0, and an exponent written with leading
zeros (`1e00000000001`) is not large. Hash columns, the only cells that reach the guard in this study (none were found
in the data with a leading-zero exponent), keep their values and object dtype either way; the guard's pattern is
narrowed to exponents with 10 or more significant digits for every later snapshot. (c) Serving parameters that differ
from the 8×H100 reference configuration and change no decoding setting: requests in flight per generation endpoint
64/128 on 40-80/≥80 GB cards (protocol 32), the judge's tensor-parallel degree chosen per node (1 on ≥90 GB cards, 2 on
44-80 GB cards, with 8,192 batched tokens when headroom is tight), and judge weight verification by marker. Each run's
`timing.seed<S>.json` records the values it used.

**A8 (2026-11-03 00:10 MST, after the confirmation tables had been read): one confirmation run rescored after a judge
fault.** (a) Found during the independent verification of the confirmation report: M050@P1 seed 44 scored 16.98 on
MathVerse against 43.65 and 47.30 on seeds 42 and 43 (MathVista 67.25 against 78.00/77.75; LogicVista 47.21 against
50.84/51.68), while its responses look like the other seeds' (length, finish reason, `\boxed{}` rate). Its scoring half
ran on grn023 (6× A800 40 GB) with the Qwen3-32B judge at tensor-parallel degree 4, the only run of the study with that
judge configuration (A7(c): every other run used degree 1 or 2). Its batched judge outputs do not match their prompts:
on MathVerse the judge-extracted answer equals the response's own final `\boxed{}` answer on 14.8% of rows, against
43–88% on all 24 other runs (smoke, development, P2 and confirmation), and one batch returns the same output ("130.5")
for two different problems. The cause inside the serving stack was not isolated. (b) Repair, a technical repair and not
a selection: the run's generation outputs are kept unchanged; its scoring outputs (the seed's score campaign directory,
campaign manifest and completion files, validation and summary files, and its 48 extraction/score archive-spool
descriptors with their payloads) are moved, not deleted, to `runs/m050-p1/confirmation/quarantine_seed44_tp4_judge/`
with a `QUARANTINE.json` record, and the scoring half is rerun (`eval_subset.sh --judge-only`) on ≥90 GB cards, where
the judge runs at degree 1. This rule was fixed before any rescored value was read. (c) Nothing locked changes: the
confirmed set, the primary family, the split, the merges and the views are as locked; the rescored run is re-ingested
and `confirm` is rerun on all twelve runs. The confirmation tables computed with the faulty scores are superseded and
are not reported.

**A9 (2026-11-03 02:10 MST, record): corrections to A7(c) and A8 found by the second verification pass.** (a) A8(a)
describes the fault as outputs that "do not match their prompts" and cites one batch returning "130.5" for two
problems. Both are wrong descriptions of the mechanism: the rescored judge returns "130.5" for the same two rows (both
responses conclude 130°), and the faulty outputs mostly concern their own prompt (61.5% begin with the row's own
`\boxed{}` answer). What went wrong is degraded generation: the faulty outputs often run on and copy text from the
judge prompt's few-shot examples (the phrase "They give the solutions to the equation" appears in 133 of 630 MathVerse
extraction outputs, against 0 of 630 after the rescore; 297 against 540 are short single-line answers). The agreement
numbers in A8(a) (14.8% against 43–88%) and the repair in A8(b) stand. The cause inside the serving stack remains not
isolated. (b) Besides MathVerse, MathVista and LogicVista, the faulty judge also changed MathVision (470 of 1,702 judge
outputs differ; seed-44 score 27.18 before, 27.92 after) and, slightly, other judge-routed benchmarks; the rescore
replaced all of them. (c) A7(c)'s "Each run's `timing.seed<S>.json` records the values it used" holds only for attempts
started after A7; for earlier attempts the judge's tensor-parallel degree is in `logs/vllm_judge.*.start.log`, which
shows degree 1 or 2 for every scoring attempt except the A8 one. (d) A8's 00:10 stamp is rounded: its text was final at
00:07:47 MST, after the quarantine move (00:07:14) and before any rescored value existed (first rescored descriptor
00:17).
