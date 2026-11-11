# TRACE–VERO follow-up v1: component transfer, co-adaptation, evaluation reliability

Study id `trace_vero_followup_v1`. Written 2026-11-05, before any hybrid checkpoint was built or any new generation
existed. Later changes are appended under "Amendments" with a reason and a timestamp; text above that heading does not
change after the first GPU task starts. This is an **exploratory** follow-up to the completed merging study
(PR maveryn/trace#14). It asks which post-training updates help when transplanted into the original model and which
gains need the visual and language components to stay adapted together. The aim is an explanation of the existing
checkpoints, not a higher-scoring merged model.

## 0. Approval and source anchor

* Approval: the user asked on 2026-11-05 to run the experiments of the execution guide "TRACE–VERO Training-Free
  Follow-up Experiments" (plan v1.0, 5 Nov 2026) with as many GPUs as needed. This record is the design/amendment the
  guide requires before GPU execution; the PR #14 decision memo did not authorize a component campaign, this request
  does.
* Code anchor: PR #14 head `ce60def7dff3613618b9b53f3e2b90b1bea11a0d` (base `af32fe0e668eb878e909daef7322ac1c4ae20f64`)
  in `farhanishmam/trace`. The follow-up lives in `experiments/trace_module_followup/` on its own branch;
  `experiments/trace_merge/` (its code, frozen design, lock, results bundle) and `rlvr/` are not modified. The
  follow-up imports the reviewed helpers (streaming safetensors I/O, compatibility audit, ownership map, view contract,
  metric adapters, statistics) and runs the reviewed GPU runner unchanged.
* Work root: a new local root (`TRACE_FOLLOWUP_ROOT`). The reviewed study's work root is read only.
* Not part of this plan: RL, SFT, distillation, solve-rate filtering, Qwen3 training, merge or coefficient search,
  averaging, uploads.

## 1. Assets

| Role | Source | Pinned revision |
|---|---|---|
| `B` | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| `T` | `maveryn/trace-qwen2.5-vl-7b` | `4d0f1ae8ee25022058090dbdbff61957ece7331d` |
| `V` | `zlab-princeton/Vero-Qwen25-7B` | `180e84be5acb2aa887cf51015b84b6a6e453ee90` |

The parent weight files must match the reviewed study's recorded sha256 (`audit/parent_weight_sha256.json` of the old
root); the architecture, token identity, tensor name/shape/dtype and effective processor audit is rerun. VERO's model
card names its base family but not the exact base revision; that provenance limitation is recorded, not resolved.

## 2. Components and the four hybrids (F1)

Stored tensors are partitioned with the audited `tensor_group` map; there is no fallback owner and an unclassified key
stops the build. `E` = `visual.patch_embed.*` + `visual.blocks.*` (expected 1 + 384), `J` = `visual.merger.*` (5),
`L` = `model.embed_tokens.weight` + `model.layers.*` + `model.norm.weight` + `lm_head.weight` (1 + 336 + 1 + 1).
Frontend `F = (E, J)` (390 tensors), language `L` (339), total 729; recomputed from the pinned checkpoints.

| Cell | Frontend | Language | Meaning |
|---|---|---|---|
| B | F_B | L_B | original model |
| T | F_T | L_T | full TRACE |
| V | F_V | L_V | full VERO |
| TF | F_T | L_B | TRACE frontend in base |
| TL | F_B | L_T | TRACE language updates with base frontend |
| VF | F_V | L_B | VERO frontend in base |
| VL | F_B | L_V | VERO language updates with base frontend |

Construction: every output tensor is the exact stored BF16 bytes of its donor (no averaging, no conversion), in B's
tensor-to-shard layout, staged and atomically finalized after completeness, index, per-tensor donor-hash equality and
finiteness checks; parents unchanged before/after. Manifests record parent pins and hashes, per-tensor source and
sha256, code hashes and output hashes. Required tests before inference: identity (all-B / all-T / all-V assignment
reproduces that parent tensor for tensor through the builder's copy path), ownership (every key one owner; TF/TL and
VF/VL partition their parent's changes relative to B), runtime (all views load under the pinned stack; smoke outputs
finite; no missing-key warnings; untied embeddings/head as in B's config; identical processor for identical inputs),
artifacts (immutable manifests; no credentials, benchmark content or weights in tracked files; no training or upload
command).

## 3. Interfaces (held fixed within each matrix)

* **P1 (primary), backbone interface**: B's non-weight files (the reviewed study's `P1_backbone` list) for all seven
  cells. T@P1 differs from archived T@P0 only by B's extra stop id; V@P1 replaces VERO's native template.
* **P2, VERO interface**: V's non-weight files (`P2_vero` list) for all seven cells, on a smaller fixed subset
  (Section 4). P2 is a matched sensitivity matrix, not a deployment choice; no cell receives whichever interface
  scored best.
* Decoding: temperature 0.6, top-p 1.0, top-k disabled, presence penalty 0, repetition penalty 1.0; 4,096 generated
  tokens; same stop-token policy within each interface; image processor bounds 3,136–12,845,056 pixels. Same protocol
  code (`af32fe0`), pinned VLMEvalKit `a8b12bf1`, Qwen3-32B judge `9216db57`, and the reviewed study's runner and
  serving amendments (A5, A7(c)). Judge tensor-parallel degree is 1 on ≥ 80 GB cards and at most 2 elsewhere (never 4,
  the A8 fault configuration); the judge-agreement health check (boxed vs extracted MathVerse answers, floor 30%) runs
  after scoring for every run, and a failing run is rescored on ≥ 90 GB cards with the failed outputs quarantined.

## 4. Data and execution stages

Row lists are the reviewed study's frozen partition (copied, hash-checked): `smoke` (135 rows), `development_screen`
(2,910 development rows), `confirmation` (26,228 rows). The confirmation partition was examined in the earlier study,
so it is **not unseen**; this reuse is exploratory and nothing here is presented as independent confirmation.
Bootstrap units are the reviewed `splits/bootstrap_units.parquet` (connected image/problem components with
near-duplicate twins merged; WeMath problem groups kept together).

| Stage | Cells | Interface | Rows | Seeds | Purpose |
|---|---|---|---|---|---|
| Smoke | all 7 | P1 and P2 | smoke | 42 | loading, prompt rendering, generation completeness, extraction, judge health |
| Screen | all 7 | P1 | development_screen | 42 | fault detection and reporting |
| Main matrix | all 7 | P1 | confirmation | 42, 43, 44 | the analysis set: 21 checkpoint–seed jobs |
| Template sensitivity | all 7 | P2 | development_screen | 42 | interface dependence (one-seed diagnostic) |

References are run contemporaneously (B, T and V regenerated under P1 in the current stack) rather than reused.
Schedule (as in the reviewed study's A4): main-matrix runs may start before the screen finishes, because nothing is
selected on the screen. A hybrid's main-matrix runs count only if its own smoke run passes technically; a run of a
model whose smoke run fails technically is discarded and rerun after the repair.

**Technical-stop rule (prespecified).** A cell is a technical stop, reported and not analysed, only for an
implementation failure: a build or verification check fails; the view does not load or logs missing/unexpected keys;
non-finite logits/outputs; or, on its smoke or screen run, more than 50% of responses empty or more than 80% truncated
at the 4,096-token limit. Low accuracy is a result, never a stop; no hybrid is dropped for performing poorly.

Failed attempts and repaired outputs stay in separate namespaces (quarantine directories), never silently replaced.
Costs (GPU-hours, generated tokens, storage, judge use, retries) are recorded from the runs' timing files.

## 5. Scoring (F0)

Official scores (pinned adapters, native aggregates: WeMath group score, TableVQABench split macro, etc.) are the
primary scale. `format-normalized-v2` is a diagnostic scale built only from development-partition responses of the
reviewed study's stored outputs, frozen (spec + code sha256 + tests) **before any hybrid outcome is inspected**, and
applied with an identical rule to every model and interface. (The guide's label is `format-normalized-v1`; v1 was
frozen first, then a blinded development check of TableVQABench found two defects: a `\boxed{}` answer could override
the pinned "both labels present = unparsed" rule on vtabfact, and LaTeX escapes such as `55\%` were kept. The rule was
bumped to v2 before any confirmation value was computed under v1 and before any new run existed; the v1 spec is kept
read-only. Frozen spec: `audit/scoring_spec.format-normalized-v2.json`, sha256 `a229a564…`.) The rule only raises
scores: multiple-choice rows touched only where the pinned parser returns no option (PuzzleVQA, VisualPuzzles), and
TableVQABench rows without full credit re-scored by the pinned split scorers after two narrow rewrites; the other 21
benchmarks keep their official score. The extractor sees the response and the allowed option
labels, never the gold answer; it reads explicit final-answer regions only; conflicting final candidates stay
unresolved and unresolved rows stay in the denominator; explicit wrong answers stay wrong. TableVQABench is audited
separately by sub-split and answer type; its official split aggregation and tolerances are kept. Official and
normalized results are always reported side by side; the normalized scale is never the official benchmark score.

## 6. Analysis

Let `S(F, L)` be one benchmark's official score or the 24-benchmark equal-weight macro (percentage points), averaged
over the three seeds, all cells on the same rows, interface (P1), seeds and scoring version. For `P ∈ {T, V}`:
`g_F^P = S(F_P, L_B) − S(F_B, L_B)`, `g_L^P = S(F_B, L_P) − S(F_B, L_B)`,
`I_P = S(F_P, L_P) − S(F_P, L_B) − S(F_B, L_P) + S(F_B, L_B)`, so `S(P) − S(B) = g_F^P + g_L^P + I_P` exactly. Also
reported: component-reversion effects `S(P) − S(F_B, L_P) = g_F^P + I_P` and `S(P) − S(F_P, L_B) = g_L^P + I_P`. The
interaction is scale-dependent, not an intrinsic measure of synergy.

* **Primary family** (prespecified): `g_F^T, g_L^T, I_T, g_F^V, g_L^V, I_V` on the official macro, Holm-adjusted
  two-sided paired bootstrap p-values. The identical family on `format-normalized-v1` is a separately identified
  sensitivity analysis; the scoring version with better significance is never selected.
* **Bootstrap**: paired cluster bootstrap over the bootstrap units, 10,000 replicates, seed 314159; all seven models
  and all three decoding seeds stay attached to each sampled unit; native benchmark metrics recomputed inside each
  replicate (not raw row means); benchmark weights fixed; number of valid groups reported. p-values from the centered
  bootstrap distribution (two-sided). Implementation tested on synthetic data with known effects.
* **No noninferiority margin is set.** No hybrid is said to "retain the full gain"; gaps and intervals are reported.
  Intervals describe evaluation uncertainty for fixed checkpoints, not training-run variation; decoding seeds are not
  training replications. Non-significance is not equivalence.
* **Exploratory, labelled so**: per-benchmark and per-category effects, cross-parent differences (e.g. `g_F^T − g_F^V`),
  P2 template matrix, behavioural outputs (corrections/regressions vs B, full-parent successes retained/lost by the
  hybrids, on validated binary-scored benchmarks only; special success sets defined on seed 42 and retention measured
  on seed 43), length/truncation/format rates.
* Prior M050/CV050/CT050 results are kept separate from the new hybrid results.

## 7. Conditional extensions (F2–F4) and stop rules

After the core matrix, a short decision record chooses **at most one** F2 extension using development evidence (screen
and P2 development runs) and the F1 pattern, with recipes locked before they are evaluated: F2a (encoder vs merger for
the informative parent), F2b (cross-parent swaps `X_VT = (F_V, L_T)`, `X_TV = (F_T, L_V)`), or F2c (coarse decoder
block-group reversion). **Stopping rule**: if effects are unstable, tiny relative to uncertainty, or explained by
scoring/template artifacts, stop after F1. F3 (one norm-matched VERO checkpoint, λ computed from full-precision audited
norms, never hard-coded) runs only if update magnitude remains an explanatory question after normalization and F1.
F4 (controlled TRACE probe) is implemented only if the module study yields a concrete question it can answer; it is
not part of this execution unless that record says so. The final memo ends with stop, targeted evaluation, or a later
training proposal; it launches no training.

## 8. Outputs

Work root (local only): hybrids, views, runs, aligned predictions, raw audit samples. Tracked (sanitized aggregates
only, no prompts, gold answers, images, responses, weights, machine paths or tokens): `design/followup_v1.md`,
`audit/component_map.json`, `audit/scoring_changes.*`, `hybrids/*/manifest.json`, `manifests/evaluation.json`,
`tables/component_effects.csv`, `tables/behavior_and_format.csv`, `reports/followup_findings.md`.

## Amendments

**A1 (2026-11-09 19:01 MST, before any rescored value existed): two scorer raises in the P2 template matrix; fix-4 and
fix-5 in a private protocol overlay; the TF@P2 and VL@P2 scoring quarantined and redone.** (a) Failures. Both runs'
generation is complete and validated; their score campaigns failed deterministically, four attempts each, with the judge
at tensor-parallel degree 1 (not the A8 fault). TF@P2: af32fe0's ChartQAPro adapter
(`run_official_vlmevalkit_saved_score._chartqapro_final_sentence`) reads the first line after the last "the answer is"
marker with `splitlines()[0]`, which raises IndexError when a response ends exactly at the marker (one row). The
official phase fails and the campaign stops before its direct and MME-Reasoning phases, so CharXiv, TableVQABench,
EvoChart, MathVision, MathVista, MathVerse, LogicVista and MME-Reasoning were never scored (not failed). VL@P2: one
MathVerse score judgement was still truncated (finish reason length) at the 1,024-token ceiling of the judge client's
token-limit retries in the first of fix-3's five attempts (temperature 0), and the client raises inside `run_cached`
before fix-3's loop can count the attempt (fix-3 left that case failing). MME-Reasoning was never scored because its
phase runs only after the direct phase succeeds. (b) Fixes, scorer-side and fail-closed like the merging study's fixes
1–3, each changing only rows on which the canonical producer raises. **fix-4** (MathVerse score stage,
`run_external_benchmark_score_queue.py`): the five score attempts call `run_cached` with `return_unresolved=True`, as
the same file's LogicVista schedule and the MME-Reasoning scorer already do, and an output still incomplete after the
token-limit retries (length/max_tokens, or empty) counts as one failed attempt without a 0/1 decision (upstream
VLMEvalKit's `MathVerse_auxeval_score` treats any unusable output so). The next attempt runs with fix-3's temperature
schedule, and a row still without a decision after five attempts is scored incorrect through fix-3's existing record.
Rows with an incomplete attempt are listed in `mathverse_score_incomplete_attempts.json` and counted in `scores.json`
(`mathverse_incomplete_score_attempt_rows`); the judge cache contract `mathverse_score` becomes
`…-judgement01-v4+trace-cert-fix-3+trace-vero-followup-fix-4`. **fix-5** (ChartQAPro adapter,
`run_official_vlmevalkit_saved_score.py`): nothing after the last final-answer marker is the empty final answer "",
which the pinned ChartQAPro scorer scores as usual (it reads the cell back as it reads an empty response); the adapter
contract becomes `chartqapro-official-final-answer-v2+trace-vero-followup-fix-5` and its record counts these rows
(`empty_final_answer_rows`). Complete judge outputs and every other response take exactly the old code path (tested on
synthetic inputs against af32fe0's behaviour in `tests/test_protocol_fixes.py`). Not broadened, recorded only: the same
`run_cached` raise can still fail CharXiv's judge (it retries on length and has no attempt loop), and an empty
MathVerse, MathVision or MathVista extraction still fails its slice (their official extraction route does not retry on
length, so a truncated extraction is used as it is); no run has reached either. (c) One rule for every cell: the fixes
apply identically to every model and interface. All 45 other completed seed-runs (smoke, development_screen and
confirmation) never took either raising path: af32fe0's adapter, rerun over their ChartQAPro score inputs, raises on no
row; their MathVerse score caches hold no output still incomplete; and no other scoring log shows either error. The
fixes cannot change their scores, so they are not rescored. A run still in progress under the shared clone that reaches
either path is repaired the same way. (d) Code: the shared protocol clone, which other campaigns use and whose code
hashes they record, is not modified. The fixes live in a private overlay (`protocol_clone_followup`, beside the work
root): a local clone of the shared clone, detached at af32fe0, with the shared clone's own post-release diff applied
(all 12,241 tracked files byte-identical to the shared clone's; its code hash `d9319e7d…` and evaluator hash `7bae61e2…`
reproduced) plus fix-4 and fix-5; `.work` (data, models) and `external/VLMEvalKit` (with fix-1) are symlinks to the
shared clone's. With the fixes the code hash is `fd8fccd5…` and the evaluator hash is unchanged. `dispatch/env_fix45.sh`
is `env.sh` with `TVM_PROTOCOL_CLONE` set to the overlay; the runner snapshot is unchanged. (e) Quarantine and
rescoring, as A8 of the merging study: the seed-42 scoring outputs of TF@P2 and VL@P2 development_screen
(`scoring/seed_42`, the campaign manifest, `validation.seed42.json`, `summary.seed42.{json,md}`, and the 33 and 44
extraction/score archive-spool descriptors with their payloads) are moved, not deleted, to
`runs/<slug>/development_screen/quarantine_seed42_scorer_raise/` with a `QUARANTINE.json` record; generation outputs and
descriptors, `validation.generation.seed42.json` and all logs stay. The scoring half is rerun with `eval_subset.sh
--judge-only` (`dispatch/wave3.tasks`), the judge at degree 1 on ≥ 80 GB cards. Rows, views, interfaces and the analysis
plan are unchanged; this rule was fixed before any rescored value existed.

**A2 (2026-11-09 19:45 MST): F2 choice, execution changes, and the two seed-43 reruns.** (a) The F2 decision record
`design/decision_F2.md` (written 18:40 MST, before any F2 checkpoint was built) selects F2c on TRACE: three exact-copy
checkpoints TR0/TR1/TR2, full TRACE with decoder layers 0–8 / 9–18 / 19–27 restored to B. They are evaluated under P1
on smoke, the development screen (seed 42) and the confirmation partition (seeds 42–44). B and T come from the F1 runs.
Ownership uses layer-range parts (`fine_parts`/`fine_partitions` in the config); the F1 recipes and their hashes are
unchanged (checked against the finished manifests). Builds are verified tensor for tensor against their donors
(`audit/build_verification_F2c.json`). F2c is reported as exploratory reversion effects `R_k = S(T) − S(TR_k)` with
descriptive Holm over the three, by `trace_followup/analyze_f2c.py`. The decision record discloses that a rough
campaign-aggregate look at the partly complete F1 confirmation runs was taken during run-health checks; the choice
uses development evidence only. (b) The first per-card filler ran in tmux on a login node, which died on 2026-11-07.
The two main-matrix runs it was tracking (V@P1 and TL@P1, confirmation seed 43) had been cut off on 2026-11-06 when
their allocations ended. Both were relaunched on 2026-11-09 with the same task lines; the runner resumes finished
requests (V@P1 kept 6,262 of 26,228 rows, TL@P1 6,264). Fillers now run as CPU host batch jobs. A generation-only mode
lets single 40–79 GB cards run one-tier main-matrix tasks. The runner itself finalizes the generation and writes its
`PENDING_JUDGE` marker there, and the filler sends the judge-only half to a judge-capable unit (TP ≤ 2, ≥ 80 GB single
cards or 2–3 card groups). This is a serving/placement change only; decoding, judge and scoring settings are as in §3.
Preempted attempts (rc 143) resume like any interrupted run.

**A3 (2026-11-09 21:05 MST, before any rescored value existed): an archive raise on V@P1 confirmation seed 43; fix-6.**
The run's generation completed (26,228 rows), but all four scoring attempts stopped in the official phase right after
VisualPuzzles scored. The protocol's local-archive sanitizer (`trace_eval_hf_archive_lib._is_media_path`) refused
extraction record 806: the adapter's extracted answer text happens to look like an image file name, and the canonical
producer raises in the same place. **fix-6** (overlay only) adds the extracted answer `normalized_extraction.value` to
the protocol's own declared opaque answer-text fields, where `model_response` and the judge texts already are. Only that
scalar string is exempt; media-named fields and nested payloads stay strict (tests). It changes nothing on a record the
canonical producer archives, because every other run archived without raising, and it changes no score. The seed-43
scoring outputs and that seed's extraction/score archive descriptors (65 paths) are moved, not deleted, to
`runs/vero-p1/confirmation/quarantine_seed43_archive_raise/`. Seeds 42 and 44 and all generation outputs stay. The
scoring half is rerun with `--judge-only` under the overlay (fix-4/5/6), judge at TP ≤ 2. The overlay's rebuild script and
delta (`dispatch/protocol_overlay/trace_vero_followup_fix6.diff`) reproduce the live overlay exactly. This rule was
fixed before any rescored value existed.

**A4 (2026-11-09 23:25 MST, before any V-NM output existed): F3 runs; the realized norm is short of its target.**
`design/decision_F3.md` (22:55 MST) records why update magnitude remains an explanatory question after F0 and F1, and
locks one norm-matched VERO checkpoint, `VNM = (1 − λ) B + λ V`. λ = ‖Δ_T‖_F / ‖Δ_V‖_F, recomputed in float64 over all
729 stored tensors: λ = 0.226576696 (‖Δ_T‖ = 1.993839, ‖Δ_V‖ = 8.799842). VNM was built by the reviewed streaming
interpolator (float32 accumulation, one BF16 cast) into `scaled/VNM` (`trace_followup/normmatch.py`;
`audit/f3_update_norms.json`, `audit/build_verification_F3.json`). After serialization, ‖VNM − B‖ = 1.5180 against
the target λ‖Δ_V‖ = 1.9938, a relative residual of −23.9%; 76.7% of stored values are bitwise equal to B's, and none
are non-finite. Rounding the small scaled update back onto B's BF16 grid absorbs about a quarter of the intended
movement. As the guide requires, this residual is reported and λ is not retuned (no accuracy had been seen). VNM is
therefore norm-matched in coefficient, not in realized norm: its realized update is 0.76 of TRACE's. Evaluation is
as locked: P1 smoke, screen and confirmation seeds 42–44, plus the P2 smoke and screen, with the F1 B/T/V runs as
references; analysis in `trace_followup/analyze_f3.py`.

**A5 (2026-11-10 00:45 MST, before any replication run existed): replicating the interface diagnostic.** Design §4
evaluates the seven cells under P2 on the development screen with one seed. The guide says to replicate important
effects before drawing a firm conclusion about interface interaction. Seed 42 showed large interface dependence:
P2 − P1 is −4.0 for B and +3.3 for V, and VERO's gain over B is +8.3 under P2. The same 2,910 development-screen rows
are therefore evaluated with seeds 43 and 44 for all seven cells under both P1 and P2 (28 runs, config tier
`replication`), so that P1 and P2 have three seeds each. The stack, interfaces, rows and scoring are unchanged. The
replication is reported with the seed-42 diagnostic as an exploratory three-seed interface table on the development
screen. It is not a confirmatory test, it selects nothing, and the confirmation-partition P1 analysis (the primary
family) is unaffected.
