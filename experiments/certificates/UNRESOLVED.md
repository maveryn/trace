# Unresolved assumptions and open decisions (M1 exit, updated 2026-10-23)

Status at M1 exit (2026-10-18), updated after the M1 evaluation-layer review (scorer
`cert-scorer-1.1.0`, adapter verifiers 1.1.0, export `cert-export-v2`; the frozen data moved from
`data/v1` to `data/v2`, see section 9). The review gate closed with PASS on 2026-10-19 (r3, see
section 9); the M2 SFT and M3 RL runners are built and smoke-tested, and section 10 lists what the M2-M3
smokes left open. Each item says what is assumed now, what evidence exists, and what would settle it.
Items marked **[user]** need a user or orchestrator decision; the rest are measured later in the plan
(mostly at the first GPU use, DESIGN 7.1).

## 1. Identity rule thresholds (pre-calibration)

- The rule is `center(b)` strictly inside the candidate and (`IoU >= 0.5` or
  `area(b ∩ c) >= 0.8 area(b)`), with one-to-one dedupe (DESIGN 2.3). The thresholds are **not
  calibrated yet**: they are frozen before M2 from zero-shot E1-dev outputs, with sensitivity
  at IoU 0.3 and 0.7. Changing them after M2 needs a new scorer version for every arm.
- Pinned behaviour that a threshold change must update: lever boxes shifted by up to 19 px
  resolve and 20 px do not (`test_shift_tolerance_matches_the_identity_rule`); a cell-sized box
  around a chess piece resolves at IoU 0.578; a box straddling two table rows does not resolve.
- Chess candidates are **piece boxes**. A box somewhat larger than its board cell fails the rule.
  Piece box versus cell box is left to E1-dev calibration (DESIGN 10).
- Path steps resolve by strict point-in-tile. A point on a tile edge or corner gets no credit by
  design; zero-shot point drift has not been measured.

## 2. Coordinate frames

- **Qwen3.5 rel1000 obedience is untested** (DESIGN 7.1, 10). Planned: a 20-row probe per model
  before the full zero-shot run. If it fails, Qwen3.5 M and G are reported as frame-confounded
  and only A is kept.
- **Qwen2.5-VL shown frame.** A perfect answer given in the 28-px resized frame and read as
  px_source loses about 4% of chess piece boxes: M_shown < 1 on 89 of 928 chess rows (9.6%) of
  the frozen E1/E2 splits. Table, lever and rpg lose none. `G_shown`/`M_shown` are logged for every
  Qwen2.5-VL row; thresholds are not loosened to absorb frame drift (DESIGN 2.3).
- `ann_v1` states no frame. Its G is reported in the family-native frame (shown frame for
  Qwen2.5-VL, rel1000 for Qwen3.5) with px_source as a diagnostic.

## 3. Readings of DESIGN that need confirmation

- **Gated product versus "fails only operand truth".** DESIGN 9 says a wrong value with
  consistent arithmetic "fails only operand truth". Under the per-term gate (D5) a wrong lever
  operand also closes the gate of `ws.product.truth` for that term, so that family fails too.
  The same holds for table `fm.sum.truth` and path `pa.total.truth`. The corruption
  expectations encode the gated behaviour.
- **Repeated `annotation_index`.** Operand and attribute credit go only to scene objects bound by
  exactly one element (`ss.attr.truth`, `ws.weight/distance/product.truth`), so a repeated index
  removes credit instead of double counting it.
- **Vacuous local check.** `ws.product.local` is 1.0 when there are no terms. Local checks never
  enter M.
- **Element-level grounding.** A lever term whose index is a bool or out of range loses its
  binding, so its error class is `grounding`, not `operand`.
- **Path SFT witness.** One optimum is drawn by `instance_seed` from the first 16 enumerated
  optima ("cap 16"). Instances with more than 16 optima (about 0.5% in the sweeps) never use the
  remaining ones as SFT targets; every optimum still scores M = 1.
- **Path legality.** Every tile of a route, the first included, must be passable, and the route
  must start on the unit's start tile; cost reads earn credit only along the longest legal simple
  prefix from the start (review EX-2/EX-3). A start-anchored route that stops short of the
  target keeps legality and its cost reads (endpoints 0.5). The
  `reachable:false` form scores M = 1 only on an unreachable oracle. Unreachable instances are
  quarantined at export (none were generated); coverage lists the form as `defined_not_exercised`.
- **Table semantic key.** It follows DESIGN 6 literally (query, headers, `values_by_row`) and
  ignores the threshold and interval bounds, so dedupe is slightly more conservative than needed.
- **Lever semantic key.** It ignores the render variant (review CONF-13), so a textured-beam
  re-render of a training problem is dropped from E2. Only SFT/RL keys and earlier rows of the
  same test split are deduped against; an E1-test and an E2-test row may still share a torque
  problem (both are evaluation rows).
- **Selection denominators count annotation boxes** (review EX-5): an unreferenced box is an
  extra selection claim (`ss.set`, `fm.rows`, `ws.terms`); operand families are unchanged.
- **A reads the scored object** (review EX-4): A, G and M read one object; `A_legacy` keeps
  TRACE's own reading (legacy extraction of the first dict in the legacy final block). They
  differ only on multi-object responses.
- **`verifier_ms` in the reward.** DESIGN 5 lists it among the reward keys and also says every
  value is in [0, 1]. It is reported in milliseconds, checked finite and >= 0, and never enters
  `overall`. All other keys are asserted in [0, 1].
- **Split names.** Extensions use lowercase split keys (`e1_dev`, `e1_test`, `e2_test`, `sft`,
  `rl`, `fixtures`) rather than DESIGN 2.6's `E1_test`.
- **go/no-go 1(b) counting.** The Section 5 report counts corruption types that keep G
  (|dG| < 0.01) and lower M on every applicable row. Typing and index corruptions (bool or float
  values, negative or out-of-range index, missing total) also satisfy this. The pre-registered
  count uses only content corruptions; both counts are reported.

## 4. Data and build

- **Build order.** The exporter drops test rows whose semantic key appears in SFT or RL, so test
  splits need the SFT/RL receipts. The integrator built E1-dev/E1-test/E2-test first with
  `--no-dedupe` (as instructed), then SFT and RL, then rebuilt the three test splits with dedupe
  from the cached chunks. The final receipts record `dedupe.enabled = true` and the drop counts.
- **Login-node resources.** The data was built on a login node whose per-user cgroup allows
  6.4 CPUs and 6.4 GiB. Twelve export workers plus the TRACE test suite exceeded the memory cap
  and the kernel killed the test run, so the integrator made the exporter stream accepted rows to
  disk (the parent no longer holds every train/trace dict) and built with 6 workers instead of
  12. Worker count never changes the accepted rows.
- **Chunk-cache invalidation.** Any edit of the certificate package or its resources changes
  `cert_code_sha256` and makes every cached chunk stale. The frozen splits record the code sha
  they were built with; a later code edit does not invalidate the frozen data, but a rebuild
  regenerates everything.
- **Private TRACE builder helpers.** Export calls `_finalize_generated_output`,
  `_candidate_validation_rejection_reason` and `CandidateValidationError` from
  `trace_tasks.core.builder`. They are pinned by `test_build_matches_trace_builder`, not by the
  legacy golden file. `prompts.legacy_prompt` calls the private `rlvr_export._build_prompt_columns`
  (pinned by `test_legacy_prompts_are_trace_exports`).
- **Strict prompt drift checks.** If TRACE changes an annotation or answer hint of a supported
  task, every row of that task is quarantined (`annotation_hint_drift`, `answer_hint_drift`)
  until `cert_prompts_v1.json` is updated. This is intended.
- **rel1000 prompts and "pixel".** The rpg question line itself contains "pixel-art". Only the
  annotation hint is stripped of frame words, so the rel1000 prompt for rpg still contains
  "pixel" inside TRACE's own question.
- **System prompt layout.** `cert_system_v1.txt` is DESIGN 4's single paragraph. TRACE's own
  system prompt file puts a blank line between the same two sentences.
- **Overflow images.** Images of candidates generated after a quota filled stay under
  `private/build/` (at most one chunk of 64 per task).
- **Weak lever holdout and table ledger legibility** (DESIGN 10): textured_beam is a modest
  render change; the ledger render also has the lowest-contrast headers, so the table E2-render
  effect mixes novelty with legibility.

## 5. Verifier cost

- Budget is 2 ms per row (DESIGN 5). On the 3,712 frozen evaluation rows the oracle-object
  score takes 0.95 ms median and 1.34 ms p99 (max 1.91 ms). With the shown-frame diagnostics on
  (eval only, or RL with `details_dir`) the table adapter reaches 1.96 ms median and 2.2 ms p99.
  Scores never depend on time; rows over 50 ms are only counted as `slow_row`.
- Each annotation box is resolved once per scoring call and frame (cached on the annotation
  view; review CONTRACT-5), and an annotation list longer than the certificate list cap is a
  deterministic `budget` envelope failure for M. Not applied: moving the table adapter's
  intersecting-cell prefilter into `identity.resolve_boxes`.

## 6. Environment and tests

- `jsonschema` is not installed in the `trace_cert` env (not installed by the review either, to
  leave the shared scoring env unchanged). Schema validation in the tests and in Section 5 uses
  the tested subset validator in `tests/test_certificate_prompts.py` (F is checked against it on
  mutated annotations, review CONF-02); the 5 jsonschema metaschema tests skip.
- Leakage test 5 (trace-tool request payload) runs only with `TRACE_TOOL_ROOT`; the release
  reward goldens and EasyR1 drift guard run only with `TRACE_RLVR_ROOT`. The M1 gate command
  (README "Tests") sets both and `CERT_GATE=1`, under which these skips are failures (review CONF-12).
- **The `trace_cert` env is not TRACE's release-pinned env.** It runs CPython 3.12 with
  numpy 2.5.4, Pillow 13.0.0, scipy 1.18.1 and networkx 3.7 on RHEL 8, while
  `constraints/release.txt` pins numpy 2.2.6, Pillow 12.2.0, scipy 1.15.3 and networkx 3.4.2 on
  an Ubuntu 24.04 host baseline. The data in `data/v1` was generated in this env; its receipts
  record the env lock sha256. Rebuilding in another env gives other images and instance ids.
  Scoring reads only the stored extensions, so it does not depend on the generation env.
- The existing TRACE suite has failures in generation, rendering and text-legibility tests
  (charts, games, graph, geometry, illustrations, ...) and in
  `test_domain_shared_ownership::test_sokoban_scene_helper_is_games_owned`. Every one of them
  also fails on a pristine worktree of the base commit d3440e95 in the same env, so none is
  caused by this extension; they are most likely the version drift above (not verified in a
  pinned env). The integrator's report lists the counts and where the failing ids are kept.

## 7. Blocking decisions carried over from DESIGN 10 [user]

1. **GPUs.** *Settled 2026-10-18/19:* a CPU-only host job runs the dispatcher and the watcher, GPU work
   goes through the use-gpu dispatcher with `trc_*` claim files, and the policy is never to place on the
   AIMO giants node or the GH200 nodes (`TRC_EXCLUDE_ALWAYS`). Allocation sharing with AIMO stays a
   standing constraint; the 2 x H200 NVL allocation is reserved for SFT and RL (section 10).
2. **Seed protocol.** Proposed: 3 training seeds for G/P (staged) and decoding seeds 42-44 on the
   cheap supported splits only; conflicts with the use-gpu skill's "3 training seeds,
   EVAL_N_REPEATS=1".
3. **N=2 EasyR1 matched configuration** (batch 64 x 8, 300 steps) instead of the release
   8 x 128 x 500.
4. **E1-dev has three roles** (SFT stop rule, identity calibration, go/no-go 1a): a mild
   selection effect, accepted because E1-test and E2-test stay untouched.
5. **Lever M has no aggregation family** by design, so M = 1 with A = 0 is possible
   (`answer_inconsistent`).
6. **Remaining M/G asymmetry.** Duplicate keys, NaN and Python literals void M but not legacy G;
   reported per arm.
7. **External-suite format shift** of JSON-only SFT models affects base/legacy comparisons, not
   P vs G; monitored, not prevented.

## 9. Review fixes that change interfaces for M2-M3 code

- Data root `data/v2` replaces `data/v1` (verifier versions 1.1.0 are in every extension's
  adapter block, so v1 stores no longer score; this is the intended version error).
- `private/build/` replaces `build/` (mode 0700); SFT records reference `sft/images/...`.
- Per-split receipts no longer carry `merged_store`; `freeze` writes
  `private/extensions_v1.jsonl.zst`, `private/rl_reward.extensions_v1.jsonl.zst` and
  `receipts/freeze.json` (which has an `outputs` and a `merged_store` block of the old shape).
- `export-sft` accepts `sft` and `rl` only. `compute_score` takes `split_phases` (default
  `{"rl": "train", "e1_dev": "val"}`), requires a bool `gated`, and checks `prompt_cert` against
  the frozen prompt of `coord_frame`.
- `score` refuses unfrozen datasets, settings other than DESIGN 7.1's and transcripts whose
  prompt is not the frozen one, unless overridden (recorded); legacy `ans` runs need TRACE's
  system prompt (stripped-text sha256 `d6105d53...`).

## 8. M2-M3 scope (was "not built yet" at M1 exit; built since)

`sft.py`, `train_rl.py`, `configs/`, `jobs/{sft.sh, sft_select.py, rl.sh, ckpt_eval.sh}`,
`trace_tool_conditions_v1.yaml.in`, `analysis/compare_cert.py` and the training env (`trace_rl`, the
README's "SFT, RL and checkpoint-evaluation GPU jobs") exist and have smoke-tested on GPUs. Not yet run at
full scale: the 3B SFT is in progress, no RL arm has started, `ckpt_eval.sh` has only seen the SFT smoke
checkpoint and a fake checkpoint, `compare_cert.py` only synthetic runs.

## 10. M2-M3 operations: what the smokes left open (2026-10-19)

- **Host RAM is the binding resource for RL [user to be aware].** One 3B arm needs about 164 GB of
  unreclaimable host RAM (245 GB at the cgroup peak with page cache) on 2 GPUs; a 2 x A100 allocation with a
  120 GB limit was OOM-killed. `hardware.min_host_ram_gb` gates it (exit 4; `TRC_RL_MIN_HOST_RAM_GB`
  overrides). Only the 2 x H200 NVL allocation (360 GB) qualifies, so every RL arm runs there, one at a time
  (the trainer uses both GPUs), and that allocation's runway bounds the schedule: at about 10.5 h per 3B arm,
  stage 1 (A, G, P seed 1) is about 32 h and stage 2 (G and P seed 2) about 21 h. The N=4 values (250 GB for 3B)
  and the 7B values (320 GB on 2 GPUs, against a 360 GB limit) are estimates; the 7B smoke has not run and may
  not fit.
- **The RL step-time projection comes from the base model.** The 3-step probe (162 s per step with a save
  every step, about 121 s without) trained the base Qwen2.5-VL-3B, whose responses are short (about 220
  tokens) and mostly invalid certificates (F about 0.5). Starting from the SFT checkpoint the responses will be
  longer and valid more often, so generation, reward and update times can move; the first real steps replace
  the projection.
- **Reward-collapse and stall thresholds are heuristics.** The watcher flags an overall-reward 10-step mean
  below 25% of the first 10 steps' mean after step 50, and 45 minutes without a new step. Neither was tuned on a
  real run; the smoke rewards (R about 0.06-0.09) are from the base model and say nothing about the SFT start.
- **Greedy dev evaluation is hardware-dependent (README, "Greedy-eval nondeterminism").** The stop rule's
  1.5 / 2 pp stability thresholds sit within what one GPU-class change moves on 32 rows; on 512 rows the effect
  is unmeasured. The 3B SFT is pinned to the 2 x H200 NVL allocation. A repeat of the SFT on another class, or a
  7B SFT elsewhere, is not comparable checkpoint for checkpoint. `sft.sh` does not enforce the pin.
- **SFT dev generations are scored with a settings override.** They decode greedily (2048 tokens), while the
  scorer expects the `trace` preset, so the dev scoring runs with `--allow-settings-mismatch` and every
  `dev_curve.json` point records `settings_problems: ["decoding 'greedy', expected the 'trace' preset"]`. This
  is by design (DESIGN 7.2) and only for E1-dev stop-rule scoring; the reported evaluations use the `trace` preset.
- **Resume after preemption is probably not bitwise identical** (rollout RNG state is not checkpointed;
  DESIGN 7.3). The H200 allocation is not known to be preempted hourly (that was reported for notchpeak), so a run
  resumes only after a kill or a stall; the receipt records every resume point.
- **Merged RL checkpoints were tested only at 2-3 steps.** The vLLM load check passes on those merged
  directories, and `ckpt_eval.sh` ran on the SFT smoke checkpoint; the first full-size merged actor (step 100)
  is the first real test of the evaluation path for RL outputs. Disk per live 3B run is about 150 GB (DESIGN 7.5).
- **`data/v1` is still on disk** (7.1 GB, mode 0700, superseded; it cannot score under the current code). It is a
  deletion candidate once nothing needs the pre-review parquets for comparison
  (`$TRACE_CERT_RUNS/CLEANUP_CANDIDATES.md`).
- **Review-gate scope.** The PASS covers the evaluation layer (parser, identity, adapters, scorer, export,
  reward) on data/v2. It does not cover `sft.py`, `train_rl.py` or the job scripts, which have CPU tests
  (`tests_sft/`, `tests_rl/`) and GPU smokes but no independent adversarial review.

## 11. External-transfer evaluation (trace_eval_v1): deviations from af32fe0 (2026-10-19, fix-3 2026-10-23)

- **MME-Reasoning official-evaluator exceptions are scored incorrect** (post-release fix-2; decided by the
  orchestrator 2026-10-19; README, "trace_eval_v1: post-release fixes and the MME-Reasoning deviation"). af32fe0 fails
  the whole run when MME-Reasoning's official per-row function raises on one row. The full 3B seed-42 runs of
  Qwen2.5-VL-3B-Instruct and the cert SFT 3B (step 564) hit `multiple_match_function`'s single-key assertion on judge
  extractions with two keys. Such a row now scores incorrect with log_score 'Failed to evaluate', upstream
  VLMEvalKit's semantics for unevaluable rows, for every model; every other failure still fails the slice. Counts
  (full, seed 42, of 1,188 MME-Reasoning rows): Qwen2.5-VL-3B-Instruct 1 (index 305), TRACE-3B 0, cert SFT 3B step 564
  2 (305, 411). In all three rows the first extracted key is not an accepted answer either, so no reading of these
  answers would score them correct; the deviation only lets the run finish. What would reopen it: a model with many
  such rows (`evaluator_exception_scored_incorrect` in `summary.seed<S>.json`), where MME-Reasoning would then measure
  answer format more than reasoning.
- **Provenance after the fix.** fix-2 changes `TRACE_EVAL_CODE_HASH` (221335f51667 to 16d37ed061fd). The three full
  3B seed-42 runs were rescored under it (generation kept), so they share one code hash; their generation verifies
  against the code it was produced under (221335f51667). The smokes and q25 probes keep scores under 221335f51667;
  they had no such row, so fix-2 would not change them. Rescoring re-runs the Qwen3-32B judge, which moved single
  judge-scored benchmarks by up to 0.89 points (base MathVerse); comparisons at that resolution need judge repeats.
- **A judge output still unusable after the official 5 attempts is scored incorrect** (post-release fix-3; decided by
  the orchestrator 2026-10-23; README, "fix-3" under "trace_eval_v1: post-release fixes and the MME-Reasoning
  deviation"). af32fe0 fails the whole run when the Qwen3-32B judge's output for one row stays unusable through its 5
  attempts; upstream VLMEvalKit scores such a row incorrect (`MathVerse_auxeval_score`: score False after "All 5
  retries failed."; `MMEReasoning_extract`: `res=''`, then 'Failed to evaluate'). fix-3 does the same for a MathVerse
  score judgement without a 0/1 decision (the strict parser is unchanged) and for an MME-Reasoning answer extraction
  that never parses; judge request failures and everything else still fail the slice. Triggers (full, seed 42, first
  scorings, now in `scoring/stale/`): rl3b-a-s3-300 MathVerse row 1880, a degenerate extraction (`4√3` plus invented
  few-shot items) whose five judgements all conclude 1 but in markdown forms the parser and upstream both reject;
  rl7b-p-s1-300 MME-Reasoning index 950, a degenerate 4,096-token answer whose extraction never parses (incorrect
  under any reading). Outcome of the rescoring (2026-10-24): row 1880 got a clean extraction and `Judgement: 1` on
  the first attempt and is scored correct (fix-3 did not fire for rl3b-a-s3-300); index 950 is scored incorrect
  (MME-Reasoning 24.33). In the 37 runs complete under d9319e7db480 by 2026-10-24 16:15 (11 full, 26 q25), fix-3
  fired 4 times, one row per run: index 950 and three MathVerse rows (rl3b-g-s3-300 seed 44 row 3240, rl3b-p-s3-300
  seed 44 row 3695, rl3b-a-s2-100 q25 row 1780). In each of the three the extraction judge hit its token limit after
  the answer, its reasoning text reached the score prompt, and the score judge answered with an answer value instead
  of 0/1. The leading extracted answer is wrong in all three, so all four rows are incorrect under any
  reading. Counts: `extraction_failed_scored_incorrect` and `mathverse_all_retries_failed_scored_incorrect` in
  `summary.seed<S>.json`. The number of such rows has no cap, so a judge fault would show up only in these counters.
  What would reopen it: a model with more than a few such rows, or a row like 1880 that is probably correct, where
  the judge's formatting rather than the model's answer would decide the score.
- **Provenance after fix-3 (2026-10-23; runs as of 2026-10-24).** fix-3 changes `TRACE_EVAL_CODE_HASH` (16d37ed061fd
  to d9319e7db480; the evaluator hash 7bae61e285d1 stays). Every scoring from 2026-10-24 on is under d9319e7db480:
  the seed-42 rescoring of rl3b-a-s3-300 and rl7b-p-s1-300 (generation kept), rl3b-ggated-s1-300 seed 42, seeds
  43/44 of rl3b-{g,p}-s{2,3}-300 and the q25 runs of the step-100/200 checkpoints. 24 complete full runs keep
  16d37ed061fd: the 16 other seed-42 runs, and seeds 43/44 of cert SFT 3B step 564 and rl3b-{a,g,p}-s1-300. They are
  not rescored, because fix-3 cannot change a run that has no row exhausted through all 5 attempts, and a run that
  completed under 16d37ed061fd has none. The full-run tables therefore mix two code hashes that score every such run
  identically, but the receipts of each verify only under their own code. Against the patched clone, the protocol's
  `verify_trace_eval.py --phase score` gives 0/24 for every 16d37ed061fd run (`receipt_invalid`, code provenance) and
  24/24 for every d9319e7db480 run; against an af32fe0 + fix-1 + fix-2 tree it gives the reverse result (all 35
  checked 2026-10-24). Rerunning an old-hash full run's task line would therefore raise `FLAG score_code_stale` and
  `FLAG protocol_verify_failed` (scores unchanged, job still COMPLETE). These 24 runs must not be rerun; they are
  reported as they are. README, "fix-3 provenance", gives the file sha256s to rebuild the old tree.
