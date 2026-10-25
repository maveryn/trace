# Phase 1 results: inference-time tools for Qwen2.5-VL on TRACE

> **Judge pass.** All numbers are official scores after the judge rerun. The first judge pass (2026-10-04) produced nothing:
> all 10,201 judge requests were rejected (HTTP 400 `can only concatenate str (not "list") to str`), because the text-only
> Qwen3-32B judge was served with `--chat-template-content-format openai`; `scripts/serve_vllm.sh` now serves text-only models
> with `string`. Judge and compare were rerun on 2026-10-05 over the same generations: of the 10,201 judge requests, 1,689 came
> back ok, 7,319 missing, 936 invalid, 255 ambiguous and 2 still rejected. Across the 64 pilot and full runs, 2,696 rows were
> resolved by the judge (421 of them correct). The text and the reports next to this file reflect that rerun.

Generated 2026-10-04 with `run_on_server.sh` (setup run separately ahead of time); judge and compare rerun 2026-10-05 over
the same generations. On TRACE validation for Qwen2.5-VL-7B-Instruct and -3B-Instruct.
The questions (Q1–Q3) and the decision rule are in [EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md). The reports behind
the numbers are next to this file, except where a section names a script or says otherwise. Their structure is described
under [Files](#files).

## Takeaway

**Zero-shot, under these prompts and loop settings, no tool condition improves on `no_tool` for Qwen2.5-VL on TRACE: every
one is significantly worse at both sizes on the seeds-pooled bootstrap CI except 7B `calculator` (−1.0 pp, CI [−2.45, +0.43],
not significant); on the per-seed Holm-McNemar tests, 7B `ocr` also misses in seed 43 (Holm p 0.239).** No condition
meets the Phase 2 decision rule. The extra token budget has no detectable effect (7B +0.05 pp, 3B +0.53 pp, n.s. in every
seed), so the loss is not a budget effect. It comes from the tool conditions (tool prompt plus tool calls), which this
design cannot separate (Q2).

In 8 of 10 conditions (7 of the 9 with a significant loss), most of the loss (57–85% of the drop) comes from episodes still calling tools at the 6-turn limit
(or truncated); for 7B `ocr` and 3B `zoom` the answered rows contribute more. With tools, the model often keeps calling
them until it reaches the 6-turn limit. Without tools that never happens. Even on episodes that do answer, accuracy is
0.2–4.7 pp below `no_tool` on the same rows (2.2–4.7 pp excluding 7B `calculator`), and the answered rows alone account for
0.2–3.1 pp of the drop (1.5–3.1 pp excluding 7B `calculator`, which is not significantly different). Both figures are
descriptive, since the tool run selects which rows it answers.

The pilot's `oracle_crop` (a crop of the ground-truth region added to the prompt) shows some headroom from looking in the right place:
+4.9 pp for 7B (p = 0.0073, Holm p = 0.036 over the pilot's 10 conditions) and +2.3 pp for 3B (p = 0.17, Holm p = 0.34).
7B is significant after Holm in the one-seed pilot; 3B is not. With one seed and possible answer leakage from the
annotations, treat it as an upper-bound diagnostic.

## Main result (full run: 2000 rows × seeds 42, 43, 44)

Official accuracy (%), seeds pooled. Δ is in pp, from `compare.md` (vs `no_tool`) and `compare_vs_budget.md` (vs `no_tool_budget`).

| condition | 7B acc | Δ vs no_tool | Δ vs no_tool_budget | 3B acc | Δ vs no_tool | Δ vs no_tool_budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_tool | 33.80 | — | | 24.25 | — | |
| no_tool_budget | 33.85 | +0.05 [−0.75, +0.88] | — | 24.78 | +0.53 [−0.27, +1.30] | — |
| calculator | 32.80 | −1.00 [−2.45, +0.43] | −1.05 | 18.45 | −5.80 [−7.15, −4.43] | −6.33 |
| ocr | 30.47 | −3.33 [−4.95, −1.77] | −3.38 | 16.60 | −7.65 [−9.12, −6.20] | −8.18 |
| zoom | 29.00 | −4.80 [−6.25, −3.37] | −4.85 | 18.65 | −5.60 [−7.02, −4.20] | −6.13 |
| all | 24.80 | −9.00 [−10.57, −7.47] | −9.05 | 17.02 | −7.23 [−8.63, −5.83] | −7.77 |
| visual_ops | 20.83 | −12.97 [−14.50, −11.43] | −13.02 | 17.03 | −7.22 [−8.62, −5.83] | −7.75 |

Brackets are the 95% bootstrap CI from `compare.md`. On the per-seed McNemar tests (Holm across the pre-registered family):
- Every tool condition is significantly *below* `no_tool` in all three seeds, except 7B `calculator` (Holm p 0.736, 1, 0.137;
  not significant in any seed) and 7B `ocr` in seed 43 (Δ −2.20 pp, Holm p 0.239). Every 3B tool condition is significant
  in every seed (max Holm p 2.6e-05).
- `no_tool_budget` is not significant in any seed for either model.

## Decision rule for Phase 2

| rule | result |
| --- | --- |
| (a) beats `no_tool` and `no_tool_budget` by ≥ 2 pp, Holm p < 0.05 | **not met** by any condition; the best is 7B `calculator` at −1.0 pp vs `no_tool` (−1.05 pp vs `no_tool_budget`), not significant |
| (b) `tool_use_rate` ≥ 10% | met by every tool condition (25–86%), so the tools were used |
| (c) the sign holds for 3B | n/a: no condition passes (a). Point estimates are negative at both sizes for every tool; 7B `calculator` is not significant |

The design also allows a small Phase 2 pilot for a weaker signal, "such as gains confined to one or two domains". No
domain-level gain survives correction for the 110 comparisons (`scripts/decompose_losses.py runs/full/<model> --by-domain`,
normal-approximation CIs).
- Of the 110 per-domain deltas (5 tools × 11 domains × 2 models), 16 are positive.
- None of the 16 positive deltas has a 95% CI excluding zero, even before correcting for multiple tests; 55 of the 110
  (24 for 7B, 31 for 3B) are significantly negative, uncorrected.
- The strongest candidate is 7B `ocr` on `illustrations`: +5.8 pp, CI [−0.04, +11.70], 120 rows. It is borderline even
  uncorrected, and `no_tool_budget` also gains +2.5 pp there, so we do not treat it as a Phase 2 signal.

## Why the tools hurt (Q3)

Each tool row is paired with the same row and seed of `no_tool`, over all 3 seeds × 2000 rows. Reproduce with
`scripts/decompose_losses.py runs/full/<model>`.

| | condition | no answer¹ | Δ (pp) | Δ from no-answer rows | Δ from answered rows | answered rows: acc | `no_tool`, same rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7B | zoom | 31.4% | −4.80 | −3.32 | −1.48 | 31.1 | 33.3 |
| 7B | ocr | 35.8% | −3.33 | −1.55 | −1.78 | 30.6 | 33.4 |
| 7B | calculator | 7.9% | −1.00 | −0.82 | −0.18 | 33.1 | 33.3 |
| 7B | visual_ops | 47.5% | −12.97 | −10.97 | −2.00 | 30.1 | 33.9 |
| 7B | all | 41.9% | −9.00 | −6.28 | −2.72 | 30.8 | 35.5 |
| 3B | zoom | 13.7% | −5.60 | −2.53 | −3.07 | 21.0 | 24.5 |
| 3B | ocr | 21.3% | −7.65 | −4.82 | −2.83 | 20.5 | 24.1 |
| 3B | calculator | 19.5% | −5.80 | −3.67 | −2.13 | 22.5 | 25.1 |
| 3B | visual_ops | 20.4% | −7.22 | −4.12 | −3.10 | 20.8 | 24.7 |
| 3B | all | 26.0% | −7.23 | −4.50 | −2.73 | 22.1 | 25.8 |

¹ Episodes whose last turn is not a final answer (`stop_reason` ≠ `answer`): still calling tools at the 6-turn limit
(`max_turns`, 94.7–99.6% of them) or truncated by the length or token budget. For 7B, the extractor still finds an answer in
the final message of 64% of these episodes overall (38–91% by condition; a minority only for `visual_ops`), and it is
scored; for 3B the extractor finds no answer in 84–91% (71–85% stay unresolved after the judge and score 0). `no_tool` and `no_tool_budget` answer in over 99% of episodes (99.5–99.8%).

- **Not stopping is the larger part of the loss in 8 of 10 conditions** (7 of the 9 with a significant loss; the 8 include
  7B `calculator`, whose −1.00 pp total is not significant). The no-answer rows make up 46.5–84.6% of the
  drop for 7B and 45.2–63.2% for 3B. For 7B `ocr` (46.5%) and 3B `zoom` (45.2%), most of the drop comes from answered rows. Like the
  answered-row comparison, this split is descriptive: the tool run decides which rows end without an answer.
- **The model asks for far more calls than are executed.** The loop executes at most 2 calls per turn and returns an error
  for the rest. 7B `visual_ops` requests 23.3 calls per episode on average, of which 3.5 execute. 7B `zoom` requests 10.0,
  of which 2.3 execute. Most of the gap (92% for `visual_ops`, 84% for `zoom`) is calls beyond the 2-per-turn cap; the rest
  are calls in the final turn (never executed, with no error returned) or malformed or unknown-tool calls. The share of episodes whose final message is a well-formed answer falls with the no-answer rate:
  from 99.5% (`no_tool`, 7B) to 80% (`zoom`), 74% (`ocr`), 63% (`all`) and 58% (`visual_ops`).
- **On episodes that do answer, the tool runs are still worse on the same rows, for every tool except 7B `calculator`**,
  which is not significantly different (−0.2 pp, 95% CI [−1.7, +1.3]; normal approximation over the answered row × seed
  pairs, clustered by row, computed from `scores.jsonl` since `decompose_losses.py` prints no CI here). The other gaps run from −2.2 pp (7B `zoom`) to −4.7 pp (7B `all`).
  This comparison is descriptive: the tool run decides which rows it answers.

## Pilot diagnostics (1000 rows, seed 42; `compare_all.md`)

| condition | 7B acc | Δ vs no_tool | 3B acc | Δ vs no_tool | reading |
| --- | ---: | ---: | ---: | ---: | --- |
| no_tool | 34.3 | — | 26.5 | — | |
| oracle_crop | 39.2 | +4.9 (p 0.0073, Holm 0.036) | 28.8 | +2.3 (p 0.17, Holm 0.34) | the right region helps 7B (significant after Holm, one seed); the 3B gain is not significant; it may also leak the answer |
| zoom_cookbook | 27.0 | −7.3 | 15.2 | −11.3 | cookbook prompt: for 7B, fewer calls (23% tool use vs 44%) and better format, but no better accuracy; for 3B, much worse |
| no_tool_lowres | 32.9 | −1.4 (n.s.) | 22.1 | −4.4 (Holm 0.025) | the 0.4 MP copy costs 3B accuracy |
| zoom_lowres | 27.3 | −5.6 vs no_tool_lowres² | 16.9 | −5.2 vs no_tool_lowres² | no sign that zoom helps more at low resolution (it still costs 5–6 pp there); the one-seed pilot cannot test the design's ZwZ-based prediction precisely |

² Paired against `no_tool_lowres` from the pilot `scores.jsonl` (exact McNemar p = 0.0010 (7B) and 0.0013 (3B), uncorrected); not in
`compare_all.md`, which compares `zoom_lowres` only with `no_tool` (−7.0 and −9.6). Zoom's effect at low resolution
(−5.6, −5.2) is about the same as at full resolution (−6.2, −5.9): the per-row difference-in-differences is +0.6 pp (7B,
paired-bootstrap 95% CI [−3.8, +5.0]) and +0.7 pp (3B, [−3.7, +5.1]), so a moderate interaction is not ruled out either way.

## Caveats

- **Zero-shot only.** This measures what the base models do with tools untrained. The design treats that as a lower bound on what RL could teach.
- **`oracle_crop` is a pilot diagnostic.** It ran on one seed and uses ground-truth annotations, so it can leak the answer.
  For 7B it is significant after Holm (+4.9 pp, Holm p = 0.036, 10-condition pilot family); for 3B it is not (+2.3 pp,
  Holm p = 0.34). Treat it as an upper bound, never as a method.
- **The answered-row comparison is descriptive.** The tool run itself decides which rows end in an answer.
- **No condition isolates the prompt.** No condition adds the tool block without tools to call (see the design). `zoom_cookbook`
  shows that prompt wording alone moves behaviour substantially.
- **The loop settings were not varied.** 6 turns, 2048 tokens per turn and 2 executed calls per turn are the design's defaults.

## How it was run

- **Models.** `Qwen/Qwen2.5-VL-7B-Instruct@cc59489` and `Qwen/Qwen2.5-VL-3B-Instruct@6628554` at TRACE's pinned revisions.
  The judge is `Qwen/Qwen3-32B@9216db5`. `judge_pending` = 0 in every run. 13,162 scored rows (10,201 judge requests,
  deduplicated within each model and stage; 9,520 distinct (answer type, response) pairs overall) needed the judge. After the 2026-10-05 rerun, 2,696 rows were resolved by the judge, with `judge_resolved` > 0 in 63 of
  64 runs (the only zero is 7B full `no_tool` seed 43). The remaining 10,466 rows are unresolved and score 0.
- **Hardware and software.** Generation (2026-10-04): one NVIDIA H200 (141 GB) per model, which served the model and then
  the judge (the pass whose requests were all rejected). Judge and compare were rerun on 2026-10-05 over the same
  generations (`STEPS="judge compare"`), one H200 per model; the judge was served with `--chat-template-content-format string`
  and a 16,384-token context. vLLM 0.30.0, torch 2.13.0 (CUDA 13.0), Python 3.12.
- **Wall clock.** About 4 h 50 min for 7B and 5 h for 3B, covering every step from check to compare; that includes a judge
  step whose requests were all rejected (about 12 min for 7B and 3 min for 3B, mostly loading the judge). The two models ran in parallel on two cards. The
  judge + compare rerun took about 17 min (7B) and 20 min (3B).
- **Code.** The runs record code commit `4bea66e`. `run_on_server.sh` is byte-identical to the one in `4689167`. That commit's
  `serve_vllm.sh` change only drops `--limit-mm-per-prompt` for the text-only judge, so the model servers ran with the same flags.
  The 2026-10-04 run itself used the `serve_vllm.sh` of `4bea66e`. The judge rerun ran from `4689167` plus an uncommitted
  change to `scripts/serve_vllm.sh` (text-only models get `--chat-template-content-format string` instead of `openai`), so the
  rerun judge differs from the first-pass judge in two flags: the `string` content format, and no `--limit-mm-per-prompt`
  (the first pass set image and video limits of 0). The generation servers are unaffected.
- **Setup step.** It was split: its installs ran once ahead of time, so the GPU runs used `STEPS="check smoke pilot full judge compare"`.
  Its GPU preflight and OCR check ran in each model's task before generation. The CPU test suite passed: 686 passed, 1 skipped.
- **Validation.** Every condition × seed has the expected rows (1000 in the pilot, 2000 in the full run), with no generation
  endpoint errors and no pending judge items; 2 of 10,201 judge requests (7B, one pilot and one full) exceeded the judge's
  16,384-token context and were scored unresolved.

## Files

`<model>/<stage>/` with `<model>` ∈ {`qwen2.5-vl-7b-instruct`, `qwen2.5-vl-3b-instruct`} and `<stage>` ∈ {`pilot`, `full`}:

| file | contents |
| --- | --- |
| `report.md` | one row per condition × seed: accuracy, format, tool use, calls, errors, turns, tokens, latency; then accuracy by domain |
| `compare.md` | paired comparison vs `no_tool`, Holm family = the core tool conditions + `no_tool_budget` (decision rule (a)) |
| `compare_vs_budget.md` | the core tool conditions vs `no_tool_budget` |
| `compare_all.md` | every condition vs `no_tool`, diagnostics included (one Holm family) |

The per-row data (`runs/`, gitignored) is not committed.
