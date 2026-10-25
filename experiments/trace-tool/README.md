# trace-tool

Tests whether giving tools to Qwen2.5-VL-3B/7B, TRACE's own base models, improves them on [TRACE](https://huggingface.co/datasets/maveryn/trace). Qwen3.5-4B/9B (thinking off) run the same grid plus five conditions that force a final answer. Qwen3-VL is also supported.

This project lives in `experiments/trace-tool/` of the TRACE repository and is self-contained: run every command below from this directory. It installs as its own package (`trace-tool`) and does not import `trace_tasks`; the TRACE scorer it needs is vendored under `src/trace_tool/vendor/trace_rlvr`. Findings: [Phase 1 (Qwen2.5-VL)](results/phase1/FINDINGS.md), [Qwen3.5](results/qwen35/FINDINGS.md).

The tools:
- the zoom-in tool from the "Zooming without Zooming" (ZwZ) tool-use baseline;
- the other image operations in ZwZ Table 9;
- a calculator;
- OCR.

The hypotheses, conditions, statistics and the conditional RL phase are in **[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md)**.

## Layout

```
configs/conditions.yaml          evaluation conditions (tools, budgets, pixel bounds, forced final answer)
configs/verl/                    tool configs for the Phase-2 RL arms
prompts/                         TRACE system prompt (sha-checked), Qwen3-VL cookbook prompt
src/trace_tool/tools/            zoom (Qwen-Agent port), rotate/flip/draw, calculator, ocr
src/trace_tool/models.py         model families: patch grid, grounding coordinates, tool-call format, thinking switch
src/trace_tool/prompting.py      tool prompts in Qwen-Agent's Nous format (= Qwen3-VL's native tool template),
                                 Qwen3.5's XML layout, and the forced-mode prompt texts
src/trace_tool/parsing.py        tool-call parsing: Hermes JSON, and Qwen3.5 XML (vLLM's qwen3 grammar, with listed differences)
src/trace_tool/agent.py          multi-turn episode loop against an OpenAI-compatible endpoint
src/trace_tool/run_eval.py       CLI: generate (resumable)
src/trace_tool/score_runs.py     CLI: score with TRACE's own code, per-run summaries, report
src/trace_tool/judge.py          CLI: TRACE's Qwen3-32B fallback extraction
src/trace_tool/compare.py        CLI: paired deltas, bootstrap CI, McNemar + Holm
src/trace_tool/rl/               Phase-2 scaffold for verl (reward, data prep, tool adapters, tool chat template for Qwen2.5-VL)
src/trace_tool/vendor/trace_rlvr TRACE scorer + extractor, vendored (Apache-2.0, pinned commit)
scripts/                         serve_vllm.sh, smoke_test.sh, run_phase1.sh, check_template.py, rl/
tests/                           CPU tests (no GPU or network needed)
```

## Setup (server)

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install vllm && pip install -e ".[ocr,json5]"
```

RapidOCR pulls `opencv-python`, which needs GUI libraries (`libGL`, `libxcb`) that headless servers lack. vLLM pulls `opencv-python-headless`. Both packages own the same `cv2/` directory, so removing just one deletes files the other needs. Remove both, then reinstall the headless build:

```bash
pip uninstall -y opencv-python opencv-python-headless && pip install opencv-python-headless
```

If the OCR backend cannot load, `trace-tool-eval` stops at startup with the error, rather than running the `ocr` and `all` conditions with a broken tool.

The dataset is fetched from the Hub on first use and pinned to `dataset-v1` (commit `4e5b5436…`, sha256-checked). For a local copy, pass `--dataset-path` to the CLIs, or set `DATASET_PATH=...` for the scripts.

## Run everything with one script

[`run_on_server.sh`](run_on_server.sh) runs the whole of Phase 1 on a Linux GPU server:
1. setup, check, smoke;
2. pilot and full, for 7B then 3B;
3. the judge pass;
4. comparisons;
5. `results_summary.tgz` to send back.

It starts and stops vLLM itself, logs to `logs/`, and resumes after a crash if you rerun it:

```bash
nohup bash run_on_server.sh > run.log 2>&1 &
```

Useful controls:
- **Stop:** `kill <pid>` stops the current step and vLLM within seconds. The script prints its pid at the start. Rerun the same command to resume.
- **Failures:** if vLLM dies, rows end in endpoint errors, or judge answers stay pending, the script stops with `INCOMPLETE` instead of producing partial results.
- **GPUs:** `GPUS=0,1,2,3` gives one replica per GPU. `JUDGE_GPUS` takes 1, 2, 4 or 8 GPUs, about 80 GB in total, for Qwen3-32B.
- **Pilot first:** to follow the design's gate, run `STEPS="setup check smoke pilot judge compare"`, read the pilot, then run `STEPS="full judge compare"`.
- **Qwen3.5:** `MODELS="Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B"` runs the same steps at the pinned Qwen3.5 revisions, with the five `*_forced` conditions added to smoke, pilot and full. The compare step then also writes `compare_forced.md` and `compare_forced_vs_unforced.md` (see [Qwen3.5](#qwen35-and-the-forced-conditions)).

The other settings are listed at the top of the file, including `VLLM_SPEC` to pin a vLLM build that matches your CUDA driver; setup checks that torch can see the GPU. The sections below are the same steps by hand.

## Run Phase 1

All scripts take the checkpoint as `MODEL` (default `Qwen/Qwen2.5-VL-7B-Instruct`) and derive the rest from it in `scripts/_model_env.sh`:
- the served name, `qwen2.5-vl-7b-instruct`, which is also the output directory, so 7B and 3B runs never mix;
- TRACE's pinned revision for its models, and ours for Qwen3.5, which you can override with `REVISION`;
- `EXTRA_CONDITIONS`, appended to the scripts' default condition lists: the five `*_forced` conditions for `Qwen/Qwen3.5-*`, nothing otherwise. Set `EXTRA_CONDITIONS=` to add none. An explicit `CONDITIONS` is used as is.
- `MAX_LEN`, the server context for `serve_vllm.sh`: 65,536 tokens for `Qwen/Qwen3.5-*`, otherwise `serve_vllm.sh`'s default of 32,768, which the Qwen2.5-VL runs used. An explicit `MAX_LEN` wins.

The model family, and with it the tools' coordinate convention, is detected from the model id.

### 1. Serve the model

Each model and port pair runs in its own shell. Pass `DP=<gpus>` for data parallelism.

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct PORT=18100 scripts/serve_vllm.sh
```

### 2. Sanity-check template rendering

Run this in the vLLM environment, which has transformers and torch; otherwise `pip install -e ".[check]"`.

```bash
python scripts/check_template.py --model Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 --endpoint http://127.0.0.1:18100 --served qwen2.5-vl-7b-instruct
```

It checks every case and exits 1 on any mismatch, and also when `/tokenize` or `/detokenize` answers with an HTTP 4xx, whose response body it prints. A connection error, an HTTP 5xx, or a 404 for an endpoint the server does not have only skips the server check (`SKIPPED`).

### 3. Smoke test (20 rows × every condition)

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/smoke_test.sh
```

### 4. Pilot (1000 rows, seed 42, all conditions), then the full run (2000 rows × 3 seeds, core conditions)

```bash
STAGE=pilot MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/run_phase1.sh
```

```bash
STAGE=full MODEL=Qwen/Qwen2.5-VL-7B-Instruct scripts/run_phase1.sh
```

### 5. Judge pass (TRACE protocol; only missing/ambiguous extractions)

TRACE used an 8k judge context for 2k-token responses. 16k fits the 8k-token responses of `no_tool_budget`. A response that still doesn't fit is recorded as `rejected` and scores 0.

```bash
MODEL=Qwen/Qwen3-32B SERVED=qwen3-32b-judge REVISION=9216db5781bf21249d130ec9da846c4624c16137 PORT=18200 MAX_IMAGES=0 MAX_LEN=16384 scripts/serve_vllm.sh
```

```bash
trace-tool-judge --pending runs/full/qwen2.5-vl-7b-instruct/judge_pending.jsonl --out runs/full/qwen2.5-vl-7b-instruct/judge_results.jsonl
```

```bash
trace-tool-score runs/full/qwen2.5-vl-7b-instruct --judge-results runs/full/qwen2.5-vl-7b-instruct/judge_results.jsonl --report runs/full/qwen2.5-vl-7b-instruct/report.md
```

### 6. Paired comparison against the TRACE-protocol baseline

```bash
trace-tool-compare runs/full/qwen2.5-vl-7b-instruct --baseline no_tool --conditions no_tool_budget,zoom,visual_ops,calculator,ocr,all --out runs/full/qwen2.5-vl-7b-instruct/compare.md
```

Repeat with `MODEL=Qwen/Qwen2.5-VL-3B-Instruct` (serve, smoke, pilot, full).

**Sanity check:** on the full run, `no_tool` seed 42 should land near TRACE's published base scores: 24.45 (3B) and 34.25 (7B), under the same protocol and judge. Expect small differences from the vLLM version and from client-side resizing. A large gap means something is off before any tool result can be trusted.

**Optional:** `MODEL=maveryn/trace-qwen2.5-vl-7b` (and 3b) are TRACE's GRPO-trained checkpoints, published at 51.55 and 41.05. They show whether tools still help after RL without tools.

### Qwen3.5 and the forced conditions

`MODEL=Qwen/Qwen3.5-4B` and `MODEL=Qwen/Qwen3.5-9B` go through the same steps (serve, check, smoke, pilot, full, judge, compare). The family is detected from the model id. Differences from Qwen2.5-VL:
- **Pinned revisions:** 4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. These are the snapshots the chat template and vision grid were checked against. Both have the same chat template.
- **Thinking off:** every request carries `chat_template_kwargs: {"enable_thinking": false}`, so the generation prompt ends with an empty `<think>` block. `run_config.json` records it, together with `tool_format`.
- **Tool calls in the template's XML format:** the tool block comes before the TRACE system prompt, where the template puts it. The model writes `<tool_call>`, `<function=NAME>`, `<parameter=KEY>`, the value, `</parameter>`, …, `</function>`, `</tool_call>`, one per line. Values are typed by the tool's JSON schema.
  - A Hermes JSON call (`{"name": ..., "arguments": ...}`) inside `<tool_call>` is also accepted, and parsed, errors included, by the Hermes rules.
  - A `<tool_call>` that holds neither an XML function nor a Hermes JSON call, or a function with an empty or garbled name, gets an error that shows the XML format. Unterminated blocks and badly typed parameter values get their own errors (`Error: unterminated <tool_call> block`, `Error: parameter 'bbox_2d' must be a JSON array`).
  - The grammar follows vLLM's qwen3 parser, with known differences. For example, a `<tool_call>` without its `</tool_call>` is an error even when its function is complete. A `<function=...>` block without the `<tool_call>` wrapper counts as a call only when it is closed by `</function>` and the reply holds no `<tool_call>` at all; otherwise it is ignored. vLLM finds a call in each of these cases. The differences found by comparing both parsers are listed in [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md#3-tools), and `tests/test_parsing.py` pins each one. The list may not be exhaustive.
- **Coordinates:** relative [0, 1000], as in Qwen3-VL, confirmed by a grounding probe on both checkpoints (see [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md#2-facts-this-design-rests-on)).
- **Decoding:** the `trace` preset, for every condition, as for Qwen2.5-VL.
- **Server context:** `scripts/_model_env.sh` sets `MAX_LEN=65536` for `Qwen/Qwen3.5-*`, so `serve_vllm.sh` serves them with a 65,536-token context instead of its 32,768 default. This leaves room for a long `visual_ops` or `all` episode, which can carry up to 11 images (the input and 10 tool outputs), so that it is not rejected for prompt length (stop reason `rejected`). An explicit `MAX_LEN` wins.

The five forced conditions, `zoom_forced`, `visual_ops_forced`, `calculator_forced`, `ocr_forced` and `all_forced`, are their base condition plus:
- a budget of 10 tool calls, stated in the system prompt;
- `[Tool calls left: N]` at the end of the last tool response of each tool turn, inside its `<tool_response>` tags;
- a final-answer notice instead, once one turn is left, the budget is spent, or only the 1,024-token answer reserve remains. The model then gets one last turn, and tool calls in it are not executed.

The budget of 10 equals 5 tool turns × 2 calls, the most the turn limit allows anyway. So in these five conditions the budget is never spent before the turn limit is reached: the final-answer notice is triggered by the turns or by the token reserve (`forced_reason` `"turns"` or `"tokens"`, never `"calls"`), and no call is skipped for the budget. The budget rule only acts when `tool_call_budget` is set below `(max_turns - 1) × max_calls_per_turn`.

The exact semantics are in [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md#forced-final-answer-qwen35). The forced conditions form their own Holm family:

```bash
trace-tool-compare runs/full/qwen3.5-9b --baseline no_tool --conditions zoom_forced,visual_ops_forced,calculator_forced,ocr_forced,all_forced --out runs/full/qwen3.5-9b/compare_forced.md
```

Each forced condition is also compared with its base, one at a time: `trace-tool-compare runs/full/qwen3.5-9b --baseline zoom --conditions zoom_forced`, and so on. `run_on_server.sh` concatenates these into `compare_forced_vs_unforced.md`.

Useful knobs:
- `trace-tool-eval --model qwen2.5-vl-7b-instruct --conditions all --limit 1 --dry-run` prints the exact first prompt per condition (no endpoint or `--out` needed). For a forced condition it shows the budget statement in the system message.
- `--conditions` picks a subset.
- `--domains charts,pages` filters domains.
- `--concurrency` sets episodes in flight.
- `--tool-threads` sets the CPU threads for tools.
- `TRACE_TOOL_OCR_THREADS` sets onnxruntime threads per OCR engine.

## Outputs

`runs/<stage>/<model>/<condition>/seed<seed>/` contains:

| file | contents |
| --- | --- |
| `run_config.json` | model, condition, every setting, system-prompt sha256, tool schemas, dataset revision, git commit |
| `responses.jsonl` | one episode per row: transcript (images as index+size), every requested tool call (args, result, latency, whether it executed), stop reason, tokens; forced conditions also record `tool_call_budget`, `tool_calls_used`, `forced_turn` and `forced_reason` |
| `scores.jsonl` | strict (RL-reward) and official (TRACE validation) correctness per row |
| `summary.json` | accuracy overall / by domain / by answer type, tool use, calls, errors, turns, tokens, latency |

Tool-use metrics:
- `tool_use_rate` is the share of episodes with at least one *successful* call to a tool the condition offers.
- `tool_attempt_rate` counts any request, including malformed calls, calls to unknown tools, and calls skipped by the turn or per-turn limits.
- `tool_error_rate` is failed calls divided by executed calls.

Forced conditions only:
- calls the model writes in the forced final turn are recorded but not executed (`skipped: "forced_answer"`), and the episode stops with `forced_tool_call`. Accuracy is still scored on the final message as written, so an answer written before the call can count. That holds for every stop reason except endpoint errors: rows with stop reason `error` are retried by a rerun of `trace-tool-eval` and left out of the scores until then. Only the loss decomposition (`scripts/decompose_losses.py`) counts `forced_tool_call` as no answer;
- calls beyond the remaining budget get an error body (`skipped: "call_budget"`). With the shipped budget of 10 this never happens (see [above](#qwen35-and-the-forced-conditions));
- `tool_calls_used` counts every call handled against the budget, which includes malformed calls and calls to unknown functions (their `tool_calls` entries have `executed: false`). It is not the number of executed calls.

Robustness:
- A run directory is locked while it is being written, so a second writer is refused.
- Resuming with a different model, model id, condition settings, dataset selection or OCR backend is refused.
- A changed code version only produces a warning.

At the root of each run you also get:
- `report.md`, one row per condition and seed;
- `compare.md`, paired deltas;
- for runs with forced conditions, `compare_forced.md` and `compare_forced_vs_unforced.md` (written by `run_on_server.sh`; the second also without a `no_tool` run);
- `judge_pending.jsonl` / `judge_results.jsonl`.

## Tests

```bash
pip install -e ".[ocr,dev,json5,scipy]"
pytest
```

| Test file | What it checks |
| --- | --- |
| `test_zoom.py` | Pixel- and box-level parity with Qwen-Agent's zoom code (`tests/reference/`) |
| `test_prompting.py`, `test_qwen25.py` | Byte-identical rendering: against the native `tools=` / `role: tool` path (Qwen3-VL template), and against Qwen-Agent's Nous layout (Qwen2.5-VL template). Also absolute-pixel coordinate mapping on the 28-px grid. |
| `test_qwen35.py` | Qwen3.5: byte-identical rendering against the native `tools=` / `role: tool` path of the vendored template (`tests/data/qwen3_5_chat_template.jinja`), with thinking off and on: one and two tool responses, an image in a tool response, bodies with edge whitespace, and a forced-mode notice. Also: XML calls rendered by the template parse back to the same arguments, and family detection. |
| `test_parsing.py` | Tool-call parsing: Hermes JSON, and Qwen3.5 XML (values typed by the tool schemas, strings never JSON-decoded, missing closing tags, calls without the `<tool_call>` wrapper, a Hermes call inside `<tool_call>`, XML-worded errors, and the listed differences from vLLM's parser). Also the stripping of hallucinated tool responses. |
| `test_conditions.py` | Existing families and conditions are unchanged: `describe()` and the first request match goldens captured before Qwen3.5 was added (`tests/data/golden_*_v1.json`). The request is checked byte for byte where the host's zlib writes the same PNG bytes as the capture host, and otherwise as text plus decoded image pixels, with a warning (`tests/golden_payload.py`). Also the forced-condition keys and their validation. |
| `test_calculator.py` | Injection and blow-up inputs are rejected |
| `test_agent.py`, `test_cli_e2e.py` | Multi-turn loop, budgets, forced final answer (countdown, call and token budgets, the order of the forcing reasons, the final turn), resume and the whole CLI chain, against a fake vLLM endpoint |
| `test_rl.py` | RL reward and data prep |

`TRACE_VALIDATION_PARQUET=/path/to/validation.parquet pytest` also checks that the gold answers of all 2,000 real rows score 1.0.

**Not tested here:**
- anything that needs a GPU, meaning real model behaviour and vLLM's own rendering (`scripts/check_template.py` and the smoke test cover these on the server; `check_template.py` also runs offline against the processor alone). The Qwen2.5-VL rendering was also checked against the real Hugging Face tokenizers at TRACE's pinned revisions;
- the verl adapters in `src/trace_tool/rl/verl_tools.py` and `scripts/rl/run_grpo_template.sh`, which are templates for Phase 2.

## Review before merging

Two parts are security-sensitive and should get a human review:
- `src/trace_tool/tools/calculator.py` evaluates model-written expressions. It uses an AST whitelist and size limits, never `eval`.
- Tool arguments are untrusted model output in general.

## Licenses

| Component | License |
| --- | --- |
| TRACE dataset | CC-BY-4.0 (cite Alam 2026) |
| Vendored TRACE scoring code (`src/trace_tool/vendor/trace_rlvr`, `LICENSE` included) | Apache-2.0 |
| Qwen-Agent-derived zoom tool and test reference; Qwen2.5-VL / Qwen3-VL chat template fixtures | Apache-2.0 |
| Qwen3.5 chat template fixture (`tests/data/qwen3_5_chat_template.jinja`, identical in the pinned Qwen3.5-4B and -9B checkpoints) | Apache-2.0 (Qwen3.5-9B `LICENSE` and model card) |
| RapidOCR | Apache-2.0 |
| onnxruntime | MIT |
| opencv-python / opencv-python-headless | Apache-2.0. The wheels bundle LGPL-2.1 FFmpeg, and the non-headless Linux wheel also bundles LGPL Qt. These are dynamically linked and fine for research use, but check them before redistributing an environment. |
| transformers (optional `check` extra) | Apache-2.0 |
| torch / torchvision (optional `check` extra) | BSD-3 |
| shapely | BSD-3 |
| pyclipper | MIT |
| omegaconf | BSD-3 |
| openai | Apache-2.0 |
| vLLM / verl | Apache-2.0 |
| json5 (optional) | Apache-2.0 |
| pytesseract and Tesseract (optional) | Apache-2.0 |

Apart from the LGPL components bundled in the OpenCV wheels, all of these are permissive. This repository does not yet have a license file of its own.
