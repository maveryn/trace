# TRACE certificates: runner side (M0-M3)

Program-verifiable reasoning certificates are an opt-in, experimental extension of TRACE.
A certificate response is one JSON object:

```json
{"annotation": <the task's legacy annotation>, "certificate": {"kind": "...", ...}, "answer": <legacy answer>}
```

The scorer checks the certificate against private oracle data built from the generated scene
(never shown to the model) and reports four signals: **A** (legacy answer), **G** (legacy
annotation score), **M** (truth checks of the certificate, gated intermediates) and **F** (strict
schema of the original response). Legacy datasets, response modes, reward ids and release
receipts are untouched.

The design is `cert_notes/DESIGN.md` (kept next to this checkout, not in it). Open assumptions
are listed in [`UNRESOLVED.md`](UNRESOLVED.md); task coverage is in
[`coverage.v1.json`](coverage.v1.json).

## Layout

| path | content |
|---|---|
| `src/trace_tasks/experimental/certificates/` | library: parser, identity rule, scorer, the four adapters, export, evaluation, EasyR1 reward, CLI (`ADAPTER_INTERFACE.md` documents the adapter contract) |
| `src/trace_tasks/resources/prompts/experimental/certificates/` | `cert_prompts_v1.json` (every public prompt text), `cert_system_v1.txt`, versioned public/private JSON schemas |
| `tests/test_certificate_*.py`, `tests/fixtures/certificates/` | unit, round-trip, counterexample, equivalence, isolation, regression and reward tests on frozen fixtures |
| `experiments/certificates/reward_entry.py` | EasyR1 `reward_function` entry (`REWARD_NAME = "trace_certificate_v1"`, batch) |
| `experiments/certificates/smoke_cpu.py` | the reproducible M1 CPU smoke |
| `experiments/certificates/analysis/section5_cpu.py` | Section 5 CPU experiments (oracle round trip, corruption suite, verifier time) |
| `experiments/certificates/analysis/split_report.py` | split counts and integrity checks of an exported data root |
| `experiments/certificates/coverage.v1.json` | copy of the audit output for the built data root |
| `experiments/certificates/trace_tool_conditions_v1.yaml.in` | trace-tool conditions `ans_v1`, `ann_v1`, `cert_v1` (template; rendered per run) |
| `experiments/certificates/jobs/` | GPU job layer: `common.sh` (env, GPUs, claims, vLLM), `zs_eval.sh` (DESIGN 7.1 zero-shot task), `sft.sh` + `sft_select.py` (M2), `rl.sh` (M3), `ckpt_eval.sh` (DESIGN 7.4), `ext_eval.sh` + `ext_eval_lib.py` (trace_eval_v1, DESIGN 6/7.5 W5), `payload_check.py`, `zs_report.py` |
| `experiments/certificates/sft.py`, `configs/sft_{3b,7b}.yaml` | M2 structured SFT: `plan`, `check`, `train` (FSDP), `gen` (offline vLLM, greedy), stop-rule inputs |
| `experiments/certificates/train_rl.py`, `configs/rl_{3b,7b}.yaml` | M3 GRPO runner around TRACE's vendored EasyR1: `check`, `smoke`, `run`, `merge`, `status`, `report` |
| `experiments/certificates/tests_sft/`, `tests_rl/`, `tests_analysis/` | CPU tests of the M2, M3 and analysis code (run with `-p no:cacheprovider` from the repository root) |
| `experiments/certificates/analysis/compare_cert.py` | paired comparison of evaluated models (DESIGN 8) |

The runner lives here, not under `rlvr/`, because the release check rejects extra files there
(DESIGN D2).

| adapter | task(s) | kind | train / E1 cells | E2 held-out cells |
|---|---|---|---|---|
| `selected_set.chess_piece_count` (control) | chess `colored_piece_kind_count`, `piece_kind_count` | `selected_set` | colored task | kind-only task (program) |
| `filtered_mean.table_filtered_column_mean` | table `filtered_column_mean` | `filtered_mean` | {above, below} x {spreadsheet, zebra, card_table} | ledger (render), interval (program), interval x ledger (both) |
| `weighted_sum.lever_side_torque` | lever `side_torque_value` | `weighted_sum` | center / offset fulcrum | textured beam (render) |
| `path.rpg_movement_cost` | rpg tactical map `movement_cost_value` | `path` | landscape, portrait | square (render) |

## Environment

Everything in M0-M1 runs on CPU in a Python env with `trace_tasks` installed editable from this
checkout plus its release dependencies (numpy, scipy, pyarrow, blake3, rfc8785, zstandard,
Pillow) and pytest. `jsonschema` is optional (a tested subset validator is used without it).
Committed files never contain machine paths; locations come from environment variables:

| variable | meaning |
|---|---|
| `TRACE_CERT_RUNS` | root of every large artifact (`data/v2`, `results/`, scratch; `data/v1` is the superseded pre-review build) |
| `TRACE_TOOL_ROOT` | optional: trace-tool checkout, enables the request-payload leakage test |
| `TRACE_RLVR_ROOT` | trace-rlvr checkout: optional for the M1 tests (release-reward goldens, EasyR1 drift guard), required by `sft.sh` and `rl.sh` (EasyR1 token-identity check, the trainer itself) |

Run commands from the repository root with that env's `python`.

## Tests

The M1 gate command (every required test runs; `CERT_GATE=1` turns the skips of the
request-payload leakage test and the release-reward pin into failures):

```bash
CERT_GATE=1 TRACE_TOOL_ROOT=<trace-tool checkout> TRACE_RLVR_ROOT=<trace-rlvr checkout> \
    python -m pytest tests/test_certificate_*.py -q -p no:cacheprovider
```

Without the two roots those tests skip. The `jsonschema` metaschema checks skip when
`jsonschema` is not installed (see `UNRESOLVED.md`); the schemas are still validated by the
tested subset validator. `tests/test_certificate_review.py` pins every finding of the M1
evaluation-layer review.

The existing TRACE suite (`python -m pytest tests -q`) must stay green apart from failures that
already occur on the base commit (see `UNRESOLVED.md`).

## M1 CPU smoke (reproducible)

```bash
python experiments/certificates/smoke_cpu.py --out "$TRACE_CERT_RUNS/scratch/smoke" --workers 4
```

It runs `audit` on the five supported tasks, `export --split fixtures` (8 rows per adapter),
writes trace-tool-shaped runs for the `cert_px`, `ans` and `ann` conditions (SFT targets, prose
before and after the object, operand corruptions, a truncation, an endpoint error and a missing
row) derived from the freshly built store, scores them with `score`, and checks the expected
aggregates (exit 0 = pass, under a minute on a fresh directory). Instance ids hash the rendered PNGs, so the responses are
derived from the store instead of being committed. The same steps by hand:

```bash
python -m trace_tasks.experimental.certificates audit --manifest <tasks.json> --out <dir>/coverage.v1.json
python -m trace_tasks.experimental.certificates export --split fixtures --out <dir>/data
python -m trace_tasks.experimental.certificates score --responses <run>/responses.jsonl \
    --store <dir>/data/private/fixtures.extensions_v1.jsonl.zst --receipt <dir>/data/receipts/fixtures.json --out <run>/cert
```

## Building the data (DESIGN 6)

```bash
D="$TRACE_CERT_RUNS/data/v2"
C="python -m trace_tasks.experimental.certificates"
nice -n 10 $C export --split fixtures --out "$D" --workers 4
nice -n 10 $C export --split sft      --out "$D" --workers 4
nice -n 10 $C export --split rl       --out "$D" --workers 4
$C export-sft --split sft --out "$D"
nice -n 10 $C export --split e1_dev   --out "$D" --workers 4
nice -n 10 $C export --split e1_test  --out "$D" --workers 4
nice -n 10 $C export --split e2_test  --out "$D" --workers 4
$C freeze --out "$D"
$C audit --out "$D/coverage.v1.json" --receipts "$D/receipts"
python experiments/certificates/analysis/split_report.py --data "$D" --out "$D/split_report.json"
```

- Splits are built one at a time into one root. A split's receipt records only its own outputs
  (and, when it replaces an earlier build of the split, that build's shas). `freeze`, run once
  after the last export, checks every receipt against disk and writes the merged store
  `private/extensions_v1.jsonl.zst`, the RL reward store `private/rl_reward.extensions_v1.jsonl.zst`
  (rl and e1_dev rows only) and `receipts/freeze.json`.
- `export-sft` writes training targets only (`--split sft|rl`): a test oracle never becomes a
  supervised target. SFT images are copied (hard-linked) to `sft/images/`.
- Test splits also drop a row whose semantic key an earlier row of the same split has; the lever
  key ignores the render variant (alternate renders of one torque problem are one scene).
- The test splits drop rows whose semantic key appears in SFT or RL, so they need the SFT and RL
  receipts. A test split may be built earlier with `--no-dedupe` and rebuilt after SFT/RL: the
  candidate chunks are cached, so the rebuild only re-runs acceptance (and generates replacements
  for dropped rows). The receipt records whether dedupe ran and how many rows it dropped.
- Builds are deterministic and resumable (chunk cache under `private/build/<split>/<adapter>/chunks`).
  Any edit of the certificate package or resources (`cert_code_sha256`), of TRACE's generator
  code and resources (`trace_core_sha256`) or of the environment (`env_lock_sha256`, a `pip
  freeze` taken without `PYTHONPATH` and without the local trace_tasks line, saved as
  `receipts/env_lock.<sha>.txt`) invalidates the cache, so freeze the code before the large builds.
- The login node caps memory at 8 GiB per user: use `--workers 4` there (each spawn worker holds
  one TRACE task generator).
- `--workers` defaults to `min(16, nproc)`; the exporter refuses to fill a quota from more than
  20x its size in candidates.

Outputs under `$D`: `private/` (mode 0700; every oracle-bearing file: `private/build/<split>/<adapter>/`
TRACE dataset dirs with `accepted.jsonl`, `quarantine.jsonl`, image manifest and chunk cache, the
per-split stores `private/<split>.extensions_v1.jsonl.zst` and the two stores `freeze` writes),
`public/<split>.{cert_px,cert_rel1000,ans,ann}.parquet` (trace-tool eval inputs with value-free
labels), `rl/<split>/<split>.parquet` (EasyR1 rows with the `answer_gt` sentinel; images
embedded), `sft/sft.jsonl` and `sft/images/`, and `receipts/` with the sha256 of every output plus
code and environment provenance. No training job needs `private/`. The E1-dev, E1-test and
E2-test parquets are frozen by the sha256 values in their receipts and in `receipts/freeze.json`.

## Section 5 CPU experiments

```bash
python experiments/certificates/analysis/section5_cpu.py --data "$D" \
    --out "$TRACE_CERT_RUNS/results/section5_cpu" --splits e1_dev,e1_test,e2_test --workers 12
```

For every frozen evaluation row: the unrounded oracle object must score A = G = M = F = 1 and the
rounded SFT target A = M = F = 1 with every box resolved (every optimal rpg path too); the
corruption suite reports per adapter x corruption mean A/G/M/F, the fraction of rows meeting
every declared expectation and the go/no-go 1(b) count (corruption types per numeric adapter
that keep G and lower M); verifier wall and CPU time per row. Exit code 0 means a 100% round
trip and no unmet corruption expectation.

## Scoring model runs (after zero-shot generation, needs GPUs)

trace-tool runs one condition per dataset: `public/<split>.cert_px.parquet` for Qwen2.5-VL,
`.cert_rel1000.parquet` for Qwen3.5, `.ans.parquet` (TRACE system prompt) and `.ann.parquet`
(`cert_system_v1.txt`). Pass `--split cert_<split>`. Then:

```bash
$C score --responses <run>/<cond>/seed<k>/responses.jsonl --store "$D/private/<split>.extensions_v1.jsonl.zst" \
    --receipt "$D/receipts/<split>.json" --out <run>/<cond>/seed<k>/cert
$C summarize --scores <run>/*/seed*/cert/cert_scores.jsonl --out <run>/summary.json
```

The condition, frame and model family are read from the run's `run_config.json`: the dataset
sha256 must be a frozen public parquet of the export receipt (`--cond` only confirms it), the
settings must be DESIGN 7.1's (system prompt of the condition, `oracle_crop: false`, no tools,
`trace` decoding), and each transcript's prompt must hash to the prompt the extension froze for
the condition and frame. `--allow-unfrozen`, `--allow-settings-mismatch` and `--no-prompt-check`
override these checks and are recorded in `score_receipt.json` and `summary.json`;
`--store-sha256` is an extra requirement on top of the receipt. Endpoint errors and server
rejections (no tokens generated) are excluded and counted, never scored as model zeros;
`summarize` sums every run's denominators (from the `summary.json` next to each scores file).

## RL reward (M3 preview)

EasyR1 loads `experiments/certificates/reward_entry.py:compute_score` by path with
`reward_function_kwargs = {store_path, store_sha256, arm: A|G|P, alpha: 0.8, format_weight: 0.05,
gated: false, coord_frame: px_source, split_phases: {rl: train, e1_dev: val}, details_dir}`;
`PYTHONPATH` must put this checkout's `src` first. `gated` must be a YAML boolean (a quoted
`'false'` raises), the reward refuses any extension whose split is not in `split_phases` (the
default), and a row's `prompt_cert` must be the prompt frozen for `coord_frame`. Use the RL
reward store `private/rl_reward.extensions_v1.jsonl.zst` (sha in `receipts/freeze.json`). Arms
differ only in `overall`; every arm logs A, G, M and F. Pass EasyR1 file paths:
`$D/rl/rl/rl.parquet` and `$D/rl/e1_dev/e1_dev.parquet` (prompt column `prompt_cert`); as a
directory, HF `datasets` reads `rl/e1_dev` as a `validation` split and EasyR1's `split="train"`
fails.

## Zero-shot GPU jobs (DESIGN 7.1, 7.5)

`jobs/zs_eval.sh MODEL [--smoke] [--dry-run]` is one use-gpu task line: it lands on one node of an
allocation that is already held, serves MODEL with trace-tool's own `scripts/serve_vllm.sh` and
`_model_env.sh` (pinned revision, TP=1, DP=<GPUs>, `.venv` with vLLM), generates with
`trace-tool-eval --config <rendered trace_tool_conditions_v1.yaml>` for e1_dev, e1_test, e2_test x
`ans_v1`, `ann_v1`, `cert_v1` x seeds 42, 43, 44 (resumable), stops vLLM, scores every finished
run on CPU with `score`/`summarize` above, writes `summary.md`, and validates row counts. Outputs:
`$TRACE_CERT_RUNS/eval/zs/<served>/<split>/<cond>/seed<k>/`. `cert_v1` reads the `cert_px` parquet
for Qwen2.5-VL (also `maveryn/trace-qwen2.5-vl-*`) and `cert_rel1000` for Qwen3.5. `--smoke` runs
20 rows (5 per task) of e1_dev `cert_v1` seed 42 into `eval/zs_smoke/` and prints A/G/M/F and the
frame-obedience diagnostics; `--dry-run` runs only the CPU preflight.

Before any GPU work the job renders the conditions file (checking `cert_system_v1.txt` against the
export receipt), checks the public parquets against their receipts, and checks the exact request
trace-tool sends (`payload_check.py`: system text, user text and one image only; TRACE decoding
preset; `enable_thinking: false` for Qwen3.5 only; no label values or ids) plus trace-tool's own
`--dry-run` rendering. On the node it refuses cards below the model's need (7B/9B: 40 GB, 3B/4B:
20 GB; exit 4), exits 3 when a selected GPU has a compute process or memory in use or a fresh claim
of another campaign (AIMO claims on idle cards are shared), writes `trc_*` claim files in
`$TRC_CLAIM_DIR` (heartbeat every 60 s, removed on exit), and logs a utilisation check 60 s after
generation starts. Exit codes: 0 done, 1 incomplete (rerun resumes), 3 GPUs busy, 4 configuration,
5 vLLM did not start.

Environment (set by an uncommitted env file next to the runs, never in this checkout):
`TRACE_CERT_RUNS`, `TRACE_TOOL_ROOT`, `HF_HOME`, `TRACE_CERT_PYTHON` (the scoring env's python) and
`TRC_CLAIM_DIR`; `common.sh` documents the optional knobs. `TRC_NVIDIA_SMI` and
`TRC_SERVE_SCRIPT` let the whole job run on CPU against stand-ins.

## SFT, RL and checkpoint-evaluation GPU jobs (DESIGN 7.2, 7.3, 7.4)

All three are single use-gpu task lines like `zs_eval.sh`: they use every GPU of the allocation they land
on, claim the cards (`trc_*` files in `$TRC_CLAIM_DIR`, heartbeat every 60 s, removed on exit), exit 3 when
the cards are busy or claimed, and resume when the same line is run again. They run in the training env
(`TRC_RL_PYTHON`: torch 2.8, vLLM 0.10.2, EasyR1 from `$TRACE_RLVR_ROOT`, read-only); scoring runs in the
`trace_cert` env on CPU. Frozen data comes from `$TRC_DATA` (default `$TRACE_CERT_RUNS/data/v2`).

```bash
source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT"
bash experiments/certificates/jobs/sft.sh 3b [--smoke] [--dry-run]
bash experiments/certificates/jobs/rl.sh --model 3b --arm P --seed 1 --sft-dir "$TRACE_CERT_RUNS/sft/qwen2.5-vl-3b/selected" [--dry-run]
bash experiments/certificates/jobs/rl.sh --model 3b --arm P --seed 1 --smoke
bash experiments/certificates/jobs/ckpt_eval.sh <ckpt dir> <served name> [--splits e1_dev] [--smoke]
```

| job | what it does | output | exit codes |
|---|---|---|---|
| `sft.sh 3b\|7b` | CPU preflight (data sha256, token lengths, EasyR1 prompt-id identity), FSDP training (global batch 32, 3 epochs, a save every half epoch), greedy 2048-token E1-dev generation of every save (offline vLLM, one process per GPU), CPU scoring, then `sft_select.py` applies the pre-registered stop rule | `$TRACE_CERT_RUNS/sft/<tag>/` (`sft_smoke/` with `--smoke`): `train_log.jsonl`, `ckpt/step_*`, `dev/step_*`, `dev_curve.json`, `receipt.json`, `selected` -> checkpoint | 0 selected; 1 incomplete (rerun resumes); 3 busy; 4 configuration |
| `rl.sh --model --arm --seed --sft-dir` | 300 GRPO steps (64 prompts x 8 rollouts), save every 25 steps, steps 100/200 hard-linked into `keep/`, steps 100/200/300 merged to bf16 HF dirs and loaded once with the serving env's vLLM | `$TRACE_CERT_RUNS/rl/<model>/<arm>/s<seed>/`: `run_receipt.json`, `progress.json`, `ckpt/experiment_log.jsonl`, `logs/`, `keep/`, `hf/global_step_<k>/` | 0 complete; 1 incomplete (rerun resumes); 3 busy; 4 configuration (including host RAM below the spec) |
| `ckpt_eval.sh CKPT NAME` | the `zs_eval.sh` pipeline for a local checkpoint: E1-dev, E1-test, E2-test x `cert_v1`, `ans_v1` x seeds 42-44, plus the released TRACE validation and protocol run | `$TRACE_CERT_RUNS/eval/ckpt/<name>/` with `summary.json`, `summary.<split>.json` and `summary.md` | as `zs_eval.sh` |

`ckpt_eval.sh` writes `summary.md` last, atomically: the per-split table of A, G, G_shown, M, F, joint,
valid-schema, completion, truncation, within-2048 rate, mean tokens, A=1 with M<1, A=0 with M=1 and verifier
time (the `zs_report.py` report, retitled "Checkpoint certificate evaluation"), a line naming the checkpoint,
base model and weight fingerprint, and the `trace_val` accuracy table. The official TRACE score of
`trace_val` needs the Qwen3-32B judge; rows the extractor cannot resolve are listed in `judge_pending.jsonl`
and count as wrong in `official_acc`, a lower bound.

### Hardware and host-RAM gates

- **GPU memory.** `rl.sh` needs `hardware.min_vram_gb[N]` per card from the spec (80 GB for 3B, N in {2, 4}); `sft.sh`
  needs `resources.min_card_mib` / `min_total_mib` from the plan (75,000 MiB per card, 150,000 MiB in total for 3B).
  Below that the job exits 4.
- **Host RAM (RL only).** EasyR1's release offload settings keep the actor's fp32 parameters and AdamW state
  and vLLM's sleeping weights in host memory. The spec key `hardware.min_host_ram_gb` (by N: 3B `{2: 200, 4: 250}`,
  7B `{2: 320, 4: 360}`; the 7B and N=4 values are estimates) is compared with the tightest memory-cgroup limit
  of the job step before any trainer starts (`train_rl.py` `run` and `smoke`); an allocation below it exits 4
  with a configuration error. `TRC_RL_MIN_HOST_RAM_GB` overrides the spec value for one invocation (`0` disables
  the gate; a lower value only makes sense for a timing probe). Measured on 2 x H200 NVL (3B, 64 x 8 rollouts):
  about 164 GB unreclaimable (resident plus shared memory), 245 GB at the cgroup peak with page cache; a 2 x A100
  allocation with a 120 GB limit was OOM-killed in the first update. Only allocations with at least about 200 GB
  of host RAM qualify; of the fleet that is the 2 x H200 NVL allocation (360 GB limit), so RL has exactly one
  place to run.
- **Memory records.** The trainer samples the cgroup every 30 s into `logs/host_mem.attempt<k>.tsv` (columns:
  epoch seconds, `used_gb` = resident + shared, `peak_gb` = cgroup peak including page cache, `limit_gb`) and
  GPU memory into `logs/gpu_mem.attempt<k>.tsv`. `progress.json` carries the running figures: `attempt`,
  `last_logged_step`, `tracker_step`, `max_steps`, `gpu_peak_mib` per GPU and `host_ram_gb`
  (`used_gb`, `peak_gb`, `limit_gb`); the run receipt's `summary.host_ram_cgroup` has the maxima over all attempts.
- **GPU class.** An RL run directory refuses another GPU type than the one it was started on
  (`TRC_RL_ALLOW_HW_CHANGE=1` overrides, which breaks per-class matching of the arms). `sft.sh` does not check
  the class: pin it in the dispatch knobs (see greedy-eval nondeterminism below).

### Measured timings (2026-10-18/19)

| job | hardware | measured | projection |
|---|---|---|---|
| SFT 3B, 564 steps | 2 x H200 NVL | 3.9-4.0 s/step (20-step smoke median 3.96 s; the full run logs 3.8-3.9 s) | about 37 min of training; with six saves, six 512-row dev generations and scoring about 0.9 h |
| SFT 3B | 2 x A100 80 GB PCIe | 9.4 s/step | about 88 min of training; about 2 h in total |
| RL 3B, 64 x 8 rollouts | 2 x H200 NVL | 3-step probe: 162 s/step median (rollout 15 s, old log-prob 20 s, actor update 87 s, checkpoint save 40 s because the smoke saves every step); without the per-step save about 121 s | 300 steps about 10.1 h; with twelve saves, six validation passes and the merges about 10.5 h per arm |
| RL 3B | 2 x A100 | not runnable (host RAM) | n/a |

GPU memory of the RL probe: 37 GB allocated, 51 GB reserved, 84 GB at the `nvidia-smi` peak (vLLM's cache).
The 15-20 h per arm estimate of DESIGN 7.5 was unmeasured; the pre-declared fallback (rollout batch 32 when a
projection exceeds 24 h) is not needed.

### Run-time fixes that the smokes forced (all in this checkout)

- `sft.sh` gives every generation shard its own `VLLM_CACHE_ROOT` on node-local disk (`TRC_SFT_VLLM_CACHE` pins
  one); concurrent shards compiling into the shared user cache raced on its renames (`BackendCompilerFailed`,
  `FileNotFoundError`). It also puts the training env's `bin/` first on `PATH` for the shards (JIT builds spawn `ninja`).
- `sft.py` casts every rank's model to fp32 and compares the parameter and buffer layout across ranks before FSDP.
  Ranks other than 0 build the model on the meta device with `from_config`, which builds Qwen2.5-VL's language
  model in bf16, so the sync-module-states broadcast had different bucket layouts per rank and hung until the 60
  min NCCL timeout (two smoke failures after 3,681 s). The cast makes the layouts equal; a mismatch now raises
  `ConfigError` instead of hanging.
- `train_rl.py` `vllm_check` puts the serving venv's `bin/` first on `PATH` and sets `VIRTUAL_ENV` (as `activate`
  does), so FlashInfer / `torch.compile` JIT finds `ninja`; before, the post-training vLLM load of the merged
  directory failed at engine initialisation. `rl.sh` and `sft.sh` export `CPATH` to the env's own `Python.h` when the
  system Python has no development headers (Triton compiles launchers at run time; the first launch failed on this).
- `train_rl.experiment_records` merges the train record and the validation record that EasyR1 writes for the
  same step instead of letting the later one replace the earlier; before, a step with validation lost its
  timing and reward metrics (the "step time median None" lines of the first A smoke). A later attempt's step
  still replaces an earlier attempt's.

### Greedy-eval nondeterminism across hardware

The same SFT smoke (config, data, seed 1, 20 steps, 2 GPUs, both runs after the fp32 fix) trained and
evaluated on 2 x A100 and on 2 x H200 NVL gives different weights in the fourth digit (step-1 loss 0.30439 versus
0.30468) and different greedy E1-dev generations on the 32 smoke rows (different `responses_sha256`; A 0.281 on
both, G 0.360 versus 0.400, M 0.309 versus 0.333, F 0.969 versus 0.938). Greedy decoding removes sampling
noise, not numerical differences between GPU architectures (kernel choice, reduction order, FlashAttention and vLLM
paths). The SFT stop rule compares consecutive checkpoints within 1.5 / 2 pp, so a rerun on another GPU class may
select a different checkpoint. **Pin the whole of one SFT (training, every dev generation, selection) to one
GPU class**, here `ONLY=grn067` (2 x H200 NVL) in the dispatch env file, and never resume it elsewhere. RL
arms of one seed are likewise pinned to one class (`rl.sh` enforces it).

### Status rows

The host watcher (outside this checkout) reads `train_log.jsonl` / `dev_curve.json` of the SFT run and
`progress.json` plus the experiment logs of each RL run and shows step / total, s/step, ETA, the last A, G, M, F
and overall reward and host RAM in `STATUS.md`. It flags a non-finite loss or reward (`nan`), a running job with no
new step for 45 minutes (`stall`) and an overall reward whose latest 10-step mean falls below 25% of its first
10-step mean after step 50 (`reward_collapse`).

## External-transfer evaluation: trace_eval_v1 (DESIGN 6, 7.4, 7.5 W5)

`jobs/ext_eval.sh MODEL_OR_DIR SERVED_NAME [--seed S] [--subset q25|full] [--smoke] [--judge-only] [--dry-run]` is one
use-gpu task line that runs TRACE's own 24-benchmark `trace_eval_v1` suite from a **separate clone** of the rlvr branch
at af32fe0 (`$TRC_EXT_ROOT/trace-rlvr`, default `$TRACE_CERT_RUNS/ext_eval/trace-rlvr`; its setup writes `.work/` and
clones VLMEvalKit, so it is never a worktree of the read-only `$TRACE_RLVR_ROOT`). It calls the protocol's components
with the arguments `run_trace_eval.sh` uses: the vLLM endpoint pool (one server per GPU), the generation queue
(temperature 0.6, top_p 1, top_k -1, max_tokens 4096, file-url media, seed S), CPU finalization, the score campaign
(deterministic routes + Qwen3-32B judge routes) on a judge pool over the node's GPUs, and for full runs the protocol's
`verify_trace_eval.py` and `summarize_trace_eval.py`. Python helpers are in `jobs/ext_eval_lib.py` (tests:
`tests_ext/`). The env is `$TRC_EXT_PYTHON` (`envs/trace_eval`).

```bash
source "$TRACE_CERT_RUNS/host/env.sh" && cd "$TRACE_ROOT"
bash experiments/certificates/jobs/ext_eval.sh Qwen/Qwen2.5-VL-3B-Instruct qwen2.5-vl-3b-instruct --seed 42
bash experiments/certificates/jobs/ext_eval.sh "$TRACE_CERT_RUNS/rl/3b/P/s1/hf/global_step_300" rl3b-p-s1-step300 --subset q25
bash experiments/certificates/jobs/ext_eval.sh Qwen/Qwen2.5-VL-3B-Instruct qwen2.5-vl-3b-instruct --smoke
```

- **Models.** An HF id must be pinned in trace-tool's `scripts/_model_env.sh`; it is served from
  `.work/models/<SERVED>` (symlinks into the HF cache plus the protocol's provenance marker). A local merged
  checkpoint gets the protocol's `register-local` marker (revision `sha256set:<content>`).
- **Subsets.** `q25` is the frozen DESIGN 6 subset: blake3 rank of `(benchmark, row_id)` within each benchmark,
  `ceil(n/4)` rows, except WeMath (whole problem groups: its official score is defined over 525 groups; the summary
  rescales by 525/132) and coverage of every `split`/`category`/`difficulty` value the TableVQABench, PuzzleVQA and
  VisualPuzzles scorers index (a no-op for q25). 8,207 rows; row list and its sha256 in
  `$TRC_EXT_ROOT/subsets/trace_eval_v1_q25.rows.json`. Subset rows keep the full run's ranks and request hashes, so
  they are byte-compatible with a full run. `--smoke` uses 4 rows per benchmark plus the same exceptions (137 rows).
- **Judge.** Qwen3-32B bf16 (61 GiB) on the smallest tensor-parallel group of 1/2/4/8 cards with >= 16 GiB headroom
  (>= 6 GiB with 8,192 batched tokens). A node that cannot host it finalizes generation, writes
  `PENDING_JUDGE.seed<S>` with the `--judge-only` task line, raises `FLAG judge_pending`, and exits 0.
- **Outputs.** `$TRC_EXT_ROOT/runs/<SERVED>/<full|q25|smoke>/`: `campaign/` (row results, predictions, workbooks),
  `scoring/` (protocol score campaign), `hf_archive/` (per-row records), `summary.seed<S>.{json,md}` (per benchmark,
  six categories, overall, natural 9 / chart-diagram-synthetic 15 / MMStar alone, deterministic-extraction failure,
  judge-needed proxy and certificate-JSON rates, rows scored incorrect under post-release fix-2 and fix-3),
  `validation.seed<S>.json`, `timing.seed<S>.json`, `logs/`.
- **Exit codes** as `zs_eval.sh`; the job runs a private copy of itself, so editing the script never corrupts a
  running job.

### trace_eval_v1: measured (2026-10-19) and projected wall-clock per model and seed

q25 probes (8,207 rows, base models, defaults): 1x RTX PRO 6000 96 GB, 3B: 41 min (generation 22 min at 6.6
rows/s, judge TP1 scoring 13 min); 8x RTX PRO 4000 24 GB, 3B: 24 min (generation 8.6 min at 20 rows/s, two TP4
judges 6.7 min); 2x A100 80 GB, 7B: 47 min (generation 22 min at 6.9 rows/s, judge TP2 13.6 min); 2x L40S, 3B: 8.7
rows/s (the allocation was preempted before scoring). Fixed costs per job: media rehash 1-5 min, vLLM start 1-1.5 min
(generation) and 1.3-3.5 min (judge), finalization 1-1.3 min, scoring floor about 4.5 min. q25 overall matches the
released full-suite seed-42 scores (3B 38.91 vs 39.29, 7B 48.16 vs 48.12). Projected full runs (32,805 rows): 3B
about 1 h on 8x RTX PRO 4000, 1.5 h on 4x RTX PRO 4000, 1.6 h on 2x A100, 2 h on 2x L40S, 2.2 h on 1x RTX PRO 6000;
7B about 2.3 h on 2x A100, 3.1 h on 2x L40S, 3.8 h on 1x RTX PRO 6000. Trained checkpoints write about 1.4x longer
answers (legacy TRACE-3B smoke: 184 vs 127 tokens per row); the workload is prefill-bound (1,591 prompt tokens per
row), so add 10-30% to generation.

Run-time findings: on Blackwell (sm_120) vLLM 0.10.2 needs flash-attn for the Qwen2.5-VL vision tower (xformers' FA3
kernel fails: "flash_fwd_launch_template.h:188: invalid argument"); on 24 GB cards a 3B server at the protocol's 0.90
memory fraction OOMs on the largest images, hence 0.80 there; the af32fe0 CharXiv scorer calls a helper the release
does not ship (post-release fix-1, below); the af32fe0 MME-Reasoning scorer fails the whole run when the official
evaluator raises on one row (post-release fix-2, below); the af32fe0 MathVerse and MME-Reasoning scorers fail the whole
run when the judge's output for one row is still unusable after its 5 attempts (post-release fix-3, below); WeMath,
TableVQABench, PuzzleVQA and VisualPuzzles scorers
fail on subsets that miss whole groups or strata (handled by the subset rule); granite-gpu allocations can be
preempted (rerun resumes from the per-row results).

### trace_eval_v1: post-release fixes and the MME-Reasoning deviation

The clone carries three fixes on top of af32fe0, applied by `$TRC_EXT_ROOT/patches/apply_post_release_fixes.sh`
(idempotent, one marker per fix; rerun it after `setup_trace_eval_env.sh`). `ext_eval.sh` exits 4 when any of them is
missing (fix-3 in either of its two scorer scripts), so every scoring from now on uses all three. Which runs carry
which code hash, and why the complete runs scored before fix-3 must not be rerun, is under "fix-3 provenance" below.

- **fix-1, CharXiv (a release gap, not a deviation).** `run_external_benchmark_score_queue.py` calls
  `_parse_charxiv_judge`, which the released `batched_vlmevalkit_qwen3vl.py` does not define. It is added with the
  semantics of that module's own JSON-object parser, in the generated VLMEvalKit copy.
- **fix-2, MME-Reasoning unevaluable rows (a deviation from af32fe0; orchestrator decision 2026-10-19, identical for
  every model).** af32fe0's `run_mme_reasoning_eval.py` fails the MME-Reasoning slice, and so the run, when the
  official per-row function raises on any row. In the full 3B runs this is `multiple_match_function`'s
  `assert len(response_dict) == 1`: the judge's extraction of the model's answer has two keys
  (`{"id": [...], "answer": ...}`, `{"id": [...], "possible_answer": [...]}`, `{"removed": [...],
  "squares_left": [...]}`), because the model's answer carries extra fields. fix-2 adopts upstream VLMEvalKit's own
  semantics for unevaluable rows: `MMEReasoning.evaluate` (`vlmeval/dataset/image_vqa.py`) scores a row it cannot
  evaluate as incorrect, with `log_score = 'Failed to evaluate'`. Upstream applies that to a missing extraction and to
  a non-boolean result and has no handler for an exception; fix-2 applies it to an exception raised by the official
  function on the extracted response (failure kind `official_evaluator_exception`, stage `deterministic_scoring`) and
  to nothing else. Infrastructure and contract failures still fail the slice: an unknown function id, missing
  `special_info`, a missing or non-JSON extraction, a non-boolean result, ImportError/OSError/MemoryError inside the
  function, judge failures, missing files and schema problems. The affected rows stay listed in the slice's
  `mme_reasoning_failures.json` (`"resolution": "scored_incorrect"`, next to `scores.json`); `scores.json` counts
  them (`evaluation.evaluator_exception_scored_incorrect` and the indices; evaluation contract
  `trace-eval-v1-mme-evaluation-v5+trace-cert-fix-2`); `summary.seed<S>.{json,md}` reports
  `evaluator_exception_scored_incorrect` per benchmark and in total. In all three affected rows below the first
  extracted key is not an accepted answer either, so reading only that key would also score the row incorrect.

  | full, seed 42 (all scored under fix-2) | rows scored incorrect by fix-2 (index) | MME-Reasoning | overall 24 | natural 9 | chart/diagram/synthetic 15 | MMStar |
  |---|---|---|---|---|---|---|
  | Qwen2.5-VL-3B-Instruct | 1 of 1,188 (305) | 22.05 | 39.08 | 47.36 | 34.11 | 53.20 |
  | TRACE-3B (released) | 0 | 25.42 | 42.34 | 49.88 | 37.81 | 56.00 |
  | cert SFT 3B, step 564 | 2 of 1,188 (305, 411) | 22.39 | 37.92 | 48.73 | 31.44 | 55.53 |

- **Provenance.** fix-2 changes `TRACE_EVAL_CODE_HASH` (221335f51667 to 16d37ed061fd; the evaluator hash
  7bae61e285d1 stays, because it fingerprints VLMEvalKit and TRACE's extension files, not TRACE's scorer scripts) and
  with it the score-campaign contract. A score campaign started before a fix therefore cannot resume after it:
  `ext_eval.sh` moves the old scoring to `scoring/stale/` and rescores all 24 benchmarks, keeping the generation. It
  does not rescore a complete run by itself (it raises `FLAG score_code_stale` when the complete scores were made
  under other code); to rescore one, move its `scoring/seed_<S>/benchmark/mme_reasoning/<SERVED>/` slice to
  `scoring/stale/` and rerun the same task line. Never delete generations. The code hash also enters every
  generation contract, so the protocol's `verify_trace_eval.py --phase generation` rejects a generation made before
  a scoring-only fix; `ext_eval.sh` then verifies the generation against the code hash it was produced under, which
  the protocol's local archive records (`ext_eval_lib.py archive-code-hash`, `verify-generation`). `hf_archive/`
  keeps the score slices of earlier scorings; each records its code hash.
- **Rescoring noise.** The three full 3B seed-42 runs were all rescored under fix-2 (2026-10-19), so they share one
  code hash (16d37ed061fd). Rescoring re-runs the judge, and judge-scored benchmarks move slightly: TRACE-3B's
  MathVision, MathVerse and SpatialVizBench by at most 0.17 points (overall 42.35 to 42.34); the base model's up to
  0.89 (MathVerse 31.85 to 32.74; MathVista +0.50, WeMath -0.28); the SFT model's at most 0.10. Runs that completed
  before fix-2 (the smokes and q25 probes) had no such row, since one would have failed the slice, so fix-2 would give
  them the same scores; their score contracts keep code 221335f51667.
- **fix-3, judge output still unusable after the official 5 attempts (a deviation from af32fe0; orchestrator decision
  2026-10-23, identical for every model; one fix in two scorer scripts, so one new code hash).** af32fe0 fails the
  benchmark slice, and so the run, when the Qwen3-32B judge's output for a single row is still unusable after its 5
  attempts (temperatures 0, 0.5, 1, 1.5, 2). Upstream VLMEvalKit scores such a row incorrect: `MathVerse_auxeval_score`
  accepts only a bare `0` or `1` and returns score False after "All 5 retries failed."; `MMEReasoning_extract` returns
  `res=''` after 5 failures and `MMEReasoning.evaluate` scores that row 'Failed to evaluate', False. fix-3 adopts these
  semantics for these two cases and nothing else:
  - *MathVerse score judgement* (`run_external_benchmark_score_queue.py`): a judgement without a 0/1 decision under the
    unchanged strict parser `_MATHVERSE_SCORE_DECISION_RE` after the 5 attempts becomes judge output `0` (raw output
    kept as `judge_output_raw`), i.e. score False, log_score `Judge output: 0`. The rows and the raw output of every
    attempt are in the run's `mathverse_score_unresolved.json`; `scores.json` counts them
    (`artifacts.mathverse_all_retries_failed_scored_incorrect` and the indices, 0 when none); the judge cache contract
    becomes `trace-eval-v1-mathverse-vlmevalkit-judgement01-v4+trace-cert-fix-3`. Trigger: rl3b-a-s3-300 seed 42,
    first scoring (now `scoring/stale/seed_42.2063874.522.1024_015748`), row 1880 (standard answer `4\sqrt{3}`). Its
    extraction was itself degenerate: `4√3` followed by three few-shot items (8-10) the extraction judge invented, and
    that text reached the score prompt. All five score judgements conclude 1, but as `### Judgement: **1** ✅`,
    `### **Judgement: 1** ✅`, `✅ **Judgement: 1**` (also judging the invented items), `## ✔️ Judgement: 1`, and
    `**Judgment**: 1 ✅` followed by `1. **Judgement**: 1` (again with the invented items), which the parser rejects.
    Upstream would reject all five as well (none is a bare `1`) and score the row False, so under fix-3 this probably
    correct row would count as incorrect (it did not recur; see the outcome below).
  - *MME-Reasoning answer extraction* (`run_mme_reasoning_eval.py`): an extraction still unparseable after the 5
    temperatures gets `res=''` and log 'All 5 retries failed.', and the row is scored incorrect ('Failed to evaluate')
    without reaching the open-answer judge or an official function. The row joins fix-2's rows in
    `mme_reasoning_failures.json` (kind `malformed_extraction`, stage `answer_extraction`, `"resolution":
    "scored_incorrect"`); `scores.json` counts it (`evaluation.extraction_failed_scored_incorrect` and the indices;
    `judge.unresolved_extractions`, until now always 0), `evaluator_exception_scored_incorrect` keeps counting fix-2's
    rows only, and the evaluation contract becomes `trace-eval-v1-mme-evaluation-v5+trace-cert-fix-2+trace-cert-fix-3`.
    Trigger: rl7b-p-s1-300 seed 42, first scoring (now `scoring/stale/seed_42.2063874.554.1024_024105`), index 950
    (a rule puzzle scored by `match_answer_function`): a degenerate 4,096-token answer listing `grid id:N:N` up to
    392; the judge's JSON extraction is cut at its token limit in attempts 0-3 and malformed in attempt 4. Incorrect
    under any reading.
  - Judge request failures (API errors, an empty or truncated MathVerse judgement after the token-limit retries),
    MME-Reasoning open-answer judgements and every other failure still fail the slice. A row with a usable judge
    output is scored exactly as before: replaying the cached judge outputs offline, the patched and the unpatched
    scorers give identical scores, tables and failure files whenever no row is exhausted. `summary.seed<S>.{json,md}`
    reports `extraction_failed_scored_incorrect` and `mathverse_all_retries_failed_scored_incorrect` per benchmark and
    in total (a summary written before fix-3 has no such keys; regenerated, it reports 0 for scores made before
    fix-3). Expected impact: about one row in a run that hits it (1 of 788 MathVerse rows is 0.13 points of MathVerse,
    1 of the 197 q25 MathVerse rows 0.51 points; 1 of 1,188 MME-Reasoning rows is 0.08 points). This is an expectation,
    not a bound: fix-3 caps nothing, so a judge fault that left many outputs unusable would score those rows incorrect
    instead of failing the slice. Read these two counters in every summary.
- **fix-3 outcome (2026-10-24).** Rescoring re-queries the judge (the failed campaigns went to `scoring/stale/`, and
  the MathVerse score cache contract changed), so a trigger need not recur. rl3b-a-s3-300's row 1880 got a clean
  extraction `4√3` and `Judgement: 1` on the first attempt and is scored correct (MathVerse 21.07, fix-3 count 0).
  rl7b-p-s1-300's index 950 failed again and is scored incorrect (MME-Reasoning 24.33,
  `extraction_failed_scored_incorrect` 1). In the 37 runs complete under d9319e7db480 by 2026-10-24 16:15 (11 full,
  26 q25), fix-3 scored 4 rows incorrect, one per run: that MME-Reasoning row and three MathVerse score judgements
  (rl3b-g-s3-300 seed 44 row 3240, rl3b-p-s3-300 seed 44 row 3695, the q25 run of rl3b-a-s2-100 row 1780). In all
  three the extraction judge hit its token limit right after the answer and went on with its own reasoning, that text
  reached the score prompt, and in all five attempts the score judge replied with an answer instead of 0/1 (for row
  3695 three times the standard answer, 720 cm³, and twice the model's 984).
  The leading extracted answer differs from the standard answer in each (28.27 vs 113.10 cm³, 984 vs 720 cm³, 13.54
  vs 18.05), so all four rows are incorrect under any reading.
- **fix-3 provenance (2026-10-23; runs as of 2026-10-24).** fix-3 changes `TRACE_EVAL_CODE_HASH` from 16d37ed061fd to
  d9319e7db480 (computed as `ext_eval.sh` does; the evaluator hash 7bae61e285d1 stays). Every scoring from 2026-10-24
  on is made under d9319e7db480. The code hash in each run's score contract:
  - 221335f51667 (af32fe0 + fix-1): the smokes and the q25 probes of the base models, as under fix-2.
  - 16d37ed061fd (af32fe0 + fix-1 + fix-2): 24 complete full runs. These are the 16 seed-42 runs scored before fix-3
    (Qwen2.5-VL-3B/7B-Instruct, TRACE-3B/7B, cert SFT 3B/7B step 564, rl3b-{a,g,p}-s1-300, rl3b-{a,g,p}-s2-300,
    rl3b-{g,p}-s3-300, rl3b-pgated-s1-300, rl7b-g-s1-300) and the 8 seed-43/44 runs of cert SFT 3B step 564 and
    rl3b-{a,g,p}-s1-300.
  - d9319e7db480 (af32fe0 + fix-1 + fix-2 + fix-3): the seed-42 rescoring of rl3b-a-s3-300 and rl7b-p-s1-300, whose
    first scoring failed on the two trigger rows. Both first campaigns were started under 16d37ed061fd, so
    `ext_eval.sh` moved them to `scoring/stale/` and rescored all 24 benchmarks, keeping the generation. Also
    rl3b-ggated-s1-300 seed 42, seeds 43 and 44 of rl3b-{g,p}-s{2,3}-300, and the `trace_eval_v1_q25` runs of the
    step-100 and step-200 checkpoints (rl3b-{a,g,p}-s{1,2,3}, rl3b-{g,p}gated-s1, rl7b-{g,p}-s1).

  The 24 complete full runs under 16d37ed061fd are not rescored. fix-3 changes only rows whose judge output stayed
  unusable through all 5 attempts, and a run that completed under 16d37ed061fd has none, because such a row failed the
  slice. Replaying the recorded judge outputs of the 16 seed-42 runs' MathVerse and MME-Reasoning slices (the only two
  that fix-3 touches) offline through both scorers gives identical scores and tables; only the contract strings and the
  new counters (0) differ. The tables therefore mix two code hashes that score these runs identically. Their score
  receipts, however, verify only against the scorer tree they were made under. The protocol's `verify_trace_eval.py
  --phase score` reports 0/24 for each of the 24 runs against the patched clone (`receipt_invalid: contract trace_eval
  code provenance does not match the active scoring surface`) and 24/24 against the 16d37ed061fd tree (below). The 11
  complete full d9319e7db480 runs give the reverse result (all 35 checked 2026-10-24).

  So rerunning such a run's task line raises `FLAG score_code_stale` and, for a full run, `FLAG
  protocol_verify_failed`. The job still ends COMPLETE with unchanged scores, and it rewrites the summary, validation
  and protocol results files (fix-3 counters at 0). Do not rerun these runs' task lines: they were verified when they
  completed, and they are reported as they are. A rerun of a 221335f51667 smoke or q25 probe raises only
  `score_code_stale`, since the protocol verifier runs for full runs only. To re-verify the old receipts, rebuild the
  16d37ed061fd tree and run the verifier from that tree's scripts. The tree is an af32fe0 checkout with fix-1 and fix-2
  only, i.e. `apply_post_release_fixes.sh` without its final `fix3` call: `run_external_benchmark_score_queue.py`
  unchanged from af32fe0 (sha256 e217a2ecfeed9ffb3e28a9a83b397e85d9a6e5882a97cd1de3132c1922fefa1b) and
  `run_mme_reasoning_eval.py` with fix-2 (sha256 819b1495e1980998350d9e2ef8d52ad03f2b97a94474e71308b97b62dd8d8381).
  In the patched clone the two files are df26ee05… and ee198256….
