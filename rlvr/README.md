# TRACE RLVR Training and Evaluation

This directory provides the GRPO training configurations used for the TRACE
checkpoints:

- `trace-qwen2.5-vl-3b`
- `trace-qwen2.5-vl-7b`

It also provides the paper's TRACE validation and 24-benchmark
`trace_eval_v1` external evaluation. The two training profiles always select
`prompt_answer` and score `answer_gt`; the dataset's optional
`trace_supervision_mode` column is advisory metadata and is not consumed.

## Environment

Run these commands from the root of a Git clone checked out to the `rlvr`
branch. The `rlvr` directory is not included in the `trace-tasks` wheel or
source distribution. The reference runs used CPython 3.10.12, CUDA 12.8, and
eight NVIDIA H100 80GB GPUs.

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0
python -m pip install --no-build-isolation -r rlvr/requirements-cu128.txt
python -m pip install -e . --no-deps
```

Direct dependency pins and source hashes are recorded in
`environment_receipt.v1.json`. EasyR1 is included as source and is not
installed as a separate `verl` distribution.

## Check the configuration

These checks import neither Torch nor Ray, perform no network access, and
write no files:

```bash
python rlvr/train.py check --config trace-qwen2.5-vl-3b
python rlvr/train.py check --config trace-qwen2.5-vl-7b
```

They validate the two training profiles, the prompt hash, the EasyR1 source
inventory, and the files required by the RLVR workflow.

## Prepare inputs

Preparation uses standard Hugging Face authentication and cache behavior. It
does not accept a token-file argument.

```bash
python rlvr/train.py prepare \
  --config trace-qwen2.5-vl-3b \
  --input-dir ../trace-inputs-3b

python rlvr/train.py prepare \
  --config trace-qwen2.5-vl-7b \
  --input-dir ../trace-inputs-7b
```

`prepare` downloads the pinned dataset and Qwen2.5-VL revisions, verifies all
17 Parquet files against the checked-in dataset-equivalence receipt, checks
the 64,000/2,000 row counts and 1,000-task cardinalities, validates the schema
and base-model weights, and writes `input_receipt.json`. Inputs are rehashed
before each run.

## Train

The training command has no semantic override flags. It requires an empty
output directory and exactly eight visible GPUs.

```bash
python rlvr/train.py run \
  --config trace-qwen2.5-vl-3b \
  --input-dir ../trace-inputs-3b \
  --output-dir ../trace-output-3b \
  --cuda-visible-devices 0,1,2,3,4,5,6,7

python rlvr/train.py run \
  --config trace-qwen2.5-vl-7b \
  --input-dir ../trace-inputs-7b \
  --output-dir ../trace-output-7b \
  --cuda-visible-devices 0,1,2,3,4,5,6,7
```

Console logging is the default. W&B telemetry is opt-in:

```bash
python rlvr/train.py run \
  --config trace-qwen2.5-vl-3b \
  --input-dir ../trace-inputs-3b \
  --output-dir ../trace-output-3b \
  --wandb
```

The command writes the resolved EasyR1 configuration and `run_receipt.json`
before launch. Training runs for 500 steps, saves and validates every 100
steps, does not resume old checkpoints, and retains one checkpoint.

## Smoke test and checkpoint merge

`smoke` uses the same data, rollout, optimizer, and reward settings for a
single diagnostic step. It saves at step 1 and disables validation:

```bash
python rlvr/train.py smoke \
  --config trace-qwen2.5-vl-3b \
  --input-dir ../trace-inputs-3b \
  --output-dir ../trace-smoke-3b
```

`merge` accepts the EasyR1 checkpoint root from a completed smoke test or full
run. It validates the run receipt, selected step, checkpoint tracker, eight
model shards, and Hugging Face config, then writes a local merged checkpoint:

```bash
python rlvr/train.py merge \
  --config trace-qwen2.5-vl-3b \
  --checkpoint-dir ../trace-output-3b/checkpoint \
  --output-dir ../trace-qwen2.5-vl-3b-merged
```

## Training settings

- Dataset: [`maveryn/trace@dataset-v1`](https://huggingface.co/datasets/maveryn/trace/tree/dataset-v1)
- 3B base: `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`
- 7B base: `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Prompt SHA-256: `f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900`
- Reward: `0.95 * exact_json_answer + 0.05 * exact_json_format`
- GRPO: 500 steps, rollout/global batch 128, eight rollouts, KL disabled,
  tensor parallelism 2, no resume, checkpoint retention 1

The configuration files also pin the released model identities. Training and
checkpoint merging do not upload models.

## TRACE validation

[`evaluation/trace_validation/README.md`](evaluation/trace_validation/README.md)
documents preparation, generation, extraction, scoring, and verification on
2,000 unseen instances from the same 1,000 task programs. The checked-in
result source contains seed-42 scores for the 3B base and TRACE checkpoints
and the six 7B checkpoints used in the paper.

Validate the frozen suite, results, and immutable archive receipt offline with:

```bash
python -m rlvr.evaluation.trace_validation.verify release
```

## External evaluation

[`evaluation/trace_eval/README.md`](evaluation/trace_eval/README.md) documents
the 24-benchmark evaluation for:

- Qwen2.5-VL-3B base and TRACE;
- Qwen2.5-VL-7B base, TRACE, VERO, Game-RL, Sphinx, and PCGRPO;
- decoding seeds 42, 43, and 44.

[`evaluation/trace_eval/results.json`](evaluation/trace_eval/results.json)
contains all 576 model/seed/benchmark scores. Validate the evaluation inputs
and aggregates offline with:

```bash
python rlvr/evaluation/scripts/validate_release_inputs.py
```
