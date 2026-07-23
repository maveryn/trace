---
description: Preflight, smoke training, full external evaluation, status, and verification for the Trace paper results.
---

# Reproducing the paper results

The `rlvr` branch contains the Qwen2.5-VL 3B/7B training profiles, TRACE
validation workflow, and the `trace_eval_v1` external evaluation. Run every
command below from the root of a Git clone checked out to that branch:

```bash
git clone https://github.com/maveryn/trace.git
cd trace
git switch rlvr
```

Repository files use the named release branch. Reproducibility identity comes
from the immutable Hugging Face revisions, receipts, and content hashes
recorded by the workflow.

## Scope and resources

- The training reference environment is CPython 3.10.12, CUDA 12.8, and eight
  NVIDIA H100 80GB GPUs. Both the smoke and canonical launchers require exactly
  eight visible GPUs.
- One full evaluation covers 32,805 examples for each model and seed. The paper
  matrix contains eight models and seeds 42, 43, and 44.
- TRACE validation separately covers 2,000 unseen instances from the same task
  programs, once for each of eight models with decoding seed 42.
- The evaluation launcher defaults to eight GPU groups and a local
  `Qwen/Qwen3-32B` judge at immutable revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Hugging Face authentication uses the standard client configuration or
  `HF_TOKEN`. Commands do not accept token-file paths.
- Work, cache, model, and result directories are local outputs. Do not commit
  them.

The canonical result source records producer revision
`5cea97310204b197fdacecdd83ef938c1e3b67cd`. Later compatibility fixes are
listed separately in the
[`post-run patch ledger`](https://github.com/maveryn/trace/blob/rlvr/rlvr/evaluation/trace_eval/post_run_patches.v1.json);
they do not change that producer identity or the reported scores.

## 1. Offline preflight

The release checks below perform no network access and do not import Torch or
Ray:

```bash
python rlvr/train.py check --config trace-qwen2.5-vl-3b
python rlvr/train.py check --config trace-qwen2.5-vl-7b
python rlvr/evaluation/scripts/validate_release_inputs.py
```

They validate the two training profiles, prompt hash, curated source inventory,
canonical result identities, 24 provenance entries, release receipts, and
public file map.

## 2. Training smoke run

Install the training runtime:

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0
python -m pip install --no-build-isolation -r rlvr/requirements-cu128.txt
python -m pip install -e . --no-deps
```

Prepare and verify the immutable 3B inputs, then launch the marked one-step
smoke profile into a new output directory:

```bash
python rlvr/train.py prepare \
  --config trace-qwen2.5-vl-3b \
  --input-dir ../trace-inputs-3b

python rlvr/train.py smoke \
  --config trace-qwen2.5-vl-3b \
  --input-dir ../trace-inputs-3b \
  --output-dir ../trace-smoke-3b \
  --cuda-visible-devices 0,1,2,3,4,5,6,7
```

The smoke run preserves the data, rollout, optimizer, and reward contracts but
is noncanonical: it runs one step, saves step 1, and disables validation. Use
`trace-qwen2.5-vl-7b`, `../trace-inputs-7b`, and `../trace-smoke-7b` for the 7B
profile. The
[`rlvr` training guide](https://github.com/maveryn/trace/blob/rlvr/rlvr/README.md)
documents the 500-step runs and checkpoint merge.

## 3. Evaluation setup

Install the composed runtime and materialize the pinned VLMEvalKit environment:

```bash
python -m pip install --no-build-isolation \
  -r rlvr/evaluation/requirements-runtime.txt
bash rlvr/evaluation/scripts/setup_trace_eval_env.sh
```

Prepare the exact 24-entry dataset view and download the pinned 3B base model
and evaluation judge:

```bash
python rlvr/evaluation/scripts/prepare_trace_eval_manifest.py

python rlvr/evaluation/scripts/prepare_trace_eval_models.py download-public \
  --model-root rlvr/evaluation/.work/models \
  --only qwen25vl3b-base \
  --only qwen3-32b-judge
```

The model preparation interface also downloads the registered 7B public
baselines and registers local merged checkpoints. Inspect its exact inputs
before preparing additional models:

```bash
python rlvr/evaluation/scripts/prepare_trace_eval_models.py --help
```

## 4. Preview the resolved evaluation

`--print-config` validates only the launcher arguments and prints the resolved
suite, run tag, seeds, model identity, and optional deltas. It does not start a
model server or write a campaign:

```bash
bash rlvr/evaluation/scripts/run_trace_eval.sh \
  --model qwen25vl3b-base \
    rlvr/evaluation/.work/models/qwen25vl3b-base \
    66285546d2b821cf421d4f5eb2576359d3770cd3 \
    Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3 \
    "Qwen2.5-VL-3B Base" \
  --seeds 42 43 44 \
  --run-tag trace_eval_v1-3b-base-reproduction \
  --print-config
```

Add one `--model SLUG LOCAL_PATH IMMUTABLE_REVISION SOURCE DISPLAY_LABEL`
group for each additional checkpoint. Every source must have the form
`owner/repository@immutable-revision`. Use `--delta` only for a declared matched
comparison.

## 5. Run the full 24-benchmark suite

Run the same reviewed command without `--print-config`:

```bash
bash rlvr/evaluation/scripts/run_trace_eval.sh \
  --model qwen25vl3b-base \
    rlvr/evaluation/.work/models/qwen25vl3b-base \
    66285546d2b821cf421d4f5eb2576359d3770cd3 \
    Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3 \
    "Qwen2.5-VL-3B Base" \
  --seeds 42 43 44 \
  --run-tag trace_eval_v1-3b-base-reproduction
```

This command evaluates one model across all 24 benchmarks and three seeds. Add
the other seven model descriptors to reproduce the full paper matrix; their
exact repository revisions are recorded in
[`results.json`](https://github.com/maveryn/trace/blob/rlvr/rlvr/evaluation/trace_eval/results.json).

The launcher verifies the environment, datasets, models, evaluator, and code
fingerprints before generation. It then runs resumable generation, extraction,
official or scoped scoring, receipt creation, local sanitized archiving, and
multi-seed summary generation. The default campaign root is
`rlvr/evaluation/.work/<run-tag>/`; Markdown and spreadsheet summaries are
written under `rlvr/evaluation/.work/results/`.

## 6. Monitor progress

In another shell with the same environment, inspect generation, score, archive,
and optional GPU progress:

```bash
python rlvr/evaluation/scripts/status_trace_eval.py \
  --campaign-root rlvr/evaluation/.work/trace_eval_v1-3b-base-reproduction \
  --dataset-manifest \
    rlvr/evaluation/.work/LMUData/trace_eval_v1_dataset_manifest.json \
  --model-slug qwen25vl3b-base \
  --seeds 42 43 44 \
  --watch 30
```

Add one `--model-slug` for every model in a multi-model campaign. Pass
`--fail-if-incomplete` for a one-shot completion gate or `--json` for
machine-readable status.

## 7. Verify the completed run

The launcher performs exact generation checks before scoring. After the run,
verify all score slices and their receipts independently:

```bash
python rlvr/evaluation/scripts/verify_trace_eval.py \
  --campaign-root rlvr/evaluation/.work/trace_eval_v1-3b-base-reproduction \
  --phase score \
  --model-slug qwen25vl3b-base \
  --seeds 42 43 44

python rlvr/evaluation/scripts/validate_release_inputs.py
```

The first command requires complete score coverage for every requested
model/seed/benchmark slice. The second revalidates the checked-in canonical
paper boundary. Neither command publishes artifacts. The guarded publication
worker remains offline unless upload and destination confirmations are supplied
explicitly.

## 8. Reproduce TRACE validation

TRACE validation uses the pinned 2,000-row validation split and a separate
generation, extraction, scoring, and verification contract from
`trace_eval_v1`. Follow the
[`trace_validation` workflow](https://github.com/maveryn/trace/tree/rlvr/rlvr/evaluation/trace_validation)
to prepare the dataset, evaluate one or more registered checkpoints, score the
responses, and verify the completed campaign. Its checked-in result source
binds all eight paper checkpoints to the immutable archive at
[`maveryn/trace-eval-runs@cf0d14aed86db2661d397ce8b68b36171873478d`](https://huggingface.co/datasets/maveryn/trace-eval-runs/tree/cf0d14aed86db2661d397ce8b68b36171873478d/runs/trace-iid-validation-2000-answer-seed42-8models-v1).

These samples are new instances from task programs represented in training;
they are not a held-out-task benchmark. Use the external evaluation above for
transfer beyond the TRACE task distributions.
