# TRACE validation

This workflow evaluates eight vision-language models on 2,000 unseen TRACE
validation instances drawn from the same 1,000 task programs as training. Each
task program contributes two non-overlapping validation instances. The frozen
roster, model revisions, prompt, decoding settings, and scoring contract are in
[`suite.v1.json`](suite.v1.json).

The final score is exact-match accuracy over all 2,000 rows. Deterministic
extraction receives only the raw response and declared answer type. Responses
that remain missing or ambiguous go to a ground-truth-blind Qwen3-32B
extraction pass; unresolved rows score zero and remain in the denominator.

## Canonical results

| Model | Accuracy (%) |
| --- | ---: |
| Qwen2.5-VL-3B Base | 24.45 |
| TRACE Qwen2.5-VL-3B | 41.05 |
| Qwen2.5-VL-7B Base | 34.25 |
| TRACE Qwen2.5-VL-7B | 51.55 |
| Game-RL Qwen2.5-VL-7B | 35.55 |
| Sphinx Qwen2.5-VL-7B | 33.50 |
| PCGRPO Qwen2.5-VL-7B | 34.10 |
| VERO Qwen2.5-VL-7B | 38.40 |

The machine-readable table is [`results.v1.json`](results.v1.json). Responses,
extractions, and row-level scores are available in the
[`maveryn/trace-eval-runs`](https://huggingface.co/datasets/maveryn/trace-eval-runs)
dataset at revision `cf0d14aed86db2661d397ce8b68b36171873478d`.
[`release_receipt.v1.json`](release_receipt.v1.json) binds this table to that
immutable dataset revision.
Public hardening applied after the archived run is listed in
[`reproducibility_patches.v1.json`](reproducibility_patches.v1.json).

The frozen suite retains the historical validation identity used by the paper
run. New reproductions use the immutable post-squash dataset revision below.
The checked-in [`dataset equivalence receipt`](../../dataset_equivalence.v1.json)
verifies that every pre-existing column, row, and embedded image is unchanged
and that `trace_supervision_mode` is the only added advisory column. Scoring and
full verification require this bridge; the canonical suite, results, and
release receipt remain unchanged.

Validate the checked-in suite, table, and receipt without network access:

```bash
python -m rlvr.evaluation.trace_validation.verify release
```

## Environment and dataset preparation

Run commands from the repository root. Install the evaluation environment and
download the pinned validation parquet:

```bash
python -m pip install --no-build-isolation \
  -r rlvr/evaluation/requirements-runtime.txt
python -m pip install -e . --no-deps

WORK=rlvr/evaluation/.work/trace_validation
DATASET_FILE=data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet
hf download maveryn/trace "$DATASET_FILE" \
  --repo-type dataset \
  --revision dataset-v1 \
  --local-dir "$WORK/inputs"

python -m rlvr.evaluation.trace_validation.prepare_dataset \
  --parquet "$WORK/inputs/$DATASET_FILE" \
  --output-root "$WORK/dataset"
```

Preparation verifies the post-squash parquet and deterministic prepared-manifest
hashes, then writes embedded image bytes unchanged to content-addressed files
beneath `$WORK/dataset`.

## Model preparation

Download the exact eight suite revisions and write content-bound provenance
markers beside each local checkpoint:

```bash
WORK=rlvr/evaluation/.work/trace_validation
MODEL_ROOT="$WORK/models"
python rlvr/evaluation/scripts/prepare_trace_eval_models.py \
  download-validation --model-root "$MODEL_ROOT"
```

The downloader verifies the complete file inventory and every file hash. It
also materializes and verifies the suite's deterministic Game-RL processor
compatibility view. To reuse an existing local runtime view, use
`register-validation-local`; registration succeeds only when its content-set
revision exactly matches the suite.

## Generation

Start one OpenAI-compatible vLLM endpoint for a model in the suite. Set
`MODEL_SLUG`, `MODEL_PATH`, and `MODEL_REVISION` from its suite entry; use
`runtime_view_revision` when that field is present.

```bash
WORK=rlvr/evaluation/.work/trace_validation
MODEL_ROOT="$WORK/models"
MODEL_SLUG=trace-qwen2.5-vl-3b
MODEL_PATH="$MODEL_ROOT/$MODEL_SLUG"
MODEL_REVISION=sha256set:fd7d9ef4dd828eb950ce29c8ccde0432ccd31420529d4f023300ede928d070a1

python rlvr/evaluation/scripts/prepare_trace_eval_models.py \
  verify-validation --entry "$MODEL_SLUG=$MODEL_PATH"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_SLUG" \
  --host 127.0.0.1 --port 18100 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":262144,"max_pixels":4194304}' \
  --generation-config vllm
```

In another shell, generate the model's resumable response file:

```bash
WORK=rlvr/evaluation/.work/trace_validation
MODEL_ROOT="$WORK/models"
MODEL_SLUG=trace-qwen2.5-vl-3b
MODEL_PATH="$MODEL_ROOT/$MODEL_SLUG"
MODEL_REVISION=sha256set:fd7d9ef4dd828eb950ce29c8ccde0432ccd31420529d4f023300ede928d070a1

python -m rlvr.evaluation.trace_validation.generate \
  --manifest "$WORK/dataset/manifest.json" \
  --output-dir "$WORK/campaign/generation/$MODEL_SLUG" \
  --endpoint-url http://127.0.0.1:18100/v1 \
  --served-model "$MODEL_SLUG" \
  --model-slug "$MODEL_SLUG" \
  --model-path "$MODEL_PATH" \
  --model-revision "$MODEL_REVISION" \
  --media-transport data-url \
  --concurrency 32
```

Repeat generation for all eight suite entries. Multiple endpoints may run in
parallel; each output directory is identity-bound and resumable. The canonical
verifier expects one unsharded output directory per model. Verify the complete
generation set before scoring:

```bash
python -m rlvr.evaluation.trace_validation.verify generation-only
```

## Extraction and scoring

Build the exact eight generation arguments and run the deterministic pass:

```bash
WORK=rlvr/evaluation/.work/trace_validation
GENERATION_ARGS=()
for MODEL_SLUG in \
  qwen2.5-vl-3b-base trace-qwen2.5-vl-3b \
  qwen2.5-vl-7b-base trace-qwen2.5-vl-7b \
  game-rl-qwen2.5-vl-7b sphinx-qwen2.5-vl-7b \
  pcgrpo-qwen2.5-vl-7b vero-qwen2.5-vl-7b
do
  GENERATION_ARGS+=(
    --generation-jsonl "$WORK/campaign/generation/$MODEL_SLUG/responses.jsonl"
  )
done

python -m rlvr.evaluation.trace_validation.score \
  --dataset-manifest "$WORK/dataset/manifest.json" \
  --suite rlvr/evaluation/trace_validation/suite.v1.json \
  "${GENERATION_ARGS[@]}" \
  --output-dir "$WORK/campaign/scoring/initial"
```

Download and verify the pinned judge checkpoint:

```bash
WORK=rlvr/evaluation/.work/trace_validation
MODEL_ROOT="$WORK/models"
JUDGE_PATH="$MODEL_ROOT/qwen3-32b-judge"
JUDGE_REVISION=9216db5781bf21249d130ec9da846c4624c16137

python rlvr/evaluation/scripts/prepare_trace_eval_models.py \
  download-public --model-root "$MODEL_ROOT" --only qwen3-32b-judge
python rlvr/evaluation/scripts/prepare_trace_eval_models.py verify --deep \
  --entry "qwen3-32b-judge=$JUDGE_PATH=$JUDGE_REVISION"
```

Serve that checkpoint with the suite's judge name:

```bash
WORK=rlvr/evaluation/.work/trace_validation
JUDGE_PATH="$WORK/models/qwen3-32b-judge"

python -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_PATH" \
  --served-model-name qwen3-32b-trace-validation-extractor \
  --host 127.0.0.1 --port 18200 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 8192 \
  --max-num-seqs 128 --max-num-batched-tokens 32768 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --generation-config vllm
```

In another shell, extract only the queued rows:

```bash
WORK=rlvr/evaluation/.work/trace_validation
python -m rlvr.evaluation.trace_validation.judge_extract \
  --pending-jsonl "$WORK/campaign/scoring/initial/judge_pending.jsonl" \
  --output-dir "$WORK/campaign/judge" \
  --api-base http://127.0.0.1:18200/v1 \
  --api-model qwen3-32b-trace-validation-extractor \
  --tokenizer-model Qwen/Qwen3-32B \
  --judge-revision 9216db5781bf21249d130ec9da846c4624c16137
```

Finalize scores and run the complete verifier:

```bash
python -m rlvr.evaluation.trace_validation.score \
  --dataset-manifest "$WORK/dataset/manifest.json" \
  --suite rlvr/evaluation/trace_validation/suite.v1.json \
  "${GENERATION_ARGS[@]}" \
  --judge-results "$WORK/campaign/judge/judge_results.jsonl" \
  --output-dir "$WORK/campaign/scoring/final"

python -m rlvr.evaluation.trace_validation.verify full
```

Use `--help` on each module to change repository-relative work locations,
endpoint concurrency, or local model storage while retaining the frozen
contracts.
