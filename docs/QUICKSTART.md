# Quickstart

## Install

From a clone of the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[test]"
```

## Inspect The Registry

```bash
trace-list --domain puzzles
```

## Build A Small Dataset

```bash
trace-generate --config examples/configs/minimal_build.yaml
```

The builder stages candidate instances, applies the configured acceptance and
validation contracts, then atomically finalizes the dataset directory. The
command reports the finalized dataset path as `dataset_root`.

## Export JSONL For RLVR

```bash
trace-export out/datasets/<dataset-id> \
  --output trace-train.jsonl \
  --format jsonl \
  --prompt-variant answer
```

Parquet export requires the optional `export` dependencies.

## Run Public Smoke Tests

```bash
pytest -q tests/test_public_smoke.py
```
