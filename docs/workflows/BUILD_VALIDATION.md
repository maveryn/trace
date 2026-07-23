# Build Validation

Trace builds are finalized only when every requested task reaches its accepted
sample count and all dataset contracts pass. Generation is staged first, then
validated and atomically finalized.

## Local Validation

Run a small deterministic build from the repository root:

```bash
trace-generate --config examples/configs/minimal_build.yaml
```

Validate the finalized dataset reported by the command:

```bash
trace-validate <dataset-root>
```

Changes to generation or validation infrastructure should check deterministic
replay, trace references, image hashes, typed answers and annotations, prompt
provenance, reward contracts, and strict-reproduction behavior.

## Completeness Authority

For a canonical dataset directory, `trace-validate` treats
`build_report.json` as the source of truth for accepted task counts and
trace-shard record counts. It checks those expectations against the training
rows, referenced trace indices, and physical trace shards. A missing or
malformed build report is an error.

An explicit `train_instances.jsonl` can be validated without a neighboring
build report when inspecting a loose export. In that mode, counts are derived
from observed rows and the validator warns that it cannot prove whether rows
were removed.

## Reconstruct The Paper Dataset Recipe

The TRACE paper checkpoints can be reproduced with
[`maveryn/trace@dataset-v1`](https://huggingface.co/datasets/maveryn/trace/tree/dataset-v1).
The training configurations resolve this release tag to an immutable revision
and verify every downloaded Parquet file against the checked-in equivalence
receipt.

`scripts/build_release_dataset.py` reconstructs the same published dataset
recipe from the current Trace source:

- 1,000 tasks;
- 64 training examples per task with generation seed `42`;
- two IID validation examples per task with generation seed `1042`;
- embedded images capped at `1,280,000` pixels;
- deterministic row shuffle seed `20260711`.

Code provenance participates in Trace instance identity. A reconstruction
therefore follows the frozen semantic recipe but is not expected to be
byte-identical to the pinned dataset revision or to reproduce its instance
identifiers.

Install the pinned export environment and inspect the resolved plan before the
full 66,000-image build:

```bash
python -m pip install -c constraints/release.txt -e ".[export]"
python scripts/build_release_dataset.py --dry-run
python scripts/build_release_dataset.py
```

The build writes 16 training Parquet shards, one validation Parquet shard,
canonical sidecars, split manifests, and a dataset manifest with SHA-256
receipts. Each split's `export_provenance.jsonl.zst` binds its embedded PNG
bytes, decoded pixels, and exported geometry to the original image hash in the
identity-bearing training record. Original image bytes are not duplicated in
the export, so artifact-only verification checks that binding but cannot replay
an image resize independently.

Verify a completed output tree in place:

```bash
python scripts/build_release_dataset.py --verify \
  --output-dir release-dataset
```

## Reproducibility Environment

The builder requires clean, committed generation inputs and the
determinism-critical versions in `constraints/release.txt` on CPython
3.10–3.12. It records the source revision, source-tree hash, dependency
versions, split recipe, and output receipts. The Python 3.14 compatibility
constraints are supported for package development but are not canonical
dataset-reconstruction provenance.

The builder writes only to local output directories and performs no upload.
Use `dataset-v1` in human-facing instructions. Machine configurations and run
receipts use the immutable commit resolved for that release.
