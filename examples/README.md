# Python API examples

These examples use the installed `trace_tasks` package directly. From a clone,
install the project first:

```bash
python -m pip install -e .
```

Generate a small dataset. Trace validates every generated record before
finalizing the dataset:

```bash
python examples/generate_and_validate.py --output outputs/python-api-example
```

The example records the installed `trace-tasks` version as code provenance by
default. Pass `--code-hash <revision>` when a source revision is available.

Use the printed `dataset_root` for verifier replay. Omitting `--response`
constructs the ground-truth answer response as a scorer smoke test; pass actual
model output with `--response` in real use:

```bash
python examples/replay_and_score.py outputs/python-api-example/datasets/<dataset-id>
```

Export the generated records to RLVR JSONL:

```bash
python examples/export_dataset.py outputs/python-api-example/datasets/<dataset-id> \
  --output outputs/python-api-example/trace-rlvr.jsonl
```

The equivalent installed commands are `trace-generate`, `trace-validate`, and
`trace-export`. When a generated dataset includes `build_report.json`,
`trace-validate` uses its task counts and trace-shard manifest to check
completeness. A standalone `train_instances.jsonl` can still be checked, but
the command warns when completeness cannot be established.
