# TRACE Evaluation v1

`trace_eval_v1` is the 24-benchmark external evaluation used for the paper. It
contains 32,805 rows per model and decoding seed, grouped into six reporting
categories. [`suite.v1.json`](suite.v1.json) defines benchmark order, aliases,
row counts, routes, generation settings, and aggregation.

| Category | Benchmarks |
| --- | --- |
| Charts & Tables | ChartQAPro, CharXivReason, TableVQABench, EvoChart |
| Visual Math | MathVision, MathVista, MathVerse, WeMath |
| Science & General | PhyX mini MC, MMMU-ProVis, RealWorldQA, MMStar |
| Spatial Reasoning | EmbSpatial, SpatialVizBench COT, CV-Bench 3D, ERQA |
| Perception & Counting | BLINK, CountBenchQA, CountQA, TreeBench |
| Puzzles & Logic | PuzzleVQA, VisualPuzzles, LogicVista, MME-Reasoning |

Seven benchmarks use scoped direct scorers, sixteen use their pinned
VLMEvalKit `dataset.evaluate` routes, and MME-Reasoning uses its dedicated
scorer. The overall score is the unweighted mean of the 24 benchmark scores
for one model and seed; each category is the mean of its four scores.

## Benchmark and result sources

The evaluator uses
[`open-compass/VLMEvalKit@a8b12bf1…`](https://github.com/open-compass/VLMEvalKit/tree/a8b12bf1c3737a33fc1de967c202f9c592b22e86),
and judge-backed routes use
`Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`.
[`benchmark_provenance.v1.json`](benchmark_provenance.v1.json) records the
source revision or release, split, row count, license or terms, citation,
prompt route, scorer, and adapter for every benchmark. Benchmark payloads,
answers, media, caches, and raw local runs are not stored in Git.

The canonical runs were produced from revision
`5cea97310204b197fdacecdd83ef938c1e3b67cd`. Three result-relevant
normalizations in that revision handle TreeBench's option-column delimiter,
MathVerse's official binary judge decision, and a single terminal period on a
LogicVista option label. Each normalization is restricted to its named
benchmark and fails on ambiguous output.

Subsequent reproducibility fixes are listed separately in
[`post_run_patches.v1.json`](post_run_patches.v1.json); they do not change the
producer revision or canonical scores.

The 576-score source and Hugging Face run receipts are in
[`results.json`](results.json),
[`release_receipts.v1.json`](release_receipts.v1.json), and
[`RESULTS.md`](RESULTS.md). Run artifacts are available from the pinned paths
in
[`maveryn/trace-eval-runs`](https://huggingface.co/datasets/maveryn/trace-eval-runs).

Validate the inputs and aggregates without network access:

```bash
python rlvr/evaluation/scripts/validate_release_inputs.py
```

## Environment and preparation

Install the training runtime and CPU-side benchmark packages, then check out
the required VLMEvalKit revision and install the evaluation adapters:

```bash
python -m pip install --no-build-isolation \
  -r rlvr/evaluation/requirements-runtime.txt
bash rlvr/evaluation/scripts/setup_trace_eval_env.sh
```

Prepare the suite's 24 datasets and a content-bound manifest under
`rlvr/evaluation/.work/`:

```bash
python rlvr/evaluation/scripts/prepare_trace_eval_manifest.py
python rlvr/evaluation/scripts/prepare_trace_eval_models.py --help
```

Hugging Face authentication, when required, is read from `HF_TOKEN`. The model
helper can download a pinned model snapshot or register a local merged
checkpoint. Each model descriptor requires a local content revision and an
immutable `owner/repository@commit` source.

## Generation, extraction, and scoring

The launcher accepts one or more model descriptors and decoding seeds. First
render the resolved configuration, then remove `--print-config` to run it:

```bash
bash rlvr/evaluation/scripts/run_trace_eval.sh \
  --model qwen2.5-vl-3b-base <local-model-path> \
    66285546d2b821cf421d4f5eb2576359d3770cd3 \
    Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3 \
    "Qwen2.5-VL-3B Base" \
  --seeds 42 43 44 \
  --print-config
```

The launcher starts resumable generation queues, preserves row identity and
raw responses, performs extraction and scoring, records code, dataset,
evaluator, model, and judge fingerprints, and writes a multi-seed summary.
Work, cache, model, GPU, and output locations can be set through environment
variables; defaults are repository-relative.

The components can also be run independently:

- `status_trace_eval.py` reports generation, scoring, archive, and optional
  GPU progress;
- `verify_trace_eval.py` checks generation or score coverage and receipts;
- `summarize_trace_eval.py` rebuilds Markdown and spreadsheet summaries;
- `trace_eval_public_export.py` creates and verifies the response,
  extraction, and score artifact tree.

Run each command with `--help` for its path and model arguments.

## Publishing run artifacts

`run_trace_eval_publish_worker.py` waits for complete verified slices, builds
the export, and remains offline unless both the upload option and exact
repository/run confirmation are supplied. Authentication comes from the
selected environment variable, which defaults to `HF_TOKEN`. Raw benchmark
prompts, answers, source rows, media paths, and local run identifiers are
rejected by the export checks.
