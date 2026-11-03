# TRACE–VERO: analysis-first, training-free merging study

This experiment compares the released TRACE (`maveryn/trace-qwen2.5-vl-7b`) and VERO (`zlab-princeton/Vero-Qwen25-7B`)
checkpoints against their shared backbone (`Qwen/Qwen2.5-VL-7B-Instruct`) on the archived `trace_eval_v1` runs. It then
tests three training-free 50:50 weight interpolations:

| id | weights | role |
|---|---|---|
| `M050` | 0.5 VERO + 0.5 TRACE | complementarity test |
| `CV050` | 0.5 VERO + 0.5 base | VERO update-shrinkage control |
| `CT050` | 0.5 TRACE + 0.5 base | TRACE update-shrinkage control |

The run trains nothing and uploads nothing. Every decision fixed before merged-model results existed is in
[`DESIGN.md`](DESIGN.md), with later changes appended as dated amendments. Analysis tables are written to the local
work root (`tables/`); [`results/`](results/) holds the sanitized aggregate bundle: reports, tables, figures and
manifests. Start with [`results/reports/analysis_report.md`](results/reports/analysis_report.md) and
[`results/reports/decision_memo.md`](results/reports/decision_memo.md).

The study lives outside `rlvr/` on purpose. The release validator and `rlvr/train.py` reject any file under `rlvr/` that
is not in the frozen source map. `python rlvr/evaluation/scripts/validate_release_inputs.py` passes before and after
this experiment.

## Layout

```text
experiments/trace_merge/
  DESIGN.md                        frozen decisions and amendments
  configs/trace_merge_qwen25_v1.yaml
  trace_merge/                     python package
    config.py                      validated config, policy switches (training/uploads/expansions must stay off)
    inventory.py ingest.py         archive integrity, content-key joins, local-run ingestion
    diagnostics.py                 protocol comparison, environment drift, length/format/judge audit
    metrics.py                     metric registry and official-metric adapters (full split and subsets)
    grouping.py                    connected image/problem groups, frozen development/confirmation split
    behavior.py statistics.py      transitions, 8-cell tables, overlap, capture/damage, cluster bootstrap, Holm
    compatibility.py merge.py      parent audit and streaming float32-accumulated interpolation
    safetensors_io.py views.py     bounded-memory safetensors I/O; inference views (weights + one parent's interface)
    evaluation.py                  task planning, run status, pre-confirmation lock
    analysis.py                    Stage B, development screen, lock check, confirmation tables
    reports.py cli.py              figures, sanitized export, markdown table helpers, command line
  jobs/                            GPU task runner (wraps the protocol's own generation and scoring components)
  scripts/trace_merge_study.py     entry point
  tests/                           unit tests (synthetic fixtures; real-data checks skip without a work root)
  results/                         sanitized aggregates and figures (committed once exported)
```

## Running

Machine paths never appear in tracked files. Commands read them from the environment:

```bash
export TRACE_MERGE_WORK_ROOT=<local work root>          # archives, splits, merges, views, runs, tables
export TRACE_MERGE_PROTOCOL_CLONE=<prepared rlvr clone>  # af32fe0 + pinned VLMEvalKit + .work/{LMUData,models}
S=experiments/trace_merge/scripts/trace_merge_study.py
python $S ingest [--runs-root <dir>]  # archive integrity, identity joins, aligned table (+ the existing B/T reruns)
python $S metrics           # metric registry and reproduction of every archived aggregate
python $S freeze-splits     # group-aware 20/80 split, development screen, smoke list
python $S analyze           # Stage B tables: official scores, corrections/regressions, 8-cell tables, overlap
python -m trace_merge.diagnostics all   # (from experiments/trace_merge) protocols, environment drift, length/format/judge audit,
                                        # judge health (judge-agreement: extracted vs boxed MathVerse answers, DESIGN.md A8)
python $S audit-compat      # parent compatibility (hard stops on any mismatch)
python $S parent-hashes     # full sha256 of every parent weight file
python $S update-stats      # parent update norms per tensor group (diagnostic)
python $S plan-merges       # print the merge plan (always a dry run)
python $S build-merges      # M050, CV050, CT050 (CPU, bounded memory, atomic finalize)
python $S build-views       # inference views (P1: backbone interface; P2: VERO interface, diagnostic)
python $S plan-eval --tier development --env-file <env.sh> --jobs-dir <jobs snapshot>
python $S lock-selection    # freezes candidates, comparisons and input hashes before confirmation
python $S plan-eval --tier confirmation --env-file <env.sh> --jobs-dir <jobs snapshot>   # checks the lock first
python $S eval-status [--tier T]        # which model x tier x seed runs are complete
python $S ingest-runs --tier T [--force]  # ingest completed runs; changed runs are ingested again
python $S screen            # development screen tables (seed 42; P1 and the P2 diagnostic separately)
python $S confirm           # confirmation tables, only while the selection lock still holds
python $S capture-reference # post hoc: the base model scored like a merge on the confirmation capture groups
python $S costs             # tables/evaluation_costs.csv: GPU-hours, generated tokens, attempts, storage
python $S figures           # Stage B, screen and (once confirmed) confirmation figures
python $S export            # copy the sanitized aggregate bundle into results/ (refuses machine paths and tokens)
```

`--dry-run` works only for `plan-merges`, `build-merges`, `freeze-splits` and `plan-eval`; other commands refuse it.
The markdown reports (`audit/audit.md`, `reports/behavior_report.md`, `reports/merge_screen.md`,
`reports/analysis_report.md`, `reports/decision_memo.md`) are written from these tables, not generated; `export` copies
them into `results/` with the tables and figures.

Each task line in a generated task file runs `jobs/run_tiers.sh` → `jobs/eval_subset.sh` on one node of an already-held
GPU allocation. That starts the protocol's vLLM endpoint pool, its generation queue restricted to the frozen rows, its
finalization, and its score campaign with the pinned Qwen3-32B judge. Tests:

```bash
cd experiments/trace_merge && python -m pytest -q tests
```

## What is local only

These stay in the work root and are never committed: archive snapshots, prepared benchmark data, aligned predictions
(which contain model responses), merged weights, run directories, and the casebook. Committed results contain
aggregates and identifiers permitted by the public export policy.
