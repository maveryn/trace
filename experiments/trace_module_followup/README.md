# TRACE–VERO follow-up: component transfer, co-adaptation and evaluation reliability

Training-free follow-up to the TRACE–VERO merging study (`experiments/trace_merge`, PR maveryn/trace#14). It builds
exact-copy component-swap hybrids of Qwen2.5-VL-7B (B), TRACE (T) and VERO (V) and evaluates them on the reviewed
study's frozen rows. It asks which post-training updates help when transplanted into the original model, and which
need the visual and language components adapted together. Nothing is trained or uploaded.

Start with [`results/reports/followup_findings.md`](results/reports/followup_findings.md). The decisions are in
[`design/followup_v1.md`](design/followup_v1.md) (amendments A1–A5), [`design/decision_F2.md`](design/decision_F2.md)
and [`design/decision_F3.md`](design/decision_F3.md).

## Layout

```text
experiments/trace_module_followup/
  configs/followup_v1.yaml   cells, component parts and partitions, hybrids, evaluation tiers, statistics
  design/                    frozen design + amendments, F2 and F3 decision records
  trace_followup/
    components.py build.py inventory.py   stored-tensor ownership, exact-copy hybrid builder, parent inventory/tests
    normmatch.py                           F3 norm-matched VERO (reviewed interpolator, float64 norms)
    views.py evaluate.py                   inference views (P1 = B's interface files, P2 = V's), task planning
    extract.py rescore.py                  F0: frozen format-normalized-v2 scorer (do not edit; spec hash in audit/)
    analyze.py analyze_f2c.py analyze_f3.py analyze_interface.py   ingest + paired cluster bootstrap analyses
    export.py cli.py                       evaluation manifest, sanitized bundle, command line
  tests/                     synthetic tests; protocol-fix tests run against the private overlay clone
  results/                   sanitized aggregate bundle (written by `export`)
```

The package imports the reviewed study's audited helpers from `experiments/trace_merge` without modifying them. GPU
evaluation runs the reviewed runner (`experiments/trace_merge/jobs`, snapshotted) with its work root set to this
study's root.

## Commands

Machine paths never appear in tracked files: the work root comes from `TRACE_FOLLOWUP_ROOT`, and the protocol clone
and reviewed work root from the reviewed study's variables.

```bash
cd experiments/trace_module_followup
python -m trace_followup inventory --reference-hashes <reviewed root>/audit/parent_weight_sha256.json
python -m trace_followup build --models TF TL VF VL        # also TR0 TR1 TR2 (F2c)
python -m trace_followup verify-builds                     # identity, ownership, donor-equal re-hash
python -m trace_followup build-views                       # 14 F1 views (+ extension cells by --labels)
python -m trace_followup plan-eval --tier analysis --env-file <env.sh> --jobs-dir <jobs snapshot>   # task file only
python -m trace_followup ingest-runs                       # reviewed ingest + F0 scoring of every complete run
python -m trace_followup analyze                           # F1 tables (primary family, P2 template, behaviour, costs)
python -m trace_followup analyze-f2c                       # decoder block-group reversion
python -m trace_followup analyze-f3                        # norm-matched VERO
python -m trace_followup analyze-interface                 # three-seed P1/P2 development-screen table (A5)
python -m trace_followup export                            # manifests/evaluation.json + results/ (sanitized)
python -m pytest -q tests
```

Planning commands write files and never start GPU work. Runs were placed on already-held allocations with a per-card
filler (machine-specific, kept in the work root).

## What is local only

These stay in the work root and are never committed: hybrid and V-NM weights, inference views, run directories,
aligned and scored rows, F0 row ledgers and blinded-check samples, quarantined outputs, and the private protocol
overlay. Committed results hold aggregates, design records and manifests only (no prompts, gold answers, images,
responses, weights, machine paths or tokens).
