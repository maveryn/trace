"""Command line for the TRACE-VERO study (handoff section 10).

Commands print their inputs/outputs and require the upstream manifests they build on. --dry-run is accepted only by
the commands that implement it (plan-merges is always one; build-merges, freeze-splits, plan-eval); any other command
refuses the flag rather than ignoring it, and --force is accepted only by ingest-runs. Nothing here trains, uploads, or
launches GPU work by itself: GPU evaluation is written as task lines for the cluster's queue runner (``plan-eval``)
and executed by ``jobs/eval_subset.sh`` on held allocations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .config import ensure_work_tree, load_config, sha256_file, write_json

COMMANDS: dict[str, Callable[[Any, argparse.Namespace], int]] = {}
DRY_RUN_COMMANDS = ("build-merges", "freeze-splits", "plan-eval", "plan-merges")
FORCE_COMMANDS = ("ingest-runs",)


def command(name: str):
    def deco(fn):
        COMMANDS[name] = fn
        return fn
    return deco


def _log(msg: str) -> None:
    print(msg, flush=True)


@command("audit-compat")
def cmd_audit_compat(cfg, args) -> int:
    from .compatibility import audit

    parents = {r: cfg.hf_snapshot(r) for r in ("B", "T", "V")}
    lineage = {}
    for r in parents:
        spec = cfg.role(r)
        lineage[r] = {"repo_id": spec["repo_id"], "source_revision": spec["source_revision"]}
    prov = parents["T"] / "trace_training_provenance.json"
    if prov.is_file():
        p = json.loads(prov.read_text(encoding="utf-8"))
        lineage["T"]["declared_base"] = f"{p.get('base_model')}@{p.get('base_revision')}"
    readme = parents["V"] / "README.md"
    if readme.is_file() and "Qwen/Qwen2.5-VL-7B-Instruct" in readme.read_text(encoding="utf-8"):
        lineage["V"]["declared_base"] = "Qwen/Qwen2.5-VL-7B-Instruct (model card; base revision not declared)"
    report = audit(parents, lineage)
    write_json(cfg.work("audit", "compatibility.json"), report)
    _log(json.dumps({"status": report["status"], "hard_stops": report["hard_stops"], "findings": report["findings"]},
                    indent=1))
    return 0 if report["status"] == "compatible" else 4


@command("parent-hashes")
def cmd_parent_hashes(cfg, args) -> int:
    """Full sha256 of every parent weight file; for LFS files the HF blob name must equal the hash."""

    out: dict[str, dict[str, str]] = {}
    bad = []
    for r in ("B", "T", "V"):
        snap = cfg.hf_snapshot(r)
        out[r] = {}
        for f in sorted(snap.glob("*.safetensors")):
            h = sha256_file(f)
            out[r][f.name] = h
            # the snapshot entry links to blobs/<sha256> for LFS files; that blob may itself link onward into a
            # deduplicated store with other names, so compare against the first link target only
            blob = Path(os.readlink(f)).name if f.is_symlink() else ""
            if len(blob) == 64 and blob != h:
                bad.append(f"{r}/{f.name}: sha256 {h} != blob {blob}")
            _log(f"[parent-hashes] {r} {f.name} {h}")
    write_json(cfg.work("audit", "parent_weight_sha256.json"), out)
    if bad:
        _log("FATAL: " + "; ".join(bad))
        return 4
    return 0


@command("update-stats")
def cmd_update_stats(cfg, args) -> int:
    from .merge import parent_update_statistics

    parents = {r: cfg.hf_snapshot(r) for r in ("B", "T", "V")}
    stats = parent_update_statistics(parents, log=_log)
    write_json(cfg.work("audit", "parent_update_statistics.json"), stats)
    return 0


@command("plan-merges")
def cmd_plan_merges(cfg, args) -> int:
    from .merge import build_configured_merges

    build_configured_merges(cfg, only=args.only, dry_run=True, log=_log)
    return 0


@command("build-merges")
def cmd_build_merges(cfg, args) -> int:
    from .merge import build_configured_merges

    if not cfg.work("audit", "compatibility.json").is_file():
        _log("FATAL: run audit-compat first")
        return 4
    out = build_configured_merges(cfg, only=args.only, dry_run=args.dry_run, log=_log)
    if not args.dry_run:
        write_json(cfg.work("manifests", "merges.json"), {k: {"content_revision": v.get("content_revision"),
                                                               "recipe_sha256": v.get("recipe_sha256")}
                                                           for k, v in out.items()})
    return 0


@command("build-views")
def cmd_build_views(cfg, args) -> int:
    from .views import build_all_views

    out = build_all_views(cfg, args.only)
    write_json(cfg.work("manifests", "views.json"), out)
    _log(json.dumps({k: v["slug"] for k, v in out.items()}, indent=1))
    return 0


@command("freeze-splits")
def cmd_freeze_splits(cfg, args) -> int:
    from .grouping import freeze_splits

    summary = freeze_splits(cfg, dry_run=args.dry_run, log=_log)
    _log(json.dumps(summary, indent=1))
    return 0


@command("plan-eval")
def cmd_plan_eval(cfg, args) -> int:
    """Write the queue-runner task file of a tier (machine paths come from --env-file / --jobs-dir, never the YAML)."""

    from .evaluation import task_lines, tier_runs

    if not args.tier or not args.env_file or not args.jobs_dir:
        _log("FATAL: plan-eval needs --tier, --env-file and --jobs-dir")
        return 2
    if args.tier == "confirmation":
        from .config import PROJECT_ROOT
        from .evaluation import verify_lock

        try:
            report = verify_lock(cfg, PROJECT_ROOT / "DESIGN.md")
        except (RuntimeError, OSError) as error:
            _log(f"FATAL: the selection lock does not hold: {error}")
            return 4
        if report.get("post_lock_amendments"):
            _log(f"DESIGN.md amendments appended after the lock: {report['post_lock_amendments']}")
    lines = task_lines(cfg, args.tier, args.env_file.resolve(), args.jobs_dir.resolve(),
                       with_smoke=args.tier == "development")
    runs = tier_runs(cfg, args.tier)
    out = cfg.work("dispatch", f"wave_{args.tier}.tasks")
    if args.dry_run:
        _log("\n".join(lines))
    else:
        out.write_text(f"# {cfg.study_id} {args.tier}: {len(runs)} runs in {len(lines)} task lines\n" + "\n".join(lines) + "\n",
                       encoding="utf-8")
        _log(f"wrote {out}")
    _log(json.dumps({"tier": args.tier, "runs": len(runs), "task_lines": len(lines)}))
    return 0


@command("eval-status")
def cmd_eval_status(cfg, args) -> int:
    from .evaluation import status

    rows = status(cfg, [args.tier] if args.tier else None)
    for r in rows:
        _log(f"{r['tier']:13s} {r['label']:10s} seed {r['seed']} {r['state']:13s} {r.get('rows')}/{r.get('expected')}")
    write_json(cfg.work("tables", "evaluation_status.json"), rows)
    return 0


@command("lock-selection")
def cmd_lock_selection(cfg, args) -> int:
    from .config import PROJECT_ROOT
    from .evaluation import lock_selection

    lock = lock_selection(cfg, PROJECT_ROOT / "DESIGN.md")
    _log(json.dumps({k: lock[k] for k in ("selected_candidates", "primary_family", "locked_at")}, indent=1))
    return 0


@command("ingest")
def cmd_ingest(cfg, args) -> int:
    """Archive integrity + identity joins (aligned table); --runs-root also ingests the existing B/T reruns."""

    from . import ingest

    argv = ["archive"]
    rc = ingest.main(argv + _passthrough(args))
    if rc == 0 and args.runs_root:
        rc = ingest.main(["existing-reruns", "--runs-root", str(args.runs_root)] + _passthrough(args))
    return rc


@command("metrics")
def cmd_metrics(cfg, args) -> int:
    """Reproduce every archived aggregate with the metric adapters and write the registry."""

    from . import metrics

    return metrics.main(["reproduce"] + _passthrough(args))


@command("analyze")
def cmd_analyze(cfg, args) -> int:
    from .analysis import stage_b

    stage_b(cfg, log=_log)
    return 0


@command("ingest-runs")
def cmd_ingest_runs(cfg, args) -> int:
    from .analysis import ingest_tier

    if not args.tier:
        _log("FATAL: ingest-runs needs --tier")
        return 2
    status = ingest_tier(cfg, args.tier, force=args.force, log=_log)
    _log(json.dumps(status, indent=1))
    return 0


@command("screen")
def cmd_screen(cfg, args) -> int:
    from .analysis import screen

    _log(json.dumps(screen(cfg, log=_log), indent=1))
    return 0


@command("confirm")
def cmd_confirm(cfg, args) -> int:
    from .analysis import confirm

    _log(json.dumps(confirm(cfg, log=_log), indent=1))
    return 0


@command("capture-reference")
def cmd_capture_reference(cfg, args) -> int:
    from .analysis import capture_base_reference

    t = capture_base_reference(cfg, log=_log)
    o = t[(t["level"] == "overall") & (t["seed_scope"] != "single_condition")]
    _log(o[["benchmark_set", "variant", "merge_label", "group", "n_group", "conditional_rate_pct",
            "conditional_rate_pct_ci_low", "conditional_rate_pct_ci_high"]].round(2).to_string(index=False))
    return 0


@command("format-sensitivity")
def cmd_format_sensitivity(cfg, args) -> int:
    from .analysis import answer_format_table

    if args.tier not in ("development", "confirmation"):
        _log("FATAL: format-sensitivity needs --tier development or --tier confirmation")
        return 2
    t = answer_format_table(cfg, args.tier, log=_log)
    cols = ["benchmark_id", "model_label", "seed", "n_rows", "official_accuracy_pct", "latex_letter_ending_pct",
            "official_zero_but_final_letter_correct_pct"]
    _log(t[cols].round(2).to_string(index=False))
    return 0


@command("figures")
def cmd_figures(cfg, args) -> int:
    from . import reports

    made = reports.stage_b_figures(cfg)
    if cfg.work("tables", "screen", "summary.json").is_file():
        made += reports.screen_figures(cfg)
    if cfg.work("tables", "confirmation", "summary.json").is_file():
        made += reports.confirmation_figures(cfg)
    for p in made:
        _log(f"[figures] {p.name}")
    return 0


@command("costs")
def cmd_costs(cfg, args) -> int:
    from .reports import write_costs

    _log(f"[costs] {write_costs(cfg)}")
    return 0


@command("export")
def cmd_export(cfg, args) -> int:
    from .reports import bundle_files, export_results

    hashes = export_results(cfg, bundle_files(cfg))
    _log(f"[export] {len(hashes)} files written to experiments/trace_merge/results")
    return 0


def _passthrough(args) -> list[str]:
    out = []
    for flag, val in (("--config", args.config), ("--work-root", args.work_root), ("--protocol-clone", args.protocol_clone)):
        if val is not None:
            out += [flag, str(val)]
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trace_merge_study", description=__doc__)
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--protocol-clone", type=Path, default=None)
    p.add_argument("--only", action="append", default=None, help="restrict to these merge ids / view labels")
    p.add_argument("--dry-run", action="store_true", help=f"only for: {', '.join(DRY_RUN_COMMANDS)}")
    p.add_argument("--force", action="store_true", help="ingest-runs: re-ingest runs that are already ingested")
    p.add_argument("--tier", choices=["smoke", "development", "confirmation"], default=None)
    p.add_argument("--env-file", type=Path, default=None, help="machine-specific env file sourced by each task line")
    p.add_argument("--jobs-dir", type=Path, default=None, help="snapshot of experiments/trace_merge/jobs used by the tasks")
    p.add_argument("--runs-root", type=Path, default=None, help="ingest: directory of the existing local B/T reruns")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for flag, allowed in (("dry_run", DRY_RUN_COMMANDS), ("force", FORCE_COMMANDS)):
        if getattr(args, flag) and args.command not in allowed:
            _log(f"FATAL: {args.command} does not support --{flag.replace('_', '-')} (only {', '.join(allowed)})")
            return 2
    cfg = load_config(args.config, work_root=args.work_root, protocol_clone=args.protocol_clone)
    if not (args.dry_run or args.command == "plan-merges"):   # read-only invocations create nothing
        ensure_work_tree(cfg)
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
