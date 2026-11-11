"""Command line of the follow-up study: python -m trace_followup <command> (run from experiments/trace_module_followup).

Planning commands write files and print what they did; none of them starts GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import ensure_work_tree, load, write_json


def _cfg(args):
    return load(args.config, work_root=args.work_root, protocol_clone=args.protocol_clone)


def cmd_inventory(args) -> int:
    from .inventory import run_inventory

    cfg = _cfg(args)
    ensure_work_tree(cfg)
    inv = run_inventory(cfg, Path(args.reference_hashes) if args.reference_hashes else None)
    print(json.dumps({k: inv[k] for k in ("inventory_sha256", "reference_hashes_matched", "compatibility_status",
                                          "component_counts")}, indent=1))
    print(json.dumps({r: v["tensors_differing_from_B"] for r, v in inv["parents"].items()}))
    return 0


def cmd_build(args) -> int:
    from .build import build_hybrid
    from .inventory import load_inventory

    cfg = _cfg(args)
    inv = load_inventory(cfg)
    for hid in args.models:
        if hid not in cfg.hybrids:
            print(f"unknown hybrid {hid}; configured: {sorted(cfg.hybrids)}", file=sys.stderr)
            return 2
    for hid in args.models:
        m = build_hybrid(cfg, hid, inv, dry_run=args.dry_run)
        if not args.dry_run:
            print(json.dumps({"hybrid_id": hid, "content_revision": m["content_revision"],
                              "tensors_by_donor": m["tensors_by_donor"],
                              "tensors_differing_from_B": m["tensors_differing_from_B"]}))
    return 0


def cmd_verify_builds(args) -> int:
    from .build import verify_hybrid
    from .inventory import identity_checks, load_inventory, ownership_checks

    cfg = _cfg(args)
    inv = load_inventory(cfg)
    out = {"ownership": ownership_checks(cfg, inv)}
    if not args.skip_identity:
        out["identity"] = identity_checks(cfg, inv)
    out["hybrids"] = {hid: verify_hybrid(cfg, hid, inv) for hid in (args.models or sorted(cfg.hybrids))}
    write_json(cfg.work("audit", "build_verification.json"), out)
    print(json.dumps(out, indent=1))
    return 0


def cmd_build_views(args) -> int:
    from .views import build_views

    cfg = _cfg(args)
    print(json.dumps(build_views(cfg, args.labels or None), indent=1))
    return 0


def cmd_plan_eval(args) -> int:
    from .evaluate import write_tasks

    cfg = _cfg(args)
    path, lines = write_tasks(cfg, args.tier, Path(args.env_file), Path(args.jobs_dir), cells=args.cells or None,
                              out=Path(args.out) if args.out else None, with_smoke=args.with_smoke)
    print(f"{len(lines)} task lines -> {path}")
    return 0


def cmd_status(args) -> int:
    from .evaluate import status

    cfg = _cfg(args)
    for row in status(cfg, args.tiers or None):
        print(json.dumps(row))
    return 0


def cmd_ingest_runs(args) -> int:
    """Ingest + F0-score every complete run (reviewed ingest, unchanged); incomplete runs are skipped with a status."""

    from .analyze import ingest_runs

    cfg = _cfg(args)
    out = ingest_runs(cfg, tiers=args.tiers or None, cells=args.cells or None, force=args.force,
                      reviewed_root=args.reviewed_root)
    for name, rec in out["runs"].items():
        print(json.dumps({"run": name, **{k: rec[k] for k in ("status", "ingest", "rows", "all_reproduced",
                                                                "not_reproduced", "max_abs_error_pp") if k in rec}}))
    print(json.dumps(out["counts"]))
    return 1 if any(r["status"].startswith(("failed", "judge_health_failed")) for r in out["runs"].values()) else 0


def cmd_analyze(args) -> int:
    from .analyze import analyze

    cfg = _cfg(args)
    summary = analyze(cfg, provisional=args.provisional, reviewed_root=args.reviewed_root,
                      n_replicates=args.replicates, behavior_replicates=args.behavior_replicates,
                      out_dir=Path(args.out_dir) if args.out_dir else None)
    print(json.dumps({k: summary[k] for k in ("status", "runs")}, indent=1, default=str))
    print(json.dumps(summary["primary"]["effects_overall"], indent=1, default=str))
    return 0


def cmd_analyze_f2c(args) -> int:
    from .analyze_f2c import analyze_f2c

    cfg = _cfg(args)
    summary = analyze_f2c(cfg, provisional=args.provisional, reviewed_root=args.reviewed_root,
                          n_replicates=args.replicates, out_dir=Path(args.out_dir) if args.out_dir else None)
    print(json.dumps({k: summary[k] for k in ("status", "runs_not_usable", "technical_stop")}, indent=1, default=str))
    print(json.dumps(summary["main"]["effects"], indent=1, default=str))
    return 0


def cmd_analyze_f3(args) -> int:
    from .analyze_f3 import analyze_f3

    cfg = _cfg(args)
    summary = analyze_f3(cfg, provisional=args.provisional, reviewed_root=args.reviewed_root,
                         n_replicates=args.replicates, out_dir=Path(args.out_dir) if args.out_dir else None)
    print(json.dumps({k: summary[k] for k in ("status", "lambda", "norms")}, indent=1, default=str))
    print(json.dumps(summary["main_P1"]["effects"], indent=1, default=str))
    return 0


def cmd_analyze_interface(args) -> int:
    from .analyze_interface import analyze_interface

    cfg = _cfg(args)
    summary = analyze_interface(cfg, provisional=args.provisional, reviewed_root=args.reviewed_root,
                                n_replicates=args.replicates, out_dir=Path(args.out_dir) if args.out_dir else None)
    print(json.dumps({k: summary[k] for k in ("status", "seeds", "missing_runs")}, indent=1, default=str))
    return 0


def cmd_export(args) -> int:
    from .export import evaluation_manifest, export_bundle

    cfg = _cfg(args)
    print(f"evaluation manifest -> {evaluation_manifest(cfg)}")
    if not args.manifest_only:
        hashes = export_bundle(cfg, Path(args.dest) if args.dest else None)
        print(f"{len(hashes)} files exported")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trace_followup")
    p.add_argument("--config", default=None)
    p.add_argument("--work-root", default=None)
    p.add_argument("--protocol-clone", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inventory")
    s.add_argument("--reference-hashes", default=None, help="the reviewed study's audit/parent_weight_sha256.json")
    s.set_defaults(fn=cmd_inventory)
    s = sub.add_parser("build")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser("verify-builds")
    s.add_argument("--models", nargs="*")
    s.add_argument("--skip-identity", action="store_true")
    s.set_defaults(fn=cmd_verify_builds)
    s = sub.add_parser("build-views")
    s.add_argument("--labels", nargs="*")
    s.set_defaults(fn=cmd_build_views)
    s = sub.add_parser("plan-eval")
    s.add_argument("--tier", required=True, choices=["smoke", "screen", "analysis", "template", "replication"])
    s.add_argument("--env-file", required=True)
    s.add_argument("--jobs-dir", required=True)
    s.add_argument("--cells", nargs="*")
    s.add_argument("--out", default=None)
    s.add_argument("--with-smoke", action="store_true", help="run each view's smoke tier first on the same line")
    s.set_defaults(fn=cmd_plan_eval)
    s = sub.add_parser("status")
    s.add_argument("--tiers", nargs="*")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("ingest-runs", help="ingest and F0-score every complete run (idempotent)")
    s.add_argument("--tiers", nargs="*", choices=["smoke", "screen", "analysis", "template", "replication"])
    s.add_argument("--cells", nargs="*", help="default: the seven F1 cells")
    s.add_argument("--force", action="store_true", help="re-ingest and re-score even if unchanged")
    s.add_argument("--reviewed-root", default=None, help="reviewed work root, read only (default TRACE_MERGE_WORK_ROOT)")
    s.set_defaults(fn=cmd_ingest_runs)
    s = sub.add_parser("analyze", help="design §6 tables (needs every run unless --provisional)")
    s.add_argument("--provisional", action="store_true",
                   help="use the complete runs only; writes tables/provisional/ with every row marked PROVISIONAL")
    s.add_argument("--replicates", type=int, default=None, help="bootstrap replicates (default: config, 10,000)")
    s.add_argument("--behavior-replicates", type=int, default=None,
                   help="replicates for the behaviour intervals (default: --replicates; 0 = no intervals)")
    s.add_argument("--out-dir", default=None)
    s.add_argument("--reviewed-root", default=None, help="reviewed work root, read only (default TRACE_MERGE_WORK_ROOT)")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("analyze-f2c", help="F2c reversion effects (design/decision_F2.md)")
    s.add_argument("--provisional", action="store_true")
    s.add_argument("--replicates", type=int, default=None)
    s.add_argument("--out-dir", default=None)
    s.add_argument("--reviewed-root", default=None)
    s.set_defaults(fn=cmd_analyze_f2c)
    s = sub.add_parser("analyze-f3", help="F3 norm-matched VERO (design/decision_F3.md)")
    s.add_argument("--provisional", action="store_true")
    s.add_argument("--replicates", type=int, default=None)
    s.add_argument("--out-dir", default=None)
    s.add_argument("--reviewed-root", default=None)
    s.set_defaults(fn=cmd_analyze_f3)
    s = sub.add_parser("analyze-interface", help="three-seed P1/P2 development-screen table (amendment A5)")
    s.add_argument("--provisional", action="store_true")
    s.add_argument("--replicates", type=int, default=None)
    s.add_argument("--out-dir", default=None)
    s.add_argument("--reviewed-root", default=None)
    s.set_defaults(fn=cmd_analyze_interface)
    s = sub.add_parser("export", help="manifests/evaluation.json and the sanitized results/ bundle")
    s.add_argument("--manifest-only", action="store_true")
    s.add_argument("--dest", default=None, help="default: experiments/trace_module_followup/results")
    s.set_defaults(fn=cmd_export)
    args = p.parse_args(argv)
    return int(args.fn(args) or 0)
