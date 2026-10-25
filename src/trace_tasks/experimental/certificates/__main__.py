"""CLI of the certificate extension: ``python -m trace_tasks.experimental.certificates``.

Subcommands (each takes explicit paths and writes a receipt):

- ``audit --out coverage.v1.json``: coverage of every TRACE task and query (DESIGN 1.1).
- ``export --split <split> --out <dir>``: build one split (DESIGN 6) and write its public,
  private, RL and build outputs plus ``receipts/<split>.json``.
- ``export-sft --split sft|rl --out <dir>``: SFT records of a built training split.
- ``freeze --out <dir>``: the merged store, the RL reward store (rl + e1_dev) and
  ``receipts/freeze.json`` naming every frozen output (run once after the last export).
- ``score --responses <run>/responses.jsonl --store <private store> --receipt <export
  receipt> --out <dir>``: ``cert_scores.jsonl`` + ``summary.json`` of one trace-tool run.
- ``summarize --scores <a.jsonl> [<b.jsonl> ...] --out summary.json``: pooled summaries.

M1 CPU smoke (DESIGN 1.1)::

    python -m trace_tasks.experimental.certificates audit --out $RUNS/coverage.v1.json
    python -m trace_tasks.experimental.certificates export --split fixtures --out $RUNS/data/v2
    python -m trace_tasks.experimental.certificates score --responses <run>/responses.jsonl \\
        --store $RUNS/data/v2/private/fixtures.extensions_v1.jsonl.zst \\
        --receipt $RUNS/data/v2/receipts/fixtures.json --out <run>/cert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

COVERAGE_VERSION = "coverage.v1"

# Static coverage facts per certificate kind (DESIGN 3.x, 10); adapters may override them
# with a ``coverage_notes`` class attribute of the same shape.
COVERAGE_NOTES: Dict[str, Dict[str, Any]] = {
    "selected_set": {
        "required_fields": [
            {"field": "scene_ir.entities[type=chess_piece].{id,color,kind,row,col,bbox_px}", "source": "trace sidecar scene_ir"},
            {"field": "scene_ir.relations.{target_piece_kind,target_piece_color,annotation_entity_ids}", "source": "trace sidecar scene_ir"},
            {"field": "answer_gt, annotation_gt", "source": "TrainInstance labels (cross-check only)"},
        ],
        "metadata_gaps": ["chess renders no coordinate labels: members bind to pieces by box geometry only"],
        "richer_than_annotation": {"value": "partly", "why": "adds per-member color/piece reads and the count; the selected set re-expresses G"},
    },
    "filtered_mean": {
        "required_fields": [
            {"field": "scene_ir.entities[entity_type=table_cell].{bbox_px, attrs.{cell_role,row_index,column_index,row_label,column_header,value}}", "source": "trace sidecar scene_ir"},
            {"field": "scene_ir.relations.{filter_variant,filter_column,target_column,threshold_value|interval_min/max,selected_row_indices}", "source": "trace sidecar scene_ir"},
        ],
        "metadata_gaps": [
            "execution_trace is sanitized (*_cell_ids, labels removed); the adapter reads scene_ir",
            "selections are never empty and means are always integers",
        ],
        "richer_than_annotation": {"value": "yes", "why": "adds filter/target operand values, the row pairing (legacy G never checks it), sum, count and mean"},
    },
    "weighted_sum": {
        "required_fields": [
            {"field": "scene_ir.entities[entity_type=physics_lever_weight].{bbox_px, meta.{side,distance_units,value,missing,relevant_to_task}}", "source": "trace sidecar scene_ir"},
            {"field": "execution_trace.{torque_side,query_side,target_answer,weight_specs}", "source": "trace sidecar execution_trace (weight_specs index-aligned, checked)"},
        ],
        "metadata_gaps": ["no per-term products or partial sums are stored; weight_specs has no ids (alignment by index, checked)"],
        "richer_than_annotation": {"value": "yes", "why": "adds weight and distance reads per block and every product; the annotation only marks the blocks"},
    },
    "path": {
        "required_fields": [
            {"field": "scene_ir.tiles[].{tile_id,row,col,terrain,movement_cost,passable,bbox,point}", "source": "trace sidecar scene_ir"},
            {"field": "scene_ir.relations.{start_tile_id,target_tile_id,terrain_movement_costs,blocked_terrain}", "source": "trace sidecar scene_ir"},
            {"field": "execution_trace.renderer.{grid_rows,grid_cols,tile_px,canvas_width,canvas_height,canvas_profile}", "source": "trace sidecar execution_trace"},
        ],
        "metadata_gaps": [
            "no set of optimal paths is stored (the verifier runs Dijkstra; shortest_path_tile_ids is one tie-break)",
            "render_spec has canvas_size but no canvas_width/canvas_height",
            "targets are never unreachable",
        ],
        "richer_than_annotation": {"value": "yes", "why": "adds the route, per-step entry costs, legality and optimality; the annotation only marks the endpoints"},
        "variants": [
            {"variant": "reachable:false", "status": "defined_not_exercised", "reason": "0 of 200 generated rpg instances are unreachable; the form is parsed and scored but never needed"},
        ],
    },
}

DEVIATIONS = [
    {"id": "D3", "deviation": "the path adapter is illustrations/rpg_tactical_map movement cost, not a graph task", "why": "graph shortest paths are unique by construction"},
    {"id": "D8", "deviation": "zero-shot Qwen3.5 annotations use rel1000 and are converted, not given in source pixels", "why": "Qwen3.5 grounds natively in rel1000; eval-only"},
    {"id": "D2", "deviation": "the runner lives in experiments/certificates/, not rlvr/experiments/certificates/", "why": "rlvr release validation rejects extra files under rlvr/"},
    {"id": "D11", "deviation": "A/G/P are matched at N=2-4 GPUs with batch 64 x 8 and 300 steps, not 8 x 128 x 500", "why": "available hardware"},
]


# ======================================================================================
# audit


def _receipt_counts(receipts_dir: Optional[Path]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{task: {split: {"quarantined": n, "reasons": {...}, "accepted": n}}}`` from export receipts."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if receipts_dir is None:
        return out
    for path in sorted(receipts_dir.glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("command") != "export":
            continue
        for counts in (receipt.get("counts") or {}).values():
            for task, tally in (counts.get("by_task") or {}).items():
                entry = out.setdefault(task, {}).setdefault(str(receipt["split"]), {"quarantined": 0, "accepted": 0, "reasons": {}})
                entry["quarantined"] += int(tally.get("quarantined", 0))
                entry["accepted"] += int(tally.get("accepted", 0))
                for reason, n in (tally.get("quarantine_reasons") or {}).items():
                    entry["reasons"][reason] = entry["reasons"].get(reason, 0) + int(n)
    return out


def build_coverage(
    *,
    receipts_dir: Optional[Path] = None,
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """coverage.v1 document: every TRACE task (or ``tasks``) with status, reason and kind."""
    from trace_tasks.core.source_layout_policy import parse_public_task_id
    from trace_tasks.tasks.registry import create_task, ensure_task_registered, is_default_dataset_task, list_task_ids

    from . import registry
    from .types import SCORER_VERSION

    task_ids = sorted(tasks) if tasks else list_task_ids()
    quarantine = _receipt_counts(receipts_dir)
    adapters: Dict[str, Any] = {}
    for name in registry.ADAPTER_MODULES:
        try:
            adapter = registry.load_adapter(name)
        except NotImplementedError as exc:
            adapters[name] = {"status": "not_implemented", "reason": str(exc)}
            continue
        adapters[name] = {"status": "implemented", **adapter.spec().to_dict(), "m_families": list(adapter.m_families)}

    entries: List[Dict[str, Any]] = []
    counts = {"supported": 0, "unsupported": 0, "defined_not_exercised": 0}
    for task_id in task_ids:
        parts = parse_public_task_id(task_id)
        ensure_task_registered(task_id)
        entry: Dict[str, Any] = {
            "task": task_id,
            "domain": parts.domain,
            "scene_id": parts.scene_id,
            "default_dataset": bool(is_default_dataset_task(task_id)),
        }
        candidate = task_id in registry.TASK_ADAPTERS or task_id in registry.KNOWN_UNSUPPORTED
        query_ids = tuple(getattr(create_task(task_id), "supported_query_ids", ()) or ()) if candidate else ()
        queries = []
        for query_id in query_ids:
            resolved = registry.resolve(task_id, str(query_id), "v0")
            if isinstance(resolved, registry.Unsupported):
                queries.append({"query_id": query_id, "status": "unsupported", "reason": resolved.reason})
            else:
                queries.append({"query_id": query_id, "status": "supported", "adapter_id": resolved.adapter_id, "kind": resolved.kind})
        supported = [q for q in queries if q["status"] == "supported"]
        if supported:
            adapter = registry.load_adapter(registry.TASK_ADAPTERS[task_id])
            notes = dict(getattr(adapter, "coverage_notes", None) or COVERAGE_NOTES.get(adapter.kind, {}))
            entry.update(
                status="supported",
                reason="adapter " + adapter.adapter_id,
                adapter_id=adapter.adapter_id,
                kind=adapter.kind,
                instance_versions=list(adapter.instance_versions),
                annotation_type=adapter.annotation_type,
                queries=queries,
                required_fields=notes.get("required_fields", []),
                metadata_gaps=notes.get("metadata_gaps", []),
                richer_than_annotation=notes.get("richer_than_annotation"),
                variants=notes.get("variants", []),
                quarantined=quarantine.get(task_id, {}),
            )
            counts["supported"] += 1
            counts["defined_not_exercised"] += sum(1 for v in entry["variants"] if v.get("status") == "defined_not_exercised")
        else:
            reason = (
                queries[0]["reason"]
                if queries
                else registry.KNOWN_UNSUPPORTED.get(task_id, "no_adapter: no certificate adapter in v1")
            )
            entry.update(status="unsupported", reason=reason, kind=None, queries=queries)
            counts["unsupported"] += 1
        entries.append(entry)
    return {
        "coverage_version": COVERAGE_VERSION,
        "scorer_version": SCORER_VERSION,
        "adapters": adapters,
        "counts": {**counts, "tasks": len(entries)},
        "deviations": DEVIATIONS,
        "known_unsupported": dict(registry.KNOWN_UNSUPPORTED),
        "tasks": entries,
    }


# ======================================================================================
# commands


def _receipt_path(out: Path) -> Path:
    return out.with_name(out.stem + ".receipt.json")


def _cmd_audit(args: argparse.Namespace) -> int:
    from .export import _write_json, code_provenance, file_sha256

    out = Path(args.out)
    tasks = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        tasks = manifest.get("tasks") if isinstance(manifest, dict) else manifest
    coverage = build_coverage(receipts_dir=Path(args.receipts) if args.receipts else None, tasks=tasks)
    _write_json(out, coverage)
    receipt = {
        "command": "audit",
        "coverage_version": COVERAGE_VERSION,
        "manifest": {"path": args.manifest, "sha256": file_sha256(args.manifest)} if args.manifest else None,
        "receipts_dir": args.receipts,
        "counts": coverage["counts"],
        "outputs": {out.name: {"sha256": file_sha256(out)}},
        "provenance": code_provenance(),
    }
    _write_json(_receipt_path(out), receipt)
    print(json.dumps(coverage["counts"]))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export_split

    adapters = [name.strip() for name in args.adapters.split(",")] if args.adapters else None
    receipt = export_split(
        args.split,
        args.out,
        adapters=adapters,
        count=args.count,
        workers=args.workers,
        chunk_size=args.chunk_size,
        dedupe=not args.no_dedupe,
    )
    print(json.dumps({"split": args.split, "totals": receipt["totals"], "receipt": f"receipts/{args.split}.json"}))
    if receipt["totals"]["roundtrip_failed"]:
        print("error: oracle round trip failed for some rows (see receipt counts.*.roundtrip_failed)", file=sys.stderr)
        return 3
    return 0


def _cmd_export_sft(args: argparse.Namespace) -> int:
    from .export import export_sft

    receipt = export_sft(args.split, args.out, system_prompt_file=args.system_prompt)
    print(json.dumps({"split": args.split, "rows": receipt["rows"], "receipt": f"receipts/{args.split}.sft.json"}))
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    from .export import FREEZE_RECEIPT, freeze

    receipt = freeze(args.out)
    print(json.dumps({
        "splits": sorted(receipt["splits"]),
        "merged_store": receipt["merged_store"]["sha256"],
        "rl_reward_store": receipt["rl_reward_store"]["sha256"],
        "receipt": f"receipts/{FREEZE_RECEIPT}",
    }))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .evaluate import score_run

    summary = score_run(
        args.responses,
        args.store,
        args.receipt,
        args.out,
        cond=args.cond,
        family=args.family,
        store_sha256=args.store_sha256,
        allow_unfrozen=args.allow_unfrozen,
        allow_settings_mismatch=args.allow_settings_mismatch,
        prompt_check=not args.no_prompt_check,
    )
    print(json.dumps({"run": summary["run"], "all": summary["all"][0] if summary["all"] else {}}, default=str)[:2000])
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    from .evaluate import summarize_files

    summary = summarize_files(args.scores, args.out)
    print(json.dumps({"groups": len(summary["all"]), "out": args.out}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from .export import PUBLIC_CONDITIONS, SFT_SPLITS, SPLIT_SEEDS

    parser = argparse.ArgumentParser(prog="python -m trace_tasks.experimental.certificates", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="write coverage.v1.json")
    audit.add_argument("--out", required=True, help="coverage.v1.json path (a .receipt.json is written next to it)")
    audit.add_argument("--receipts", help="export receipts dir: fills quarantine counts")
    audit.add_argument("--manifest", help="JSON list (or {'tasks': [...]}) of task ids to audit; default all tasks")
    audit.set_defaults(func=_cmd_audit)

    export = sub.add_parser("export", help="build one split")
    export.add_argument("--split", required=True, choices=list(SPLIT_SEEDS))
    export.add_argument("--out", required=True, help="data root, e.g. $TRACE_CERT_RUNS/data/v1")
    export.add_argument("--adapters", help="comma list of adapter modules (default: every implemented adapter)")
    export.add_argument("--count", type=int, help="rows per adapter (default: DESIGN 6 count of the split)")
    export.add_argument("--workers", type=int, help="worker processes (default min(16, nproc))")
    export.add_argument("--chunk-size", type=int, default=64, help="seed indices per cached chunk")
    export.add_argument("--no-dedupe", action="store_true", help="do not drop test rows seen in SFT/RL (recorded)")
    export.set_defaults(func=_cmd_export)

    sft = sub.add_parser("export-sft", help="SFT records of a built training split")
    sft.add_argument("--split", required=True, choices=list(SFT_SPLITS))
    sft.add_argument("--out", required=True, help="data root the split was exported to")
    sft.add_argument("--system-prompt", help="system prompt file (default: resources cert_system_v1.txt)")
    sft.set_defaults(func=_cmd_export_sft)

    frz = sub.add_parser("freeze", help="write the merged/RL stores and receipts/freeze.json")
    frz.add_argument("--out", required=True, help="data root every split was exported to")
    frz.set_defaults(func=_cmd_freeze)

    score = sub.add_parser("score", help="score one trace-tool responses.jsonl")
    score.add_argument("--responses", required=True)
    score.add_argument("--store", required=True, help="private extension store (*.jsonl.zst)")
    score.add_argument("--receipt", required=True, help="export receipt of the split (receipts/<split>.json)")
    score.add_argument("--out", required=True, help="output dir for cert_scores.jsonl and summary.json")
    score.add_argument("--cond", choices=list(PUBLIC_CONDITIONS),
                       help="must agree with the condition of the run's frozen dataset (with --allow-unfrozen: the condition)")
    score.add_argument("--family", choices=["qwen2.5-vl", "qwen3-vl", "qwen3.5", "other"], help="override the model family")
    score.add_argument("--store-sha256", help="also require this store sha256 (the store must still match the receipt)")
    score.add_argument("--allow-unfrozen", action="store_true",
                       help="score a run whose dataset sha256 is not a parquet of the receipt (needs --cond; recorded)")
    score.add_argument("--allow-settings-mismatch", action="store_true",
                       help="score a run whose system prompt / oracle_crop / tools / decoding differ from DESIGN 7.1 (recorded)")
    score.add_argument("--no-prompt-check", action="store_true",
                       help="do not compare each transcript's prompt with the frozen prompt sha256 (recorded)")
    score.set_defaults(func=_cmd_score)

    summarize = sub.add_parser("summarize", help="pool cert_scores.jsonl files")
    summarize.add_argument("--scores", required=True, nargs="+")
    summarize.add_argument("--out", required=True, help="summary json path (a .receipt.json is written next to it)")
    summarize.set_defaults(func=_cmd_summarize)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
