"""Command-line interfaces for public Trace generation and validation."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

from . import __version__
from .core.builder import BuildError, build_dataset, resolve_build_paths
from .core.config import BuildConfig, BuildTaskConfig, load_build_config
from .core.rlvr_export import (
    export_trace_dataset_to_rlvr,
    resolve_train_instances_source,
)
from .core.source_layout_policy import parse_public_task_id
from .core.validation import is_safe_trace_shard_id, validate_dataset
from .tasks.registry import list_default_task_ids, list_task_ids


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object on line {line_number} of {path}")
            records.append(value)
    return records


def _canonical_validation_expectations(
    build_report: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """Read immutable validation expectations from a canonical build report."""

    raw_task_counts = build_report.get("accepted_counts_by_task")
    if not isinstance(raw_task_counts, dict) or not raw_task_counts:
        raise ValueError("build_report.json must contain accepted_counts_by_task")

    task_counts: dict[str, int] = {}
    for raw_task_id, raw_count in raw_task_counts.items():
        task_id = str(raw_task_id).strip()
        if not task_id or isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError(
                "build_report.json accepted_counts_by_task must map non-empty task ids "
                "to non-negative integers"
            )
        if raw_count < 0:
            raise ValueError(
                "build_report.json accepted_counts_by_task must map non-empty task ids "
                "to non-negative integers"
            )
        task_counts[task_id] = raw_count

    raw_manifest = build_report.get("trace_shard_manifest")
    if not isinstance(raw_manifest, dict):
        raise ValueError("build_report.json must contain trace_shard_manifest")
    raw_shards = raw_manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError(
            "build_report.json trace_shard_manifest.shards must be non-empty"
        )

    shard_counts: dict[str, int] = {}
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, dict):
            raise ValueError("build_report.json trace shard entries must be objects")
        shard_id = raw_shard.get("shard_id")
        path_text = raw_shard.get("path")
        record_count = raw_shard.get("record_count")
        if (
            not is_safe_trace_shard_id(shard_id)
            or not isinstance(path_text, str)
            or path_text != f"traces/{shard_id}"
            or isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count < 0
        ):
            raise ValueError(
                "build_report.json trace shard entries require a unique shard_id, "
                "a traces/<shard_id> path, and a non-negative integer record_count"
            )
        if shard_id in shard_counts:
            raise ValueError("build_report.json trace shard ids must be unique")
        shard_counts[shard_id] = record_count

    if sum(task_counts.values()) != sum(shard_counts.values()):
        raise ValueError(
            "build_report.json task and trace-shard record counts must have equal totals"
        )
    return task_counts, shard_counts


def _task_row(task_id: str) -> dict[str, str]:
    parts = parse_public_task_id(task_id)
    return {
        "task_id": task_id,
        "domain": parts.domain,
        "scene_id": parts.scene_id,
        "objective_contract": parts.objective_contract,
    }


def list_main(argv: Sequence[str] | None = None) -> int:
    """List public task ids."""

    parser = argparse.ArgumentParser(description="List public Trace tasks")
    parser.add_argument("--domain", help="Only list tasks in this domain")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.add_argument(
        "--all", action="store_true", help="Include non-default registered tasks"
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    ids = list_task_ids() if args.all else list_default_task_ids()
    rows = [_task_row(task_id) for task_id in ids]
    if args.domain:
        rows = [row for row in rows if row["domain"] == args.domain]
        if not rows:
            parser.error(f"no tasks found for domain {args.domain!r}")

    if args.json:
        print(_json_dump({"count": len(rows), "tasks": rows}))
    else:
        for row in rows:
            print(row["task_id"])
        print(f"{len(rows)} task(s)", file=sys.stderr)
    return 0


def _selected_task_ids(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[str]:
    available = list_default_task_ids()
    if args.task:
        unknown = sorted(set(args.task) - set(available))
        if unknown:
            parser.error(f"unknown or inactive task id(s): {', '.join(unknown)}")
        return sorted(set(args.task))
    if args.domain:
        selected = [
            task_id
            for task_id in available
            if parse_public_task_id(task_id).domain == args.domain
        ]
        if not selected:
            parser.error(f"unknown or empty domain: {args.domain!r}")
        return selected
    if args.all_tasks:
        return available
    parser.error("select tasks with --task, --domain, --all, or provide --config")
    raise AssertionError("unreachable")


def _build_generate_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> BuildConfig:
    if args.config:
        if args.task or args.domain or args.all_tasks:
            parser.error("--config cannot be combined with --task, --domain, or --all")
        config = load_build_config(args.config)
        overrides: dict[str, Any] = {}
        if args.output is not None:
            overrides["output_root"] = str(args.output)
        if args.workers is not None:
            overrides["workers"] = int(args.workers)
        if args.max_in_flight is not None:
            overrides["max_in_flight"] = int(args.max_in_flight)
        if args.strict_repro:
            overrides["strict_repro"] = True
        return replace(config, **overrides) if overrides else config

    task_ids = _selected_task_ids(args, parser)
    if args.samples_per_task < 1:
        parser.error("--samples-per-task must be at least 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    return BuildConfig(
        output_root=str(args.output),
        dataset_name=str(args.dataset_name),
        instance_version="v0",
        image_format=str(args.image_format),
        tasks=[
            BuildTaskConfig(task_id=task_id, count=int(args.samples_per_task))
            for task_id in task_ids
        ],
        strict_repro=bool(args.strict_repro),
        max_attempts_per_instance=int(args.max_attempts),
        sampling_seed=int(args.seed),
        workers=int(args.workers if args.workers is not None else 1),
        max_in_flight=int(args.max_in_flight if args.max_in_flight is not None else 0),
    )


def generate_main(argv: Sequence[str] | None = None) -> int:
    """Generate a deterministic dataset from tasks or a YAML config."""

    parser = argparse.ArgumentParser(
        description="Generate a deterministic Trace dataset"
    )
    parser.add_argument(
        "--config", type=Path, help="Use a Trace YAML build configuration"
    )
    parser.add_argument(
        "--task", action="append", help="Public task id; repeat to select several"
    )
    parser.add_argument("--domain", help="Generate every default task in one domain")
    parser.add_argument(
        "--all",
        dest="all_tasks",
        action="store_true",
        help="Generate all default tasks",
    )
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-name", default="trace-generated")
    parser.add_argument("--image-format", choices=("png", "jpg", "jpeg"), default="png")
    parser.add_argument(
        "--workers", type=int, help="Worker processes; 0 uses all visible CPUs"
    )
    parser.add_argument(
        "--max-in-flight", type=int, help="Maximum queued generation attempts"
    )
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--strict-repro", action="store_true")
    parser.add_argument(
        "--code-hash",
        default=f"trace-tasks-{__version__}",
        help="Code provenance value stored in generated records",
    )
    parser.add_argument(
        "--format", choices=("canonical", "parquet"), default="canonical"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without generating",
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if args.output is None and not args.config:
        args.output = Path("trace-output")

    config = _build_generate_config(args, parser)
    paths = resolve_build_paths(config)
    plan = {
        "dataset_id": paths.dataset_id,
        "dataset_name": config.dataset_name,
        "output_root": str(Path(config.output_root).expanduser().resolve()),
        "task_count": len(config.tasks),
        "instance_count": sum(int(task.count or 0) for task in config.tasks),
        "sampling_seed": config.sampling_seed,
        "workers": config.workers,
        "format": args.format,
    }
    if args.dry_run:
        print(_json_dump(plan))
        return 0

    try:
        dataset_root = build_dataset(config, code_hash=str(args.code_hash))
    except BuildError as exc:
        print(f"Trace build failed: {exc}", file=sys.stderr)
        return 1

    result: dict[str, Any] = {**plan, "dataset_root": str(dataset_root.resolve())}
    if args.format == "parquet":
        export_path = (
            Path(config.output_root) / "exports" / f"{paths.dataset_id}.parquet"
        )
        exported = export_trace_dataset_to_rlvr(
            dataset_root,
            export_path,
            output_format="parquet",
            prompt_variant="active",
            image_storage_mode="embedded_bytes",
            parquet_cpu_count=(config.workers or os.cpu_count() or 1),
            max_embedded_image_pixels=1_280_000,
        )
        result["parquet_path"] = str(exported.output_path)
    print(_json_dump(result))
    return 0


def validate_main(argv: Sequence[str] | None = None) -> int:
    """Validate a generated canonical dataset."""

    parser = argparse.ArgumentParser(description="Validate a generated Trace dataset")
    parser.add_argument(
        "source", type=Path, help="Dataset root or train_instances.jsonl"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the complete validation report"
    )
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    try:
        dataset_root, train_instances_path = resolve_train_instances_source(args.source)
        instances = _read_jsonl(train_instances_path)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.error(f"could not read validation source: {exc}")
    if not instances:
        parser.error("dataset contains no instances")
    build_report_path = dataset_root / "build_report.json"
    build_report: dict[str, Any] = {}
    expected_trace_shard_counts: dict[str, int] | None = None
    if build_report_path.exists():
        try:
            loaded_report = json.loads(build_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            parser.error(f"could not read build_report.json: {exc}")
        if not isinstance(loaded_report, dict):
            parser.error("build_report.json must contain a JSON object")
        build_report = loaded_report
        try:
            counts, expected_trace_shard_counts = _canonical_validation_expectations(
                build_report
            )
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if args.source.expanduser().is_dir():
            parser.error(
                "canonical dataset roots require build_report.json; pass an explicit "
                "train_instances.jsonl file for weaker standalone validation"
            )
        counts = Counter(str(item.get("task", "")) for item in instances)
        print(
            "warning: build_report.json not found; validating standalone JSONL with "
            "observed counts, so completeness cannot be checked",
            file=sys.stderr,
        )
    report = validate_dataset(
        instances,
        staging_root=dataset_root,
        expected_task_counts=counts,
        dataset_id=str(build_report.get("dataset_id") or dataset_root.name),
        expected_instance_version=str(instances[0].get("instance_version") or "v0"),
        expected_trace_shard_counts=expected_trace_shard_counts,
    )
    if args.json or int(report["total_errors"]) > 0:
        print(_json_dump(report))
    else:
        print(f"valid: {len(instances)} instance(s), 0 errors")
    return 0 if int(report["total_errors"]) == 0 else 1


def export_main(argv: Sequence[str] | None = None) -> int:
    """Export a canonical dataset to RLVR JSONL or Parquet."""

    parser = argparse.ArgumentParser(
        description="Export Trace data to JSONL or Parquet"
    )
    parser.add_argument(
        "source", type=Path, help="Dataset root or train_instances.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("jsonl", "parquet"))
    parser.add_argument(
        "--prompt-variant",
        choices=(
            "active",
            "answer",
            "answer_only",
            "annotation",
            "answer_and_annotation",
        ),
        default="active",
    )
    parser.add_argument(
        "--image-path-mode",
        choices=("relative", "absolute", "dataset_relative"),
        default="relative",
    )
    parser.add_argument(
        "--image-storage-mode",
        choices=("path_dict", "embedded_bytes"),
        default="path_dict",
    )
    parser.add_argument("--parquet-cpu-count", type=int)
    parser.add_argument("--max-embedded-image-pixels", type=int)
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)

    # The export library reports progress through tqdm.write. Keep stdout
    # machine-readable for callers that consume this command's JSON result.
    with redirect_stdout(sys.stderr):
        result = export_trace_dataset_to_rlvr(
            args.source,
            args.output,
            output_format=args.format,
            prompt_variant=args.prompt_variant,
            image_path_mode=args.image_path_mode,
            image_storage_mode=args.image_storage_mode,
            parquet_cpu_count=args.parquet_cpu_count,
            max_embedded_image_pixels=args.max_embedded_image_pixels,
        )
    print(
        _json_dump(
            {
                "output_path": str(result.output_path),
                "row_count": result.row_count,
                "format": result.output_format,
                "prompt_variant": result.prompt_variant,
                "image_path_mode": result.image_path_mode,
            }
        )
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """Run generation when invoked as ``python -m trace_tasks.cli``."""

    return generate_main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
