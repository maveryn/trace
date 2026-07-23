#!/usr/bin/env python3
"""Report generation, scoring, archive, and GPU progress for trace_eval_v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_queue_lib import extract_score_and_rows, run_dir, score_path, spec_by_key
from trace_eval_evaluator_provenance import DEFAULT_VLMEVAL_ROOT, evaluator_provenance_sha256
from trace_eval_score_receipts import ScoreReceiptError, validate_score_campaign_receipts
from trace_eval_suite import TraceEvalSuite, load_trace_eval_suite


REPO_ROOT = Path(__file__).resolve().parents[3]


def _manifest_rows(path: Path, suite: TraceEvalSuite) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    views = payload.get("dataset_views") or {}
    source_view = views.get(suite.dataset_manifest_view)
    if source_view != list(suite.benchmark_keys):
        raise ValueError(
            "dataset manifest must contain the exact ordered trace_eval_v1 view"
        )
    datasets = payload.get("datasets") or {}
    if set(datasets) != set(suite.benchmark_keys):
        raise ValueError("dataset manifest receipts must exactly match trace_eval_v1")
    rows = {
        key: int((datasets.get(key) or {}).get("rows", -1))
        for key in suite.benchmark_keys
    }
    mismatched = {
        key: {"expected": suite.rows_by_benchmark[key], "actual": rows[key]}
        for key in suite.benchmark_keys
        if rows[key] != suite.rows_by_benchmark[key]
    }
    if mismatched:
        raise ValueError(f"dataset manifest row counts do not match trace_eval_v1: {mismatched}")
    return rows


def _row_result_progress(path: Path, *, recent_after: float) -> tuple[int, int]:
    count = 0
    recent = 0
    try:
        entries = os.scandir(path)
    except FileNotFoundError:
        return 0, 0
    with entries:
        for entry in entries:
            if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                continue
            count += 1
            try:
                if entry.stat(follow_symlinks=False).st_mtime >= recent_after:
                    recent += 1
            except FileNotFoundError:
                continue
    return count, recent


def _score_complete(path: Path, expected_rows: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        score, rows = extract_score_and_rows(payload)
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        return False
    return (
        score is not None
        and rows is not None
        and math.isfinite(float(score))
        and int(rows) == expected_rows
    )


def _archive_status(spool_root: Path) -> dict[str, int]:
    ledger = spool_root / "ledger.sqlite3"
    result = {"descriptors": 0, "discovered": 0, "built": 0, "uploaded": 0, "failed": 0}
    ready = spool_root / "ready"
    if ready.is_dir():
        result["descriptors"] = sum(1 for _ in ready.glob("*.ready.json"))
    if not ledger.is_file():
        return result
    try:
        with sqlite3.connect(ledger) as connection:
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM slices GROUP BY status"
            ):
                if status in result:
                    result[str(status)] = int(count)
    except sqlite3.Error:
        result["failed"] += 1
    return result


def _gpu_status() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    records: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        records.append(
            {
                "index": int(fields[0]),
                "utilization": int(fields[1]),
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
            }
        )
    return records


def collect_status(
    args: argparse.Namespace,
    suite: TraceEvalSuite | None = None,
) -> dict[str, Any]:
    suite_manifest = getattr(args, "suite_manifest", None)
    active_suite = suite or (
        load_trace_eval_suite(suite_manifest) if suite_manifest else load_trace_eval_suite()
    )
    expected_by_benchmark = _manifest_rows(args.dataset_manifest, active_suite)
    expected_per_model_seed = active_suite.rows_per_model_seed
    recent_window = max(30.0, float(args.rate_window_seconds))
    recent_after = time.time() - recent_window
    records: list[dict[str, Any]] = []
    recent_rows = 0
    durable_rows = 0
    score_slices = 0
    observed_score_slices = 0
    score_receipt_errors: dict[int, str] = {}
    current_evaluator_hash = getattr(args, "_evaluator_provenance_sha256_cache", None)
    if current_evaluator_hash is None:
        current_evaluator_hash = evaluator_provenance_sha256(
            repo_root=REPO_ROOT,
            vlmeval_root=Path(getattr(args, "vlmeval_root", DEFAULT_VLMEVAL_ROOT)).resolve(),
        )
        # A watch process binds to one startup fingerprint; do not rehash the
        # evaluator worktree on every status poll.
        setattr(args, "_evaluator_provenance_sha256_cache", current_evaluator_hash)
    for seed in args.seeds:
        run_root = args.campaign_root / f"seed_{seed}" / "runs"
        benchmark_root = args.score_root / f"seed_{seed}" / "benchmark"
        verified_scores: dict[tuple[str, str], dict[str, Any]] = {}
        try:
            verified_scores = validate_score_campaign_receipts(
                score_root=args.score_root,
                seed=seed,
                model_slugs=args.model_slugs,
                suite=active_suite,
                evaluator_sha256=current_evaluator_hash,
                repo_root=REPO_ROOT,
            )
        except (ScoreReceiptError, OSError, ValueError, TypeError) as error:
            score_receipt_errors[int(seed)] = str(error)
        for model_slug in args.model_slugs:
            model_rows = 0
            model_recent = 0
            model_scores = 0
            model_observed_scores = 0
            complete_benchmarks = 0
            for key in active_suite.benchmark_keys:
                spec = spec_by_key(key)
                output_dir = run_dir(spec, model_slug, run_root)
                summary_path = output_dir / "generation_summary.json"
                expected_rows = expected_by_benchmark[key]
                rows, recent = _row_result_progress(
                    output_dir / "api_row_results", recent_after=recent_after
                )
                if summary_path.is_file():
                    try:
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        if int(summary["rows"]) == int(summary["expected_rows"]) == expected_rows:
                            rows = expected_rows
                            complete_benchmarks += 1
                    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                        pass
                model_rows += min(rows, expected_rows)
                model_recent += recent
                if _score_complete(score_path(spec, model_slug, benchmark_root), expected_rows):
                    model_observed_scores += 1
                if (model_slug, key) in verified_scores:
                    model_scores += 1
            durable_rows += model_rows
            recent_rows += model_recent
            score_slices += model_scores
            observed_score_slices += model_observed_scores
            records.append(
                {
                    "seed": seed,
                    "model_slug": model_slug,
                    "durable_rows": model_rows,
                    "expected_rows": expected_per_model_seed,
                    "generation_benchmarks": complete_benchmarks,
                    "expected_benchmarks": len(active_suite.benchmark_keys),
                    "score_slices": model_scores,
                    "observed_score_files": model_observed_scores,
                    "expected_score_slices": len(active_suite.benchmark_keys),
                }
            )

    combinations = len(args.seeds) * len(args.model_slugs)
    expected_rows_total = expected_per_model_seed * combinations
    expected_score_slices = len(active_suite.benchmark_keys) * combinations
    expected_archive_slices = expected_score_slices * 3
    rate = recent_rows / recent_window
    remaining = max(0, expected_rows_total - durable_rows)
    eta_seconds = remaining / rate if rate > 0 and remaining else (0.0 if not remaining else None)
    gpus = _gpu_status() if args.gpu else []
    warnings: list[str] = []
    if remaining and gpus and max(item["utilization"] for item in gpus) < args.low_gpu_threshold:
        warnings.append(
            f"generation incomplete while every GPU is below {args.low_gpu_threshold}% utilization"
        )
    for seed, error in score_receipt_errors.items():
        warnings.append(f"seed {seed} score receipt is incomplete or invalid: {error}")
    return {
        "suite": active_suite.suite_id,
        "suite_manifest_sha256": active_suite.manifest_sha256,
        "campaign_root": str(args.campaign_root),
        "score_root": str(args.score_root),
        "models": list(args.model_slugs),
        "seeds": list(args.seeds),
        "benchmarks": len(active_suite.benchmark_keys),
        "rows_per_model_seed": expected_per_model_seed,
        "durable_rows": durable_rows,
        "expected_rows": expected_rows_total,
        "recent_rows": recent_rows,
        "recent_rows_per_second": rate,
        "generation_eta_seconds": eta_seconds,
        "score_slices": score_slices,
        "observed_score_files": observed_score_slices,
        "expected_score_slices": expected_score_slices,
        "score_receipt_errors": score_receipt_errors,
        "expected_archive_slices": expected_archive_slices,
        "records": records,
        "archive": _archive_status(args.archive_spool_root),
        "gpus": gpus,
        "warnings": warnings,
        "complete": durable_rows == expected_rows_total and score_slices == expected_score_slices,
    }


def _duration(value: float | None) -> str:
    if value is None:
        return "unknown"
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_status(report: dict[str, Any]) -> None:
    print(
        "[trace-eval-status] "
        f"generation={report['durable_rows']}/{report['expected_rows']} "
        f"rate={report['recent_rows_per_second']:.2f} rows/s "
        f"eta={_duration(report['generation_eta_seconds'])} "
        f"scores={report['score_slices']}/{report['expected_score_slices']}"
        f" observed_score_files={report['observed_score_files']}"
    )
    for record in report["records"]:
        print(
            "[trace-eval-status:slice] "
            f"seed={record['seed']} model={record['model_slug']} "
            f"rows={record['durable_rows']}/{record['expected_rows']} "
            f"generated={record['generation_benchmarks']}/{record['expected_benchmarks']} "
            f"scored={record['score_slices']}/{record['expected_score_slices']}"
            f" observed={record['observed_score_files']}"
        )
    archive = report["archive"]
    print(
        "[trace-eval-status:archive] "
        f"descriptors={archive['descriptors']} expected={report['expected_archive_slices']} "
        f"built={archive['built']} uploaded={archive['uploaded']} failed={archive['failed']}"
    )
    if report["gpus"]:
        print(
            "[trace-eval-status:gpus] "
            + " ".join(
                f"gpu{item['index']}={item['utilization']}%/{item['memory_used_mib']}MiB"
                for item in report["gpus"]
            )
        )
    for warning in report["warnings"]:
        print(f"[trace-eval-status:warning] {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--suite-manifest", type=Path, default=None)
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--archive-spool-root", type=Path)
    parser.add_argument("--model-slug", action="append", dest="model_slugs", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rate-window-seconds", type=float, default=300.0)
    parser.add_argument("--low-gpu-threshold", type=int, default=10)
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    parser.add_argument("--fail-if-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.campaign_root = args.campaign_root.expanduser().resolve()
    args.score_root = (
        args.score_root.expanduser().resolve()
        if args.score_root
        else args.campaign_root / "scoring"
    )
    args.dataset_manifest = args.dataset_manifest.expanduser().resolve()
    args.vlmeval_root = args.vlmeval_root.expanduser().resolve()
    args.suite_manifest = (
        args.suite_manifest.expanduser().resolve()
        if args.suite_manifest
        else None
    )
    args.archive_spool_root = (
        args.archive_spool_root.expanduser().resolve()
        if args.archive_spool_root
        else args.campaign_root / "hf_archive"
    )
    args.model_slugs = tuple(dict.fromkeys(args.model_slugs))
    while True:
        report = collect_status(args)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print_status(report)
        if not args.watch or report["complete"]:
            raise SystemExit(1 if args.fail_if_incomplete and not report["complete"] else 0)
        time.sleep(max(1.0, args.watch))


if __name__ == "__main__":
    main()
