#!/usr/bin/env python3
"""Materialize and verify the native trace_eval_v1 dataset manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from trace_eval_media_contract import (  # noqa: E402
    DATASET_MANIFEST_SCHEMA,
    manifest_snapshot_sha256,
)
from prepare_trace_eval_datasets import (  # noqa: E402
    _configure_environment,
    _git_commit,
    _install_import_paths,
    _materialize_dataset,
    _receipt_is_complete,
    _utc_now,
    _write_json_atomic,
)
from trace_eval_suite import load_trace_eval_suite  # noqa: E402


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LMU_ROOT = EVALUATION_ROOT / ".work" / "LMUData"
DEFAULT_VLMEVAL_ROOT = REPO_ROOT / "external" / "VLMEvalKit"
DEFAULT_MANIFEST_NAME = "trace_eval_v1_dataset_manifest.json"


def _new_manifest(*, lmu_root: Path, vlmeval_root: Path) -> dict[str, Any]:
    suite = load_trace_eval_suite()
    payload = json.loads(suite.path.read_text(encoding="utf-8"))
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "suite_id": suite.suite_id,
        "suite_path": str(suite.path),
        "suite_sha256": suite.manifest_sha256,
        "vlmevalkit_repository": payload["vlmevalkit"]["repository"],
        "vlmevalkit_expected_commit": payload["vlmevalkit"]["commit"],
        "vlmevalkit_commit": _git_commit(vlmeval_root),
        "lmu_data_root": str(lmu_root.resolve()),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "dataset_views": {suite.dataset_manifest_view: list(suite.benchmark_keys)},
        "view_snapshot_sha256": {},
        "datasets": {},
    }


def _load_or_initialize_manifest(
    path: Path, *, lmu_root: Path, vlmeval_root: Path
) -> dict[str, Any]:
    fresh = _new_manifest(lmu_root=lmu_root, vlmeval_root=vlmeval_root)
    if not path.exists():
        return fresh
    previous = json.loads(path.read_text(encoding="utf-8"))
    stable_fields = (
        "schema_version",
        "suite_id",
        "suite_sha256",
        "vlmevalkit_expected_commit",
        "vlmevalkit_commit",
        "lmu_data_root",
        "dataset_views",
    )
    mismatched = [field for field in stable_fields if previous.get(field) != fresh.get(field)]
    if mismatched:
        print(
            f"[trace-eval-dataset:manifest-reset] changed={','.join(mismatched)} path={path}",
            flush=True,
        )
        return fresh
    fresh["created_at"] = previous.get("created_at", fresh["created_at"])
    fresh["datasets"] = previous.get("datasets", {})
    fresh["view_snapshot_sha256"] = previous.get("view_snapshot_sha256", {})
    return fresh


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    suite = load_trace_eval_suite()
    datasets = manifest.get("datasets", {})
    manifest["updated_at"] = _utc_now()
    manifest["ready"] = sum(item.get("status") == "ready" for item in datasets.values())
    manifest["failed"] = sum(item.get("status") == "error" for item in datasets.values())
    snapshot = manifest_snapshot_sha256(
        suite_sha256=suite.manifest_sha256,
        vlmevalkit_commit=str(manifest.get("vlmevalkit_commit") or ""),
        datasets=datasets,
        keys=suite.benchmark_keys,
    ) if all(
        (datasets.get(key) or {}).get("status") == "ready"
        and (datasets.get(key) or {}).get("dataset_snapshot_sha256")
        for key in suite.benchmark_keys
    ) else None
    manifest["view_snapshot_sha256"] = (
        {suite.dataset_manifest_view: snapshot} if snapshot is not None else {}
    )
    manifest["dataset_snapshot_sha256"] = snapshot
    _write_json_atomic(path, manifest)


def validate_trace_eval_manifest(payload: dict[str, Any]) -> None:
    suite = load_trace_eval_suite()
    if payload.get("schema_version") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported TRACE evaluation dataset manifest schema")
    if payload.get("suite_id") != suite.suite_id:
        raise ValueError("dataset manifest suite id mismatch")
    if payload.get("suite_sha256") != suite.manifest_sha256:
        raise ValueError("dataset manifest suite hash mismatch")
    if (payload.get("dataset_views") or {}) != {
        suite.dataset_manifest_view: list(suite.benchmark_keys)
    }:
        raise ValueError("dataset manifest must contain only the trace_eval_v1 view")
    datasets = payload.get("datasets") or {}
    if set(datasets) != set(suite.benchmark_keys):
        raise ValueError("dataset manifest receipts must exactly match trace_eval_v1")
    for benchmark in suite.benchmarks:
        receipt = datasets.get(benchmark.key) or {}
        if not _receipt_is_complete(
            receipt,
            alias=benchmark.official_alias,
            expected_rows=benchmark.rows,
        ):
            raise ValueError(f"dataset receipt is incomplete or stale: {benchmark.key}")
    snapshot = (payload.get("view_snapshot_sha256") or {}).get(
        suite.dataset_manifest_view
    )
    if not isinstance(snapshot, str) or len(snapshot) != 64:
        raise ValueError("dataset manifest has no valid trace_eval_v1 snapshot")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--lmu-root", type=Path, default=DEFAULT_LMU_ROOT)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    suite = load_trace_eval_suite()
    lmu_root = args.lmu_root.expanduser().resolve()
    hf_home = (args.hf_home or (lmu_root / ".hf-cache")).expanduser().resolve()
    vlmeval_root = args.vlmeval_root.expanduser().resolve()
    manifest_path = (
        args.manifest or (lmu_root / DEFAULT_MANIFEST_NAME)
    ).expanduser().resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    requested = set(args.only)
    unknown = requested - set(suite.benchmark_keys)
    if unknown:
        raise SystemExit(f"unknown trace_eval_v1 benchmark keys: {sorted(unknown)}")

    token = os.environ.get(args.token_env, "").strip() or None
    _configure_environment(lmu_root, hf_home, token)
    _install_import_paths(vlmeval_root)
    actual_commit = _git_commit(vlmeval_root)
    suite_payload = json.loads(suite.path.read_text(encoding="utf-8"))
    expected_commit = str(suite_payload["vlmevalkit"]["commit"])
    if actual_commit != expected_commit:
        raise SystemExit(
            f"VLMEvalKit commit mismatch: {actual_commit} != {expected_commit} ({vlmeval_root})"
        )

    manifest = _load_or_initialize_manifest(
        manifest_path, lmu_root=lmu_root, vlmeval_root=vlmeval_root
    )
    if not args.verify_only:
        _save_manifest(manifest_path, manifest)
    keys = [key for key in suite.benchmark_keys if not requested or key in requested]
    benchmarks = {item.key: item for item in suite.benchmarks}
    failures: list[str] = []
    for position, key in enumerate(keys, start=1):
        benchmark = benchmarks[key]
        previous = manifest["datasets"].get(key, {})
        if not args.force and _receipt_is_complete(
            previous,
            alias=benchmark.official_alias,
            expected_rows=benchmark.rows,
        ):
            print(
                f"[trace-eval-dataset:skip] {position}/{len(keys)} key={key} "
                f"rows={benchmark.rows} media={previous.get('unique_media')}",
                flush=True,
            )
            continue
        if args.verify_only:
            failures.append(key)
            print(
                f"[trace-eval-dataset:verify-error] {position}/{len(keys)} key={key} "
                "receipt or content hash is stale",
                flush=True,
            )
            continue
        manifest["datasets"][key] = {
            "status": "preparing",
            "key": key,
            "alias": benchmark.official_alias,
            "expected_rows": benchmark.rows,
            "started_at": _utc_now(),
        }
        _save_manifest(manifest_path, manifest)
        print(
            f"[trace-eval-dataset:start] {position}/{len(keys)} key={key} "
            f"alias={benchmark.official_alias} expected_rows={benchmark.rows}",
            flush=True,
        )
        try:
            receipt = _materialize_dataset(
                key=key,
                alias=benchmark.official_alias,
                expected_rows=benchmark.rows,
                lmu_root=lmu_root,
                workers=args.workers,
                token=token,
            )
        except KeyboardInterrupt:
            manifest["datasets"][key] = {
                **manifest["datasets"][key],
                "status": "interrupted",
                "completed_at": _utc_now(),
            }
            _save_manifest(manifest_path, manifest)
            raise
        except Exception as exc:
            failures.append(key)
            manifest["datasets"][key] = {
                **manifest["datasets"][key],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "completed_at": _utc_now(),
            }
            _save_manifest(manifest_path, manifest)
            print(
                f"[trace-eval-dataset:error] key={key} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                break
            continue
        manifest["datasets"][key] = receipt
        _save_manifest(manifest_path, manifest)
        print(
            f"[trace-eval-dataset:done] key={key} rows={receipt['rows']} "
            f"unique_media={receipt['unique_media']} media_bytes={receipt['media_bytes']}",
            flush=True,
        )

    selected_ready = sum(
        _receipt_is_complete(
            manifest["datasets"].get(key, {}),
            alias=benchmarks[key].official_alias,
            expected_rows=benchmarks[key].rows,
        )
        for key in keys
    )
    print(
        f"[trace-eval-dataset:summary] ready={selected_ready}/{len(keys)} "
        f"failures={failures} manifest={manifest_path}",
        flush=True,
    )
    if failures or selected_ready != len(keys):
        raise SystemExit(1)
    if not requested:
        validate_trace_eval_manifest(manifest)


if __name__ == "__main__":
    main()
