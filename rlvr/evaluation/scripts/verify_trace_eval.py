#!/usr/bin/env python3
"""Verify exact generation or scoring coverage for trace_eval_v1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark_queue_lib import run_dir, score_path, spec_by_key  # noqa: E402
from trace_eval_campaign_verification import (  # noqa: E402
    _expected_generation_contract_hash,
    _generation_complete,
    _parse_model_entries,
)
from trace_eval_suite import TraceEvalSuite, load_trace_eval_suite  # noqa: E402
from trace_eval_code_provenance import trace_eval_code_sha256  # noqa: E402
from trace_eval_evaluator_provenance import (  # noqa: E402
    DEFAULT_VLMEVAL_ROOT,
    evaluator_provenance_sha256,
)
from trace_eval_score_receipts import (  # noqa: E402
    ScoreReceiptError,
    validate_score_campaign_receipts,
)


DEFAULT_DATASET_MANIFEST = Path(
    os.environ.get(
        "TRACE_EVAL_DATASET_MANIFEST",
        str(EVALUATION_ROOT / ".work" / "LMUData" / "trace_eval_v1_dataset_manifest.json"),
    )
)


def _load_dataset_contract(
    path: Path, suite: TraceEvalSuite
) -> tuple[str, str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("suite_id") != suite.suite_id:
        raise ValueError("dataset manifest suite id mismatch")
    if payload.get("suite_sha256") != suite.manifest_sha256:
        raise ValueError("dataset manifest suite hash mismatch")
    if (payload.get("dataset_views") or {}) != {
        suite.dataset_manifest_view: list(suite.benchmark_keys)
    }:
        raise ValueError("dataset manifest must contain only the exact trace_eval_v1 view")
    snapshot = str(
        (payload.get("view_snapshot_sha256") or {}).get(suite.dataset_manifest_view) or ""
    )
    if len(snapshot) != 64:
        raise ValueError("dataset manifest has no valid trace_eval_v1 snapshot")
    datasets = payload.get("datasets") or {}
    if set(datasets) != set(suite.benchmark_keys):
        raise ValueError("dataset manifest receipts must exactly match trace_eval_v1")
    rows = {
        key: int((datasets.get(key) or {}).get("rows", -1))
        for key in suite.benchmark_keys
    }
    if rows != suite.rows_by_benchmark:
        raise ValueError("dataset manifest row counts do not match trace_eval_v1")
    schema = str(payload.get("schema_version") or "")
    if not schema:
        raise ValueError("dataset manifest has no schema version")
    return snapshot, f"{schema}:{snapshot}", rows


def verify_campaign(
    args: argparse.Namespace,
    suite: TraceEvalSuite | None = None,
) -> dict[str, Any]:
    active_suite = suite or load_trace_eval_suite(args.suite_manifest)
    snapshot, manifest_revision, rows_by_benchmark = _load_dataset_contract(
        args.dataset_manifest, active_suite
    )
    dataset_snapshot = args.dataset_snapshot_sha256 or snapshot
    dataset_revision = args.dataset_revision or manifest_revision
    if dataset_snapshot != snapshot:
        raise ValueError("requested dataset snapshot does not match the native manifest")
    if dataset_revision != manifest_revision:
        raise ValueError("requested dataset revision does not match the native manifest")

    model_entries = _parse_model_entries(args.model_entry)
    model_slugs = list(dict.fromkeys([*args.model_slugs, *model_entries]))
    if not model_slugs:
        raise ValueError("pass at least one --model-slug or --model-entry")
    vlmeval_root = Path(getattr(args, "vlmeval_root", DEFAULT_VLMEVAL_ROOT)).resolve()
    current_evaluator_hash = evaluator_provenance_sha256(
        repo_root=REPO_ROOT,
        vlmeval_root=vlmeval_root,
    )
    if args.phase == "generation":
        missing = sorted(set(model_slugs) - set(model_entries))
        if missing:
            raise ValueError(f"generation verification is missing model entries for: {missing}")
        if not args.code_hash:
            raise ValueError("generation verification requires --code-hash")
        recorded_evaluator_hash = str(
            getattr(args, "evaluator_provenance_sha256", None) or ""
        )
        if recorded_evaluator_hash != current_evaluator_hash:
            raise ValueError(
                "generation evaluator provenance does not match the current evaluator worktree"
            )
        current_code_hash = trace_eval_code_sha256(
            repo_root=REPO_ROOT,
            vlmeval_root=vlmeval_root,
            evaluator_sha256=current_evaluator_hash,
        )
        if args.code_hash != current_code_hash:
            raise ValueError(
                "generation code hash does not match the current trace_eval_v1 code contract"
            )

    score_root = args.score_root or args.campaign_root / "scoring"
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_root = args.campaign_root / f"seed_{seed}" / "runs"
        benchmark_root = score_root / f"seed_{seed}" / "benchmark"
        verified_scores: dict[tuple[str, str], dict[str, Any]] = {}
        receipt_error: str | None = None
        if args.phase == "score":
            try:
                verified_scores = validate_score_campaign_receipts(
                    score_root=score_root,
                    seed=seed,
                    model_slugs=model_slugs,
                    suite=active_suite,
                    evaluator_sha256=current_evaluator_hash,
                    repo_root=REPO_ROOT,
                )
            except (ScoreReceiptError, OSError, ValueError, TypeError) as error:
                receipt_error = f"receipt_invalid:{error}"
        for model_slug in model_slugs:
            for benchmark in active_suite.benchmarks:
                spec = spec_by_key(benchmark.key)
                if args.phase == "generation":
                    path = run_dir(spec, model_slug, run_root) / "generation_summary.json"
                    model, revision = model_entries[model_slug]
                    contract_hash = _expected_generation_contract_hash(
                        model=model,
                        model_slug=model_slug,
                        model_revision=revision,
                        seed=seed,
                        dataset_snapshot_sha256=dataset_snapshot,
                        dataset_revision=dataset_revision,
                        final25_code_hash=args.code_hash,
                        max_tokens=4096,
                    )
                    complete, detail = _generation_complete(
                        path,
                        seed,
                        dataset_revision,
                        expected_model_revision=revision,
                        expected_contract_hash=contract_hash,
                        expected_max_tokens=4096,
                    )
                    if complete:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        exact_rows = (
                            int(payload.get("rows", -1)) == benchmark.rows
                            and int(payload.get("expected_rows", -1)) == benchmark.rows
                        )
                        if not exact_rows:
                            complete = False
                            detail += f" suite_rows={benchmark.rows}"
                else:
                    path = score_path(spec, model_slug, benchmark_root)
                    verified = verified_scores.get((model_slug, benchmark.key))
                    complete = verified is not None and receipt_error is None
                    detail = (
                        f"score={verified['score']} rows={verified['rows']}/{rows_by_benchmark[benchmark.key]}"
                        if complete
                        else str(receipt_error or "receipt_missing_slice")
                    )
                records.append(
                    {
                        "seed": seed,
                        "model_slug": model_slug,
                        "benchmark_key": benchmark.key,
                        "phase": args.phase,
                        "complete": complete,
                        "detail": detail,
                        "path": str(path),
                    }
                )

    incomplete = [record for record in records if not record["complete"]]
    return {
        "schema_version": "trace-eval-verification-v1",
        "suite_id": active_suite.suite_id,
        "suite_manifest_sha256": active_suite.manifest_sha256,
        "phase": args.phase,
        "models": model_slugs,
        "seeds": list(args.seeds),
        "complete": not incomplete,
        "completed_slices": len(records) - len(incomplete),
        "expected_slices": len(records),
        "incomplete": incomplete,
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, default=None)
    parser.add_argument("--phase", choices=("generation", "score"), required=True)
    parser.add_argument("--model-slug", action="append", dest="model_slugs", default=[])
    parser.add_argument(
        "--model-entry",
        action="append",
        default=[],
        metavar="SLUG=MODEL=REVISION",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--suite-manifest", type=Path, default=None)
    parser.add_argument("--dataset-revision", default=os.environ.get("TRACE_EVAL_DATASET_REVISION"))
    parser.add_argument(
        "--dataset-snapshot-sha256",
        default=os.environ.get("TRACE_EVAL_DATASET_SNAPSHOT"),
    )
    parser.add_argument("--code-hash", default=os.environ.get("TRACE_EVAL_CODE_HASH"))
    parser.add_argument(
        "--evaluator-provenance-sha256",
        default=os.environ.get("TRACE_EVAL_EVALUATOR_HASH"),
    )
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.campaign_root = args.campaign_root.expanduser().resolve()
    args.score_root = args.score_root.expanduser().resolve() if args.score_root else None
    args.dataset_manifest = args.dataset_manifest.expanduser().resolve()
    args.vlmeval_root = args.vlmeval_root.expanduser().resolve()
    if args.suite_manifest is None:
        args.suite_manifest = load_trace_eval_suite().path
    else:
        args.suite_manifest = args.suite_manifest.expanduser().resolve()
    try:
        report = verify_campaign(args)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"[trace-eval-verify] phase={report['phase']} "
            f"complete={report['completed_slices']}/{report['expected_slices']} "
            f"incomplete={len(report['incomplete'])}"
        )
        for record in report["incomplete"][:25]:
            print(
                "[trace-eval-verify:missing] "
                f"seed={record['seed']} model={record['model_slug']} "
                f"benchmark={record['benchmark_key']} detail={record['detail']}"
            )
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
