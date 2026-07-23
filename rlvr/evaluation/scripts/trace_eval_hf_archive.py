#!/usr/bin/env python3
"""Manage content-addressed trace_eval_v1 slices and local coverage."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from huggingface_hub import HfApi

try:
    from scripts.trace_eval_hf_archive_lib import (
        DEFAULT_REPO_ID,
        DEFAULT_REVISION,
        ArchiveDaemon,
        install_stop_signals,
        redact_secret,
        reconstruct_remote,
        verify_expected_slice_coverage,
    )
except ModuleNotFoundError:  # Supports direct ``python scripts/...`` invocation.
    from trace_eval_hf_archive_lib import (
        DEFAULT_REPO_ID,
        DEFAULT_REVISION,
        ArchiveDaemon,
        install_stop_signals,
        redact_secret,
        reconstruct_remote,
        verify_expected_slice_coverage,
    )


def _parser() -> argparse.ArgumentParser:
    evaluation_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spool-root",
        type=Path,
        default=evaluation_root / ".work" / "hf_archive",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--upload-threads", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--retry-cap-seconds", type=float, default=300.0)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create and verify the private dataset repository.")
    commands.add_parser(
        "build",
        help="Build every ready slice locally without a token or network access.",
    )
    daemon = commands.add_parser("daemon", help="Poll the spool and upload completed slices.")
    daemon.add_argument("--poll-seconds", type=float, default=30.0)
    commands.add_parser("flush", help="Build and upload every currently ready slice.")
    verify = commands.add_parser("verify", help="Verify local completeness and remote content hashes.")
    verify.add_argument("--expect-run-id")
    verify.add_argument("--expect-suite", action="store_true")
    verify.add_argument("--expect-benchmark", action="append", default=[])
    verify.add_argument("--expect-campaign-config-hash")
    verify.add_argument("--expect-dataset-revision")
    verify.add_argument("--expect-model-slug", action="append", default=[])
    verify.add_argument("--expect-seed", action="append", type=int, default=[])
    verify.add_argument(
        "--expect-stage",
        action="append",
        choices=("generation", "extraction", "score"),
        default=[],
    )
    coverage = commands.add_parser(
        "coverage", help="Check local trace_eval_v1 coverage without contacting Hugging Face."
    )
    coverage.add_argument("--expect-run-id", required=True)
    coverage.add_argument("--expect-suite", action="store_true")
    coverage.add_argument("--expect-benchmark", action="append", default=[])
    coverage.add_argument("--expect-model-slug", action="append", required=True)
    coverage.add_argument("--expect-seed", action="append", type=int, required=True)
    coverage.add_argument("--expect-campaign-config-hash")
    coverage.add_argument("--expect-dataset-revision")
    coverage.add_argument(
        "--expect-stage",
        action="append",
        choices=("generation", "extraction", "score"),
        default=[],
    )
    reconstruct = commands.add_parser("reconstruct", help="Rebuild long-form Parquet from remote slices.")
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--stage", choices=("generation", "extraction", "score"))
    reconstruct.add_argument("--run-id")
    reconstruct.add_argument("--model")
    reconstruct.add_argument("--seed", type=int)
    reconstruct.add_argument("--benchmark")
    return parser


def _expected_coverage(args: argparse.Namespace) -> dict[str, int]:
    if not args.expect_run_id or not args.expect_model_slug or not args.expect_seed:
        raise ValueError(
            "coverage requires --expect-run-id, --expect-model-slug, and --expect-seed"
        )
    benchmarks = list(args.expect_benchmark)
    if not benchmarks:
        try:
            from scripts.benchmark_queue_lib import TRACE_EVAL_V1_BENCHMARKS
        except ModuleNotFoundError:
            from benchmark_queue_lib import TRACE_EVAL_V1_BENCHMARKS
        benchmarks = list(TRACE_EVAL_V1_BENCHMARKS)
    return verify_expected_slice_coverage(
        args.spool_root,
        run_id=args.expect_run_id,
        model_slugs=args.expect_model_slug,
        seeds=args.expect_seed,
        benchmarks=benchmarks,
        stages=args.expect_stage or ("generation", "extraction", "score"),
        campaign_config_hash=args.expect_campaign_config_hash,
        dataset_revision=args.expect_dataset_revision,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token: str | None = None
    try:
        if args.command == "coverage":
            print(json.dumps(_expected_coverage(args), sort_keys=True))
            return 0
        if args.command == "build":
            daemon = ArchiveDaemon(
                spool_root=args.spool_root,
                repo_id=args.repo_id,
                revision=args.revision,
                token=None,
                api=object(),
                batch_size=args.batch_size,
                upload_threads=args.upload_threads,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_cap_seconds=args.retry_cap_seconds,
            )
            built = daemon.build_ready()
            failed = len(daemon.ledger.rows(("failed",)))
            print(
                json.dumps(
                    {"built": built, "failed": failed, "spool_root": str(args.spool_root)},
                    sort_keys=True,
                )
            )
            return 1 if failed else 0
        token = os.environ.get(args.token_env, "").strip()
        if not token:
            raise ValueError(f"{args.token_env} must contain a Hugging Face token")
        api = HfApi(token=token)
        if args.command == "reconstruct":
            rows = reconstruct_remote(
                api=api,
                repo_id=args.repo_id,
                revision=args.revision,
                token=token,
                output=args.output,
                stage=args.stage,
                run_id=args.run_id,
                model_slug=args.model,
                seed=args.seed,
                benchmark=args.benchmark,
            )
            print(json.dumps({"output": str(args.output), "rows": rows}, sort_keys=True))
            return 0

        daemon = ArchiveDaemon(
            spool_root=args.spool_root,
            repo_id=args.repo_id,
            revision=args.revision,
            token=token,
            api=api,
            batch_size=args.batch_size,
            upload_threads=args.upload_threads,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            retry_cap_seconds=args.retry_cap_seconds,
        )
        if args.command == "init":
            daemon.initialize_repo()
            print(json.dumps({"private": True, "repo_id": args.repo_id}, sort_keys=True))
            return 0
        if args.command == "flush":
            report = daemon.flush_all()
            print(json.dumps(report.__dict__, sort_keys=True))
            return 1 if report.failed else 0
        if args.command == "verify":
            report = daemon.verify()
            coverage_requested = bool(args.expect_suite or args.expect_benchmark)
            if not coverage_requested and (
                args.expect_campaign_config_hash or args.expect_dataset_revision
            ):
                raise ValueError(
                    "--expect-campaign-config-hash and --expect-dataset-revision "
                    "require --expect-suite or --expect-benchmark"
                )
            if coverage_requested:
                report.update(_expected_coverage(args))
            print(json.dumps(report, sort_keys=True))
            return 0
        if args.command == "daemon":
            stop_event = threading.Event()
            install_stop_signals(stop_event)
            daemon.run(poll_seconds=args.poll_seconds, stop_event=stop_event)
            return 0
    except Exception as error:
        message = redact_secret(error, token)
        print(f"archive error: {message}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
