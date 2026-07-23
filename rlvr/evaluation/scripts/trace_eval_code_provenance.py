#!/usr/bin/env python3
"""Compute the exact trace_eval_v1 generation-code contract hash."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any

from trace_eval_evaluator_provenance import (
    DEFAULT_VLMEVAL_ROOT,
    REPO_ROOT,
    evaluator_provenance_sha256,
    sha256_file,
)


SCHEMA_VERSION = "trace-eval-code-provenance-v1"
SCRIPT_PATTERNS = (
    "*benchmark*queue*.py",
    "trace_eval_*.py",
    "run_trace_eval*.py",
    "run_mme_reasoning_eval.py",
)
EXPLICIT_PATHS = (
    "rlvr/requirements-cu128.txt",
    "rlvr/evaluation/requirements-runtime.txt",
    "rlvr/evaluation/requirements.txt",
    "rlvr/evaluation/trace_eval/post_run_patches.v1.json",
    "rlvr/evaluation/trace_eval/suite.v1.json",
    "rlvr/examples/prompts/chat_template_no_think.jinja",
    "rlvr/evaluation/scripts/apply_vlmevalkit_trace_extensions.py",
    "rlvr/evaluation/scripts/prepare_trace_eval_manifest.py",
    "rlvr/evaluation/scripts/prepare_trace_eval_datasets.py",
    "rlvr/evaluation/scripts/prepare_trace_eval_models.py",
    "rlvr/evaluation/scripts/run_official_vlmevalkit_saved_score.py",
    "rlvr/evaluation/scripts/trace_eval_suite.py",
    "rlvr/evaluation/scripts/trace_eval_evaluator_provenance.py",
    "rlvr/evaluation/scripts/trace_eval_code_provenance.py",
    "rlvr/evaluation/scripts/trace_eval_score_receipts.py",
    "rlvr/evaluation/scripts/trace_eval_scoring_contract.py",
    "rlvr/evaluation/scripts/trace_benchmark_answer_parsing.py",
    "rlvr/evaluation/scripts/run_trace_eval.sh",
    "rlvr/evaluation/scripts/run_trace_eval_score_campaign.py",
    "rlvr/evaluation/scripts/setup_trace_eval_env.sh",
    "rlvr/evaluation/scripts/start_vllm_endpoint_pool.sh",
    "rlvr/evaluation/scripts/status_trace_eval.py",
    "rlvr/evaluation/scripts/stop_vllm_endpoint_pool.sh",
    "rlvr/evaluation/scripts/verify_trace_eval.py",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def trace_eval_code_manifest(
    *,
    repo_root: Path = REPO_ROOT,
    vlmeval_root: Path = DEFAULT_VLMEVAL_ROOT,
    evaluator_sha256: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    scripts_root = repo_root / "rlvr" / "evaluation" / "scripts"
    selected = {repo_root / path for path in EXPLICIT_PATHS}
    for path in scripts_root.iterdir():
        if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in SCRIPT_PATTERNS):
            selected.add(path)
    missing = sorted(str(path) for path in selected if not path.is_file())
    if missing:
        raise RuntimeError(f"trace_eval_v1 code provenance files are missing: {missing}")
    files = {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in sorted(selected)
    }
    evaluator_hash = evaluator_sha256 or evaluator_provenance_sha256(
        repo_root=repo_root,
        vlmeval_root=vlmeval_root,
    )
    if len(evaluator_hash) != 64:
        raise RuntimeError("invalid evaluator provenance SHA-256")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "evaluator_provenance_sha256": evaluator_hash,
    }
    manifest["sha256"] = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    return manifest


def trace_eval_code_sha256(
    *,
    repo_root: Path = REPO_ROOT,
    vlmeval_root: Path = DEFAULT_VLMEVAL_ROOT,
    evaluator_sha256: str | None = None,
) -> str:
    return str(
        trace_eval_code_manifest(
            repo_root=repo_root,
            vlmeval_root=vlmeval_root,
            evaluator_sha256=evaluator_sha256,
        )["sha256"]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--evaluator-sha256")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = trace_eval_code_manifest(
        repo_root=args.repo_root,
        vlmeval_root=args.vlmeval_root,
        evaluator_sha256=args.evaluator_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True) if args.json else manifest["sha256"])


if __name__ == "__main__":
    main()
