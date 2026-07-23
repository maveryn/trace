#!/usr/bin/env python3
"""Score saved trace_eval_v1 campaigns without touching generation outputs.

This is orchestration only. It stages prediction workbooks into a clean tree,
then delegates benchmark behavior to the pinned VLMEvalKit evaluator, the
shared direct scorer, or the dedicated MME-Reasoning scorer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trace_eval_media_contract import (
    GENERATION_CONTRACT_VERSION,
    MEDIA_CONTRACT_VERSION,
    MEDIA_TRANSPORT,
    QWEN_MAX_IMAGE_PIXELS,
    QWEN_MIN_IMAGE_PIXELS,
)
from trace_eval_evaluator_provenance import build_evaluator_provenance
from trace_eval_code_provenance import trace_eval_code_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = REPO_ROOT / "rlvr" / "evaluation"
DEFAULT_WORK_ROOT = EVALUATION_ROOT / ".work"
SCRIPTS_ROOT = Path(__file__).resolve().parent
DEFAULT_VLMEVAL_ROOT = REPO_ROOT / "external" / "VLMEvalKit"
DEFAULT_EVAL_DEPS = DEFAULT_WORK_ROOT / "eval_deps"
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_LMU_DATA = Path(os.environ.get("LMUData", DEFAULT_WORK_ROOT / "LMUData"))
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", DEFAULT_LMU_DATA / ".hf-cache"))
DEFAULT_JUDGE_MODEL = Path(
    os.environ.get(
        "TRACE_EVAL_JUDGE_MODEL",
        DEFAULT_WORK_ROOT / "models" / "qwen3-32b-judge",
    )
)
PINNED_VLMEVALKIT_COMMIT = "a8b12bf1c3737a33fc1de967c202f9c592b22e86"
CONTRACT_VERSION = "trace-eval-score-campaign-v1"
TRACE_EVAL_SUITE_PATH = (
    REPO_ROOT / "rlvr" / "evaluation" / "trace_eval" / "suite.v1.json"
)

# These routes are intentionally explicit. Do not derive them from the legacy
# generic-extraction registry: this campaign replaces that route with the
# pinned dataset.evaluate implementation.
OFFICIAL_SCORE_KEYS: tuple[str, ...] = ()
DIRECT_SCORE_KEYS = (
    "charxivreason",
    "tablevqabench",
    "evochart",
    "mathvision",
    "mathvista",
    "mathverse",
    "logicvista",
)
MME_SCORE_KEY = "mme_reasoning"
ALL_SCORE_KEYS = (*DIRECT_SCORE_KEYS, *OFFICIAL_SCORE_KEYS, MME_SCORE_KEY)
ACTIVE_SUITE = "trace_eval_v1"
ACTIVE_RUN_SET = "trace_eval_v1"
ACTIVE_DATASET_VIEW = "trace_eval_v1"
GENERATION_MAX_TOKENS_BY_KEY: dict[str, int] = {}
DETERMINISTIC_OFFICIAL_SCORE_KEYS = frozenset({"embspatial", "realworldqa"})


def _activate_suite(suite: str) -> None:
    global ACTIVE_SUITE, ACTIVE_RUN_SET, ACTIVE_DATASET_VIEW, CONTRACT_VERSION
    global OFFICIAL_SCORE_KEYS, DIRECT_SCORE_KEYS, ALL_SCORE_KEYS
    global GENERATION_MAX_TOKENS_BY_KEY

    if suite != "trace_eval_v1":
        raise ValueError(f"Unknown suite {suite!r}")
    from trace_eval_suite import load_trace_eval_suite

    selected = load_trace_eval_suite(TRACE_EVAL_SUITE_PATH)
    ACTIVE_SUITE = selected.suite_id
    ACTIVE_RUN_SET = selected.suite_id
    ACTIVE_DATASET_VIEW = selected.dataset_manifest_view
    CONTRACT_VERSION = "trace-eval-score-campaign-v1"
    OFFICIAL_SCORE_KEYS = selected.routes["official_vlmevalkit"]
    DIRECT_SCORE_KEYS = selected.routes["direct_score"]
    dedicated = selected.routes["dedicated_score"]
    if dedicated != (MME_SCORE_KEY,):
        raise ValueError(
            "trace_eval_v1 dedicated route must contain only MME-Reasoning"
        )
    GENERATION_MAX_TOKENS_BY_KEY = {}
    ALL_SCORE_KEYS = (*DIRECT_SCORE_KEYS, *OFFICIAL_SCORE_KEYS, MME_SCORE_KEY)


@dataclass(frozen=True)
class Campaign:
    model: str
    slug: str
    root: Path


@dataclass(frozen=True)
class Workbook:
    benchmark_key: str
    alias: str
    run_name: str
    source: Path
    staged: Path
    sha256: str
    primary: bool


@dataclass(frozen=True)
class OfficialJob:
    campaign: Campaign
    workbook: Workbook
    output_dir: Path
    judge_kwargs: dict[str, Any]
    primary_metric: str | None
    primary_value_scale: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _default_endpoints() -> list[str]:
    return [f"http://127.0.0.1:{18100 + offset}/v1" for offset in range(8)]


def _normalize_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Judge endpoint must be HTTP(S): {value!r}")
    return endpoint.rstrip("/")


def _chat_completions_url(endpoint: str) -> str:
    return _normalize_endpoint(endpoint) + "/chat/completions"


def _load_specs() -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    from benchmark_queue_lib import spec_by_key
    from trace_eval_scoring_contract import (
        DEDICATED_SCORE_KEYS as CONTRACT_DEDICATED_KEYS,
        DIRECT_SCORE_KEYS as CONTRACT_DIRECT_KEYS,
        OFFICIAL_VLMEVAL_SCORE_KEYS as CONTRACT_OFFICIAL_KEYS,
    )

    expected_direct = CONTRACT_DIRECT_KEYS
    expected_official = CONTRACT_OFFICIAL_KEYS
    expected_dedicated = CONTRACT_DEDICATED_KEYS
    if set(DIRECT_SCORE_KEYS) != set(expected_direct):
        raise RuntimeError(
            "active direct route disagrees with the pinned scoring contract"
        )
    if set(OFFICIAL_SCORE_KEYS) != set(expected_official):
        raise RuntimeError(
            "active official route disagrees with the pinned scoring contract"
        )
    if set(expected_dedicated) != {MME_SCORE_KEY}:
        raise RuntimeError("active dedicated route must contain only MME-Reasoning")

    return {key: spec_by_key(key) for key in ALL_SCORE_KEYS}


def _source_run_root(campaign: Campaign, seed: int) -> Path:
    return campaign.root / f"seed_{seed}" / "runs"


def _run_dir(root: Path, key: str, slug: str, run_name: str) -> Path:
    return root / key / slug / run_name


def _canonical_prediction_files(spec: Any, source_dir: Path) -> tuple[Path, list[Path]]:
    alias_prediction = source_dir / f"{spec.alias}_predictions.xlsx"
    if not alias_prediction.is_file():
        raise FileNotFoundError(
            f"Missing canonical prediction workbook for {spec.key}: expected {alias_prediction}"
        )
    return alias_prediction, []


def _discover_workbooks(
    campaigns: list[Campaign],
    *,
    seed: int,
    staged_run_root: Path,
    specs: dict[str, Any],
) -> dict[str, list[Workbook]]:
    result: dict[str, list[Workbook]] = {}
    failures: list[str] = []
    for campaign in campaigns:
        campaign_workbooks: list[Workbook] = []
        source_root = _source_run_root(campaign, seed)
        if not source_root.is_dir():
            failures.append(
                f"{campaign.slug}: missing generated run root {source_root}"
            )
            result[campaign.slug] = campaign_workbooks
            continue
        for key in ALL_SCORE_KEYS:
            spec = specs[key]
            source_dir = _run_dir(source_root, key, campaign.slug, spec.run_name)
            try:
                primary, extras = _canonical_prediction_files(spec, source_dir)
            except FileNotFoundError as error:
                failures.append(str(error))
                continue
            staged_dir = _run_dir(staged_run_root, key, campaign.slug, spec.run_name)
            for source in (primary, *extras):
                campaign_workbooks.append(
                    Workbook(
                        benchmark_key=key,
                        alias=spec.alias,
                        run_name=spec.run_name,
                        source=source.resolve(),
                        staged=staged_dir / source.name,
                        sha256=_sha256(source),
                        primary=source == primary,
                    )
                )
        result[campaign.slug] = campaign_workbooks
    if failures:
        preview = "\n".join(f"  - {item}" for item in failures)
        raise FileNotFoundError(
            f"trace_eval_v1 prediction preflight failed:\n{preview}"
        )
    return result


def _primary_workbook(
    workbooks: dict[str, list[Workbook]], slug: str, key: str
) -> Workbook:
    matches = [
        item for item in workbooks[slug] if item.benchmark_key == key and item.primary
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one primary workbook for {slug}/{key}, found {len(matches)}"
        )
    return matches[0]


def _validate_media_generation_contract(generation: dict[str, Any]) -> None:
    expected = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        "media_transport": MEDIA_TRANSPORT,
        "min_image_pixels": QWEN_MIN_IMAGE_PIXELS,
        "max_image_pixels": QWEN_MAX_IMAGE_PIXELS,
    }
    for field, value in expected.items():
        if generation.get(field) != value:
            raise ValueError(
                f"generation.{field}={generation.get(field)!r}, expected {value!r}"
            )


def _validate_generation_inputs(
    campaigns: list[Campaign],
    workbooks: dict[str, list[Workbook]],
    *,
    seed: int,
    expected_dataset_snapshot: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    import pandas as pd

    validated: dict[str, dict[str, dict[str, Any]]] = {}
    failures: list[str] = []
    expected_generation_base = {
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": -1,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": seed,
    }
    for campaign in campaigns:
        campaign_inputs: dict[str, dict[str, Any]] = {}
        snapshots: set[str] = set()
        for key in ALL_SCORE_KEYS:
            workbook = _primary_workbook(workbooks, campaign.slug, key)
            summary_path = workbook.source.parent / "generation_summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows = int(summary["rows"])
                expected_rows = int(summary["expected_rows"])
                if rows <= 0 or rows != expected_rows:
                    raise ValueError(f"rows={rows} expected_rows={expected_rows}")
                if str(summary.get("model")) != campaign.model:
                    raise ValueError(f"model={summary.get('model')!r}")
                if str(summary.get("model_slug")) != campaign.slug:
                    raise ValueError(f"model_slug={summary.get('model_slug')!r}")

                generation = summary.get("generation") or {}
                expected_generation = {
                    **expected_generation_base,
                    "max_tokens": GENERATION_MAX_TOKENS_BY_KEY.get(key, 4096),
                }
                for field, expected in expected_generation.items():
                    if generation.get(field) != expected:
                        raise ValueError(
                            f"generation.{field}={generation.get(field)!r}, expected {expected!r}"
                        )
                if generation.get("api_model") != campaign.slug:
                    raise ValueError(
                        f"generation.api_model={generation.get('api_model')!r}"
                    )
                _validate_media_generation_contract(generation)
                snapshot = str(generation.get("dataset_snapshot_sha256") or "")
                if len(snapshot) != 64:
                    raise ValueError(f"invalid dataset snapshot {snapshot!r}")
                if snapshot != expected_dataset_snapshot:
                    raise ValueError(
                        f"dataset snapshot {snapshot!r} does not match active "
                        f"{ACTIVE_DATASET_VIEW} manifest snapshot {expected_dataset_snapshot!r}"
                    )
                snapshots.add(snapshot)

                finish_reason = summary.get("finish_reason") or {}
                if not isinstance(finish_reason, dict) or set(finish_reason) - {
                    "stop",
                    "length",
                }:
                    raise ValueError(f"unexpected finish reasons {finish_reason!r}")
                if sum(int(value) for value in finish_reason.values()) != rows:
                    raise ValueError(
                        f"finish reasons do not cover all {rows} rows: {finish_reason!r}"
                    )

                workbook_rows = len(pd.read_excel(workbook.source, usecols=[0]))
                if workbook_rows != rows:
                    raise ValueError(
                        f"workbook rows={workbook_rows}, summary rows={rows}"
                    )
                artifact = Path(
                    str((summary.get("artifacts") or {}).get("eval_file", ""))
                )
                if artifact.resolve() != workbook.source:
                    raise ValueError(
                        f"summary eval_file={artifact}, workbook={workbook.source}"
                    )
            except Exception as error:
                failures.append(f"{campaign.slug}/{key}: {error}")
                continue

            campaign_inputs[key] = {
                "rows": rows,
                "generation_summary": str(summary_path),
                "generation_summary_sha256": _sha256(summary_path),
                "dataset_snapshot_sha256": snapshot,
                "finish_reason": finish_reason,
            }
        if len(snapshots) > 1:
            failures.append(
                f"{campaign.slug}: inconsistent dataset snapshots across generation summaries: "
                f"{sorted(snapshots)}"
            )
        validated[campaign.slug] = campaign_inputs
    if failures:
        raise RuntimeError(
            "Generation input preflight failed:\n  - " + "\n  - ".join(failures)
        )
    return validated


def _validate_dataset_manifest(path: Path, lmu_data: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("vlmevalkit_commit") != PINNED_VLMEVALKIT_COMMIT:
        raise RuntimeError(
            "Dataset manifest VLMEvalKit commit mismatch: "
            f"{manifest.get('vlmevalkit_commit')!r}"
        )
    if int(manifest.get("failed", -1)) != 0 or int(manifest.get("ready", -1)) < len(
        ALL_SCORE_KEYS
    ):
        raise RuntimeError(
            f"Dataset manifest is not ready for {ACTIVE_DATASET_VIEW}: ready={manifest.get('ready')} "
            f"failed={manifest.get('failed')}"
        )
    view = manifest.get("dataset_views", {}).get(ACTIVE_DATASET_VIEW)
    expected_keys = set(ALL_SCORE_KEYS)
    view_matches = isinstance(view, list) and set(view) == expected_keys
    if not view_matches:
        raise RuntimeError(
            f"Dataset manifest {ACTIVE_DATASET_VIEW} view does not match the scoring route"
        )
    datasets = manifest.get("datasets", {})
    not_ready = [
        key for key in ALL_SCORE_KEYS if datasets.get(key, {}).get("status") != "ready"
    ]
    if not_ready:
        raise RuntimeError(f"Dataset receipts are not ready: {not_ready}")
    view_snapshot = str(
        (manifest.get("view_snapshot_sha256") or {}).get(ACTIVE_DATASET_VIEW) or ""
    )
    if len(view_snapshot) != 64:
        raise RuntimeError(
            f"Dataset manifest has no valid {ACTIVE_DATASET_VIEW} view snapshot: "
            f"{view_snapshot!r}"
        )
    recorded_root = Path(str(manifest.get("lmu_data_root", ""))).resolve()
    if recorded_root != lmu_data.resolve():
        raise RuntimeError(
            f"Dataset manifest uses LMUData={recorded_root}, requested {lmu_data.resolve()}"
        )
    return manifest


def _base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if not args.emit_archive:
        for key in list(env):
            if key.startswith("TRACE_EVAL_HF_"):
                env.pop(key, None)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(key, None)

    required_python_paths = [
        args.eval_deps,
        REPO_ROOT,
        SCRIPTS_ROOT,
        args.vlmeval_root,
        args.vlmeval_root / "scripts",
    ]
    previous = [item for item in env.get("PYTHONPATH", "").split(os.pathsep) if item]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in required_python_paths] + previous
    )
    env["LMUData"] = str(args.lmu_data)
    env["HF_HOME"] = str(args.hf_home)
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Pinned VLMEvalKit enables its OpenAI-compatible judge path only for
    # syntactically valid ``sk-`` keys. The endpoint is still localhost-only.
    env["OPENAI_API_KEY"] = "sk-local"
    env["LOCAL_LLM"] = args.judge_api_model
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _validate_runtime(args: argparse.Namespace, env: dict[str, str]) -> None:
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"Evaluation Python is not executable: {args.python}")
    if not args.eval_deps.is_dir():
        raise FileNotFoundError(
            f"Evaluation dependency target is missing: {args.eval_deps}"
        )
    if not args.vlmeval_root.is_dir():
        raise FileNotFoundError(f"VLMEvalKit checkout is missing: {args.vlmeval_root}")
    commit = _git(args.vlmeval_root, "rev-parse", "HEAD")
    if commit != PINNED_VLMEVALKIT_COMMIT:
        raise RuntimeError(
            f"VLMEvalKit must be pinned to {PINNED_VLMEVALKIT_COMMIT}, found {commit}"
        )
    if not args.lmu_data.is_dir():
        raise FileNotFoundError(f"LMUData root is missing: {args.lmu_data}")
    if not args.hf_home.is_dir():
        raise FileNotFoundError(f"HF_HOME cache is missing: {args.hf_home}")
    if not args.judge_model.is_dir():
        raise FileNotFoundError(f"Qwen3 judge model is missing: {args.judge_model}")
    probe = subprocess.run(
        [
            str(args.python),
            "-c",
            "import pandas, tabulate; import vlmeval",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        detail = (probe.stderr or probe.stdout).strip()
        raise RuntimeError(f"Evaluation dependency preflight failed: {detail}")


def _judge_kwargs(args: argparse.Namespace, key: str) -> dict[str, Any]:
    if key in DETERMINISTIC_OFFICIAL_SCORE_KEYS:
        return {"model": "exact_matching", "nproc": args.eval_nproc}
    kwargs: dict[str, Any] = {
        "model": args.judge_api_model,
        "nproc": args.eval_nproc,
        "temperature": 0,
        "max_tokens": args.judge_max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # Pinned PhyX requires an explicit official validation mode. This is the
    # multiple-choice benchmark, so use its string-level official scorer.
    if key == "phyx_mini_mc":
        kwargs["valid_type"] = "STR"
    return kwargs


def _primary_metric_contract(key: str) -> tuple[str | None, str]:
    # Most datasets expose one primary metric through report_primary_metric.
    # Explicit exceptions are kept here rather than changing evaluator output.
    if key == "wemath":
        return "Score (Strict)", "percent"
    return None, "auto"


def _contract(
    args: argparse.Namespace,
    campaigns: list[Campaign],
    workbooks: dict[str, list[Workbook]],
    generation_inputs: dict[str, dict[str, dict[str, Any]]],
    dataset_manifest: dict[str, Any],
    endpoints: list[str],
) -> dict[str, Any]:
    sources = []
    for campaign in campaigns:
        for workbook in workbooks[campaign.slug]:
            sources.append(
                {
                    "model": campaign.model,
                    "model_slug": campaign.slug,
                    "campaign_root": str(campaign.root),
                    "benchmark_key": workbook.benchmark_key,
                    "alias": workbook.alias,
                    "run_name": workbook.run_name,
                    "source": str(workbook.source),
                    "staged": str(workbook.staged),
                    "sha256": workbook.sha256,
                    "primary": workbook.primary,
                }
            )
    scoring_paths = [
        Path(__file__).resolve(),
        SCRIPTS_ROOT / "run_official_vlmevalkit_saved_score.py",
        SCRIPTS_ROOT / "run_external_benchmark_score_queue.py",
        SCRIPTS_ROOT / "run_mme_reasoning_eval.py",
        SCRIPTS_ROOT / "trace_eval_scoring_contract.py",
    ]
    selection: dict[str, Any] | None = None
    if ACTIVE_SUITE == "trace_eval_v1":
        from trace_eval_suite import load_trace_eval_suite

        suite = load_trace_eval_suite(TRACE_EVAL_SUITE_PATH)
        scoring_paths.extend(
            (
                SCRIPTS_ROOT / "trace_eval_suite.py",
                SCRIPTS_ROOT / "trace_eval_evaluator_provenance.py",
                SCRIPTS_ROOT / "trace_eval_score_receipts.py",
                SCRIPTS_ROOT / "run_trace_eval_score_campaign.py",
                suite.path,
            )
        )
        selection = {
            "path": str(suite.path),
            "sha256": suite.manifest_sha256,
            "benchmarks": list(suite.benchmark_keys),
        }

    evaluator_provenance = build_evaluator_provenance(
        repo_root=REPO_ROOT,
        vlmeval_root=args.vlmeval_root,
    )
    trace_code_provenance = (
        trace_eval_code_manifest(
            repo_root=REPO_ROOT,
            vlmeval_root=args.vlmeval_root,
            evaluator_sha256=evaluator_provenance["sha256"],
        )
        if ACTIVE_SUITE == "trace_eval_v1"
        else None
    )

    contract = {
        "contract_version": CONTRACT_VERSION,
        "suite": ACTIVE_SUITE,
        "selection": selection,
        "dataset_view": ACTIVE_DATASET_VIEW,
        "seed": args.seed,
        "routes": {
            "official_vlmevalkit": list(OFFICIAL_SCORE_KEYS),
            "direct": list(DIRECT_SCORE_KEYS),
            "mme_reasoning": [MME_SCORE_KEY],
        },
        "campaigns": [
            {
                "model": item.model,
                "model_slug": item.slug,
                "campaign_root": str(item.root),
            }
            for item in campaigns
        ],
        "sources": sources,
        "generation_inputs": generation_inputs,
        "dataset_manifest": {
            "path": str(args.dataset_manifest),
            "sha256": _sha256(args.dataset_manifest),
            "dataset_snapshot_sha256": dataset_manifest.get("dataset_snapshot_sha256"),
            "dataset_view": ACTIVE_DATASET_VIEW,
            "view_snapshot_sha256": dataset_manifest.get(
                "view_snapshot_sha256", {}
            ).get(ACTIVE_DATASET_VIEW),
            "vlmevalkit_commit": dataset_manifest.get("vlmevalkit_commit"),
        },
        "evaluator_provenance": {
            "schema_version": evaluator_provenance["schema_version"],
            "sha256": evaluator_provenance["sha256"],
            "vlmevalkit_git_head": evaluator_provenance["vlmevalkit"]["git_head"],
            "vlmevalkit_file_records": len(evaluator_provenance["vlmevalkit"]["files"]),
            "trace_extension_file_records": len(
                evaluator_provenance["trace_extensions"]["files"]
            ),
        },
        "trace_eval_code_provenance": (
            {
                "schema_version": trace_code_provenance["schema_version"],
                "sha256": trace_code_provenance["sha256"],
                "evaluator_provenance_sha256": trace_code_provenance[
                    "evaluator_provenance_sha256"
                ],
            }
            if trace_code_provenance is not None
            else None
        ),
        "judge": {
            "api_model": args.judge_api_model,
            "model_path": str(args.judge_model),
            "endpoints": endpoints,
            "kwargs_by_benchmark": {
                key: _judge_kwargs(args, key) for key in OFFICIAL_SCORE_KEYS
            },
            "api_parallelism": args.judge_api_parallelism,
            "api_batch_size": args.judge_api_batch_size,
            "api_batches_per_endpoint": args.judge_api_batches_per_endpoint,
            "api_max_batch_chars": args.judge_api_max_batch_chars,
            "cache_contract_version": args.judge_cache_contract_version,
        },
        "scoring_implementation": (
            trace_code_provenance["files"]
            if trace_code_provenance is not None
            else {
                str(path.relative_to(REPO_ROOT)): _sha256(path)
                for path in scoring_paths
            }
        ),
        "runtime": {
            "python": str(args.python),
            "eval_deps": str(args.eval_deps),
            "vlmeval_root": str(args.vlmeval_root),
            "lmu_data": str(args.lmu_data),
            "hf_home": str(args.hf_home),
        },
    }
    return contract


def _prepare_score_root(
    score_root: Path,
    contract: dict[str, Any],
    resume: bool,
    *,
    shared_seed_root: bool = False,
) -> str:
    seed = int(contract["seed"])
    manifest_name = (
        f"score_campaign_manifest_seed_{seed}.json"
        if shared_seed_root
        else "score_campaign_manifest.json"
    )
    manifest_path = score_root / manifest_name
    if shared_seed_root:
        score_root.mkdir(parents=True, exist_ok=True)
        if manifest_path.is_file():
            if not resume:
                raise FileExistsError(
                    f"Score contract already exists: {manifest_path}; pass --resume"
                )
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("contract") != contract:
                raise RuntimeError(
                    f"Existing seed-{seed} score manifest does not match source hashes/judge contract"
                )
            return str(existing["contract_sha256"])
        seed_root = score_root / f"seed_{seed}"
        if seed_root.exists() and any(seed_root.iterdir()):
            raise RuntimeError(
                f"Cannot create a seed-{seed} contract over existing outputs without {manifest_path}"
            )
    elif score_root.exists() and any(score_root.iterdir()):
        if not resume:
            raise FileExistsError(
                f"Score root is not empty: {score_root}; pass --resume only for this exact contract"
            )
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume without {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract") != contract:
            raise RuntimeError(
                "Existing score root manifest does not match source hashes/judge contract"
            )
        return str(existing["contract_sha256"])

    score_root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    contract_sha256 = hashlib.sha256(encoded).hexdigest()
    _write_json(
        manifest_path,
        {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "contract_sha256": contract_sha256,
            "contract": contract,
        },
    )
    return contract_sha256


def _stage_workbooks(workbooks: dict[str, list[Workbook]], *, resume: bool) -> None:
    for items in workbooks.values():
        for workbook in items:
            workbook.staged.parent.mkdir(parents=True, exist_ok=True)
            if workbook.staged.exists():
                if not resume:
                    raise FileExistsError(workbook.staged)
                if _sha256(workbook.staged) != workbook.sha256:
                    shutil.copy2(workbook.source, workbook.staged)
                    if _sha256(workbook.staged) != workbook.sha256:
                        raise RuntimeError(
                            f"Restaged workbook hash mismatch: {workbook.staged}"
                        )
                    print(f"[stage:restore] {workbook.staged}", flush=True)
                continue
            shutil.copy2(workbook.source, workbook.staged)
            if _sha256(workbook.staged) != workbook.sha256:
                raise RuntimeError(f"Staged workbook hash mismatch: {workbook.staged}")


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[score:start] {label} log={log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{dt.datetime.now(dt.timezone.utc).isoformat()}] {_command_text(command)}\n"
        )
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; see {log_path}")
    print(f"[score:done] {label}", flush=True)


def _archive_value(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except Exception:
        pass
    return value


def _archive_identity(value: Any) -> str | None:
    value = _archive_value(value)
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _archive_score_value(value: Any) -> float | None:
    value = _archive_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "correct"}:
            return 1.0
        if normalized in {"false", "incorrect"}:
            return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _archive_normalized_extraction(
    adapted_row: Any,
    prediction: Any,
    fallback_method: str,
) -> dict[str, Any]:
    status = _archive_identity(adapted_row.get("trace_extraction_status"))
    if status in {"resolved", "ambiguous", "invalid"}:
        method = (
            _archive_identity(adapted_row.get("trace_extraction_method"))
            or fallback_method
        )
        candidates: Any = []
        raw_candidates = _archive_value(adapted_row.get("trace_extraction_candidates"))
        if isinstance(raw_candidates, str):
            try:
                candidates = json.loads(raw_candidates)
            except json.JSONDecodeError:
                candidates = []
        elif isinstance(raw_candidates, (list, tuple)):
            candidates = list(raw_candidates)
        return {
            "status": status,
            "value": prediction if status == "resolved" else None,
            "method": method,
            "candidates": candidates,
        }

    present = bool(_archive_identity(prediction))
    return {
        "status": "resolved" if present else "invalid",
        "value": prediction if present else None,
        "method": fallback_method,
    }


def _official_archive_row_scores(
    summary: dict[str, Any],
    source: Any,
) -> dict[int, dict[str, Any]]:
    import pandas as pd

    score_columns = ("eval_score", "hit", "correct", "score", "match")
    identity_columns = ("source_row_hash", "request_hash", "source_ordinal", "index")
    source_identities = {
        column: [_archive_identity(value) for value in source[column].tolist()]
        for column in identity_columns
        if column in source and source[column].notna().all()
    }
    best: dict[int, dict[str, Any]] = {}
    for value in (summary.get("artifacts") or {}).get("official_outputs", []):
        path = Path(str(value))
        if not path.is_file() or path.suffix.lower() not in {".xlsx", ".pkl"}:
            continue
        try:
            table = (
                pd.read_excel(path)
                if path.suffix.lower() == ".xlsx"
                else pd.read_pickle(path)
            )
        except Exception:
            continue
        if not isinstance(table, pd.DataFrame):
            continue
        score_column = next(
            (column for column in score_columns if column in table), None
        )
        if score_column is None:
            continue
        for identity_column, source_values in source_identities.items():
            if identity_column not in table:
                continue
            source_lookup = {
                identity: ordinal
                for ordinal, identity in enumerate(source_values)
                if identity is not None
            }
            candidate_values = [
                _archive_identity(item) for item in table[identity_column].tolist()
            ]
            if len(source_lookup) != len(source_values) or len(
                set(candidate_values)
            ) != len(candidate_values):
                continue
            matched: dict[int, dict[str, Any]] = {}
            for candidate_identity, score_value in zip(
                candidate_values, table[score_column].tolist()
            ):
                if candidate_identity not in source_lookup:
                    continue
                score = _archive_score_value(score_value)
                if score is None:
                    continue
                matched[source_lookup[candidate_identity]] = {
                    "score": score,
                    "artifact": str(path.resolve()),
                    "score_column": score_column,
                    "identity_column": identity_column,
                }
            if len(matched) > len(best):
                best = matched
            if len(best) == len(source):
                return best
    return best


def _official_complete(job: OfficialJob) -> bool:
    scores_path = job.output_dir / "scores.json"
    if not scores_path.is_file():
        return False
    summary = json.loads(scores_path.read_text(encoding="utf-8"))
    provenance = summary.get("provenance", {})
    source_sha256 = provenance.get(
        "source_prediction_sha256", provenance.get("prediction_sha256")
    )
    if source_sha256 != job.workbook.sha256:
        raise RuntimeError(f"Official score source hash mismatch: {scores_path}")
    if provenance.get("judge_kwargs") != job.judge_kwargs:
        raise RuntimeError(f"Official score judge contract mismatch: {scores_path}")
    return True


def _archive_official_job(args: argparse.Namespace, job: OfficialJob) -> None:
    if not args.emit_archive:
        return
    scores_path = job.output_dir / "scores.json"
    summary = json.loads(scores_path.read_text(encoding="utf-8"))
    prediction_path = Path(summary["artifacts"]["prediction_table"])

    import pandas as pd
    from trace_eval_archive_hooks import (
        emit_extraction_slice,
        emit_score_slice,
        resolve_model_revision,
        resolve_model_source,
    )

    source = pd.read_excel(
        job.workbook.staged,
        converters={"request_hash": str, "source_row_hash": str},
    )
    adapted = pd.read_excel(prediction_path)
    if len(source) != len(adapted):
        raise RuntimeError(
            f"official archive row mismatch for {job.workbook.benchmark_key}: "
            f"source={len(source)} adapted={len(adapted)}"
        )
    adapter = (summary.get("provenance") or {}).get("prediction_adapter") or {}
    method = str(adapter.get("contract") or "pinned_vlmevalkit_dataset_evaluate_input")
    official_row_scores = _official_archive_row_scores(summary, source)
    extraction_records: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    for ordinal, ((_, source_row), (_, adapted_row)) in enumerate(
        zip(source.iterrows(), adapted.iterrows())
    ):
        raw_response = source_row.get(
            "raw_prediction", source_row.get("prediction", "")
        )
        prediction = adapted_row.get("prediction", "")
        source_index = str(source_row.get("index", ordinal))
        source_row_hash = str(source_row.get("source_row_hash") or "")
        if not source_row_hash:
            material = {
                "index": source_index,
                "question": source_row.get("question"),
                "answer": source_row.get("answer"),
                "options": {
                    key: source_row.get(key)
                    for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    if key in source and not pd.isna(source_row.get(key))
                },
            }
            source_row_hash = hashlib.sha256(
                json.dumps(material, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        request_hash = str(source_row.get("request_hash") or "")
        if not request_hash:
            request_hash = hashlib.sha256(
                f"{source_row_hash}\0{raw_response}".encode("utf-8")
            ).hexdigest()
        common = {
            "source_index": source_index,
            "source_ordinal": ordinal,
            "source_row_hash": source_row_hash,
            "request_hash": request_hash,
            "question": source_row.get("question"),
            "ground_truth": source_row.get("answer"),
            "options": {
                key: source_row.get(key)
                for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                if key in source and not pd.isna(source_row.get(key))
            },
        }
        extraction_records.append(
            {
                **common,
                "model_response": raw_response,
                "judge_prompt": "",
                "judge_response": "",
                "normalized_extraction": _archive_normalized_extraction(
                    adapted_row, prediction, method
                ),
                "retries": {"count": 0},
            }
        )
        score_receipt = official_row_scores.get(ordinal)
        row_score = score_receipt["score"] if score_receipt is not None else None
        if row_score is None:
            for candidate in ("eval_score", "hit", "correct", "score", "match"):
                row_score = _archive_score_value(adapted_row.get(candidate))
                if row_score is not None:
                    score_receipt = {
                        "artifact": str(prediction_path.resolve()),
                        "score_column": candidate,
                        "identity_column": "row_order",
                    }
                    break
        if row_score is not None:
            score_records.append(
                {
                    **common,
                    "metadata": {
                        "official_score_artifact": score_receipt["artifact"],
                        "official_score_column": score_receipt["score_column"],
                        "official_score_identity": score_receipt["identity_column"],
                    },
                    "prediction": prediction,
                    "score": row_score,
                    "scorer": "pinned_vlmevalkit.dataset.evaluate",
                    "excluded": False,
                }
            )

    identity = {
        "model": resolve_model_source(job.campaign.slug, job.campaign.model),
        "model_slug": job.campaign.slug,
        "model_revision": resolve_model_revision(job.campaign.slug, job.campaign.model),
        "seed": int(args.seed),
        "benchmark": job.workbook.benchmark_key,
        "dataset_alias": job.workbook.alias,
        "dataset_split": "default",
        "dataset_revision": os.environ.get("TRACE_EVAL_DATASET_REVISION", "unknown"),
    }
    aggregate = {
        "rows": int(summary["rows"]),
        "score": float(summary["score"]),
        "primary_metric": summary.get("primary_metric"),
        "contract": (summary.get("provenance") or {}).get("contract"),
    }
    if len(score_records) != len(source):
        score_records = [
            {
                "source_index": "__aggregate__",
                "source_ordinal": 0,
                "source_row_hash": job.workbook.sha256,
                "request_hash": _sha256(scores_path),
                "question": None,
                "ground_truth": None,
                "metadata": {
                    "scope": "aggregate",
                    "row_level_scores_available": False,
                    "evaluated_rows": int(summary["rows"]),
                },
                "prediction": None,
                "score": float(summary["score"]),
                "scorer": "pinned_vlmevalkit.dataset.evaluate.aggregate",
                "excluded": False,
            }
        ]
    emit_extraction_slice(
        records=extraction_records,
        contract_version="trace-eval-v1-official-extraction-v1",
        aggregate=aggregate,
        **identity,
    )
    emit_score_slice(
        records=score_records,
        contract_version="trace-eval-v1-official-score-v1",
        aggregate=aggregate,
        **identity,
    )


def _validate_score_outputs(
    campaigns: list[Campaign],
    keys: tuple[str, ...],
    *,
    benchmark_root: Path,
    specs: dict[str, Any],
    generation_inputs: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    failures: list[str] = []
    for campaign in campaigns:
        for key in keys:
            path = (
                _run_dir(
                    benchmark_root,
                    key,
                    campaign.slug,
                    specs[key].run_name,
                )
                / "scores.json"
            )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                score_key = "accuracy" if key == MME_SCORE_KEY else "score"
                score = float(payload[score_key])
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score {score!r}")
                rows = int(payload["rows"])
                expected_rows = int(generation_inputs[campaign.slug][key]["rows"])
                if rows != expected_rows:
                    raise ValueError(f"rows={rows}, expected {expected_rows}")
                if (
                    payload.get("model") is not None
                    and str(payload["model"]) != campaign.model
                ):
                    raise ValueError(f"model={payload.get('model')!r}")
                if (
                    payload.get("model_slug") is not None
                    and str(payload["model_slug"]) != campaign.slug
                ):
                    raise ValueError(f"model_slug={payload.get('model_slug')!r}")
            except Exception as error:
                failures.append(f"{campaign.slug}/{key}: {path}: {error}")
                continue
            completed.append(
                {
                    "benchmark_key": key,
                    "model": campaign.model,
                    "model_slug": campaign.slug,
                    "rows": rows,
                    "score": score,
                    "scores_path": str(path),
                    "scores_sha256": _sha256(path),
                }
            )
    if failures:
        raise RuntimeError(
            "Score output validation failed:\n  - " + "\n  - ".join(failures)
        )
    return completed


def _official_command(args: argparse.Namespace, job: OfficialJob) -> list[str]:
    command = [
        str(args.python),
        str(SCRIPTS_ROOT / "run_official_vlmevalkit_saved_score.py"),
        "--benchmark-key",
        job.workbook.benchmark_key,
        "--prediction-xlsx",
        str(job.workbook.staged),
        "--output-dir",
        str(job.output_dir),
        "--model",
        job.campaign.model,
        "--model-slug",
        job.campaign.slug,
        "--run-name",
        job.workbook.run_name,
        "--judge-kwargs-json",
        json.dumps(job.judge_kwargs, sort_keys=True, separators=(",", ":")),
        "--primary-value-scale",
        job.primary_value_scale,
        "--vlmeval-root",
        str(args.vlmeval_root),
    ]
    if job.primary_metric:
        command.extend(["--primary-metric", job.primary_metric])
    return command


def _direct_command(
    args: argparse.Namespace,
    campaign: Campaign,
    benchmark_key: str,
    *,
    staged_run_root: Path,
    benchmark_root: Path,
    queue_root: Path,
    endpoints: list[str],
    contract_sha256: str,
) -> list[str]:
    command = [
        str(args.python),
        str(SCRIPTS_ROOT / "run_external_benchmark_score_queue.py"),
        "--model",
        campaign.model,
        "--model-slug",
        campaign.slug,
        "--run-set",
        ACTIVE_RUN_SET,
        "--seed",
        str(args.seed),
        "--run-root",
        str(staged_run_root),
        "--benchmark-root",
        str(benchmark_root),
        "--queue-root",
        str(queue_root),
        "--queue-name",
        f"official-{ACTIVE_SUITE}-{campaign.slug}-{benchmark_key}-{contract_sha256[:16]}",
        "--worker-id",
        f"official-{ACTIVE_SUITE}-{campaign.slug}-{benchmark_key}-direct",
        "--only",
        benchmark_key,
        "--eval-judge-model",
        "exact_matching",
        "--eval-nproc",
        str(args.eval_nproc),
        "--judge-model",
        str(args.judge_model),
        "--judge-api-model",
        args.judge_api_model,
        "--judge-api-tokenizer-model",
        str(args.judge_model),
        "--judge-api-parallelism",
        str(args.judge_api_parallelism),
        "--judge-api-batch-size",
        str(args.judge_api_batch_size),
        "--judge-api-batches-per-endpoint",
        str(args.judge_api_batches_per_endpoint),
        "--judge-api-max-batch-chars",
        str(args.judge_api_max_batch_chars),
        "--judge-cache-contract-version",
        args.judge_cache_contract_version,
        "--stop-on-error",
    ]
    for endpoint in endpoints:
        command.extend(["--judge-api-base", endpoint])
    return command


def _mme_command(
    args: argparse.Namespace,
    campaign: Campaign,
    *,
    staged_run_root: Path,
    benchmark_root: Path,
    endpoints: list[str],
) -> list[str]:
    command = [
        str(args.python),
        str(SCRIPTS_ROOT / "run_mme_reasoning_eval.py"),
        "score",
        "--model",
        campaign.model,
        "--model-slug",
        campaign.slug,
        "--seed",
        str(args.seed),
        "--run-root",
        str(staged_run_root),
        "--benchmark-root",
        str(benchmark_root),
        "--judge-model",
        str(args.judge_model),
        "--judge-api-model",
        args.judge_api_model,
        "--judge-api-tokenizer-model",
        str(args.judge_model),
        "--judge-api-parallelism",
        str(args.judge_api_parallelism),
        "--judge-api-batch-size",
        str(args.judge_api_batch_size),
        "--judge-api-batches-per-endpoint",
        str(args.judge_api_batches_per_endpoint),
        "--judge-api-max-batch-chars",
        str(args.judge_api_max_batch_chars),
        "--judge-cache-contract-version",
        args.judge_cache_contract_version,
    ]
    for endpoint in endpoints:
        command.extend(["--judge-api-base", endpoint])
    return command


def _run_direct_phase(
    args: argparse.Namespace,
    campaigns: list[Campaign],
    *,
    staged_run_root: Path,
    benchmark_root: Path,
    queue_root: Path,
    endpoints: list[str],
    contract_sha256: str,
    env: dict[str, str],
    log_root: Path,
    specs: dict[str, Any],
    generation_inputs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    jobs = [(campaign, key) for campaign in campaigns for key in DIRECT_SCORE_KEYS]
    endpoint_pool: queue.Queue[str] = queue.Queue()
    for endpoint in endpoints:
        endpoint_pool.put(endpoint)

    def execute(job: tuple[Campaign, str]) -> None:
        campaign, key = job
        score_file = (
            _run_dir(benchmark_root, key, campaign.slug, specs[key].run_name)
            / "scores.json"
        )
        if args.resume and score_file.is_file():
            try:
                _validate_score_outputs(
                    [campaign],
                    (key,),
                    benchmark_root=benchmark_root,
                    specs=specs,
                    generation_inputs=generation_inputs,
                )
            except RuntimeError:
                print(f"[score:rerun] direct/{campaign.slug}/{key}", flush=True)
            else:
                print(f"[score:skip] direct/{campaign.slug}/{key}", flush=True)
                return
        endpoint = endpoint_pool.get()
        try:
            command = _direct_command(
                args,
                campaign,
                key,
                staged_run_root=staged_run_root,
                benchmark_root=benchmark_root,
                queue_root=queue_root,
                endpoints=[endpoint],
                contract_sha256=contract_sha256,
            )
            _run_logged(
                command,
                env=env,
                log_path=log_root / "direct" / campaign.slug / f"{key}.log",
                label=f"direct/{campaign.slug}/{key}",
            )
            _validate_score_outputs(
                [campaign],
                (key,),
                benchmark_root=benchmark_root,
                specs=specs,
                generation_inputs=generation_inputs,
            )
        finally:
            endpoint_pool.put(endpoint)

    errors: list[str] = []
    workers = min(args.direct_workers, len(endpoints), len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            campaign, key = futures[future]
            try:
                future.result()
            except Exception as error:
                errors.append(f"{campaign.slug}/{key}: {error}")
    if errors:
        raise RuntimeError("Direct scoring failures:\n  - " + "\n  - ".join(errors))


def _partition_mme_endpoints(endpoints: list[str], workers: int) -> list[list[str]]:
    if workers < 1:
        return []
    if not endpoints:
        raise ValueError("MME-Reasoning scoring requires at least one judge endpoint")
    group_count = min(workers, len(endpoints))
    groups = [[] for _ in range(group_count)]
    for offset, endpoint in enumerate(endpoints):
        groups[offset % group_count].append(endpoint)
    return groups


def _run_mme_phase(
    args: argparse.Namespace,
    campaigns: list[Campaign],
    *,
    staged_run_root: Path,
    benchmark_root: Path,
    endpoints: list[str],
    env: dict[str, str],
    log_root: Path,
    specs: dict[str, Any],
    generation_inputs: dict[str, dict[str, dict[str, Any]]],
) -> None:
    pending: list[Campaign] = []
    for campaign in campaigns:
        score_file = (
            _run_dir(
                benchmark_root,
                MME_SCORE_KEY,
                campaign.slug,
                specs[MME_SCORE_KEY].run_name,
            )
            / "scores.json"
        )
        if args.resume and score_file.is_file():
            try:
                _validate_score_outputs(
                    [campaign],
                    (MME_SCORE_KEY,),
                    benchmark_root=benchmark_root,
                    specs=specs,
                    generation_inputs=generation_inputs,
                )
            except RuntimeError:
                print(f"[score:rerun] mme/{campaign.slug}", flush=True)
            else:
                print(f"[score:skip] mme/{campaign.slug}", flush=True)
                continue
        pending.append(campaign)
    if not pending:
        return

    workers = min(args.mme_workers, len(endpoints), len(pending))
    endpoint_pool: queue.Queue[list[str]] = queue.Queue()
    for group in _partition_mme_endpoints(endpoints, workers):
        endpoint_pool.put(group)

    def execute(campaign: Campaign) -> None:
        endpoint_group = endpoint_pool.get()
        try:
            _run_logged(
                _mme_command(
                    args,
                    campaign,
                    staged_run_root=staged_run_root,
                    benchmark_root=benchmark_root,
                    endpoints=endpoint_group,
                ),
                env=env,
                log_path=log_root / "mme_reasoning" / f"{campaign.slug}.log",
                label=f"mme/{campaign.slug}",
            )
            _validate_score_outputs(
                [campaign],
                (MME_SCORE_KEY,),
                benchmark_root=benchmark_root,
                specs=specs,
                generation_inputs=generation_inputs,
            )
        finally:
            endpoint_pool.put(endpoint_group)

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, campaign): campaign for campaign in pending}
        for future in concurrent.futures.as_completed(futures):
            campaign = futures[future]
            try:
                future.result()
            except Exception as error:
                errors.append(f"{campaign.slug}: {error}")
    if errors:
        raise RuntimeError(
            "MME-Reasoning scoring failures:\n  - " + "\n  - ".join(errors)
        )


def _check_endpoints(endpoints: list[str], model: str) -> None:
    failures: list[str] = []
    for endpoint in endpoints:
        url = endpoint.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.load(response)
            models = [str(item.get("id")) for item in payload.get("data", [])]
            if model not in models:
                failures.append(f"{url}: served models={models}")
        except Exception as error:
            failures.append(f"{url}: {error}")
    if failures:
        raise RuntimeError(
            "Judge endpoint preflight failed:\n  - " + "\n  - ".join(failures)
        )


def _official_jobs(
    args: argparse.Namespace,
    campaigns: list[Campaign],
    workbooks: dict[str, list[Workbook]],
    benchmark_root: Path,
) -> list[OfficialJob]:
    jobs: list[OfficialJob] = []
    for campaign in campaigns:
        for key in OFFICIAL_SCORE_KEYS:
            workbook = _primary_workbook(workbooks, campaign.slug, key)
            primary_metric, value_scale = _primary_metric_contract(key)
            jobs.append(
                OfficialJob(
                    campaign=campaign,
                    workbook=workbook,
                    output_dir=_run_dir(
                        benchmark_root,
                        key,
                        campaign.slug,
                        workbook.run_name,
                    ),
                    judge_kwargs=_judge_kwargs(args, key),
                    primary_metric=primary_metric,
                    primary_value_scale=value_scale,
                )
            )
    return jobs


def _run_official_phase(
    args: argparse.Namespace,
    jobs: list[OfficialJob],
    *,
    endpoints: list[str],
    env: dict[str, str],
    log_root: Path,
) -> None:
    pending = [job for job in jobs if not (args.resume and _official_complete(job))]
    completed = [job for job in jobs if job not in pending]
    skipped = len(jobs) - len(pending)
    print(f"[official:plan] pending={len(pending)} skipped={skipped}", flush=True)
    for job in completed:
        _archive_official_job(args, job)
    if not pending:
        return

    endpoint_pool: queue.Queue[str] = queue.Queue()
    for endpoint in endpoints:
        endpoint_pool.put(endpoint)

    def execute(job: OfficialJob) -> None:
        endpoint = endpoint_pool.get()
        try:
            job_env = env.copy()
            job_env["OPENAI_API_BASE"] = _chat_completions_url(endpoint)
            label = f"official/{job.campaign.slug}/{job.workbook.benchmark_key}"
            _run_logged(
                _official_command(args, job),
                env=job_env,
                log_path=log_root
                / "official"
                / job.campaign.slug
                / f"{job.workbook.benchmark_key}.log",
                label=label,
            )
            _archive_official_job(args, job)
        finally:
            endpoint_pool.put(endpoint)

    errors: list[str] = []
    workers = min(args.official_workers, len(endpoints), len(pending))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, job): job for job in pending}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as error:
                errors.append(
                    f"{job.campaign.slug}/{job.workbook.benchmark_key}: {error}"
                )
    if errors:
        raise RuntimeError("Official scoring failures:\n  - " + "\n  - ".join(errors))


def _print_plan(
    args: argparse.Namespace,
    campaigns: list[Campaign],
    jobs: list[OfficialJob],
    *,
    staged_run_root: Path,
    benchmark_root: Path,
    queue_root: Path,
    endpoints: list[str],
    contract_sha256: str,
) -> None:
    print(
        f"[preflight:ok] campaigns={len(campaigns)} benchmarks={len(ALL_SCORE_KEYS)} "
        f"official_jobs={len(jobs)} direct_jobs={len(campaigns) * len(DIRECT_SCORE_KEYS)} "
        f"mme_jobs={len(campaigns)} endpoints={len(endpoints)} contract={contract_sha256}"
    )
    for job in jobs:
        print(
            f"[plan:official] endpoint=<leased> "
            f"{_command_text(_official_command(args, job))}"
        )
    for campaign in campaigns:
        for key in DIRECT_SCORE_KEYS:
            print(
                "[plan:direct] "
                + _command_text(
                    _direct_command(
                        args,
                        campaign,
                        key,
                        staged_run_root=staged_run_root,
                        benchmark_root=benchmark_root,
                        queue_root=queue_root,
                        endpoints=[endpoints[0]],
                        contract_sha256=contract_sha256,
                    )
                )
            )
        print(
            "[plan:mme] "
            + _command_text(
                _mme_command(
                    args,
                    campaign,
                    staged_run_root=staged_run_root,
                    benchmark_root=benchmark_root,
                    endpoints=endpoints,
                )
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one or more saved TRACE campaigns in an isolated evaluation tree."
    )
    parser.add_argument(
        "--campaign",
        action="append",
        nargs=3,
        required=True,
        metavar=("MODEL", "MODEL_SLUG", "CAMPAIGN_ROOT"),
        help="Repeat once or more; CAMPAIGN_ROOT contains seed_<seed>/runs.",
    )
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("trace_eval_v1",),
        default="trace_eval_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--eval-deps", type=Path, default=DEFAULT_EVAL_DEPS)
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    parser.add_argument("--lmu-data", type=Path, default=DEFAULT_LMU_DATA)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--judge-model", type=Path, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-api-model", default="qwen3-32b-judge")
    parser.add_argument("--judge-endpoint", action="append", default=[])
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--eval-nproc", type=int, default=16)
    parser.add_argument("--official-workers", type=int, default=8)
    parser.add_argument("--direct-workers", type=int, default=8)
    parser.add_argument("--mme-workers", type=int, default=3)
    parser.add_argument("--judge-api-parallelism", type=int, default=64)
    parser.add_argument("--judge-api-batch-size", type=int, default=64)
    parser.add_argument("--judge-api-batches-per-endpoint", type=int, default=1)
    parser.add_argument("--judge-api-max-batch-chars", type=int, default=200_000)
    parser.add_argument(
        "--judge-cache-contract-version", default="trace-persistent-judge-v2"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--shared-seed-root",
        action="store_true",
        help="Store independent per-seed contracts under one score root.",
    )
    parser.add_argument(
        "--emit-archive",
        action="store_true",
        help="Emit extraction and score descriptors to the configured asynchronous HF spool.",
    )
    parser.add_argument(
        "--preflight", "--dry-run", action="store_true", dest="preflight"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _activate_suite(args.suite)
    if not args.campaign:
        raise SystemExit("error: pass at least one --campaign descriptor")
    if (
        args.seed < 0
        or args.eval_nproc < 1
        or args.official_workers < 1
        or args.direct_workers < 1
        or args.mme_workers < 1
    ):
        raise SystemExit(
            "error: seed must be non-negative and worker counts must be positive"
        )
    if args.judge_max_tokens < 1:
        raise SystemExit("error: --judge-max-tokens must be positive")

    campaigns = [
        Campaign(model=model, slug=slug, root=Path(root).expanduser().resolve())
        for model, slug, root in args.campaign
    ]
    if len({item.slug for item in campaigns}) != len(campaigns):
        raise SystemExit("error: campaign model slugs must be unique")

    args.score_root = args.score_root.expanduser().resolve()
    # Keep the venv launcher path; resolving its symlink would silently select
    # the system interpreter and drop the venv site-packages.
    args.python = args.python.expanduser().absolute()
    args.eval_deps = args.eval_deps.expanduser().resolve()
    args.vlmeval_root = args.vlmeval_root.expanduser().resolve()
    args.lmu_data = args.lmu_data.expanduser().resolve()
    args.hf_home = args.hf_home.expanduser().resolve()
    args.judge_model = args.judge_model.expanduser().resolve()
    args.dataset_manifest = (
        args.dataset_manifest.expanduser().resolve()
        if args.dataset_manifest
        else args.lmu_data / "trace_eval_v1_dataset_manifest.json"
    )
    endpoints = list(
        dict.fromkeys(
            _normalize_endpoint(item)
            for item in (args.judge_endpoint or _default_endpoints())
        )
    )
    if not endpoints:
        raise SystemExit("error: at least one judge endpoint is required")

    staged_run_root = args.score_root / f"seed_{args.seed}" / "runs"
    benchmark_root = args.score_root / f"seed_{args.seed}" / "benchmark"
    queue_root = args.score_root / f"seed_{args.seed}" / "queues"
    log_root = args.score_root / "logs"

    specs = _load_specs()
    env = _base_env(args)
    _validate_runtime(args, env)
    dataset_manifest = _validate_dataset_manifest(args.dataset_manifest, args.lmu_data)
    workbooks = _discover_workbooks(
        campaigns,
        seed=args.seed,
        staged_run_root=staged_run_root,
        specs=specs,
    )
    generation_inputs = _validate_generation_inputs(
        campaigns,
        workbooks,
        seed=args.seed,
        expected_dataset_snapshot=dataset_manifest["view_snapshot_sha256"][
            ACTIVE_DATASET_VIEW
        ],
    )
    contract = _contract(
        args,
        campaigns,
        workbooks,
        generation_inputs,
        dataset_manifest,
        endpoints,
    )
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    jobs = _official_jobs(args, campaigns, workbooks, benchmark_root)

    if args.preflight:
        _check_endpoints(endpoints, args.judge_api_model)
        _print_plan(
            args,
            campaigns,
            jobs,
            staged_run_root=staged_run_root,
            benchmark_root=benchmark_root,
            queue_root=queue_root,
            endpoints=endpoints,
            contract_sha256=contract_sha256,
        )
        return

    _check_endpoints(endpoints, args.judge_api_model)
    contract_sha256 = _prepare_score_root(
        args.score_root,
        contract,
        args.resume,
        shared_seed_root=args.shared_seed_root,
    )
    _stage_workbooks(workbooks, resume=args.resume)

    _run_official_phase(
        args,
        jobs,
        endpoints=endpoints,
        env=env,
        log_root=log_root,
    )
    _validate_score_outputs(
        campaigns,
        OFFICIAL_SCORE_KEYS,
        benchmark_root=benchmark_root,
        specs=specs,
        generation_inputs=generation_inputs,
    )
    _run_direct_phase(
        args,
        campaigns,
        staged_run_root=staged_run_root,
        benchmark_root=benchmark_root,
        queue_root=queue_root,
        endpoints=endpoints,
        contract_sha256=contract_sha256,
        env=env,
        log_root=log_root,
        specs=specs,
        generation_inputs=generation_inputs,
    )
    _run_mme_phase(
        args,
        campaigns,
        staged_run_root=staged_run_root,
        benchmark_root=benchmark_root,
        endpoints=endpoints,
        env=env,
        log_root=log_root,
        specs=specs,
        generation_inputs=generation_inputs,
    )
    completed = _validate_score_outputs(
        campaigns,
        ALL_SCORE_KEYS,
        benchmark_root=benchmark_root,
        specs=specs,
        generation_inputs=generation_inputs,
    )
    _write_json(
        args.score_root
        / (
            f"score_campaign_completion_seed_{args.seed}.json"
            if args.shared_seed_root
            else "score_campaign_completion.json"
        ),
        {
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "contract_sha256": contract_sha256,
            "expected_slices": len(campaigns) * len(ALL_SCORE_KEYS),
            "completed_slices": len(completed),
            "slices": completed,
        },
    )
    print(f"[campaign:done] score_root={args.score_root}", flush=True)


if __name__ == "__main__":
    main()
