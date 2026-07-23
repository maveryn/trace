#!/usr/bin/env python3
"""Verify the pinned TRACE validation generation and scoring campaign."""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

# Keep direct-script execution working alongside ``python -m`` invocation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlvr.evaluation.trace_validation import prepare_dataset as dataset_prep

WORK_ROOT = REPO_ROOT / "rlvr" / "evaluation" / ".work" / "trace_validation"
DEFAULT_SUITE = REPO_ROOT / "rlvr" / "evaluation" / "trace_validation" / "suite.v1.json"
DEFAULT_CAMPAIGN_ROOT = WORK_ROOT / "campaign"
DEFAULT_DATASET_MANIFEST = WORK_ROOT / "dataset" / "manifest.json"
DEFAULT_DATASET_PARQUET = (
    WORK_ROOT
    / "inputs"
    / "data"
    / "validation"
    / "trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
)
DEFAULT_RESULTS = (
    REPO_ROOT / "rlvr" / "evaluation" / "trace_validation" / "results.v1.json"
)
DEFAULT_RELEASE_RECEIPT = (
    REPO_ROOT / "rlvr" / "evaluation" / "trace_validation" / "release_receipt.v1.json"
)
DEFAULT_DATASET_EQUIVALENCE_RECEIPT = dataset_prep.DATASET_EQUIVALENCE_RECEIPT

RELEASE_SOURCE_REVISION = "cf0d14aed86db2661d397ce8b68b36171873478d"
RELEASE_SOURCE_RUN_ID = "trace-iid-validation-2000-answer-seed42-8models-v1"
RELEASE_SOURCE_RESULTS_SHA256 = (
    "c35240e88e681434ed26de321015ef62b3ee9a438e20613d606e16b393263da9"
)
EXPECTED_RELEASE_RECEIPT_SHA256 = (
    "865c7901f70578b21f3ce8c0c74621b30fc9bfe64f957ba4bd38c8e830aa971f"
)

EXPECTED_SUITE_SHA256 = (
    "f9cccdcdddb6135c16d3a9d434f985b51e4105c07ff0c74a54a71a4dfe7c85c7"
)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "ff483e38c2dd2f618e6467950c0c2bab5048cca0d6bffa5b3d1fd45fcb4b0b69"
)
EXPECTED_DATASET_FILE_SHA256 = (
    "051d1441b3f65f291841384962f16d6b5063f236072805f6c330f49afc02c4d1"
)
EXPECTED_SYSTEM_PROMPT_FILE_SHA256 = (
    "f394927d9abcfb7a1e43ef48a30c29b8c70e6facdbda314d7b27c59d8c3ae900"
)
EXPECTED_JUDGE_SYSTEM_PROMPT_SHA256 = (
    "6ada68518b48c2678f595e563f47979b837933e7cd6ec54975dd7ffe0578a962"
)

GENERATION_RUN_SCHEMA = "trace-validation-generation-run-v1"
GENERATION_RECEIPT_SCHEMA = "trace-validation-generation-receipt-v1"
DATASET_MANIFEST_SCHEMA = "trace-validation-dataset-manifest-v1"
SCORING_CONTRACT_VERSION = "trace-validation-answer-scoring-v2"
JUDGE_CONTRACT_VERSION = "trace-validation-qwen3-answer-extractor-v2"
ANSWER_EXTRACTION_VERSION = "trace-validation-answer-extraction-v1"

EXPECTED_DECODING: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "max_tokens": 2_048,
    "seed": 42,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "n": 1,
    "stream": False,
}
EXPECTED_MM_PROCESSOR_KWARGS = {
    "min_pixels": 262_144,
    "max_pixels": 4_194_304,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMMUTABLE_REVISION_RE = re.compile(
    r"(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256set:[0-9a-f]{64})"
)
_ANSWER_TYPES = {"integer", "number", "option_letter", "string"}
_AGGREGATE_METRICS = (
    "historical_answer_correct",
    "historical_format_correct",
    "terminal_rfc_json",
    "historical_reward",
    "deterministic_semantic_correct",
    "combined_semantic_correct",
    "deterministic_found",
    "judge_requested",
    "judge_resolved",
    "unresolved",
)


class VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelPin:
    slug: str
    repo_id: str
    revision: str
    runtime_revision: str | None = None

    @property
    def inference_revision(self) -> str:
        return self.runtime_revision or self.revision


@dataclass(frozen=True)
class DatasetPin:
    repo_id: str
    revision: str
    file: str
    file_sha256: str
    file_size_bytes: int
    manifest_sha256: str

    def normalized_identity(self, *, rows: int) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "split": "validation",
            "file": self.file,
            "file_sha256": self.file_sha256,
            "file_size_bytes": self.file_size_bytes,
            "row_count": rows,
        }


@dataclass(frozen=True)
class VerificationPolicy:
    suite_sha256: str
    suite_id: str
    historical_dataset: DatasetPin
    reproduction_dataset: DatasetPin
    equivalence_compared_revision: str
    equivalence_receipt_sha256: str | None
    rows: int
    tasks: int
    samples_per_task: int
    models: tuple[ModelPin, ...]
    require_unique_images: bool = True


PRODUCTION_POLICY = VerificationPolicy(
    suite_sha256=EXPECTED_SUITE_SHA256,
    suite_id="trace-validation-2000-seed42-v1",
    historical_dataset=DatasetPin(
        repo_id="maveryn/trace",
        revision="e317b746b258630682367cc6a9d87dedd195113c",
        file=(
            "data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
        ),
        file_sha256=(
            "0cb46bcf858ae3e9f39b88f60a24549a5de133976b9e8b74a45b4e6e4d699470"
        ),
        file_size_bytes=238_177_343,
        manifest_sha256=(
            "825215fe98d1af3178c07449603d653cccb30a4eef63e4b9dc1cd45c3e43ce36"
        ),
    ),
    reproduction_dataset=DatasetPin(
        repo_id="maveryn/trace",
        revision="4e5b54361360296a855542b40cfd8b7f81b355fe",
        file=(
            "data/validation/trace_rlvr_validation_iid_2000_all1000_seed1042.parquet"
        ),
        file_sha256=EXPECTED_DATASET_FILE_SHA256,
        file_size_bytes=238_174_013,
        manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
    ),
    equivalence_compared_revision="78f09b5482abc8e447a0a722cdf39e7d32f483c8",
    equivalence_receipt_sha256=(
        "40afcb59e0c67d7b7f47b78bb673a6c152320eb0656a990910ff52c634d43ae7"
    ),
    rows=2_000,
    tasks=1_000,
    samples_per_task=2,
    models=(
        ModelPin(
            "qwen2.5-vl-3b-base",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "66285546d2b821cf421d4f5eb2576359d3770cd3",
        ),
        ModelPin(
            "trace-qwen2.5-vl-3b",
            "maveryn/trace-qwen2.5-vl-3b",
            "2ec2374d5c219e6b12e26bda93d3b3adeb1e30c5",
            "sha256set:fd7d9ef4dd828eb950ce29c8ccde0432ccd31420529d4f023300ede928d070a1",
        ),
        ModelPin(
            "qwen2.5-vl-7b-base",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        ),
        ModelPin(
            "trace-qwen2.5-vl-7b",
            "maveryn/trace-qwen2.5-vl-7b",
            "4d0f1ae8ee25022058090dbdbff61957ece7331d",
            "sha256set:28421ef2be848d24e2a9fa363d885f42c651bfb6fa986de3d07faa9d78da47cf",
        ),
        ModelPin(
            "game-rl-qwen2.5-vl-7b",
            "OpenMOSS-Team/Game-RL-Qwen2.5-VL-7B",
            "205b5934ce70504cfd6ae26b16f705d0b98b9306",
            "sha256set:2a805cbedc07225555712644c3569019da15a30108bb50b2dbe60d9562d24b2f",
        ),
        ModelPin(
            "sphinx-qwen2.5-vl-7b",
            "xashru/sphinx_qwen7b_500",
            "6ffefb03d5cb0767683bfb42a084ea86b707ef9a",
        ),
        ModelPin(
            "pcgrpo-qwen2.5-vl-7b",
            "armenjeddi/PCGRPO-Qwen2.5-VL-7B-Jigsaw-with-curriculum-with-grpo-care",
            "921bbced4176f5d362e98c843a57656c5d78dad7",
        ),
        ModelPin(
            "vero-qwen2.5-vl-7b",
            "zlab-princeton/Vero-Qwen25-7B",
            "180e84be5acb2aa887cf51015b84b6a6e453ee90",
        ),
    ),
)


@dataclass
class DatasetArtifacts:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    system_prompt: str
    system_prompt_file_sha256: str
    equivalence: dict[str, Any]


@dataclass
class GenerationArtifacts:
    response_paths: dict[str, Path]
    metadata: dict[str, dict[str, Any]]
    records: dict[tuple[str, int], dict[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256((_canonical_json(value) + "\n").encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot load JSON artifact: {path}") from exc
    _require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"cannot open JSONL artifact: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"blank JSONL line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
            _require(
                isinstance(value, dict),
                f"JSONL row is not an object at {path}:{line_number}",
            )
            rows.append(value)
    return rows


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _safe_relative_path(root: Path, relative_path: Any) -> Path:
    _require(
        isinstance(relative_path, str) and bool(relative_path), "empty relative path"
    )
    pure = PurePosixPath(relative_path)
    _require(
        not pure.is_absolute() and ".." not in pure.parts,
        f"unsafe path: {relative_path}",
    )
    resolved_root = root.resolve()
    path = resolved_root.joinpath(*pure.parts).resolve()
    _require(path.is_relative_to(resolved_root), f"path escapes root: {relative_path}")
    return path


def _verify_suite(suite_path: Path, policy: VerificationPolicy) -> dict[str, Any]:
    suite_path = suite_path.expanduser().resolve()
    _require(suite_path.is_file(), f"missing suite: {suite_path}")
    suite_sha256 = _sha256_file(suite_path)
    _require(
        suite_sha256 == policy.suite_sha256,
        f"suite SHA-256 mismatch: {suite_sha256} != {policy.suite_sha256}",
    )
    suite = _load_json(suite_path)
    _require(
        suite.get("schema_version") == "trace-validation-suite-v1", "wrong suite schema"
    )
    _require(suite.get("suite_id") == policy.suite_id, "wrong suite_id")

    dataset = suite.get("dataset")
    _require(isinstance(dataset, dict), "suite has no dataset contract")
    historical_dataset = policy.historical_dataset
    expected_dataset = {
        "repo_id": historical_dataset.repo_id,
        "revision": historical_dataset.revision,
        "split": "validation",
        "file": historical_dataset.file,
        "sha256": historical_dataset.file_sha256,
        "rows": policy.rows,
        "tasks": policy.tasks,
        "samples_per_task": policy.samples_per_task,
        "distribution": "same_task_programs_nonoverlapping_samples",
        "prompt_key": "prompt_answer",
        "image_key": "images",
    }
    _require(
        dataset == expected_dataset,
        "suite dataset contract differs from the pinned validation split",
    )

    prompt = suite.get("prompt")
    _require(isinstance(prompt, dict), "suite has no prompt contract")
    _require(
        prompt.get("system_prompt_sha256") == EXPECTED_SYSTEM_PROMPT_FILE_SHA256,
        "suite system prompt SHA-256 mismatch",
    )
    _require(
        prompt.get("chat_template") == "native_checkpoint_template",
        "wrong chat template",
    )
    _require(
        prompt.get("add_generation_prompt") is True, "generation prompt is not enabled"
    )

    generation = suite.get("generation")
    expected_suite_generation = {
        "seed": EXPECTED_DECODING["seed"],
        "responses_per_prompt": EXPECTED_DECODING["n"],
        "temperature": EXPECTED_DECODING["temperature"],
        "top_p": EXPECTED_DECODING["top_p"],
        "top_k": EXPECTED_DECODING["top_k"],
        "presence_penalty": EXPECTED_DECODING["presence_penalty"],
        "frequency_penalty": EXPECTED_DECODING["frequency_penalty"],
        "repetition_penalty": EXPECTED_DECODING["repetition_penalty"],
        "max_tokens": EXPECTED_DECODING["max_tokens"],
        "min_image_pixels": EXPECTED_MM_PROCESSOR_KWARGS["min_pixels"],
        "max_image_pixels": EXPECTED_MM_PROCESSOR_KWARGS["max_pixels"],
        "server_generation_config": "vllm",
    }
    _require(
        generation == expected_suite_generation, "suite generation contract mismatch"
    )

    scoring = suite.get("scoring")
    _require(isinstance(scoring, dict), "suite has no scoring contract")
    _require(
        scoring.get("deterministic_extraction") == ANSWER_EXTRACTION_VERSION,
        "wrong extractor",
    )
    _require(
        scoring.get("judge_only_on") == ["missing", "ambiguous"], "wrong judge policy"
    )
    _require(
        scoring.get("unresolved_rows_score") == 0, "unresolved rows must score zero"
    )
    _require(scoring.get("drop_failed_rows") is False, "failed rows may not be dropped")

    judge = suite.get("judge")
    _require(isinstance(judge, dict), "suite has no judge contract")
    _require(
        judge.get("temperature") == 0.0 and judge.get("top_p") == 1.0,
        "wrong judge decoding",
    )
    _require(judge.get("thinking") is False, "judge thinking must be disabled")
    _require(
        judge.get("input_fields") == ["raw_response", "answer_type"],
        "judge inputs changed",
    )
    _require(
        judge.get("forbidden_fields")
        == ["image", "question", "choices", "answer_gt_value"],
        "judge forbidden-field contract changed",
    )

    models = suite.get("models")
    _require(isinstance(models, list), "suite models is not a list")
    expected_slugs = [model.slug for model in policy.models]
    actual_slugs = [model.get("slug") for model in models if isinstance(model, dict)]
    _require(actual_slugs == expected_slugs, f"suite roster mismatch: {actual_slugs}")
    _require(
        len(set(actual_slugs)) == len(actual_slugs), "suite has duplicate model slugs"
    )
    for model_document, pin in zip(models, policy.models):
        _require(
            model_document.get("repo_id") == pin.repo_id, f"{pin.slug}: repo mismatch"
        )
        _require(
            model_document.get("revision") == pin.revision,
            f"{pin.slug}: revision mismatch",
        )
        _require(
            model_document.get("runtime_view_revision") == pin.runtime_revision,
            f"{pin.slug}: runtime revision mismatch",
        )
        _require(bool(model_document.get("label")), f"{pin.slug}: empty display label")
    return suite


def verify_release_metadata(
    *,
    suite_path: Path = DEFAULT_SUITE,
    results_path: Path = DEFAULT_RESULTS,
    receipt_path: Path = DEFAULT_RELEASE_RECEIPT,
) -> dict[str, Any]:
    """Validate the checked-in result table against its immutable-source receipt."""

    suite_path = suite_path.expanduser().resolve()
    results_path = results_path.expanduser().resolve()
    receipt_path = receipt_path.expanduser().resolve()
    _verify_suite(suite_path, PRODUCTION_POLICY)
    _require(results_path.is_file(), f"missing results: {results_path}")
    _require(receipt_path.is_file(), f"missing release receipt: {receipt_path}")
    _require(
        _sha256_file(receipt_path) == EXPECTED_RELEASE_RECEIPT_SHA256,
        "release receipt SHA-256 mismatch",
    )

    results_sha256 = _sha256_file(results_path)
    results = _load_json(results_path)
    receipt = _load_json(receipt_path)
    _require(
        results.get("schema_version") == "trace-validation-results-v1",
        "wrong results schema",
    )
    _require(
        receipt.get("schema_version") == "trace-validation-release-receipt-v1",
        "wrong release receipt schema",
    )

    immutable_source = receipt.get("immutable_source")
    _require(
        isinstance(immutable_source, dict), "release receipt has no immutable source"
    )
    _require(
        immutable_source.get("dataset_repo_id") == "maveryn/trace-eval-runs",
        "wrong release source repository",
    )
    _require(
        immutable_source.get("dataset_revision") == RELEASE_SOURCE_REVISION,
        "wrong release source revision",
    )
    _require(
        immutable_source.get("run_id") == RELEASE_SOURCE_RUN_ID,
        "wrong release source run",
    )
    source_files = immutable_source.get("files")
    _require(isinstance(source_files, dict), "release receipt has no source file pins")
    benchmark_scores = source_files.get("benchmark_scores")
    _require(
        isinstance(benchmark_scores, dict), "release receipt has no score source pin"
    )
    _require(
        benchmark_scores.get("sha256") == RELEASE_SOURCE_RESULTS_SHA256,
        "wrong immutable score-source hash",
    )

    result_source = results.get("source")
    _require(isinstance(result_source, dict), "results have no source receipt")
    _require(
        result_source.get("dataset_repo_id") == immutable_source.get("dataset_repo_id")
        and result_source.get("dataset_revision") == RELEASE_SOURCE_REVISION
        and result_source.get("run_id") == RELEASE_SOURCE_RUN_ID
        and result_source.get("results_sha256") == RELEASE_SOURCE_RESULTS_SHA256,
        "results source and release receipt differ",
    )

    public_release = receipt.get("public_release")
    _require(
        isinstance(public_release, dict), "release receipt has no public-release pins"
    )
    _require(
        public_release.get("suite_sha256") == _sha256_file(suite_path),
        "release receipt suite hash mismatch",
    )
    _require(
        public_release.get("results_sha256") == results_sha256,
        "release receipt results hash mismatch",
    )

    evaluation = results.get("evaluation")
    _require(isinstance(evaluation, dict), "results have no evaluation contract")
    _require(evaluation.get("rows_per_model") == 2_000, "wrong result denominator")
    _require(evaluation.get("seed") == 42, "wrong result seed")
    _require(evaluation.get("score_unit") == "percent", "wrong result score unit")
    result_rows = results.get("results")
    _require(isinstance(result_rows, list), "results rows are not a list")
    expected_models = [pin.slug for pin in PRODUCTION_POLICY.models]
    actual_models = [
        row.get("model_id") for row in result_rows if isinstance(row, dict)
    ]
    _require(actual_models == expected_models, "result model roster or order changed")
    _require(len(result_rows) == 8, "results must contain exactly eight models")
    for row in result_rows:
        _require(isinstance(row, dict), "result row is not an object")
        _require(row.get("evaluated_rows") == 2_000, "result row denominator changed")
        score_value = row.get("score")
        _require(
            isinstance(score_value, (int, float))
            and not isinstance(score_value, bool)
            and math.isfinite(score_value)
            and 0.0 <= float(score_value) <= 100.0,
            "result score is invalid",
        )

    campaign = receipt.get("campaign_verification")
    _require(isinstance(campaign, dict), "release receipt has no campaign verification")
    _require(
        campaign.get("status") == "ok"
        and campaign.get("models") == 8
        and campaign.get("seed") == 42
        and campaign.get("rows_per_model") == 2_000
        and campaign.get("generation_rows") == 16_000
        and campaign.get("scored_rows") == 16_000,
        "campaign verification summary is inconsistent",
    )
    return {
        "phase": "release",
        "status": "ok",
        "source_revision": RELEASE_SOURCE_REVISION,
        "models": len(result_rows),
        "rows_per_model": 2_000,
        "results_sha256": results_sha256,
    }


def _verify_dataset(
    manifest_path: Path,
    parquet_path: Path,
    suite: Mapping[str, Any],
    policy: VerificationPolicy,
    equivalence_receipt_path: Path,
) -> DatasetArtifacts:
    manifest_path = manifest_path.expanduser().resolve()
    parquet_path = parquet_path.expanduser().resolve()
    _require(manifest_path.is_file(), f"missing dataset manifest: {manifest_path}")
    manifest_sha256 = _sha256_file(manifest_path)
    _require(
        manifest_sha256 == policy.reproduction_dataset.manifest_sha256,
        "dataset manifest SHA-256 mismatch",
    )
    _require(parquet_path.is_file(), f"missing pinned parquet: {parquet_path}")
    _require(
        _sha256_file(parquet_path) == policy.reproduction_dataset.file_sha256,
        "dataset parquet SHA mismatch",
    )
    _require(
        parquet_path.stat().st_size == policy.reproduction_dataset.file_size_bytes,
        "dataset parquet byte-size mismatch",
    )

    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == DATASET_MANIFEST_SCHEMA,
        "wrong manifest schema",
    )
    identity = manifest.get("dataset")
    _require(isinstance(identity, dict), "manifest has no dataset identity")
    reproduction_dataset = policy.reproduction_dataset
    expected_identity = {
        "repo_id": reproduction_dataset.repo_id,
        "revision": reproduction_dataset.revision,
        "config": "default",
        "split": "validation",
        "file": reproduction_dataset.file,
        "file_sha256": reproduction_dataset.file_sha256,
        "file_size_bytes": reproduction_dataset.file_size_bytes,
        "row_count": policy.rows,
    }
    _require(
        identity == expected_identity,
        "manifest dataset identity differs from the reproduction dataset",
    )
    try:
        equivalence = dataset_prep.validate_dataset_equivalence_bridge(
            equivalence_receipt_path,
            historical_identity=policy.historical_dataset.normalized_identity(
                rows=policy.rows
            ),
            reproduction_identity=reproduction_dataset.normalized_identity(
                rows=policy.rows
            ),
            historical_manifest_sha256=(policy.historical_dataset.manifest_sha256),
            reproduction_manifest_sha256=manifest_sha256,
            expected_compared_revision=policy.equivalence_compared_revision,
            expected_receipt_sha256=policy.equivalence_receipt_sha256,
        )
    except dataset_prep.DatasetEquivalenceError as exc:
        raise VerificationError(f"dataset equivalence failed: {exc}") from exc
    media_contract = manifest.get("media")
    _require(isinstance(media_contract, dict), "manifest has no media contract")
    _require(
        media_contract.get("reencoded") is False, "validation media was re-encoded"
    )
    _require(media_contract.get("resized") is False, "validation media was resized")

    rows = manifest.get("rows")
    _require(
        isinstance(rows, list) and len(rows) == policy.rows, "wrong manifest row count"
    )
    instance_ids: set[str] = set()
    image_hashes: set[str] = set()
    task_counts: collections.Counter[str] = collections.Counter()
    verified_media: dict[str, Path] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"manifest row {index} is not an object")
        _require(
            row.get("row_index") == index, f"manifest row order changed at {index}"
        )
        instance_id = row.get("instance_id")
        _require(
            isinstance(instance_id, str) and bool(instance_id),
            f"row {index}: no instance_id",
        )
        _require(
            instance_id not in instance_ids,
            f"duplicate dataset instance_id: {instance_id}",
        )
        instance_ids.add(instance_id)
        task = row.get("task")
        _require(isinstance(task, str) and bool(task), f"row {index}: no task")
        task_counts[task] += 1
        _require(
            isinstance(row.get("domain"), str) and bool(row["domain"]),
            f"row {index}: no domain",
        )
        _require(
            row.get("answer_type") in _ANSWER_TYPES, f"row {index}: invalid answer type"
        )
        answer_gt = row.get("answer_gt")
        _require(isinstance(answer_gt, dict), f"row {index}: invalid answer_gt")
        _require(
            answer_gt.get("type") == row["answer_type"] and "value" in answer_gt,
            f"row {index}: answer mismatch",
        )
        answer_value = answer_gt["value"]
        if row["answer_type"] == "integer":
            _require(
                isinstance(answer_value, int) and not isinstance(answer_value, bool),
                f"row {index}: invalid integer GT",
            )
        elif row["answer_type"] == "number":
            _require(
                isinstance(answer_value, (int, float))
                and not isinstance(answer_value, bool),
                f"row {index}: invalid number GT",
            )
            _require(
                math.isfinite(float(answer_value)), f"row {index}: nonfinite number GT"
            )
        elif row["answer_type"] == "option_letter":
            _require(
                isinstance(answer_value, str)
                and re.fullmatch(r"[A-Z]", answer_value) is not None,
                f"row {index}: invalid option GT",
            )
        elif row["answer_type"] == "string":
            _require(
                isinstance(answer_value, str) and bool(answer_value.strip()),
                f"row {index}: invalid string GT",
            )
        prompt = row.get("prompt_answer")
        images = row.get("images")
        _require(isinstance(prompt, str) and bool(prompt), f"row {index}: no prompt")
        _require(isinstance(images, list) and bool(images), f"row {index}: no images")
        _require(
            prompt.count("<image>") == len(images),
            f"row {index}: image marker mismatch",
        )
        for image_index, image in enumerate(images):
            _require(isinstance(image, dict), f"row {index}: invalid image metadata")
            _require(
                image.get("image_index") == image_index,
                f"row {index}: image order mismatch",
            )
            digest = image.get("sha256")
            _require(_is_sha256(digest), f"row {index}: invalid image hash")
            _require(
                isinstance(image.get("size_bytes"), int) and image["size_bytes"] > 0,
                f"row {index}: invalid image size",
            )
            path = _safe_relative_path(manifest_path.parent, image.get("relative_path"))
            _require(path.is_file(), f"row {index}: missing image {path}")
            if digest not in verified_media:
                _require(
                    path.stat().st_size == image["size_bytes"],
                    f"row {index}: image byte-size drift",
                )
                _require(
                    _sha256_file(path) == digest, f"row {index}: image content drift"
                )
                verified_media[digest] = path
            else:
                _require(
                    verified_media[digest] == path, f"image {digest}: inconsistent path"
                )
            image_hashes.add(str(digest))
    _require(len(instance_ids) == policy.rows, "dataset instance IDs are not unique")
    _require(len(task_counts) == policy.tasks, "dataset task count mismatch")
    _require(
        set(task_counts.values()) == {policy.samples_per_task},
        "dataset is not the expected equal samples-per-task validation split",
    )
    if policy.require_unique_images:
        _require(
            len(image_hashes) == policy.rows,
            "dataset images are not uniquely content-addressed",
        )

    prompt_path = (REPO_ROOT / suite["prompt"]["system_prompt_file"]).resolve()
    _require(
        prompt_path.is_relative_to(REPO_ROOT), "system prompt path escapes repository"
    )
    _require(prompt_path.is_file(), f"missing system prompt: {prompt_path}")
    system_prompt_file_sha256 = _sha256_file(prompt_path)
    _require(
        system_prompt_file_sha256 == suite["prompt"]["system_prompt_sha256"],
        "system prompt file SHA mismatch",
    )
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    _require(bool(system_prompt), "system prompt is empty")
    return DatasetArtifacts(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        rows=rows,
        system_prompt=system_prompt,
        system_prompt_file_sha256=system_prompt_file_sha256,
        equivalence=equivalence,
    )


def _verify_model_marker(model_path: Path, pin: ModelPin, model_revision: str) -> None:
    from rlvr.evaluation.scripts.prepare_trace_eval_models import (
        verify_model_directory,
    )

    try:
        verify_model_directory(
            model_path,
            pin.slug,
            pin.repo_id,
            pin.revision,
            model_revision,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise VerificationError(f"{pin.slug}: {exc}") from exc


def _expected_user_content(
    row: Mapping[str, Any],
    dataset: DatasetArtifacts,
) -> list[dict[str, Any]]:
    prompt = str(row["prompt_answer"])
    images = list(row["images"])
    content: list[dict[str, Any]] = []
    segments = prompt.split("<image>")
    for index, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if index < len(images):
            image = images[index]
            path = _safe_relative_path(
                dataset.manifest_path.parent,
                image["relative_path"],
            )
            payload = path.read_bytes()
            _require(
                hashlib.sha256(payload).hexdigest() == image.get("sha256"),
                f"prepared image content changed: {path}",
            )
            mime_type = image.get("mime_type")
            _require(
                isinstance(mime_type, str) and mime_type.startswith("image/"),
                f"prepared image has invalid MIME type: {mime_type!r}",
            )
            data_url = (
                f"data:{mime_type};base64,"
                f"{base64.b64encode(payload).decode('ascii')}"
            )
            content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def _expected_request_hash(
    row: Mapping[str, Any],
    dataset: DatasetArtifacts,
    *,
    served_model: str,
) -> str:
    payload = {
        "model": served_model,
        "messages": [
            {"role": "system", "content": dataset.system_prompt},
            {"role": "user", "content": _expected_user_content(row, dataset)},
        ],
        "mm_processor_kwargs": dict(EXPECTED_MM_PROCESSOR_KWARGS),
        **EXPECTED_DECODING,
    }
    return _canonical_hash(payload)


def _contains_forbidden_ground_truth_key(value: Any) -> bool:
    if isinstance(value, dict):
        if {"answer_gt", "annotation_gt", "reward_contract"}.intersection(value):
            return True
        return any(
            _contains_forbidden_ground_truth_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_ground_truth_key(item) for item in value)
    return False


def _verify_generation_model(
    generation_dir: Path,
    pin: ModelPin,
    dataset: DatasetArtifacts,
    policy: VerificationPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reproduction_dataset = policy.reproduction_dataset
    metadata_path = generation_dir / "run_metadata.json"
    responses_path = generation_dir / "responses.jsonl"
    receipts_path = generation_dir / "receipts"
    metadata = _load_json(metadata_path)
    _require(
        metadata.get("schema_version") == GENERATION_RUN_SCHEMA,
        f"{pin.slug}: bad run schema",
    )
    _require(
        metadata.get("status") == "complete", f"{pin.slug}: generation is not complete"
    )
    _require(
        metadata.get("model_slug") == pin.slug, f"{pin.slug}: metadata slug mismatch"
    )
    _require(
        metadata.get("served_model") == pin.slug, f"{pin.slug}: served model mismatch"
    )
    _require(
        metadata.get("model_revision") == pin.inference_revision,
        f"{pin.slug}: wrong inference revision",
    )
    _require(
        _IMMUTABLE_REVISION_RE.fullmatch(str(metadata["model_revision"])) is not None,
        f"{pin.slug}: mutable revision",
    )
    _require(
        metadata.get("manifest_sha256") == dataset.manifest_sha256,
        f"{pin.slug}: manifest hash mismatch",
    )
    _require(
        Path(str(metadata.get("manifest_path"))).resolve() == dataset.manifest_path,
        f"{pin.slug}: manifest path mismatch",
    )
    _require(
        metadata.get("dataset_rows") == policy.rows,
        f"{pin.slug}: dataset row count mismatch",
    )
    _require(
        metadata.get("dataset_repo_id") == reproduction_dataset.repo_id,
        f"{pin.slug}: dataset repo mismatch",
    )
    _require(
        metadata.get("dataset_revision") == reproduction_dataset.revision,
        f"{pin.slug}: dataset revision mismatch",
    )
    _require(
        metadata.get("dataset_file") == reproduction_dataset.file,
        f"{pin.slug}: dataset file mismatch",
    )
    _require(
        metadata.get("dataset_file_sha256") == reproduction_dataset.file_sha256,
        f"{pin.slug}: dataset SHA mismatch",
    )
    _require(
        metadata.get("decoding") == EXPECTED_DECODING,
        f"{pin.slug}: decoding contract mismatch",
    )
    _require(
        metadata.get("mm_processor_kwargs") == EXPECTED_MM_PROCESSOR_KWARGS,
        f"{pin.slug}: processor contract mismatch",
    )
    _require(
        metadata.get("system_prompt_file_sha256") == dataset.system_prompt_file_sha256,
        f"{pin.slug}: prompt hash mismatch",
    )
    _require(
        metadata.get("media_transport") == "data-url",
        f"{pin.slug}: production media transport is not data-url",
    )
    _require(
        metadata.get("allowed_local_media_root") is None,
        f"{pin.slug}: data-url transport must not declare a local media root",
    )
    _require(
        metadata.get("shard_index") == 0 and metadata.get("shard_count") == 1,
        f"{pin.slug}: unexpected row shard",
    )
    _require(
        metadata.get("selected_rows") == policy.rows,
        f"{pin.slug}: selected row count mismatch",
    )
    _require(
        metadata.get("completed_rows") == policy.rows,
        f"{pin.slug}: completed row count mismatch",
    )
    _require(metadata.get("error_rows") == 0, f"{pin.slug}: generation contains errors")
    _require(
        Path(str(metadata.get("responses_path"))).resolve() == responses_path.resolve(),
        f"{pin.slug}: responses path mismatch",
    )
    _require(
        Path(str(metadata.get("receipts_path"))).resolve() == receipts_path.resolve(),
        f"{pin.slug}: receipts path mismatch",
    )
    endpoint = urlparse(str(metadata.get("endpoint_url") or ""))
    _require(
        endpoint.scheme in {"http", "https"} and endpoint.hostname is not None,
        f"{pin.slug}: bad endpoint",
    )
    _require(
        endpoint.path.endswith("/v1/chat/completions"),
        f"{pin.slug}: endpoint is not chat completions",
    )
    model_path = Path(str(metadata.get("model_path"))).resolve()
    _require(model_path.is_dir(), f"{pin.slug}: missing model path")
    _verify_model_marker(model_path, pin, str(metadata["model_revision"]))

    expected_run_contract = {
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_repo_id": reproduction_dataset.repo_id,
        "dataset_revision": reproduction_dataset.revision,
        "dataset_file_sha256": reproduction_dataset.file_sha256,
        "system_prompt_file_sha256": dataset.system_prompt_file_sha256,
        "system_prompt_sha256": hashlib.sha256(
            dataset.system_prompt.encode("utf-8")
        ).hexdigest(),
        "endpoint_url": metadata["endpoint_url"],
        "served_model": pin.slug,
        "model_slug": pin.slug,
        "model_path": str(model_path),
        "model_revision": pin.inference_revision,
        "media_transport": "data-url",
        "allowed_local_media_root": None,
        "decoding": dict(EXPECTED_DECODING),
        "mm_processor_kwargs": dict(EXPECTED_MM_PROCESSOR_KWARGS),
        "shard_index": 0,
        "shard_count": 1,
    }
    _require(
        metadata.get("run_contract") == expected_run_contract,
        f"{pin.slug}: run contract mismatch",
    )
    run_identity = _canonical_hash(expected_run_contract)
    _require(
        metadata.get("run_identity_sha256") == run_identity,
        f"{pin.slug}: run identity hash mismatch",
    )

    _require(responses_path.is_file(), f"{pin.slug}: missing responses.jsonl")
    _require(
        metadata.get("responses_sha256") == _sha256_file(responses_path),
        f"{pin.slug}: responses file SHA mismatch",
    )
    records = _load_jsonl(responses_path)
    _require(len(records) == policy.rows, f"{pin.slug}: responses row count mismatch")
    expected_receipt_names = {f"{index:06d}.json" for index in range(policy.rows)}
    actual_receipt_names = {path.name for path in receipts_path.glob("*.json")}
    _require(
        actual_receipt_names == expected_receipt_names,
        f"{pin.slug}: receipt file set mismatch",
    )
    instance_ids: set[str] = set()
    request_hashes: set[str] = set()
    for index, (record, dataset_row) in enumerate(zip(records, dataset.rows)):
        prefix = f"{pin.slug} row {index}"
        _require(
            record.get("schema_version") == GENERATION_RECEIPT_SCHEMA,
            f"{prefix}: bad receipt schema",
        )
        _require(record.get("status") == "complete", f"{prefix}: incomplete response")
        _require(record.get("row_index") == index, f"{prefix}: row order mismatch")
        _require(
            record.get("instance_id") == dataset_row["instance_id"],
            f"{prefix}: instance mismatch",
        )
        _require(record.get("model_slug") == pin.slug, f"{prefix}: model mismatch")
        for field in ("task", "domain", "answer_type"):
            _require(
                record.get(field) == dataset_row[field], f"{prefix}: {field} mismatch"
            )
        _require(
            record.get("run_identity_sha256") == run_identity,
            f"{prefix}: run identity mismatch",
        )
        _require(
            not _contains_forbidden_ground_truth_key(record),
            f"{prefix}: ground truth leaked into generation",
        )
        raw_response = record.get("raw_response")
        _require(
            isinstance(raw_response, str) and bool(raw_response),
            f"{prefix}: empty raw response",
        )
        raw_hash = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
        _require(
            record.get("raw_response_sha256") == raw_hash,
            f"{prefix}: raw response hash mismatch",
        )
        api_response = record.get("api_response")
        _require(isinstance(api_response, dict), f"{prefix}: missing API response")
        _require(
            record.get("response_sha256") == _canonical_hash(api_response),
            f"{prefix}: API response hash mismatch",
        )
        choices = api_response.get("choices")
        _require(
            isinstance(choices, list) and len(choices) == 1,
            f"{prefix}: response choice count mismatch",
        )
        choice = choices[0]
        _require(
            isinstance(choice, dict) and isinstance(choice.get("message"), dict),
            f"{prefix}: invalid choice",
        )
        _require(
            choice["message"].get("content") == raw_response,
            f"{prefix}: raw/API response mismatch",
        )
        _require(
            choice.get("finish_reason") == record.get("finish_reason"),
            f"{prefix}: finish reason mismatch",
        )
        _require(api_response.get("model") == pin.slug, f"{prefix}: API model mismatch")
        request_summary = record.get("request")
        _require(
            isinstance(request_summary, dict), f"{prefix}: missing request summary"
        )
        _require(
            request_summary.get("prompt_sha256")
            == hashlib.sha256(dataset_row["prompt_answer"].encode("utf-8")).hexdigest(),
            f"{prefix}: prompt hash mismatch",
        )
        _require(
            request_summary.get("ordered_image_sha256")
            == [image["sha256"] for image in dataset_row["images"]],
            f"{prefix}: image hash order mismatch",
        )
        expected_request_hash = _expected_request_hash(
            dataset_row, dataset, served_model=pin.slug
        )
        _require(
            record.get("request_hash") == expected_request_hash,
            f"{prefix}: request hash mismatch",
        )
        _require(
            expected_request_hash not in request_hashes,
            f"{prefix}: duplicate request hash",
        )
        request_hashes.add(expected_request_hash)
        _require(record.get("error") is None, f"{prefix}: response records an error")
        _require(
            isinstance(record.get("attempt_count"), int)
            and record["attempt_count"] >= 1,
            f"{prefix}: bad attempt count",
        )
        receipt = _load_json(receipts_path / f"{index:06d}.json")
        _require(receipt == record, f"{prefix}: receipt differs from responses.jsonl")
        _require(
            record["instance_id"] not in instance_ids,
            f"{prefix}: duplicate instance ID",
        )
        instance_ids.add(record["instance_id"])
    _require(
        len(instance_ids) == policy.rows,
        f"{pin.slug}: incomplete unique instance coverage",
    )
    expected_response_bytes = b"".join(
        (_canonical_json(record) + "\n").encode("utf-8") for record in records
    )
    _require(
        responses_path.read_bytes() == expected_response_bytes,
        f"{pin.slug}: responses.jsonl is not the byte-exact ordered receipt concatenation",
    )
    return metadata, records


def verify_generation_phase(
    *,
    suite_path: Path,
    dataset_manifest: Path,
    dataset_parquet: Path,
    generation_root: Path,
    dataset_equivalence_receipt: Path = DEFAULT_DATASET_EQUIVALENCE_RECEIPT,
    policy: VerificationPolicy = PRODUCTION_POLICY,
) -> tuple[dict[str, Any], DatasetArtifacts, GenerationArtifacts]:
    suite = _verify_suite(suite_path, policy)
    dataset = _verify_dataset(
        dataset_manifest,
        dataset_parquet,
        suite,
        policy,
        dataset_equivalence_receipt,
    )
    generation_root = generation_root.expanduser().resolve()
    _require(generation_root.is_dir(), f"missing generation root: {generation_root}")
    expected_slugs = {pin.slug for pin in policy.models}
    actual_slugs = {
        child.name
        for child in generation_root.iterdir()
        if child.is_dir() and (child / "run_metadata.json").exists()
    }
    _require(
        actual_slugs == expected_slugs,
        f"generation roster mismatch: {sorted(actual_slugs)}",
    )
    response_paths: dict[str, Path] = {}
    metadata_by_slug: dict[str, dict[str, Any]] = {}
    records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for pin in policy.models:
        model_dir = generation_root / pin.slug
        metadata, records = _verify_generation_model(model_dir, pin, dataset, policy)
        metadata_by_slug[pin.slug] = metadata
        response_paths[pin.slug] = model_dir / "responses.jsonl"
        for index, record in enumerate(records):
            key = (pin.slug, index)
            _require(key not in records_by_key, f"duplicate generation key: {key}")
            records_by_key[key] = record
    _require(
        len(records_by_key) == policy.rows * len(policy.models),
        "generation coverage is not rows × exact roster",
    )
    return (
        suite,
        dataset,
        GenerationArtifacts(response_paths, metadata_by_slug, records_by_key),
    )


def _canonical_answer(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return value
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _aggregate(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key, members in sorted(
        groups.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        record = {field: value for field, value in zip(group_fields, key)}
        route_counts = collections.Counter(
            str(row["final_extraction_route"]) for row in members
        )
        means = {
            field: float(sum(float(row[field]) for row in members) / len(members))
            for field in _AGGREGATE_METRICS
        }
        judge_requested_rows = sum(int(row["judge_requested"]) for row in members)
        judge_resolved_rows = sum(int(row["judge_resolved"]) for row in members)
        unresolved_rows = sum(int(row["unresolved"]) for row in members)
        record.update(
            {
                "rows": len(members),
                "accuracy_denominator_rows": len(members),
                "historical_answer_accuracy": means["historical_answer_correct"],
                "historical_format_rate": means["historical_format_correct"],
                "terminal_rfc_json_rate": means["terminal_rfc_json"],
                "historical_mean_reward": means["historical_reward"],
                "deterministic_semantic_accuracy": means[
                    "deterministic_semantic_correct"
                ],
                "combined_semantic_accuracy": means["combined_semantic_correct"],
                "deterministic_found_rate": means["deterministic_found"],
                "judge_fallback_rate": means["judge_requested"],
                "judge_requested_rows": judge_requested_rows,
                "judge_resolved_rows": judge_resolved_rows,
                "judge_resolved_overall_rate": means["judge_resolved"],
                "judge_resolution_rate": (
                    judge_resolved_rows / judge_requested_rows
                    if judge_requested_rows
                    else 0.0
                ),
                "unresolved_rows": unresolved_rows,
                "unresolved_rate": means["unresolved"],
                "route_counts": dict(sorted(route_counts.items())),
            }
        )
        output.append(record)
    return output


def _summary_markdown(overall: list[dict[str, Any]], labels: dict[str, str]) -> str:
    lines = [
        "# TRACE validation results",
        "",
        "The split contains 2,000 unseen validation instances from the same 1,000 task programs used for training.",
        "",
        "| Model | Strict answer | Terminal format | Deterministic semantic | Combined semantic | Judge fallback | Unresolved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| {label} | {ha:.2%} | {fmt:.2%} | {det:.2%} | {comb:.2%} | {judge:.2%} | {unresolved:.2%} |".format(
                label=labels.get(str(row["model_slug"]), str(row["model_slug"])),
                ha=row["historical_answer_accuracy"],
                fmt=row["historical_format_rate"],
                det=row["deterministic_semantic_accuracy"],
                comb=row["combined_semantic_accuracy"],
                judge=row["judge_fallback_rate"],
                unresolved=row["unresolved_rate"],
            )
        )
    lines.extend(
        [
            "",
            "Strict answer accuracy uses the training reward scorer. Combined semantic accuracy applies the same deterministic and Qwen3 extraction policy to every model; unresolved rows remain incorrect and in the denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_judge_answer(result: Mapping[str, Any], pending: Mapping[str, Any]) -> None:
    status = result.get("judge_status")
    _require(status in {"ok", "missing", "ambiguous", "failed"}, "invalid judge status")
    answer = result.get("answer")
    evidence = result.get("evidence")
    if status != "ok":
        _require(
            answer is None and evidence == "",
            "judge abstention/failure payload is invalid",
        )
        return
    _require(isinstance(evidence, str) and bool(evidence), "judge evidence is empty")
    _require(evidence in pending["raw_response"], "judge evidence is not verbatim")
    answer_type = pending["answer_type"]
    if answer_type == "integer":
        _require(
            isinstance(answer, int) and not isinstance(answer, bool),
            "judge integer is invalid",
        )
    elif answer_type == "number":
        _require(
            isinstance(answer, (int, float)) and not isinstance(answer, bool),
            "judge number is invalid",
        )
        _require(math.isfinite(float(answer)), "judge number is nonfinite")
    elif answer_type == "option_letter":
        _require(
            isinstance(answer, str) and re.fullmatch(r"[A-Z]", answer) is not None,
            "judge option is invalid",
        )
    elif answer_type == "string":
        _require(
            isinstance(answer, str) and bool(answer.strip()), "judge string is invalid"
        )


def _verify_final_scoring(
    *,
    score_dir: Path,
    judge_results_path: Path,
    suite_path: Path,
    suite: dict[str, Any],
    dataset: DatasetArtifacts,
    generation: GenerationArtifacts,
    policy: VerificationPolicy,
) -> dict[str, Any]:
    from .answer_extraction import extract_answer
    from .judge_extract import validate_judge_output
    from .score import _historical_scores

    score_dir = score_dir.expanduser().resolve()
    judge_results_path = judge_results_path.expanduser().resolve()
    scored_path = score_dir / "scored_rows.jsonl"
    pending_path = score_dir / "judge_pending.jsonl"
    summary_path = score_dir / "summary.json"
    by_task_path = score_dir / "by_task.jsonl"
    markdown_path = score_dir / "summary.md"
    for path in (
        scored_path,
        pending_path,
        summary_path,
        by_task_path,
        markdown_path,
        judge_results_path,
    ):
        _require(path.is_file(), f"missing final artifact: {path}")

    scored = _load_jsonl(scored_path)
    expected_total = policy.rows * len(policy.models)
    _require(
        len(scored) == expected_total, "scored row count is not rows × exact roster"
    )
    expected_order = sorted(generation.records)
    actual_order = [
        (str(row.get("model_slug")), int(row.get("row_index", -1))) for row in scored
    ]
    _require(
        actual_order == expected_order,
        "scored rows are missing, duplicated, or out of order",
    )
    scored_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    expected_pending_keys: set[tuple[str, int]] = set()
    for row, key in zip(scored, actual_order):
        model_slug, row_index = key
        generation_row = generation.records[key]
        dataset_row = dataset.rows[row_index]
        prefix = f"scored {model_slug} row {row_index}"
        _require(
            row.get("schema_version") == "trace-validation-scored-row-v1",
            f"{prefix}: bad schema",
        )
        _require(
            row.get("scoring_contract_version") == SCORING_CONTRACT_VERSION,
            f"{prefix}: scoring version mismatch",
        )
        _require(
            row.get("deterministic_extraction_version") == ANSWER_EXTRACTION_VERSION,
            f"{prefix}: extractor version mismatch",
        )
        for field in ("instance_id", "task", "domain", "answer_type"):
            _require(
                row.get(field) == dataset_row[field], f"{prefix}: {field} mismatch"
            )
        _require(
            row.get("answer_gt") == dataset_row["answer_gt"],
            f"{prefix}: answer_gt mismatch",
        )
        _require(
            row.get("raw_response_sha256") == generation_row["raw_response_sha256"],
            f"{prefix}: response hash mismatch",
        )
        _require(
            row.get("generation_finish_reason") == generation_row["finish_reason"],
            f"{prefix}: finish reason mismatch",
        )
        _require(
            row.get("generation_request_hash") == generation_row["request_hash"],
            f"{prefix}: request hash mismatch",
        )
        for field in (
            "historical_answer_correct",
            "historical_format_correct",
            "terminal_rfc_json",
            "deterministic_found",
            "deterministic_semantic_correct",
            "judge_requested",
            "judge_resolved",
            "combined_semantic_correct",
            "unresolved",
        ):
            _require(row.get(field) in {0, 1}, f"{prefix}: {field} is not binary")
        expected_historical = _historical_scores(
            generation_row["raw_response"], dataset_row["answer_gt"]
        )
        for field, expected_value in expected_historical.items():
            _require(
                row.get(field) == expected_value,
                f"{prefix}: {field} mismatch",
            )
        _require(
            row["judge_requested"] == 1 - row["deterministic_found"],
            f"{prefix}: judge routing mismatch",
        )
        _require(
            row["judge_resolved"] <= row["judge_requested"],
            f"{prefix}: impossible judge resolution",
        )
        _require(
            row["unresolved"]
            == int(row["judge_requested"] and not row["judge_resolved"]),
            f"{prefix}: unresolved mismatch",
        )
        extraction = row.get("deterministic_extraction")
        _require(
            isinstance(extraction, dict), f"{prefix}: no deterministic extraction audit"
        )
        independent_extraction = extract_answer(
            generation_row["raw_response"],
            answer_type=dataset_row["answer_type"],
        ).as_dict()
        _require(
            extraction == independent_extraction,
            f"{prefix}: deterministic extraction audit mismatch",
        )
        if row["deterministic_found"]:
            _require(
                extraction.get("status") == "found", f"{prefix}: found/status mismatch"
            )
            candidate = extraction.get("typed_candidate")
            correct = int(
                _canonical_answer(candidate)
                == _canonical_answer(dataset_row["answer_gt"]["value"])
            )
            _require(
                row["deterministic_semantic_correct"] == correct,
                f"{prefix}: deterministic score mismatch",
            )
            _require(
                row["final_extraction_route"]
                == f"deterministic:{extraction.get('route')}",
                f"{prefix}: deterministic route mismatch",
            )
        else:
            _require(
                extraction.get("status") in {"missing", "ambiguous"},
                f"{prefix}: invalid fallback status",
            )
            _require(
                row["deterministic_semantic_correct"] == 0,
                f"{prefix}: fallback row has a deterministic score",
            )
            expected_pending_keys.add(key)
        scored_by_key[key] = row

    pending_rows = _load_jsonl(pending_path)
    pending_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for pending in pending_rows:
        key = (str(pending.get("model_slug")), int(pending.get("row_index", -1)))
        _require(key not in pending_by_key, f"duplicate judge pending key: {key}")
        _require(key in expected_pending_keys, f"unexpected judge pending key: {key}")
        generation_row = generation.records[key]
        dataset_row = dataset.rows[key[1]]
        _require(
            pending.get("instance_id") == dataset_row["instance_id"],
            f"pending {key}: instance mismatch",
        )
        _require(
            pending.get("answer_type") == dataset_row["answer_type"],
            f"pending {key}: type mismatch",
        )
        _require(
            pending.get("raw_response") == generation_row["raw_response"],
            f"pending {key}: response mismatch",
        )
        _require(
            pending.get("raw_response_sha256") == generation_row["raw_response_sha256"],
            f"pending {key}: hash mismatch",
        )
        _require(
            pending.get("deterministic_status") in {"missing", "ambiguous"},
            f"pending {key}: status mismatch",
        )
        _require(
            pending.get("deterministic_extraction_version")
            == ANSWER_EXTRACTION_VERSION,
            f"pending {key}: extractor version mismatch",
        )
        _require(
            set(pending)
            == {
                "model_slug",
                "row_index",
                "instance_id",
                "answer_type",
                "raw_response",
                "raw_response_sha256",
                "deterministic_status",
                "deterministic_extraction_version",
            },
            f"pending {key}: fields differ from the GT-blind contract",
        )
        _require(
            not _contains_forbidden_ground_truth_key(pending),
            f"pending {key}: ground truth leak",
        )
        pending_by_key[key] = pending
    _require(
        set(pending_by_key) == expected_pending_keys, "judge pending coverage mismatch"
    )

    judge_rows = _load_jsonl(judge_results_path)
    judge_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    request_hashes: set[str] = set()
    for result in judge_rows:
        key = (str(result.get("model_slug")), int(result.get("row_index", -1)))
        _require(key not in judge_by_key, f"duplicate judge result key: {key}")
        _require(key in pending_by_key, f"judge result has no pending row: {key}")
        pending = pending_by_key[key]
        _require(
            result.get("schema_version") == "trace-validation-judge-receipt-v1",
            f"judge {key}: bad schema",
        )
        _require(
            result.get("contract_version") == JUDGE_CONTRACT_VERSION,
            f"judge {key}: contract mismatch",
        )
        _require(
            result.get("judge_revision") == suite["judge"]["revision"],
            f"judge {key}: revision mismatch",
        )
        _require(
            result.get("judge_model") == suite["judge"]["served_model_name"],
            f"judge {key}: model mismatch",
        )
        _require(
            result.get("tokenizer_model") == suite["judge"]["repo_id"],
            f"judge {key}: tokenizer mismatch",
        )
        _require(
            result.get("instance_id") == pending["instance_id"],
            f"judge {key}: instance mismatch",
        )
        _require(
            result.get("answer_type") == pending["answer_type"],
            f"judge {key}: type mismatch",
        )
        _require(
            result.get("deterministic_status") == pending["deterministic_status"],
            f"judge {key}: deterministic status mismatch",
        )
        _require(
            result.get("deterministic_extraction_version")
            == pending["deterministic_extraction_version"],
            f"judge {key}: extractor version mismatch",
        )
        _require(
            result.get("raw_response_sha256") == pending["raw_response_sha256"],
            f"judge {key}: raw hash mismatch",
        )
        _require(
            result.get("system_prompt_sha256") == EXPECTED_JUDGE_SYSTEM_PROMPT_SHA256,
            f"judge {key}: prompt hash mismatch",
        )
        _require(
            _is_sha256(result.get("rendered_prompt_sha256")),
            f"judge {key}: rendered hash invalid",
        )
        _require(
            _is_sha256(result.get("tokenizer_template_sha256")),
            f"judge {key}: template hash invalid",
        )
        request_hash = result.get("request_hash")
        _require(_is_sha256(request_hash), f"judge {key}: request hash invalid")
        expected_request_contract = {
            "contract_version": JUDGE_CONTRACT_VERSION,
            "model_slug": key[0],
            "row_index": key[1],
            "instance_id": pending["instance_id"],
            "raw_response_sha256": pending["raw_response_sha256"],
            "answer_type": pending["answer_type"],
            "deterministic_status": pending["deterministic_status"],
            "deterministic_extraction_version": ANSWER_EXTRACTION_VERSION,
            "judge_model": suite["judge"]["served_model_name"],
            "judge_revision": suite["judge"]["revision"],
            "tokenizer_model": suite["judge"]["repo_id"],
            "system_prompt_sha256": EXPECTED_JUDGE_SYSTEM_PROMPT_SHA256,
            "rendered_prompt_sha256": result["rendered_prompt_sha256"],
            "tokenizer_template_sha256": result["tokenizer_template_sha256"],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "retry_token_limits": suite["judge"]["max_token_retries"],
        }
        _require(
            result.get("request_contract") == expected_request_contract,
            f"judge {key}: request contract mismatch",
        )
        expected_judge_request_hash = hashlib.sha256(
            _canonical_json(expected_request_contract).encode("utf-8")
        ).hexdigest()
        _require(
            request_hash == expected_judge_request_hash,
            f"judge {key}: request hash mismatch",
        )
        _require(
            request_hash not in request_hashes,
            f"judge {key}: duplicate request hash/receipt",
        )
        request_hashes.add(str(request_hash))
        _verify_judge_answer(result, pending)
        attempts = result.get("attempts")
        _require(
            isinstance(attempts, list) and bool(attempts),
            f"judge {key}: no attempt audit",
        )
        retry_limits = suite["judge"]["max_token_retries"]
        _require(len(attempts) <= len(retry_limits), f"judge {key}: too many attempts")
        validated_attempts: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
        for attempt_number, attempt in enumerate(attempts):
            _require(isinstance(attempt, dict), f"judge {key}: invalid attempt audit")
            _require(
                attempt.get("retry_number") == attempt_number,
                f"judge {key}: retry order mismatch",
            )
            _require(
                attempt.get("max_tokens") == retry_limits[attempt_number],
                f"judge {key}: retry token mismatch",
            )
            _require(
                isinstance(attempt.get("raw_output"), str),
                f"judge {key}: non-string raw output",
            )
            try:
                parsed_attempt = validate_judge_output(
                    attempt["raw_output"],
                    raw_response=pending["raw_response"],
                    answer_type=pending["answer_type"],
                )
            except Exception:
                _require(
                    isinstance(attempt.get("validation_error"), str)
                    and bool(attempt["validation_error"]),
                    f"judge {key}: invalid attempt lacks validation error",
                )
            else:
                _require(
                    attempt.get("validation_error") is None,
                    f"judge {key}: valid attempt is marked as failed",
                )
                validated_attempts.append((attempt, parsed_attempt))
        if result["judge_status"] == "failed":
            _require(
                len(attempts) == len(retry_limits),
                f"judge {key}: failed row lacks all retries",
            )
            _require(
                not validated_attempts,
                f"judge {key}: failed row contains valid attempt",
            )
        else:
            _require(
                len(validated_attempts) == 1
                and validated_attempts[0][0] is attempts[-1],
                f"judge {key}: successful attempt audit mismatch",
            )
            parsed_attempt = validated_attempts[0][1]
            _require(
                parsed_attempt
                == {
                    "status": result["judge_status"],
                    "answer": result["answer"],
                    "evidence": result["evidence"],
                },
                f"judge {key}: receipt does not match validated raw attempt",
            )
        receipt_path = judge_results_path.parent / "receipts" / f"{request_hash}.json"
        _require(
            _load_json(receipt_path) == result,
            f"judge {key}: receipt differs from results",
        )
        judge_by_key[key] = result
    _require(
        set(judge_by_key) == expected_pending_keys,
        "judge results are not one-to-one with pending",
    )
    actual_judge_receipts = {
        path.name for path in (judge_results_path.parent / "receipts").glob("*.json")
    }
    expected_judge_receipts = {
        f"{request_hash}.json" for request_hash in request_hashes
    }
    _require(
        actual_judge_receipts == expected_judge_receipts,
        "judge receipt file set mismatch",
    )

    for key, row in scored_by_key.items():
        if key not in expected_pending_keys:
            _require(
                row.get("judge_status") is None and row.get("judge_answer") is None,
                f"scored {key}: unexpected judge data",
            )
            _require(
                row.get("judge_evidence") == "",
                f"scored {key}: unexpected judge evidence",
            )
            _require(
                row["combined_semantic_correct"]
                == row["deterministic_semantic_correct"],
                f"scored {key}: combined mismatch",
            )
            continue
        judge = judge_by_key[key]
        _require(
            row.get("judge_status") == judge["judge_status"],
            f"scored {key}: judge status mismatch",
        )
        _require(
            row.get("judge_answer") == judge["answer"],
            f"scored {key}: judge answer mismatch",
        )
        _require(
            row.get("judge_evidence") == judge["evidence"],
            f"scored {key}: judge evidence mismatch",
        )
        resolved = int(judge["judge_status"] == "ok")
        _require(
            row["judge_resolved"] == resolved, f"scored {key}: judge resolved mismatch"
        )
        expected_correct = (
            int(
                _canonical_answer(judge["answer"])
                == _canonical_answer(dataset.rows[key[1]]["answer_gt"]["value"])
            )
            if resolved
            else 0
        )
        _require(
            row["combined_semantic_correct"] == expected_correct,
            f"scored {key}: combined score mismatch",
        )
        expected_route = "judge" if resolved else f"judge:{judge['judge_status']}"
        _require(
            row["final_extraction_route"] == expected_route,
            f"scored {key}: final route mismatch",
        )

    summary = _load_json(summary_path)
    expected_overall = _aggregate(scored, ("model_slug",))
    expected_by_domain = _aggregate(scored, ("model_slug", "domain"))
    expected_by_answer_type = _aggregate(scored, ("model_slug", "answer_type"))
    expected_by_task = _aggregate(scored, ("model_slug", "task"))
    _require(
        summary.get("overall") == expected_overall, "overall summary is inconsistent"
    )
    _require(
        summary.get("by_domain") == expected_by_domain, "domain summary is inconsistent"
    )
    _require(
        summary.get("by_answer_type") == expected_by_answer_type,
        "answer-type summary is inconsistent",
    )
    _require(
        _load_jsonl(by_task_path) == expected_by_task, "task summary is inconsistent"
    )
    labels = {str(model["slug"]): str(model["label"]) for model in suite["models"]}
    _require(
        markdown_path.read_text(encoding="utf-8")
        == _summary_markdown(expected_overall, labels),
        "summary.md is inconsistent",
    )

    provenance = summary.get("provenance")
    _require(isinstance(provenance, dict), "summary has no provenance")
    _require(
        provenance.get("schema_version") == "trace-validation-score-provenance-v1",
        "bad score provenance schema",
    )
    _require(
        provenance.get("scoring_contract_version") == SCORING_CONTRACT_VERSION,
        "score provenance version mismatch",
    )
    _require(
        Path(str(provenance.get("dataset_manifest"))).resolve()
        == dataset.manifest_path,
        "score dataset path mismatch",
    )
    _require(
        provenance.get("dataset_manifest_sha256") == dataset.manifest_sha256,
        "score dataset hash mismatch",
    )
    _require(
        provenance.get("dataset_identity") == dataset.manifest["dataset"],
        "score dataset identity mismatch",
    )
    _require(
        provenance.get("dataset_equivalence") == dataset.equivalence,
        "score dataset-equivalence provenance mismatch",
    )
    _require(
        Path(str(provenance.get("suite"))).resolve() == suite_path.resolve(),
        "score suite path mismatch",
    )
    _require(
        provenance.get("suite_sha256") == _sha256_file(suite_path),
        "score suite hash mismatch",
    )
    generation_files = provenance.get("generation_files")
    _require(
        isinstance(generation_files, list), "score generation provenance is invalid"
    )
    actual_generation_files = {
        Path(str(item.get("path"))).resolve(): item.get("sha256")
        for item in generation_files
        if isinstance(item, dict)
    }
    expected_generation_files = {
        path.resolve(): _sha256_file(path)
        for path in generation.response_paths.values()
    }
    _require(
        actual_generation_files == expected_generation_files,
        "score generation hashes mismatch",
    )
    judge_provenance = provenance.get("judge_results")
    _require(isinstance(judge_provenance, dict), "final score has no judge provenance")
    _require(
        Path(str(judge_provenance.get("path"))).resolve() == judge_results_path,
        "score judge path mismatch",
    )
    _require(
        judge_provenance.get("sha256") == _sha256_file(judge_results_path),
        "score judge hash mismatch",
    )
    _require(
        provenance.get("models") == len(policy.models), "score model count mismatch"
    )
    _require(
        provenance.get("model_slugs") == sorted(pin.slug for pin in policy.models),
        "score model roster mismatch",
    )
    _require(
        provenance.get("rows_per_model") == policy.rows, "score rows-per-model mismatch"
    )
    _require(provenance.get("total_rows") == expected_total, "score total-row mismatch")
    _require(
        provenance.get("accuracy_denominator_rows") == expected_total,
        "score denominator mismatch",
    )
    _require(
        provenance.get("judge_requested_rows") == len(expected_pending_keys),
        "score requested count mismatch",
    )
    _require(
        provenance.get("judge_pending_rows") == len(expected_pending_keys),
        "score pending count mismatch",
    )
    _require(
        provenance.get("judge_result_rows") == len(judge_rows),
        "score judge-result count mismatch",
    )
    _require(
        provenance.get("judge_resolved_rows")
        == sum(int(row["judge_resolved"]) for row in scored),
        "score judge-resolved count mismatch",
    )
    _require(
        provenance.get("unresolved_rows")
        == sum(int(row["unresolved"]) for row in scored),
        "score unresolved count mismatch",
    )
    _require(
        provenance.get("deterministic_extraction_contract_version")
        == ANSWER_EXTRACTION_VERSION,
        "score extractor provenance mismatch",
    )
    _require(
        provenance.get("judge_extraction_contract_version") == JUDGE_CONTRACT_VERSION,
        "score judge contract provenance mismatch",
    )
    _require(
        provenance.get("judge_model")
        == {
            "repo_id": suite["judge"]["repo_id"],
            "revision": suite["judge"]["revision"],
            "served_model_name": suite["judge"]["served_model_name"],
        },
        "score judge model provenance mismatch",
    )
    _require(
        provenance.get("denominator_policy")
        == {
            "accuracy": "all_generation_rows",
            "judge_fallback_rate": "all_generation_rows",
            "judge_resolution_rate": "judge_requested_rows",
            "unresolved_rows_score": 0,
            "drop_failed_rows": False,
        },
        "score denominator policy mismatch",
    )
    _require(provenance.get("judge_finalized") is True, "score is not judge-finalized")
    code_paths = {
        "score": REPO_ROOT / "rlvr/evaluation/trace_validation/score.py",
        "answer_extraction": REPO_ROOT
        / "rlvr/evaluation/trace_validation/answer_extraction.py",
        "judge_extraction": REPO_ROOT
        / "rlvr/evaluation/trace_validation/judge_extract.py",
        "reward_scoring": REPO_ROOT / "src/trace_tasks/core/reward_scoring.py",
        "dataset_preparation": REPO_ROOT
        / "rlvr/evaluation/trace_validation/prepare_dataset.py",
    }
    expected_code_sha = {key: _sha256_file(path) for key, path in code_paths.items()}
    _require(
        provenance.get("code_sha256") == expected_code_sha,
        "scoring code provenance mismatch",
    )
    return {
        "scored_rows": len(scored),
        "judge_pending_rows": len(pending_rows),
        "judge_result_rows": len(judge_rows),
        "unresolved_rows": sum(int(row["unresolved"]) for row in scored),
        "summary_sha256": _sha256_file(summary_path),
    }


def verify_full_phase(
    *,
    suite_path: Path,
    dataset_manifest: Path,
    dataset_parquet: Path,
    generation_root: Path,
    score_dir: Path,
    judge_results: Path,
    dataset_equivalence_receipt: Path = DEFAULT_DATASET_EQUIVALENCE_RECEIPT,
    policy: VerificationPolicy = PRODUCTION_POLICY,
) -> dict[str, Any]:
    suite, dataset, generation = verify_generation_phase(
        suite_path=suite_path,
        dataset_manifest=dataset_manifest,
        dataset_parquet=dataset_parquet,
        generation_root=generation_root,
        dataset_equivalence_receipt=dataset_equivalence_receipt,
        policy=policy,
    )
    scoring_report = _verify_final_scoring(
        score_dir=score_dir,
        judge_results_path=judge_results,
        suite_path=suite_path,
        suite=suite,
        dataset=dataset,
        generation=generation,
        policy=policy,
    )
    return {
        "phase": "full",
        "status": "ok",
        "suite_sha256": policy.suite_sha256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_equivalence_receipt_sha256": dataset.equivalence["receipt_sha256"],
        "models": len(policy.models),
        "rows_per_model": policy.rows,
        "generation_rows": len(generation.records),
        **scoring_report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("release", "generation-only", "full"))
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--release-receipt", type=Path, default=DEFAULT_RELEASE_RECEIPT)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--dataset-parquet", type=Path, default=DEFAULT_DATASET_PARQUET)
    parser.add_argument(
        "--dataset-equivalence-receipt",
        type=Path,
        default=DEFAULT_DATASET_EQUIVALENCE_RECEIPT,
    )
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT / "generation",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT / "scoring" / "final",
    )
    parser.add_argument(
        "--judge-results",
        type=Path,
        default=DEFAULT_CAMPAIGN_ROOT / "judge" / "judge_results.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.phase == "release":
            report = verify_release_metadata(
                suite_path=args.suite,
                results_path=args.results,
                receipt_path=args.release_receipt,
            )
        elif args.phase == "generation-only":
            _, dataset, generation = verify_generation_phase(
                suite_path=args.suite,
                dataset_manifest=args.dataset_manifest,
                dataset_parquet=args.dataset_parquet,
                generation_root=args.generation_root,
                dataset_equivalence_receipt=args.dataset_equivalence_receipt,
            )
            report = {
                "phase": "generation-only",
                "status": "ok",
                "suite_sha256": EXPECTED_SUITE_SHA256,
                "dataset_manifest_sha256": dataset.manifest_sha256,
                "dataset_equivalence_receipt_sha256": dataset.equivalence[
                    "receipt_sha256"
                ],
                "models": len(PRODUCTION_POLICY.models),
                "rows_per_model": PRODUCTION_POLICY.rows,
                "generation_rows": len(generation.records),
            }
        else:
            report = verify_full_phase(
                suite_path=args.suite,
                dataset_manifest=args.dataset_manifest,
                dataset_parquet=args.dataset_parquet,
                generation_root=args.generation_root,
                score_dir=args.score_dir,
                judge_results=args.judge_results,
                dataset_equivalence_receipt=args.dataset_equivalence_receipt,
            )
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(
            _canonical_json(
                {
                    "phase": args.phase,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
