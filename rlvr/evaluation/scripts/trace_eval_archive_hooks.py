#!/usr/bin/env python3
"""Non-blocking archive hooks for completed TRACE evaluation slices.

The evaluation runners should call the stage-specific ``emit_*_slice`` helpers
after their normal local artifact is durable.  The helpers are a no-op unless
``TRACE_EVAL_HF_SPOOL_ROOT`` is set.  When enabled, they validate provenance,
remove benchmark media and credentials from source rows, and delegate the
atomic local snapshot to :mod:`trace_eval_hf_archive_lib`.

Uploading is deliberately outside this module.  ``trace_eval_hf_archive.py
daemon`` consumes the resulting descriptors without blocking evaluation GPUs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from scripts.trace_eval_hf_archive_lib import (
        MEDIA_SUFFIXES,
        OPAQUE_TEXT_RECORD_KEYS,
        OPAQUE_TEXT_RECORD_PATHS,
        SAFE_MEDIA_METADATA_KEYS,
        SECRET_KEY_PARTS,
        ArchiveValidationError,
        canonical_json,
        emit_slice_ready as _emit_slice_ready,
        sanitize_archive_value,
    )
except ModuleNotFoundError:  # Supports direct ``python scripts/...`` imports.
    from trace_eval_hf_archive_lib import (
        MEDIA_SUFFIXES,
        OPAQUE_TEXT_RECORD_KEYS,
        OPAQUE_TEXT_RECORD_PATHS,
        SAFE_MEDIA_METADATA_KEYS,
        SECRET_KEY_PARTS,
        ArchiveValidationError,
        canonical_json,
        emit_slice_ready as _emit_slice_ready,
        sanitize_archive_value,
    )


HOOK_VERSION = "trace-final25-archive-hooks-v1"
SPOOL_ROOT_ENV = "TRACE_EVAL_HF_SPOOL_ROOT"
RUN_ID_ENV = "TRACE_EVAL_RUN_ID"
TRACE_COMMIT_ENVS = ("TRACE_EVAL_TRACE_GIT_COMMIT", "TRACE_GIT_COMMIT")
VLMEVALKIT_COMMIT_ENVS = (
    "TRACE_EVAL_VLMEVALKIT_GIT_COMMIT",
    "TRACE_VLMEVALKIT_GIT_COMMIT",
    "VLMEVALKIT_GIT_COMMIT",
)
CAMPAIGN_HASH_ENV = "TRACE_EVAL_CAMPAIGN_CONFIG_HASH"
CAMPAIGN_PATH_ENV = "TRACE_EVAL_CAMPAIGN_CONFIG_PATH"
WANDB_RUN_ID_ENVS = ("TRACE_EVAL_WANDB_RUN_ID", "WANDB_RUN_ID")
CONTRACT_VERSION_ENV = "TRACE_EVAL_CONTRACT_VERSION"
MODEL_REVISIONS_ENV = "TRACE_EVAL_MODEL_REVISIONS_JSON"
MODEL_SOURCES_ENV = "TRACE_EVAL_MODEL_SOURCES_JSON"
REPO_ROOT = Path(__file__).resolve().parents[3]

_OMIT = object()


@dataclass(frozen=True)
class ArchiveHookContext:
    spool_root: Path
    run_id: str
    trace_git_commit: str
    vlmevalkit_git_commit: str
    campaign_config_hash: str
    wandb_run_id: str | None = None


def _first_nonempty(env: Mapping[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = str(env.get(name, "")).strip()
        if value:
            return value
    return None


def archive_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether local archive emission is explicitly enabled."""

    values = os.environ if env is None else env
    return bool(str(values.get(SPOOL_ROOT_ENV, "")).strip())


def _resolve_model_metadata(
    environment_name: str,
    model_slug: str,
    fallback: str,
    env: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if env is None else env
    encoded = str(values.get(environment_name, "")).strip()
    if encoded:
        try:
            configured_values = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ArchiveValidationError(f"{environment_name} must be valid JSON") from error
        if not isinstance(configured_values, Mapping):
            raise ArchiveValidationError(f"{environment_name} must be a JSON object")
        configured = str(configured_values.get(model_slug, "")).strip()
        if configured:
            return configured
    return str(fallback)


def resolve_model_revision(
    model_slug: str,
    fallback: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve a per-model immutable revision from the campaign environment."""

    values = os.environ if env is None else env
    configured = _resolve_model_metadata(MODEL_REVISIONS_ENV, model_slug, "", values)
    if configured:
        return configured
    single = str(values.get("TRACE_EVAL_MODEL_REVISION", "")).strip()
    return single or str(fallback)


def resolve_model_source(
    model_slug: str,
    fallback: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the stable HF repository or training-run source for a model."""

    return _resolve_model_metadata(MODEL_SOURCES_ENV, model_slug, fallback, env)


def _git_commit(path: Path, label: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ArchiveValidationError(
            f"{label} commit is required; set its archive environment variable"
        ) from error
    commit = completed.stdout.strip()
    if not commit:
        raise ArchiveValidationError(
            f"{label} commit is required; set its archive environment variable"
        )
    return commit


def _campaign_config_hash(
    env: Mapping[str, str], campaign_config: Mapping[str, Any] | None
) -> str:
    configured = str(env.get(CAMPAIGN_HASH_ENV, "")).strip()
    if configured:
        return configured
    config_path = str(env.get(CAMPAIGN_PATH_ENV, "")).strip()
    if config_path:
        try:
            return hashlib.sha256(Path(config_path).expanduser().read_bytes()).hexdigest()
        except OSError as error:
            raise ArchiveValidationError(
                f"cannot hash campaign config from {CAMPAIGN_PATH_ENV}"
            ) from error
    if campaign_config is not None:
        safe = sanitize_archive_value(campaign_config, path="campaign_config")
        return hashlib.sha256(canonical_json(safe).encode("utf-8")).hexdigest()
    raise ArchiveValidationError(
        f"archive enabled but {CAMPAIGN_HASH_ENV} or {CAMPAIGN_PATH_ENV} is not set"
    )


def load_archive_context(
    *,
    env: Mapping[str, str] | None = None,
    campaign_config: Mapping[str, Any] | None = None,
) -> ArchiveHookContext | None:
    """Load run-wide archive provenance, or return ``None`` when disabled."""

    values = os.environ if env is None else env
    spool_text = str(values.get(SPOOL_ROOT_ENV, "")).strip()
    if not spool_text:
        return None
    run_id = str(values.get(RUN_ID_ENV, "")).strip()
    if not run_id:
        raise ArchiveValidationError(f"archive enabled but {RUN_ID_ENV} is not set")
    trace_commit = _first_nonempty(values, TRACE_COMMIT_ENVS) or _git_commit(
        REPO_ROOT, "TRACE git"
    )
    vlmevalkit_commit = _first_nonempty(values, VLMEVALKIT_COMMIT_ENVS) or _git_commit(
        REPO_ROOT / "external" / "VLMEvalKit", "VLMEvalKit git"
    )
    return ArchiveHookContext(
        spool_root=Path(spool_text).expanduser(),
        run_id=run_id,
        trace_git_commit=trace_commit,
        vlmevalkit_git_commit=vlmevalkit_commit,
        campaign_config_hash=_campaign_config_hash(values, campaign_config),
        wandb_run_id=_first_nonempty(values, WANDB_RUN_ID_ENVS),
    )


def _key_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_secret_key(key: str) -> bool:
    return key in SECRET_KEY_PARTS or key.endswith(("_access_token", "_api_key"))


def _is_media_key(key: str) -> bool:
    if key in SAFE_MEDIA_METADATA_KEYS:
        return False
    tokens = set(key.split("_"))
    return bool(tokens & {"image", "images", "video", "videos", "audio", "media"}) or key.endswith(
        ("_image_path", "_video_path", "_media_path")
    )


def _looks_like_media_path(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    if re.search(
        r"data\s*:\s*(?:image|video|audio|application/octet-stream)[^,]*;base64,",
        lowered,
    ):
        return True
    if lowered.startswith("file://"):
        return True
    suffix = Path(lowered.split("?", 1)[0].split("#", 1)[0]).suffix
    return suffix in MEDIA_SUFFIXES and (
        stripped.startswith(("/", "~/", "./", "../", "\\\\"))
        or "/" in stripped
        or "\\" in stripped
    )


def _scrub_benchmark_value(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = _key_name(key)
            if _is_secret_key(normalized) or _is_media_key(normalized):
                continue
            scrubbed = _scrub_benchmark_value(child, path=f"{path}.{key}")
            if scrubbed is not _OMIT:
                result[key] = scrubbed
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, child in enumerate(value):
            scrubbed = _scrub_benchmark_value(child, path=f"{path}[{index}]")
            if scrubbed is not _OMIT:
                result.append(scrubbed)
        return result
    if isinstance(value, str) and _looks_like_media_path(value):
        return _OMIT
    try:
        return sanitize_archive_value(value, path=path)
    except ArchiveValidationError:
        # Source rows are intentionally lossy: binary/media values are already
        # represented by their separately retained hashes and dimensions.
        return _OMIT


def sanitize_benchmark_source_row(source_row: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return the reconstructable, non-media portion of a benchmark row."""

    if not isinstance(source_row, Mapping) and hasattr(source_row, "to_dict"):
        source_row = source_row.to_dict()
    if not isinstance(source_row, Mapping):
        raise ArchiveValidationError("source_row must be a mapping")
    scrubbed = _scrub_benchmark_value(source_row, path="source_row")
    if not isinstance(scrubbed, dict):  # Defensive; mappings always produce a dict above.
        raise ArchiveValidationError("source_row did not normalize to a mapping")
    return sanitize_archive_value(scrubbed, path="source_row")


def _required(record: Mapping[str, Any], key: str, row_number: int) -> Any:
    if key not in record:
        raise ArchiveValidationError(f"archive record {row_number} is missing {key}")
    return record[key]


def _source_ordinal(value: Any, row_number: int) -> int:
    if isinstance(value, bool):
        raise ArchiveValidationError(f"archive record {row_number} source_ordinal must be >= 0")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArchiveValidationError(
            f"archive record {row_number} source_ordinal must be >= 0"
        ) from error
    if result < 0 or result != value:
        raise ArchiveValidationError(f"archive record {row_number} source_ordinal must be >= 0")
    return result


def _common_record(record: Mapping[str, Any], row_number: int) -> dict[str, Any]:
    common = {
        "source_index": _required(record, "source_index", row_number),
        "source_ordinal": _source_ordinal(
            _required(record, "source_ordinal", row_number), row_number
        ),
        "source_row_hash": _required(record, "source_row_hash", row_number),
        "request_hash": _required(record, "request_hash", row_number),
        "question": record.get("question"),
        "ground_truth": record.get("ground_truth"),
    }
    for key in ("options", "metadata"):
        if key in record:
            scrubbed = _scrub_benchmark_value(record[key], path=key)
            if scrubbed is not _OMIT:
                common[key] = scrubbed
    return common


def generation_archive_record(record: Mapping[str, Any], row_number: int = 0) -> dict[str, Any]:
    """Normalize one completed generation row to the archive contract."""

    result = _common_record(record, row_number)
    result.update(
        {
            "source_row": sanitize_benchmark_source_row(
                _required(record, "source_row", row_number)
            ),
            "prompt": _required(record, "prompt", row_number),
            "model_response": _required(record, "model_response", row_number),
            "sampling": record.get("sampling", {}),
            "finish_reason": record.get("finish_reason"),
            "usage": record.get("usage", {}),
        }
    )
    return sanitize_archive_value(
        result,
        path=f"generation_records[{row_number}]",
        opaque_text_keys=OPAQUE_TEXT_RECORD_KEYS["generation"],
    )


def extraction_archive_record(record: Mapping[str, Any], row_number: int = 0) -> dict[str, Any]:
    """Normalize one completed answer-extraction row to the archive contract."""

    result = _common_record(record, row_number)
    result.update(
        {
            "model_response": _required(record, "model_response", row_number),
            "judge_prompt": _required(record, "judge_prompt", row_number),
            "judge_response": _required(record, "judge_response", row_number),
            "normalized_extraction": _required(
                record, "normalized_extraction", row_number
            ),
            "retries": record.get("retries", []),
        }
    )
    return sanitize_archive_value(
        result,
        path=f"extraction_records[{row_number}]",
        opaque_text_keys=OPAQUE_TEXT_RECORD_KEYS["extraction"],
        opaque_text_paths=OPAQUE_TEXT_RECORD_PATHS["extraction"],
    )


def score_archive_record(record: Mapping[str, Any], row_number: int = 0) -> dict[str, Any]:
    """Normalize one completed scoring row to the archive contract."""

    result = _common_record(record, row_number)
    result.update(
        {
            "prediction": _required(record, "prediction", row_number),
            "score": _required(record, "score", row_number),
            "scorer": _required(record, "scorer", row_number),
            "excluded": record.get("excluded", False),
        }
    )
    return sanitize_archive_value(
        result,
        path=f"score_records[{row_number}]",
        opaque_text_keys=OPAQUE_TEXT_RECORD_KEYS["score"],
    )


def _provenance(
    context: ArchiveHookContext,
    *,
    contract_version: str | None,
    env: Mapping[str, str],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    version = str(contract_version or env.get(CONTRACT_VERSION_ENV, "")).strip()
    if not version:
        raise ArchiveValidationError(
            f"contract_version is required (argument or {CONTRACT_VERSION_ENV})"
        )
    result: dict[str, Any] = {
        "trace_git_commit": context.trace_git_commit,
        "vlmevalkit_git_commit": context.vlmevalkit_git_commit,
        "contract_version": version,
        "campaign_config_hash": context.campaign_config_hash,
        "archive_hook_version": HOOK_VERSION,
    }
    code_hash = str(env.get("TRACE_EVAL_CODE_HASH", "")).strip()
    if code_hash:
        result["final25_code_hash"] = code_hash
    if context.wandb_run_id:
        result["wandb_run_id"] = context.wandb_run_id
    if extra:
        collisions = sorted(set(result) & set(extra))
        if collisions:
            raise ArchiveValidationError(
                "provenance_extra cannot override: " + ", ".join(collisions)
            )
        result.update(extra)
    return sanitize_archive_value(result, path="provenance")


def _emit_stage(
    *,
    stage: str,
    normalizer: Any,
    records: Iterable[Mapping[str, Any]],
    model: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    contract_version: str | None,
    aggregate: Mapping[str, Any] | None,
    model_slug: str | None,
    provenance_extra: Mapping[str, Any] | None,
    campaign_config: Mapping[str, Any] | None,
    env: Mapping[str, str] | None,
) -> Path | None:
    values = os.environ if env is None else env
    context = load_archive_context(env=values, campaign_config=campaign_config)
    if context is None:
        return None
    if isinstance(records, Mapping):
        record_rows: Iterable[Mapping[str, Any]] = (records,)
    elif hasattr(records, "to_dict"):
        try:
            record_rows = records.to_dict(orient="records")
        except TypeError:
            record_rows = records
    else:
        record_rows = records
    normalized = [normalizer(record, index) for index, record in enumerate(record_rows)]
    return _emit_slice_ready(
        context.spool_root,
        stage=stage,
        run_id=context.run_id,
        model=model,
        model_revision=model_revision,
        seed=seed,
        benchmark=benchmark,
        dataset_alias=dataset_alias,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        records=normalized,
        provenance=_provenance(
            context,
            contract_version=contract_version,
            env=values,
            extra=provenance_extra,
        ),
        aggregate=aggregate or {},
        model_slug=model_slug,
    )


def emit_records(
    *,
    stage: str,
    records: Iterable[Mapping[str, Any]],
    model: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    contract_version: str | None = None,
    aggregate: Mapping[str, Any] | None = None,
    model_slug: str | None = None,
    provenance_extra: Mapping[str, Any] | None = None,
    campaign_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Spool mappings or a pandas DataFrame using the named stage contract."""

    normalizers = {
        "generation": generation_archive_record,
        "extraction": extraction_archive_record,
        "score": score_archive_record,
    }
    try:
        normalizer = normalizers[stage]
    except KeyError as error:
        raise ArchiveValidationError(f"unsupported archive stage: {stage!r}") from error
    return _emit_stage(
        stage=stage,
        normalizer=normalizer,
        records=records,
        model=model,
        model_revision=model_revision,
        seed=seed,
        benchmark=benchmark,
        dataset_alias=dataset_alias,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        contract_version=contract_version,
        aggregate=aggregate,
        model_slug=model_slug,
        provenance_extra=provenance_extra,
        campaign_config=campaign_config,
        env=env,
    )


def emit_generation_slice(
    *,
    records: Iterable[Mapping[str, Any]],
    model: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    contract_version: str | None = None,
    aggregate: Mapping[str, Any] | None = None,
    model_slug: str | None = None,
    provenance_extra: Mapping[str, Any] | None = None,
    campaign_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Atomically spool one completed generation slice when archival is enabled."""

    return _emit_stage(
        stage="generation",
        normalizer=generation_archive_record,
        records=records,
        model=model,
        model_revision=model_revision,
        seed=seed,
        benchmark=benchmark,
        dataset_alias=dataset_alias,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        contract_version=contract_version,
        aggregate=aggregate,
        model_slug=model_slug,
        provenance_extra=provenance_extra,
        campaign_config=campaign_config,
        env=env,
    )


def emit_extraction_slice(
    *,
    records: Iterable[Mapping[str, Any]],
    model: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    contract_version: str | None = None,
    aggregate: Mapping[str, Any] | None = None,
    model_slug: str | None = None,
    provenance_extra: Mapping[str, Any] | None = None,
    campaign_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Atomically spool one completed extraction slice when archival is enabled."""

    return _emit_stage(
        stage="extraction",
        normalizer=extraction_archive_record,
        records=records,
        model=model,
        model_revision=model_revision,
        seed=seed,
        benchmark=benchmark,
        dataset_alias=dataset_alias,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        contract_version=contract_version,
        aggregate=aggregate,
        model_slug=model_slug,
        provenance_extra=provenance_extra,
        campaign_config=campaign_config,
        env=env,
    )


def emit_score_slice(
    *,
    records: Iterable[Mapping[str, Any]],
    model: str,
    model_revision: str,
    seed: int,
    benchmark: str,
    dataset_alias: str,
    dataset_split: str,
    dataset_revision: str,
    contract_version: str | None = None,
    aggregate: Mapping[str, Any] | None = None,
    model_slug: str | None = None,
    provenance_extra: Mapping[str, Any] | None = None,
    campaign_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Atomically spool one completed score slice when archival is enabled."""

    return _emit_stage(
        stage="score",
        normalizer=score_archive_record,
        records=records,
        model=model,
        model_revision=model_revision,
        seed=seed,
        benchmark=benchmark,
        dataset_alias=dataset_alias,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        contract_version=contract_version,
        aggregate=aggregate,
        model_slug=model_slug,
        provenance_extra=provenance_extra,
        campaign_config=campaign_config,
        env=env,
    )
