#!/usr/bin/env python3
"""Build and verify the neutral ``trace_eval_v1`` evaluation artifact set.

The source archive is intentionally treated as private publication input.  The
public output contains model responses, normalized extractions, and scores, but
never redistributes benchmark prompts, questions, answers, options, source
rows, media paths, source request hashes, or private archive identifiers.

Generation slices are the sole authority for raw response text.  Extraction
and score slices are joined to generation through the private record identity,
then receive a new public sample identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "trace_eval_v1"
EXPORT_PLAN_VERSION = "trace_eval_export_plan_v1"
EXPORT_MANIFEST_VERSION = "trace_eval_export_manifest_v1"
PART_MANIFEST_VERSION = "trace_eval_part_manifest_v1"
RESULTS_VERSION = "trace_eval_results_v1"
SUITE_METADATA_VERSION = "trace_eval_suite_v1"
MODEL_METADATA_VERSION = "trace_eval_model_v1"
RUN_METADATA_VERSION = "trace_eval_run_v1"
RESPONSE_CONTRACT_VERSION = "trace_eval_response_v1"
EXTRACTION_CONTRACT_VERSION = "trace_eval_extraction_v1"
SCORE_CONTRACT_VERSION = "trace_eval_score_v1"
REQUEST_CONTRACT_VERSION = "trace_eval_request_v1"
PROMPT_CONTRACT_VERSION = "trace_eval_prompt_contract_v1"
MEDIA_CONTRACT_VERSION = "trace_eval_media_v1"
SOURCE_SLICE_SET_CONTRACT_VERSION = "trace_eval_source_slice_set_v1"
SOURCE_SELECTION_CONTRACT_VERSION = "trace_eval_source_selection_v1"
DATASET_REVISION_PREFIX = "trace-eval-datasets-v1"
CONFIG_NAMES = ("responses", "extractions", "scores")
SOURCE_STAGES = {
    "responses": "generation",
    "extractions": "extraction",
    "scores": "score",
}
PUBLIC_CONTRACTS = {
    "responses": RESPONSE_CONTRACT_VERSION,
    "extractions": EXTRACTION_CONTRACT_VERSION,
    "scores": SCORE_CONTRACT_VERSION,
}
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|sha256set:[0-9a-f]{64})$")
PUBLIC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
REPOSITORY_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
INTERNAL_MARKERS = (
    "all31",
    "final24",
    "final25",
    "final26",
    "final31",
    "trace_final",
)
FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "ground_truth",
        "ground_truth_json",
        "image_path",
        "judge_prompt",
        "media_path",
        "model",
        "model_slug",
        "options",
        "options_json",
        "prompt",
        "prompt_text",
        "question",
        "raw_prediction",
        "record_json",
        "source_model",
        "source_model_id",
        "source_row",
        "source_row_json",
        "source_run",
        "source_run_id",
    }
)
OPAQUE_PUBLIC_FIELDS = frozenset(
    {
        "response_text",
        "extraction_value_json",
        "extraction_candidates_json",
        "judge_output",
        "score_value_json",
    }
)
PRIVATE_OUTPUT_PREFIXES = tuple(
    f"/{prefix}/" for prefix in ("dev" + "/shm", "home", "root")
) + ("file://",)
MEDIA_SETTING_KEYS = (
    "media_contract_version",
    "source_media_contract_sha256",
    "media_transport",
    "min_image_pixels",
    "max_image_pixels",
    "max_image_side",
    "image_jpeg_quality",
)


class PublicExportError(RuntimeError):
    """The source archive or requested public export violates the contract."""


class PublicExportIntegrityError(PublicExportError):
    """A content digest, identity, or derived artifact does not match."""


@dataclass(frozen=True)
class ModelMapping:
    source_model_id: str
    source_revision: str
    model_id: str
    model_revision: str
    display_name: str
    repository_id: str
    repository_revision: str


@dataclass(frozen=True)
class JudgeMapping:
    source_model_id: str
    model_id: str
    model_revision: str


@dataclass(frozen=True)
class ExportPlan:
    source_run_id: str
    source_selection_sha256: str
    source_slice_set_sha256: str
    run_id: str
    suite_id: str
    benchmarks: tuple[str, ...]
    categories: tuple[tuple[str, tuple[str, ...]], ...]
    seeds: tuple[int, ...]
    models: tuple[ModelMapping, ...]
    judge: JudgeMapping

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExportPlan":
        if value.get("schema_version") != EXPORT_PLAN_VERSION:
            raise PublicExportError(
                f"export plan schema must be {EXPORT_PLAN_VERSION!r}"
            )
        source = _require_mapping(value.get("source"), "source")
        public = _require_mapping(value.get("public"), "public")
        source_run_id = _require_text(source.get("run_id"), "source.run_id")
        source_selection_sha256 = _require_sha256(
            source.get("selection_sha256"), "source.selection_sha256"
        )
        source_slice_set_digest = _require_sha256(
            source.get("slice_set_sha256"), "source.slice_set_sha256"
        )
        run_id = _require_public_id(public.get("run_id"), "public.run_id")
        suite_id = _require_public_id(public.get("suite_id"), "public.suite_id")
        if run_id == source_run_id:
            raise PublicExportError("public.run_id must not equal source.run_id")

        benchmarks_raw = public.get("benchmarks")
        if not isinstance(benchmarks_raw, list) or not benchmarks_raw:
            raise PublicExportError("public.benchmarks must be a nonempty list")
        benchmarks = tuple(
            _require_public_id(item, f"public.benchmarks[{index}]")
            for index, item in enumerate(benchmarks_raw)
        )
        if len(set(benchmarks)) != len(benchmarks):
            raise PublicExportError("public.benchmarks contains duplicates")

        categories_raw = _require_mapping(public.get("categories"), "public.categories")
        categories: list[tuple[str, tuple[str, ...]]] = []
        categorized: list[str] = []
        for name, members_raw in categories_raw.items():
            category_name = _require_text(name, "category name")
            if not isinstance(members_raw, list) or not members_raw:
                raise PublicExportError(f"category {category_name!r} must be a nonempty list")
            members = tuple(
                _require_public_id(item, f"category {category_name!r}")
                for item in members_raw
            )
            categories.append((category_name, members))
            categorized.extend(members)
        if tuple(categorized) != benchmarks:
            raise PublicExportError(
                "category membership and order must exactly equal public.benchmarks"
            )

        seeds_raw = public.get("seeds")
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise PublicExportError("public.seeds must be a nonempty list")
        seeds: list[int] = []
        for index, seed in enumerate(seeds_raw):
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise PublicExportError(f"public.seeds[{index}] must be an integer")
            seeds.append(seed)
        if len(set(seeds)) != len(seeds):
            raise PublicExportError("public.seeds contains duplicates")

        models_raw = value.get("models")
        if not isinstance(models_raw, list) or not models_raw:
            raise PublicExportError("models must be a nonempty list")
        models: list[ModelMapping] = []
        for index, raw in enumerate(models_raw):
            item = _require_mapping(raw, f"models[{index}]")
            mapping = ModelMapping(
                source_model_id=_require_text(
                    item.get("source_model_id"), f"models[{index}].source_model_id"
                ),
                source_revision=_require_revision(
                    item.get("source_revision"), f"models[{index}].source_revision"
                ),
                model_id=_require_public_id(
                    item.get("model_id"), f"models[{index}].model_id"
                ),
                model_revision=_require_revision(
                    item.get("model_revision"), f"models[{index}].model_revision"
                ),
                display_name=_require_text(
                    item.get("display_name"), f"models[{index}].display_name"
                ),
                repository_id=_require_repository_id(
                    item.get("repository_id"), f"models[{index}].repository_id"
                ),
                repository_revision=_require_revision(
                    item.get("repository_revision"),
                    f"models[{index}].repository_revision",
                ),
            )
            if mapping.source_model_id == mapping.model_id:
                raise PublicExportError(
                    f"models[{index}].model_id must not equal its source identifier"
                )
            models.append(mapping)
        if len({item.source_model_id for item in models}) != len(models):
            raise PublicExportError("source model identifiers must be unique")
        if len({item.model_id for item in models}) != len(models):
            raise PublicExportError("public model identifiers must be unique")

        judge_raw = _require_mapping(value.get("judge"), "judge")
        judge = JudgeMapping(
            source_model_id=_require_public_id(
                judge_raw.get("source_model_id"), "judge.source_model_id"
            ),
            model_id=_require_repository_id(
                judge_raw.get("model_id"), "judge.model_id"
            ),
            model_revision=_require_revision(
                judge_raw.get("model_revision"), "judge.model_revision"
            ),
        )

        plan = cls(
            source_run_id=source_run_id,
            source_selection_sha256=source_selection_sha256,
            source_slice_set_sha256=source_slice_set_digest,
            run_id=run_id,
            suite_id=suite_id,
            benchmarks=benchmarks,
            categories=tuple(categories),
            seeds=tuple(seeds),
            models=tuple(models),
            judge=judge,
        )
        _assert_no_internal_markers(
            {
                "run_id": plan.run_id,
                "suite_id": plan.suite_id,
                "benchmarks": plan.benchmarks,
                "categories": plan.categories,
                "model_ids": [item.model_id for item in plan.models],
                "display_names": [item.display_name for item in plan.models],
                "repositories": [
                    {
                        "repository_id": item.repository_id,
                        "repository_revision": item.repository_revision,
                    }
                    for item in plan.models
                ],
                "judge": {
                    "model_id": judge.model_id,
                    "model_revision": judge.model_revision,
                },
            },
            # Source model slugs can legitimately be substrings of canonical
            # public repository names. Their exact replacement is enforced by
            # the explicit model map and slice identity checks instead.
            dynamic_markers=(plan.source_run_id,),
        )
        return plan

    @property
    def source_model_map(self) -> dict[str, ModelMapping]:
        return {item.source_model_id: item for item in self.models}

    @property
    def expected_identities(self) -> set[tuple[str, str, int, str]]:
        return {
            (config, model.model_id, seed, benchmark)
            for config in CONFIG_NAMES
            for model in self.models
            for seed in self.seeds
            for benchmark in self.benchmarks
        }


@dataclass(frozen=True)
class PublicFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class VerifiedPublicExport:
    manifest: dict[str, Any]
    manifest_sha256: str
    files: tuple[PublicFile, ...]


@dataclass(frozen=True)
class _SourceSlice:
    config_name: str
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest_size: int
    source_parquet_path: Path
    source_parquet_sha256: str
    source_parquet_size: int
    manifest: dict[str, Any]
    model: ModelMapping


@dataclass(frozen=True)
class _SampleContext:
    source_record_id: str
    public_row: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def source_slice_set_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact private source slice selection without publishing it."""

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for ordinal, raw in enumerate(records):
        config_name = _require_text(
            raw.get("config_name"), f"slice {ordinal} config"
        )
        if config_name not in CONFIG_NAMES:
            raise PublicExportError(
                f"slice {ordinal} has unsupported config {config_name!r}"
            )
        model_id = _require_public_id(
            raw.get("model_id"), f"slice {ordinal} model"
        )
        seed = raw.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PublicExportError(f"slice {ordinal} seed must be an integer")
        benchmark_id = _require_public_id(
            raw.get("benchmark_id"), f"slice {ordinal} benchmark"
        )
        identity = (config_name, model_id, seed, benchmark_id)
        if identity in seen:
            raise PublicExportError(f"duplicate source slice identity: {identity}")
        seen.add(identity)
        item: dict[str, Any] = {
            "config_name": config_name,
            "model_id": model_id,
            "seed": seed,
            "benchmark_id": benchmark_id,
        }
        for kind in ("manifest", "parquet"):
            file_record = _require_mapping(
                raw.get(kind), f"slice {ordinal} {kind}"
            )
            try:
                size = int(file_record.get("size"))
            except (TypeError, ValueError, OverflowError) as error:
                raise PublicExportError(
                    f"slice {ordinal} {kind} size must be an integer"
                ) from error
            if size < 0:
                raise PublicExportError(f"slice {ordinal} {kind} size is negative")
            item[kind] = {
                "sha256": _require_sha256(
                    file_record.get("sha256"), f"slice {ordinal} {kind} digest"
                ),
                "size": size,
            }
        normalized.append(item)
    normalized.sort(
        key=lambda item: (
            item["config_name"],
            item["model_id"],
            item["seed"],
            item["benchmark_id"],
        )
    )
    return canonical_sha256(
        {
            "contract_version": SOURCE_SLICE_SET_CONTRACT_VERSION,
            "slices": normalized,
        }
    )


def prompt_sha256(prompt_text: str) -> str:
    if not isinstance(prompt_text, str):
        raise PublicExportError("prompt text must be a string")
    return sha256_bytes(prompt_text.encode("utf-8"))


def response_sha256(response_text: str) -> str:
    if not isinstance(response_text, str):
        raise PublicExportError("generation response must be a string")
    return sha256_bytes(response_text.encode("utf-8"))


def media_set_sha256(ordered_media_sha256: Sequence[str]) -> str:
    ordered = []
    for index, digest in enumerate(ordered_media_sha256):
        ordered.append(
            {
                "type": "image",
                "sha256": _require_sha256(digest, f"ordered media hash {index}"),
            }
        )
    return canonical_sha256(ordered)


def source_record_sha256(source_row_sha256: str, media_sha256: str) -> str:
    return canonical_sha256(
        {
            "source_row_hash": _require_sha256(source_row_sha256, "source_row_sha256"),
            "media_set_sha256": _require_sha256(media_sha256, "media_set_sha256"),
        }
    )


def normalize_generation_settings(
    sampling: Mapping[str, Any], *, model_id: str, media_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "model_id": _require_public_id(model_id, "model_id"),
        "temperature": _optional_finite_float(sampling.get("temperature")),
        "top_p": _optional_finite_float(sampling.get("top_p")),
        "top_k": _optional_int(sampling.get("top_k")),
        "presence_penalty": _optional_finite_float(sampling.get("presence_penalty")),
        "repetition_penalty": _optional_finite_float(sampling.get("repetition_penalty")),
        "max_output_tokens": _optional_int(
            sampling.get("max_output_tokens", sampling.get("max_tokens"))
        ),
        "seed": _optional_int(sampling.get("seed")),
    }
    if set(media_inputs) != set(MEDIA_SETTING_KEYS):
        raise PublicExportError("normalized media inputs are incomplete")
    result.update(dict(media_inputs))
    return result


def _source_media_inputs(source: _SourceSlice) -> dict[str, Any]:
    aggregate = _require_mapping(
        source.manifest.get("aggregate"), "generation aggregate"
    )
    generation = _require_mapping(
        aggregate.get("generation"), "generation aggregate settings"
    )
    source_contract = _require_text(
        generation.get("media_contract_version"), "source media contract version"
    )
    transport = _require_text(
        generation.get("media_transport"), "source media transport"
    )
    if transport not in {"file-url", "data-url"}:
        raise PublicExportError(f"unsupported source media transport: {transport!r}")
    minimum = _optional_int(generation.get("min_image_pixels"))
    maximum = _optional_int(generation.get("max_image_pixels"))
    if minimum is None or maximum is None or minimum < 1 or maximum < minimum:
        raise PublicExportError("source media pixel bounds are invalid")
    maximum_side = _optional_int(generation.get("max_image_side"))
    jpeg_quality = _optional_int(generation.get("image_jpeg_quality"))
    if maximum_side is not None and maximum_side < 1:
        raise PublicExportError("source maximum image side is invalid")
    if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
        raise PublicExportError("source JPEG quality is invalid")
    return {
        "media_contract_version": MEDIA_CONTRACT_VERSION,
        "source_media_contract_sha256": sha256_bytes(source_contract.encode("utf-8")),
        "media_transport": transport,
        "min_image_pixels": minimum,
        "max_image_pixels": maximum,
        "max_image_side": maximum_side,
        "image_jpeg_quality": jpeg_quality,
    }


def _media_inputs_from_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: settings.get(key) for key in MEDIA_SETTING_KEYS}
    if result["media_contract_version"] != MEDIA_CONTRACT_VERSION:
        raise PublicExportIntegrityError("public media contract version mismatch")
    _require_sha256(
        result["source_media_contract_sha256"], "source media contract digest"
    )
    if result["media_transport"] not in {"file-url", "data-url"}:
        raise PublicExportIntegrityError("public media transport is invalid")
    minimum = _optional_int(result["min_image_pixels"])
    maximum = _optional_int(result["max_image_pixels"])
    if minimum is None or maximum is None or minimum < 1 or maximum < minimum:
        raise PublicExportIntegrityError("public media pixel bounds are invalid")
    maximum_side = _optional_int(result["max_image_side"])
    jpeg_quality = _optional_int(result["image_jpeg_quality"])
    if maximum_side is not None and maximum_side < 1:
        raise PublicExportIntegrityError("public maximum image side is invalid")
    if jpeg_quality is not None and not 1 <= jpeg_quality <= 100:
        raise PublicExportIntegrityError("public JPEG quality is invalid")
    return result


def public_request_sha256_from_hashes(
    *,
    prompt_digest: str,
    ordered_media_sha256: Sequence[str],
    generation_settings: Mapping[str, Any],
) -> str:
    prompt_digest = _require_sha256(prompt_digest, "prompt_sha256")
    media_content = [
        {"type": "image", "sha256": _require_sha256(item, "media sha256")}
        for item in ordered_media_sha256
    ]
    material = {
        "contract_version": REQUEST_CONTRACT_VERSION,
        "messages": [
            {
                "role": "user",
                "content": [
                    *media_content,
                    {"type": "text", "sha256": prompt_digest},
                ],
            }
        ],
        "generation": dict(generation_settings),
    }
    return canonical_sha256(material)


def public_request_sha256(
    *,
    prompt_text: str,
    ordered_media_sha256: Sequence[str],
    generation_settings: Mapping[str, Any],
) -> str:
    return public_request_sha256_from_hashes(
        prompt_digest=prompt_sha256(prompt_text),
        ordered_media_sha256=ordered_media_sha256,
        generation_settings=generation_settings,
    )


def prompt_contract_sha256(
    *,
    dataset_sha256: str,
    source_row_sha256: str,
    source_record_digest: str,
    media_digest: str,
    prompt_digest: str,
    request_digest: str,
) -> str:
    material = {
        "contract_version": PROMPT_CONTRACT_VERSION,
        "dataset_sha256": _require_sha256(dataset_sha256, "dataset_sha256"),
        "source_row_sha256": _require_sha256(source_row_sha256, "source_row_sha256"),
        "source_record_sha256": _require_sha256(
            source_record_digest, "source_record_sha256"
        ),
        "media_set_sha256": _require_sha256(media_digest, "media_set_sha256"),
        "prompt_sha256": _require_sha256(prompt_digest, "prompt_sha256"),
        "request_sha256": _require_sha256(request_digest, "request_sha256"),
    }
    return canonical_sha256(material)


def verify_prompt_contract(
    row: Mapping[str, Any],
    *,
    prompt_text: str,
    ordered_media_sha256: Sequence[str] | None = None,
    generation_settings: Mapping[str, Any] | None = None,
) -> None:
    """Verify a prompt privately without storing it in the public artifact.

    A caller reconstructs the benchmark prompt and supplies it here.  Ordered
    media hashes and neutral generation settings default to the public response
    record, but may be supplied independently by a reproduction harness.
    """

    actual_prompt_hash = prompt_sha256(prompt_text)
    if actual_prompt_hash != row.get("prompt_sha256"):
        raise PublicExportIntegrityError("prompt SHA-256 does not match")
    media = list(
        ordered_media_sha256
        if ordered_media_sha256 is not None
        else (row.get("ordered_media_sha256") or [])
    )
    settings = dict(
        generation_settings
        if generation_settings is not None
        else json.loads(_require_text(row.get("generation_settings_json"), "generation settings"))
    )
    actual_media_hash = media_set_sha256(media)
    if actual_media_hash != row.get("media_set_sha256"):
        raise PublicExportIntegrityError("media-set SHA-256 does not match")
    actual_source_record = source_record_sha256(
        str(row.get("source_row_sha256")), actual_media_hash
    )
    if actual_source_record != row.get("source_record_sha256"):
        raise PublicExportIntegrityError("source-record SHA-256 does not match")
    actual_request = public_request_sha256_from_hashes(
        prompt_digest=actual_prompt_hash,
        ordered_media_sha256=media,
        generation_settings=settings,
    )
    if actual_request != row.get("request_sha256"):
        raise PublicExportIntegrityError("public request SHA-256 does not match")
    actual_contract = prompt_contract_sha256(
        dataset_sha256=str(row.get("dataset_sha256")),
        source_row_sha256=str(row.get("source_row_sha256")),
        source_record_digest=actual_source_record,
        media_digest=actual_media_hash,
        prompt_digest=actual_prompt_hash,
        request_digest=actual_request,
    )
    if actual_contract != row.get("prompt_contract_sha256"):
        raise PublicExportIntegrityError("prompt-contract SHA-256 does not match")


COMMON_SAMPLE_FIELDS = [
    pa.field("schema_version", pa.string(), nullable=False),
    pa.field("run_id", pa.string(), nullable=False),
    pa.field("model_id", pa.string(), nullable=False),
    pa.field("model_revision", pa.string(), nullable=False),
    pa.field("seed", pa.int64(), nullable=False),
    pa.field("benchmark_id", pa.string(), nullable=False),
    pa.field("dataset_split", pa.string(), nullable=False),
    pa.field("dataset_revision", pa.string(), nullable=False),
    pa.field("sample_id", pa.string(), nullable=False),
    pa.field("response_id", pa.string(), nullable=False),
    pa.field("source_index", pa.string(), nullable=False),
    pa.field("source_ordinal", pa.int64(), nullable=False),
    pa.field("dataset_sha256", pa.string(), nullable=False),
    pa.field("source_row_sha256", pa.string(), nullable=False),
    pa.field("source_record_sha256", pa.string(), nullable=False),
    pa.field("media_set_sha256", pa.string(), nullable=False),
    pa.field("ordered_media_sha256", pa.list_(pa.string()), nullable=False),
    pa.field("prompt_sha256", pa.string(), nullable=False),
    pa.field("request_sha256", pa.string(), nullable=False),
    pa.field("prompt_contract_sha256", pa.string(), nullable=False),
    pa.field("response_sha256", pa.string(), nullable=False),
]


RESPONSE_SCHEMA = pa.schema(
    [
        *COMMON_SAMPLE_FIELDS,
        pa.field("contract_version", pa.string(), nullable=False),
        pa.field("response_text", pa.string(), nullable=False),
        pa.field("finish_reason", pa.string()),
        pa.field("generation_settings_json", pa.string(), nullable=False),
        pa.field("prompt_tokens", pa.int64()),
        pa.field("completion_tokens", pa.int64()),
    ],
    metadata={
        b"trace_eval_schema": SCHEMA_VERSION.encode("ascii"),
        b"trace_eval_config": b"responses",
    },
)


EXTRACTION_SCHEMA = pa.schema(
    [
        *COMMON_SAMPLE_FIELDS,
        pa.field("contract_version", pa.string(), nullable=False),
        pa.field("extraction_status", pa.string(), nullable=False),
        pa.field("extraction_value_type", pa.string(), nullable=False),
        pa.field("extraction_value_json", pa.string(), nullable=False),
        pa.field("extraction_candidates_json", pa.string(), nullable=False),
        pa.field("extraction_sha256", pa.string(), nullable=False),
        pa.field("extraction_method", pa.string(), nullable=False),
        pa.field("used_judge", pa.bool_(), nullable=False),
        pa.field("judge_model_id", pa.string()),
        pa.field("judge_model_revision", pa.string()),
        pa.field("judge_output", pa.string()),
        pa.field("judge_output_sha256", pa.string()),
        pa.field("retry_count", pa.int64(), nullable=False),
    ],
    metadata={
        b"trace_eval_schema": SCHEMA_VERSION.encode("ascii"),
        b"trace_eval_config": b"extractions",
    },
)


SCORE_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("benchmark_id", pa.string(), nullable=False),
        pa.field("dataset_split", pa.string(), nullable=False),
        pa.field("dataset_revision", pa.string(), nullable=False),
        pa.field("score_id", pa.string(), nullable=False),
        pa.field("score_scope", pa.string(), nullable=False),
        pa.field("sample_id", pa.string()),
        pa.field("response_id", pa.string()),
        pa.field("source_index", pa.string()),
        pa.field("source_ordinal", pa.int64()),
        pa.field("dataset_sha256", pa.string(), nullable=False),
        pa.field("source_row_sha256", pa.string()),
        pa.field("source_record_sha256", pa.string()),
        pa.field("media_set_sha256", pa.string()),
        pa.field("prompt_sha256", pa.string()),
        pa.field("request_sha256", pa.string()),
        pa.field("prompt_contract_sha256", pa.string()),
        pa.field("response_sha256", pa.string()),
        pa.field("contract_version", pa.string(), nullable=False),
        pa.field("scorer_id", pa.string(), nullable=False),
        pa.field("metric_id", pa.string(), nullable=False),
        pa.field("score_unit", pa.string(), nullable=False),
        pa.field("score_value", pa.float64()),
        pa.field("score_value_json", pa.string(), nullable=False),
        pa.field("excluded", pa.bool_(), nullable=False),
        pa.field("evaluated_rows", pa.int64()),
    ],
    metadata={
        b"trace_eval_schema": SCHEMA_VERSION.encode("ascii"),
        b"trace_eval_config": b"scores",
    },
)


PUBLIC_SCHEMAS = {
    "responses": RESPONSE_SCHEMA,
    "extractions": EXTRACTION_SCHEMA,
    "scores": SCORE_SCHEMA,
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicExportError(f"{label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicExportError(f"{label} must be a nonempty string")
    return value.strip()


def _require_public_id(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if not PUBLIC_ID_RE.fullmatch(text):
        raise PublicExportError(f"{label} is not a valid public identifier: {text!r}")
    return text


def _require_repository_id(value: Any, label: str) -> str:
    text = _require_text(value, label)
    if not REPOSITORY_ID_RE.fullmatch(text):
        raise PublicExportError(f"{label} is not a valid repository identifier")
    return text


def _require_sha256(value: Any, label: str) -> str:
    text = _require_text(value, label).lower()
    if not HEX_64_RE.fullmatch(text):
        raise PublicExportError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _require_revision(value: Any, label: str) -> str:
    text = _require_text(value, label).lower()
    if not HEX_REVISION_RE.fullmatch(text):
        raise PublicExportError(f"{label} must be a commit or content digest")
    return text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PublicExportError("boolean is not an integer generation setting")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PublicExportError(f"invalid integer generation setting: {value!r}") from error


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PublicExportError("boolean is not a floating-point generation setting")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PublicExportError(f"invalid numeric generation setting: {value!r}") from error
    if not math.isfinite(result):
        raise PublicExportError(f"non-finite generation setting: {value!r}")
    return result


def _json_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PublicExportError("non-finite JSON number is not exportable")
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    raise PublicExportError(f"unsupported JSON value type: {type(value).__name__}")


def _assert_no_internal_markers(
    value: Any,
    *,
    dynamic_markers: Sequence[str] = (),
    path: str = "$",
    opaque_fields: frozenset[str] = OPAQUE_PUBLIC_FIELDS,
) -> None:
    markers = tuple(item.lower() for item in (*INTERNAL_MARKERS, *dynamic_markers) if item)
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_PUBLIC_FIELDS:
                raise PublicExportError(f"forbidden public field at {path}.{key_text}")
            if key_text in opaque_fields:
                continue
            _assert_no_internal_markers(
                child,
                dynamic_markers=markers,
                path=f"{path}.{key_text}",
                opaque_fields=opaque_fields,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_internal_markers(
                child,
                dynamic_markers=markers,
                path=f"{path}[{index}]",
                opaque_fields=opaque_fields,
            )
        return
    if isinstance(value, str):
        lowered = value.lower()
        for marker in markers:
            if marker and marker in lowered:
                raise PublicExportError(f"internal identifier {marker!r} leaked at {path}")
        if value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
            raise PublicExportError(f"absolute path leaked at {path}")


def _assert_private_output_safe(
    row: Mapping[str, Any],
    *,
    private_run_id: str | None = None,
    private_markers: Sequence[str] = (),
) -> None:
    markers = [item.lower() for item in (*PRIVATE_OUTPUT_PREFIXES, *INTERNAL_MARKERS)]
    if private_run_id:
        markers.append(private_run_id.lower())
    markers.extend(item.lower() for item in private_markers if item)
    for field in OPAQUE_PUBLIC_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).lower()
        for marker in markers:
            if marker and marker in text:
                raise PublicExportError(
                    f"private identifier or local path leaked in {field}"
                )


def validate_public_schema(schema: pa.Schema, config_name: str) -> None:
    expected = PUBLIC_SCHEMAS.get(config_name)
    if expected is None:
        raise PublicExportError(f"unsupported public config: {config_name!r}")
    names = set(schema.names)
    forbidden = sorted(names & FORBIDDEN_PUBLIC_FIELDS)
    if forbidden:
        raise PublicExportError(
            f"public {config_name} schema contains forbidden fields: {forbidden}"
        )
    metadata_matches = dict(schema.metadata or {}) == dict(expected.metadata or {})
    if not schema.remove_metadata().equals(
        expected.remove_metadata(), check_metadata=False
    ) or not metadata_matches:
        raise PublicExportError(f"public {config_name} schema does not match {SCHEMA_VERSION}")


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-._")
    if not result:
        raise PublicExportError(f"cannot form a public slug from {value!r}")
    return result


def _sample_id(
    *,
    run_id: str,
    model_id: str,
    model_revision: str,
    seed: int,
    benchmark_id: str,
    source_index: str,
    source_ordinal: int,
    source_row_sha256: str,
) -> str:
    digest = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "seed": seed,
            "benchmark_id": benchmark_id,
            "source_index": source_index,
            "source_ordinal": source_ordinal,
            "source_row_sha256": source_row_sha256,
        }
    )
    return f"sample-{digest}"


def _score_id(identity: Mapping[str, Any], *, scope: str, sample_id: str | None) -> str:
    return "score-" + canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": identity["run_id"],
            "model_id": identity["model_id"],
            "seed": identity["seed"],
            "benchmark_id": identity["benchmark_id"],
            "scope": scope,
            "sample_id": sample_id,
        }
    )


def _response_id(sample_id: str, response_digest: str) -> str:
    return "response-" + canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "response_sha256": _require_sha256(response_digest, "response_sha256"),
        }
    )


def _resolve_source_root(path: Path | str) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (root, root / "staged")
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "metadata" / "slices").is_dir():
            return candidate
    raise PublicExportError(
        f"source root must contain data/ and metadata/slices/: {root}"
    )


def _safe_source_path(root: Path, relative: Any, label: str) -> Path:
    text = _require_text(relative, label)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PublicExportError(f"{label} is not a safe relative path")
    resolved = (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise PublicExportError(f"{label} escapes source root")
    return resolved


def _discover_source_slices(
    source_root: Path,
    plan: ExportPlan,
    *,
    verify_slice_set: bool = True,
) -> dict[tuple[str, str, int, str], _SourceSlice]:
    source_models = plan.source_model_map
    selected: dict[tuple[str, str, int, str], _SourceSlice] = {}
    manifest_paths = (source_root / "metadata" / "slices").rglob(
        "*.manifest.json"
    )
    for path in sorted(manifest_paths):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PublicExportError(
                f"cannot read source manifest {path}: {error}"
            ) from error
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping) or identity.get("run_id") != plan.source_run_id:
            continue
        source_model_id = str(identity.get("model_slug") or "")
        model = source_models.get(source_model_id)
        if model is None:
            continue
        benchmark = str(identity.get("benchmark") or "")
        if benchmark not in plan.benchmarks:
            continue
        seed = identity.get("seed")
        if seed not in plan.seeds:
            continue
        source_stage = str(identity.get("stage") or "")
        matches = [name for name, stage in SOURCE_STAGES.items() if stage == source_stage]
        if len(matches) != 1:
            continue
        config_name = matches[0]
        if str(identity.get("model_revision") or "") != model.source_revision:
            raise PublicExportError(
                f"source revision mismatch for {source_model_id}: "
                f"{identity.get('model_revision')!r} != {model.source_revision!r}"
            )
        expected_repository_reference = (
            f"{model.repository_id}@{model.repository_revision}"
        )
        if str(identity.get("model") or "") != expected_repository_reference:
            raise PublicExportError(
                f"source repository identity mismatch for {source_model_id}: "
                f"expected {expected_repository_reference!r}"
            )
        parquet_path = _safe_source_path(
            source_root, manifest.get("parquet_path"), f"parquet_path in {path}"
        )
        if not parquet_path.is_file():
            raise PublicExportError(f"source Parquet is missing: {parquet_path}")
        parquet_sha = _require_sha256(
            manifest.get("parquet_sha256"), f"parquet_sha256 in {path}"
        )
        if sha256_file(parquet_path) != parquet_sha:
            raise PublicExportIntegrityError(f"source Parquet digest mismatch: {parquet_path}")
        key = (config_name, model.model_id, int(seed), benchmark)
        if key in selected:
            raise PublicExportError(f"duplicate source slice for {key}")
        selected[key] = _SourceSlice(
            config_name=config_name,
            source_manifest_path=path,
            source_manifest_sha256=sha256_file(path),
            source_manifest_size=path.stat().st_size,
            source_parquet_path=parquet_path,
            source_parquet_sha256=parquet_sha,
            source_parquet_size=parquet_path.stat().st_size,
            manifest=dict(manifest),
            model=model,
        )
    missing = sorted(plan.expected_identities - set(selected))
    extra = sorted(set(selected) - plan.expected_identities)
    if missing or extra:
        raise PublicExportError(
            f"source coverage mismatch: missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:3]} first_extra={extra[:3]}"
        )
    actual_slice_set = _discovered_source_slice_set_sha256(selected)
    if verify_slice_set and actual_slice_set != plan.source_slice_set_sha256:
        raise PublicExportIntegrityError(
            "selected source slice-set digest does not match the sealed export plan"
        )
    return selected


def _discovered_source_slice_set_sha256(
    selected: Mapping[tuple[str, str, int, str], _SourceSlice]
) -> str:
    return source_slice_set_sha256(
        [
            {
                "config_name": config_name,
                "model_id": model_id,
                "seed": seed,
                "benchmark_id": benchmark,
                "manifest": {
                    "sha256": source.source_manifest_sha256,
                    "size": source.source_manifest_size,
                },
                "parquet": {
                    "sha256": source.source_parquet_sha256,
                    "size": source.source_parquet_size,
                },
            }
            for (config_name, model_id, seed, benchmark), source in selected.items()
        ]
    )


def _source_records(source: _SourceSlice) -> list[dict[str, Any]]:
    try:
        table = pq.ParquetFile(source.source_parquet_path).read(columns=["record_id", "record_json"])
    except Exception as error:
        raise PublicExportError(f"cannot read source Parquet {source.source_parquet_path}: {error}") from error
    expected_rows = int(source.manifest.get("rows", -1))
    if table.num_rows != expected_rows:
        raise PublicExportIntegrityError(
            f"source row count mismatch for {source.source_parquet_path}: "
            f"{table.num_rows} != {expected_rows}"
        )
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(table.to_pylist()):
        try:
            record = json.loads(row["record_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise PublicExportError(
                f"invalid record_json at row {ordinal} in {source.source_parquet_path}"
            ) from error
        if not isinstance(record, dict):
            raise PublicExportError(f"source record {ordinal} is not an object")
        if record.get("record_id") not in (None, row["record_id"]):
            raise PublicExportIntegrityError(
                f"source record identity mismatch at row {ordinal} in {source.source_parquet_path}"
            )
        record["record_id"] = str(row["record_id"])
        records.append(record)
    records.sort(
        key=lambda item: (
            int(item.get("source_ordinal", 0)),
            str(item.get("source_index", "")),
            str(item["record_id"]),
        )
    )
    return records


def _dataset_revision(dataset_sha: str) -> str:
    return f"{DATASET_REVISION_PREFIX}:{_require_sha256(dataset_sha, 'dataset_sha256')}"


def _base_identity(source: _SourceSlice, plan: ExportPlan) -> dict[str, Any]:
    identity = source.manifest["identity"]
    split = _require_public_id(str(identity.get("dataset_split") or "default"), "dataset split")
    return {
        "run_id": plan.run_id,
        "model_id": source.model.model_id,
        "model_revision": source.model.model_revision,
        "seed": int(identity["seed"]),
        "benchmark_id": str(identity["benchmark"]),
        "dataset_split": split,
    }


def _generation_common(
    record: Mapping[str, Any],
    source: _SourceSlice,
    plan: ExportPlan,
    media_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    metadata = _require_mapping(record.get("metadata"), "generation metadata")
    dataset_sha = _require_sha256(
        metadata.get("dataset_snapshot_sha256"), "dataset_snapshot_sha256"
    )
    source_row_sha = _require_sha256(record.get("source_row_hash"), "source_row_hash")
    ordered_media_raw = metadata.get("image_hash")
    if not isinstance(ordered_media_raw, list):
        raise PublicExportError("generation metadata.image_hash must be a list")
    ordered_media = [
        _require_sha256(item, f"metadata.image_hash[{index}]")
        for index, item in enumerate(ordered_media_raw)
    ]
    media_sha = media_set_sha256(ordered_media)
    persisted_media = _require_sha256(metadata.get("media_hash"), "metadata.media_hash")
    if media_sha != persisted_media:
        raise PublicExportIntegrityError("generation media-set digest mismatch")
    source_record_digest = source_record_sha256(source_row_sha, media_sha)
    persisted_source_record = _require_sha256(
        metadata.get("source_record_sha256"), "metadata.source_record_sha256"
    )
    if source_record_digest != persisted_source_record:
        raise PublicExportIntegrityError("generation source-record digest mismatch")

    prompt = record.get("prompt")
    if not isinstance(prompt, str):
        raise PublicExportError("generation prompt must be a string")
    response = record.get("model_response")
    if not isinstance(response, str):
        raise PublicExportError(
            "generation model_response must be text; extraction values must not replace it"
        )
    sampling = _require_mapping(record.get("sampling"), "generation sampling")
    generation_settings = normalize_generation_settings(
        sampling,
        model_id=source.model.model_id,
        media_inputs=media_inputs,
    )
    prompt_digest = prompt_sha256(prompt)
    request_digest = public_request_sha256_from_hashes(
        prompt_digest=prompt_digest,
        ordered_media_sha256=ordered_media,
        generation_settings=generation_settings,
    )
    contract_digest = prompt_contract_sha256(
        dataset_sha256=dataset_sha,
        source_row_sha256=source_row_sha,
        source_record_digest=source_record_digest,
        media_digest=media_sha,
        prompt_digest=prompt_digest,
        request_digest=request_digest,
    )
    identity = _base_identity(source, plan)
    source_index = str(record.get("source_index"))
    source_ordinal = record.get("source_ordinal")
    if isinstance(source_ordinal, bool) or not isinstance(source_ordinal, int) or source_ordinal < 0:
        raise PublicExportError("source_ordinal must be a nonnegative integer")
    sample_id = _sample_id(
        run_id=identity["run_id"],
        model_id=identity["model_id"],
        model_revision=identity["model_revision"],
        seed=identity["seed"],
        benchmark_id=identity["benchmark_id"],
        source_index=source_index,
        source_ordinal=source_ordinal,
        source_row_sha256=source_row_sha,
    )
    response_digest = response_sha256(response)
    common = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "dataset_revision": _dataset_revision(dataset_sha),
        "sample_id": sample_id,
        "response_id": _response_id(sample_id, response_digest),
        "source_index": source_index,
        "source_ordinal": source_ordinal,
        "dataset_sha256": dataset_sha,
        "source_row_sha256": source_row_sha,
        "source_record_sha256": source_record_digest,
        "media_set_sha256": media_sha,
        "ordered_media_sha256": ordered_media,
        "prompt_sha256": prompt_digest,
        "request_sha256": request_digest,
        "prompt_contract_sha256": contract_digest,
        "response_sha256": response_digest,
    }
    return common, response, canonical_json(generation_settings)


def _response_rows(
    source: _SourceSlice,
    plan: ExportPlan,
) -> tuple[list[dict[str, Any]], dict[str, _SampleContext], set[str]]:
    rows: list[dict[str, Any]] = []
    contexts: dict[str, _SampleContext] = {}
    settings_values: set[str] = set()
    media_inputs = _source_media_inputs(source)
    for record in _source_records(source):
        common, response, settings_json = _generation_common(
            record, source, plan, media_inputs
        )
        usage = record.get("usage") if isinstance(record.get("usage"), Mapping) else {}
        row = {
            **common,
            "contract_version": RESPONSE_CONTRACT_VERSION,
            "response_text": response,
            "finish_reason": (
                None if record.get("finish_reason") is None else str(record.get("finish_reason"))
            ),
            "generation_settings_json": settings_json,
            "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
            "completion_tokens": _optional_int(usage.get("completion_tokens")),
        }
        source_record_id = _require_text(record.get("record_id"), "source record id")
        if source_record_id in contexts:
            raise PublicExportError(f"duplicate generation record id: {source_record_id}")
        rows.append(row)
        contexts[source_record_id] = _SampleContext(source_record_id, row)
        settings_values.add(settings_json)
    return rows, contexts, settings_values


def _common_sample_fields(context: _SampleContext) -> dict[str, Any]:
    return {field.name: context.public_row[field.name] for field in COMMON_SAMPLE_FIELDS}


def _validate_stage_record_link(
    record: Mapping[str, Any], context: _SampleContext, *, stage: str
) -> None:
    expected = context.public_row
    checks = {
        "source_index": str(record.get("source_index")),
        "source_ordinal": record.get("source_ordinal"),
        "source_row_sha256": record.get("source_row_hash"),
    }
    for field, actual in checks.items():
        if actual != expected[field]:
            raise PublicExportIntegrityError(
                f"{stage} record identity does not match generation field {field}"
            )


def _neutral_extraction_method(method: Any, *, used_judge: bool) -> str:
    text = str(method or "").lower()
    if used_judge or "judge" in text:
        return "judge"
    if "determin" in text or "regex" in text or "parse" in text:
        return "deterministic_parser"
    if text:
        return "benchmark_adapter"
    return "unknown"


def _retry_count(retries: Any) -> int:
    if not isinstance(retries, Mapping):
        return 0
    for key in ("count", "total_retries", "retry_count"):
        if retries.get(key) is not None:
            value = _optional_int(retries.get(key))
            return max(0, value or 0)
    events = retries.get("events")
    return max(0, len(events) - 1) if isinstance(events, list) else 0


def _validate_source_judge_reference(
    source: _SourceSlice, judge: JudgeMapping, *, used_judge: bool
) -> None:
    if not used_judge:
        return
    aggregate = source.manifest.get("aggregate")
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    source_reference = aggregate.get("judge_model")
    if source_reference in (None, ""):
        # Some dedicated scorers omit this aggregate field. A nonempty
        # row-level judge prompt and output still prove judge use, while the
        # private plan supplies the immutable public identity.
        return
    text = str(source_reference)
    normalized = text.rstrip("/").rsplit("/", 1)[-1]
    if text != judge.model_id and normalized != judge.source_model_id:
        raise PublicExportIntegrityError(
            f"source judge identity {text!r} does not match the export plan"
        )


def _extraction_rows(
    source: _SourceSlice,
    contexts: Mapping[str, _SampleContext],
    judge: JudgeMapping,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in _source_records(source):
        source_record_id = _require_text(record.get("record_id"), "source record id")
        context = contexts.get(source_record_id)
        if context is None:
            raise PublicExportIntegrityError(
                f"extraction has no generation response: {source_record_id}"
            )
        _validate_stage_record_link(record, context, stage="extraction")
        if source_record_id in seen:
            raise PublicExportError(f"duplicate extraction record id: {source_record_id}")
        seen.add(source_record_id)
        normalized = _require_mapping(
            record.get("normalized_extraction"), "normalized extraction"
        )
        status = _require_text(normalized.get("status"), "extraction status").lower()
        if status not in {"resolved", "ambiguous", "invalid", "excluded"}:
            raise PublicExportError(f"unsupported extraction status: {status!r}")
        value = normalized.get("value")
        candidates = normalized.get("candidates") or []
        if not isinstance(candidates, list):
            raise PublicExportError("extraction candidates must be a list")
        value_json = canonical_json(value)
        candidates_json = canonical_json(candidates)
        judge_output_raw = record.get("judge_response")
        judge_output = (
            str(judge_output_raw) if judge_output_raw not in (None, "") else None
        )
        used_judge = judge_output is not None or bool(record.get("judge_prompt"))
        if used_judge and judge_output is None:
            raise PublicExportIntegrityError(
                "judge-backed extraction has no judge output"
            )
        extraction_material = {
            "contract_version": EXTRACTION_CONTRACT_VERSION,
            "sample_id": context.public_row["sample_id"],
            "status": status,
            "value": value,
            "candidates": candidates,
        }
        rows.append(
            {
                **_common_sample_fields(context),
                "contract_version": EXTRACTION_CONTRACT_VERSION,
                "extraction_status": status,
                "extraction_value_type": _json_value_type(value),
                "extraction_value_json": value_json,
                "extraction_candidates_json": candidates_json,
                "extraction_sha256": canonical_sha256(extraction_material),
                "extraction_method": _neutral_extraction_method(
                    normalized.get("method"), used_judge=used_judge
                ),
                "used_judge": used_judge,
                "judge_model_id": judge.model_id if used_judge else None,
                "judge_model_revision": judge.model_revision if used_judge else None,
                "judge_output": judge_output,
                "judge_output_sha256": (
                    sha256_bytes(judge_output.encode("utf-8")) if judge_output is not None else None
                ),
                "retry_count": _retry_count(record.get("retries")),
            }
        )
    if seen != set(contexts):
        raise PublicExportIntegrityError(
            f"extraction coverage mismatch: {len(seen)} != {len(contexts)} responses"
        )
    _validate_source_judge_reference(
        source, judge, used_judge=any(row["used_judge"] for row in rows)
    )
    return rows


def _aggregate_score(source: _SourceSlice) -> tuple[float, int]:
    aggregate = _require_mapping(source.manifest.get("aggregate"), "score aggregate")
    value = aggregate.get("score")
    if value is None:
        value = aggregate.get("accuracy")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PublicExportError("score aggregate has no numeric primary value") from error
    if not math.isfinite(score):
        raise PublicExportError("score aggregate is non-finite")
    try:
        evaluated_rows = int(aggregate["rows"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise PublicExportError("score aggregate has no evaluated row count") from error
    if evaluated_rows < 1:
        raise PublicExportError("score aggregate evaluated row count must be positive")
    return score, evaluated_rows


def _score_numeric(value: Any, *, excluded: bool) -> float | None:
    if value is None:
        if excluded:
            return None
        raise PublicExportError("non-excluded row score is null")
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise PublicExportError(f"row score is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise PublicExportError("row score is non-finite")
    return result


def _score_rows(
    source: _SourceSlice,
    contexts: Mapping[str, _SampleContext],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    records = _source_records(source)
    aggregate_value, evaluated_rows = _aggregate_score(source)
    first_context = next(iter(contexts.values()), None)
    if first_context is None:
        raise PublicExportError("cannot export a score slice without generation rows")
    if evaluated_rows != len(contexts):
        raise PublicExportIntegrityError(
            f"aggregate evaluated rows do not match generation responses: "
            f"{evaluated_rows} != {len(contexts)}"
        )
    identity = {
        "run_id": first_context.public_row["run_id"],
        "model_id": source.model.model_id,
        "model_revision": source.model.model_revision,
        "seed": int(source.manifest["identity"]["seed"]),
        "benchmark_id": str(source.manifest["identity"]["benchmark"]),
        "dataset_split": first_context.public_row["dataset_split"],
    }
    dataset_sha = first_context.public_row["dataset_sha256"]
    dataset_revision = first_context.public_row["dataset_revision"]
    source_aggregate_only = all(
        str(record.get("source_index")) == "__aggregate__"
        or (
            isinstance(record.get("metadata"), Mapping)
            and (
                record["metadata"].get("scope") == "aggregate"
                or record["metadata"].get("score_contract") == "aggregate_only"
            )
        )
        for record in records
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    scorer_id = f"{identity['benchmark_id']}-primary-v1"
    if not source_aggregate_only:
        for record in records:
            source_record_id = _require_text(record.get("record_id"), "source record id")
            context = contexts.get(source_record_id)
            if context is None:
                raise PublicExportIntegrityError(
                    f"row score has no generation response: {source_record_id}"
                )
            _validate_stage_record_link(record, context, stage="score")
            if source_record_id in seen:
                raise PublicExportError(f"duplicate row score record: {source_record_id}")
            seen.add(source_record_id)
            excluded_raw = record.get("excluded")
            excluded = bool(excluded_raw)
            score_raw = record.get("score")
            score_value = _score_numeric(score_raw, excluded=excluded)
            sample_id = context.public_row["sample_id"]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    **identity,
                    "dataset_revision": dataset_revision,
                    "score_id": _score_id(identity, scope="row", sample_id=sample_id),
                    "score_scope": "row",
                    "sample_id": sample_id,
                    "response_id": context.public_row["response_id"],
                    "source_index": context.public_row["source_index"],
                    "source_ordinal": context.public_row["source_ordinal"],
                    "dataset_sha256": dataset_sha,
                    "source_row_sha256": context.public_row["source_row_sha256"],
                    "source_record_sha256": context.public_row["source_record_sha256"],
                    "media_set_sha256": context.public_row["media_set_sha256"],
                    "prompt_sha256": context.public_row["prompt_sha256"],
                    "request_sha256": context.public_row["request_sha256"],
                    "prompt_contract_sha256": context.public_row["prompt_contract_sha256"],
                    "response_sha256": context.public_row["response_sha256"],
                    "contract_version": SCORE_CONTRACT_VERSION,
                    "scorer_id": scorer_id,
                    "metric_id": "primary",
                    "score_unit": "fraction",
                    "score_value": score_value,
                    "score_value_json": canonical_json(score_raw),
                    "excluded": excluded,
                    "evaluated_rows": None,
                }
            )
        if seen != set(contexts):
            raise PublicExportIntegrityError(
                f"row-score coverage mismatch: {len(seen)} != {len(contexts)} responses"
            )
    else:
        for record in records:
            metadata = record.get("metadata")
            is_explicit_aggregate = str(record.get("source_index")) == "__aggregate__" or (
                isinstance(metadata, Mapping) and metadata.get("scope") == "aggregate"
            )
            if is_explicit_aggregate:
                continue
            source_record_id = _require_text(record.get("record_id"), "source record id")
            context = contexts.get(source_record_id)
            if context is None:
                raise PublicExportIntegrityError(
                    f"aggregate placeholder has no generation response: {source_record_id}"
                )
            _validate_stage_record_link(record, context, stage="score placeholder")

    aggregate_row = {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "dataset_revision": dataset_revision,
        "score_id": _score_id(identity, scope="aggregate", sample_id=None),
        "score_scope": "aggregate",
        "sample_id": None,
        "response_id": None,
        "source_index": None,
        "source_ordinal": None,
        "dataset_sha256": dataset_sha,
        "source_row_sha256": None,
        "source_record_sha256": None,
        "media_set_sha256": None,
        "prompt_sha256": None,
        "request_sha256": None,
        "prompt_contract_sha256": None,
        "response_sha256": None,
        "contract_version": SCORE_CONTRACT_VERSION,
        "scorer_id": scorer_id,
        "metric_id": "primary",
        "score_unit": "percent",
        "score_value": aggregate_value,
        "score_value_json": canonical_json(aggregate_value),
        "excluded": False,
        "evaluated_rows": evaluated_rows,
    }
    rows.append(aggregate_row)
    scope = "aggregate_only" if source_aggregate_only else "row_and_aggregate"
    return rows, scope, aggregate_row


def _placeholder_score_record_count(source: _SourceSlice) -> int:
    records = _source_records(source)
    if records and all(
        isinstance(record.get("metadata"), Mapping)
        and record["metadata"].get("score_contract") == "aggregate_only"
        for record in records
    ):
        return len(records)
    return 0


def _digest_or_hash(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    if HEX_64_RE.fullmatch(text.lower()):
        return text.lower()
    return sha256_bytes(text.encode("utf-8"))


def _public_provenance(source: _SourceSlice, dataset_sha256: str) -> dict[str, Any]:
    private = source.manifest.get("provenance")
    private = private if isinstance(private, Mapping) else {}
    contract = private.get("contract_version")
    producer_revision = str(private.get("trace_git_commit") or "").lower() or None
    harness_revision = str(private.get("vlmevalkit_git_commit") or "").lower() or None
    result = {
        "source_archive_manifest_sha256": source.source_manifest_sha256,
        "source_archive_manifest_size": source.source_manifest_size,
        "source_archive_part_sha256": source.source_parquet_sha256,
        "source_archive_part_size": source.source_parquet_size,
        "source_payload_sha256": _digest_or_hash(source.manifest.get("payload_sha256")),
        "source_contract_sha256": _digest_or_hash(contract),
        "campaign_config_sha256": _digest_or_hash(private.get("campaign_config_hash")),
        "producer_code_revision": (
            producer_revision if producer_revision and re.fullmatch(r"[0-9a-f]{40}", producer_revision) else None
        ),
        "evaluation_harness_revision": (
            harness_revision if harness_revision and re.fullmatch(r"[0-9a-f]{40}", harness_revision) else None
        ),
        "producer_code_sha256": _digest_or_hash(private.get("final25_code_hash")),
        "dataset_sha256": _require_sha256(dataset_sha256, "dataset_sha256"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _safe_public_relative(path: str) -> Path:
    candidate = Path(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        raise PublicExportError(f"unsafe public path: {path!r}")
    return candidate


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_public_parquet(
    root: Path,
    *,
    config_name: str,
    identity: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    private_run_id: str,
) -> tuple[str, str, int]:
    schema = PUBLIC_SCHEMAS[config_name]
    expected_keys = set(schema.names)
    for ordinal, row in enumerate(rows):
        actual_keys = set(row)
        if actual_keys != expected_keys:
            raise PublicExportError(
                f"{config_name} row {ordinal} fields mismatch: "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        _assert_no_internal_markers(row)
        _assert_private_output_safe(row, private_run_id=private_run_id)
    table = pa.Table.from_pylist(list(rows), schema=schema)
    validate_public_schema(table.schema, config_name)
    partition = Path(
        f"run={identity['run_id']}",
        f"model={identity['model_id']}",
        f"seed={identity['seed']}",
        f"benchmark={identity['benchmark_id']}",
    )
    directory = root / "data" / config_name / partition
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "part.pending.parquet"
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        compression_level=9,
        data_page_version="2.0",
        version="2.6",
        use_dictionary=True,
        write_statistics=True,
    )
    digest = sha256_file(temporary)
    final = directory / f"part-{digest}.parquet"
    temporary.replace(final)
    return final.relative_to(root).as_posix(), digest, final.stat().st_size


def _write_part_manifest(
    root: Path,
    *,
    config_name: str,
    identity: Mapping[str, Any],
    rows: int,
    parquet_path: str,
    parquet_sha256: str,
    parquet_size: int,
    provenance: Mapping[str, Any],
    scoring_scope: str | None = None,
    evaluated_rows: int | None = None,
    placeholder_records_removed: int = 0,
) -> tuple[str, str, int]:
    manifest: dict[str, Any] = {
        "schema_version": PART_MANIFEST_VERSION,
        "config_name": config_name,
        "contract_version": PUBLIC_CONTRACTS[config_name],
        "identity": dict(identity),
        "rows": rows,
        "parquet_path": parquet_path,
        "parquet_sha256": parquet_sha256,
        "parquet_size": parquet_size,
        "provenance": dict(provenance),
    }
    if config_name == "scores":
        manifest.update(
            {
                "scoring_scope": scoring_scope,
                "evaluated_rows": evaluated_rows,
                "normalization": {
                    "placeholder_records_removed": placeholder_records_removed,
                },
            }
        )
    _assert_no_internal_markers(manifest)
    partition = Path(
        f"run={identity['run_id']}",
        f"model={identity['model_id']}",
        f"seed={identity['seed']}",
        f"benchmark={identity['benchmark_id']}",
    )
    relative = (
        Path("metadata", "parts", config_name)
        / partition
        / f"part-{parquet_sha256}.manifest.json"
    )
    path = root / relative
    _write_json(path, manifest)
    return relative.as_posix(), sha256_file(path), path.stat().st_size


def _artifact_entry(
    *,
    config_name: str,
    identity: Mapping[str, Any],
    rows: int,
    parquet: tuple[str, str, int],
    part_manifest: tuple[str, str, int],
    scoring_scope: str | None = None,
    evaluated_rows: int | None = None,
    placeholder_records_removed: int = 0,
) -> dict[str, Any]:
    result = {
        "config_name": config_name,
        "run_id": identity["run_id"],
        "model_id": identity["model_id"],
        "seed": identity["seed"],
        "benchmark_id": identity["benchmark_id"],
        "rows": rows,
        "parquet_path": parquet[0],
        "parquet_sha256": parquet[1],
        "parquet_size": parquet[2],
        "manifest_path": part_manifest[0],
        "manifest_sha256": part_manifest[1],
        "manifest_size": part_manifest[2],
    }
    if config_name == "scores":
        result.update(
            {
                "scoring_scope": scoring_scope,
                "evaluated_rows": evaluated_rows,
                "placeholder_records_removed": placeholder_records_removed,
            }
        )
    return result


def _summary_rows(
    rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(float(row["score"]))
    result: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        item = dict(zip(group_keys, key))
        item.update(
            {
                "mean": statistics.fmean(values),
                "stddev": statistics.stdev(values) if len(values) > 1 else None,
                "seed_count": len(values),
            }
        )
        result.append(item)
    return result


def build_results(
    plan: ExportPlan, aggregate_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {
        (model.model_id, seed, benchmark)
        for model in plan.models
        for seed in plan.seeds
        for benchmark in plan.benchmarks
    }
    actual: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in aggregate_rows:
        if row.get("score_scope") != "aggregate" or row.get("score_unit") != "percent":
            raise PublicExportIntegrityError("results require aggregate percent score rows")
        key = (str(row["model_id"]), int(row["seed"]), str(row["benchmark_id"]))
        if key in actual:
            raise PublicExportIntegrityError(f"duplicate aggregate score for {key}")
        actual[key] = row
    if set(actual) != expected:
        raise PublicExportIntegrityError(
            f"aggregate result coverage mismatch: {len(actual)} != {len(expected)}"
        )

    benchmark_scores: list[dict[str, Any]] = []
    for model in plan.models:
        for seed in plan.seeds:
            for benchmark in plan.benchmarks:
                row = actual[(model.model_id, seed, benchmark)]
                benchmark_scores.append(
                    {
                        "model_id": model.model_id,
                        "seed": seed,
                        "benchmark_id": benchmark,
                        "score": float(row["score_value"]),
                        "evaluated_rows": int(row["evaluated_rows"]),
                        "scoring_scope": "aggregate",
                    }
                )

    category_scores: list[dict[str, Any]] = []
    overall_scores: list[dict[str, Any]] = []
    for model in plan.models:
        for seed in plan.seeds:
            by_benchmark = {
                row["benchmark_id"]: float(row["score"])
                for row in benchmark_scores
                if row["model_id"] == model.model_id and row["seed"] == seed
            }
            for category_name, members in plan.categories:
                values = [by_benchmark[item] for item in members]
                category_scores.append(
                    {
                        "model_id": model.model_id,
                        "seed": seed,
                        "category_id": _slug(category_name),
                        "category_name": category_name,
                        "score": statistics.fmean(values),
                        "benchmark_count": len(values),
                    }
                )
            values = [by_benchmark[item] for item in plan.benchmarks]
            overall_scores.append(
                {
                    "model_id": model.model_id,
                    "seed": seed,
                    "score": statistics.fmean(values),
                    "benchmark_count": len(values),
                }
            )

    result = {
        "schema_version": RESULTS_VERSION,
        "suite_id": plan.suite_id,
        "run_id": plan.run_id,
        "score_unit": "percent",
        "aggregation": "unweighted_macro_mean",
        "benchmark_scores": benchmark_scores,
        "category_scores": category_scores,
        "overall_scores": overall_scores,
        "benchmark_summaries": _summary_rows(
            benchmark_scores, ("model_id", "benchmark_id")
        ),
        "category_summaries": _summary_rows(
            category_scores,
            ("model_id", "category_id", "category_name", "benchmark_count"),
        ),
        "overall_summaries": _summary_rows(
            overall_scores, ("model_id", "benchmark_count")
        ),
    }
    _assert_no_internal_markers(result)
    return result


def load_export_plan(path: Path | str) -> ExportPlan:
    plan_path = Path(path)
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicExportError(f"cannot read export plan {plan_path}: {error}") from error
    return ExportPlan.from_mapping(_require_mapping(raw, "export plan"))


def _metadata_file(root: Path, relative: str, value: Any) -> dict[str, Any]:
    _assert_no_internal_markers(value)
    path = root / _safe_public_relative(relative)
    _write_json(path, value)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _part_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "model_id": row["model_id"],
        "model_revision": row["model_revision"],
        "seed": row["seed"],
        "benchmark_id": row["benchmark_id"],
        "dataset_split": row["dataset_split"],
        "dataset_revision": row["dataset_revision"],
    }


def _readme(plan: ExportPlan, config_stats: Mapping[str, Mapping[str, int]]) -> str:
    lines = [
        "---",
        "license: other",
        "pretty_name: Trace Evaluation Runs",
        "configs:",
    ]
    for config in CONFIG_NAMES:
        lines.extend(
            [
                f"- config_name: {config}",
                "  data_files:",
                "  - split: test",
                f"    path: data/{config}/**/*.parquet",
            ]
        )
    lines.extend(
        [
            "---",
            "",
            "# Trace Evaluation Runs",
            "",
            "This repository contains response, extraction, and scoring artifacts for the Trace visual-reasoning evaluation suite.",
            "Benchmark prompts, questions, reference answers, answer options, media, and local file paths are not redistributed.",
            "Prompt and media digests preserve deterministic linkage for authorized reconstruction.",
            "",
            "## Configurations",
            "",
        ]
    )
    for config in CONFIG_NAMES:
        stats = config_stats[config]
        lines.append(
            f"- `{config}`: {stats['rows']:,} records in {stats['parts']:,} parts."
        )
    removed = int(config_stats["scores"].get("placeholder_records_removed", 0))
    lines.extend(
        [
            "",
            "Scores contain one aggregate record per model, seed, and benchmark. Sample-level score records are included only when the scorer emitted actual per-sample judgments.",
            f"Aggregate-only placeholder records were normalized out ({removed:,} records removed).",
            "",
            "## Benchmarks",
            "",
        ]
    )
    for category_name, benchmarks in plan.categories:
        lines.append(f"### {category_name}")
        lines.append("")
        lines.append(", ".join(f"`{item}`" for item in benchmarks))
        lines.append("")
    lines.extend(
        [
            "## Models",
            "",
            *[
                f"- `{model.model_id}`: {model.display_name} "
                f"(`{model.repository_id}@{model.repository_revision}`)"
                for model in plan.models
            ],
            "",
            "## Integrity",
            "",
            "`metadata/manifest.json` is the release allowlist. Each Parquet part and part manifest is content-addressed, and `metadata/results/benchmark_scores.json` is recomputed from aggregate score records during verification.",
            "",
            "Prompt text can be verified locally with the exporter CLI when an authorized benchmark reconstruction is available. Dataset reconstruction remains an external release gate and is not claimed by this artifact set.",
            "",
        ]
    )
    text = "\n".join(lines)
    _assert_no_internal_markers(text)
    return text


def _tree_files(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def export_public_artifacts(
    source_root: Path | str,
    plan: ExportPlan | Mapping[str, Any] | Path | str,
    output_root: Path | str,
) -> VerifiedPublicExport:
    """Create an atomic, deterministic public artifact tree.

    The inputs are intentionally generic: any canonical archive campaign with
    the same generation/extraction/score slice contracts can be exported by
    providing a new explicit source-to-public model map and public suite plan.
    """

    if isinstance(plan, ExportPlan):
        export_plan = plan
    elif isinstance(plan, Mapping):
        export_plan = ExportPlan.from_mapping(plan)
    else:
        export_plan = load_export_plan(plan)
    source = _resolve_source_root(source_root)
    destination = Path(output_root).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".trace-eval-public-", dir=destination.parent)
    )
    try:
        selected = _discover_source_slices(source, export_plan)
        artifacts: list[dict[str, Any]] = []
        aggregate_rows: list[dict[str, Any]] = []
        generation_settings: set[str] = set()
        media_inputs_values: set[str] = set()
        config_stats: dict[str, dict[str, int]] = {
            config: {"rows": 0, "parts": 0, "placeholder_records_removed": 0}
            for config in CONFIG_NAMES
        }

        for model in export_plan.models:
            for seed in export_plan.seeds:
                for benchmark in export_plan.benchmarks:
                    response_source = selected[("responses", model.model_id, seed, benchmark)]
                    extraction_source = selected[("extractions", model.model_id, seed, benchmark)]
                    score_source = selected[("scores", model.model_id, seed, benchmark)]
                    response_rows, contexts, settings = _response_rows(
                        response_source, export_plan
                    )
                    generation_settings.update(settings)
                    media_inputs_values.add(
                        canonical_json(_source_media_inputs(response_source))
                    )
                    extraction_rows = _extraction_rows(
                        extraction_source, contexts, export_plan.judge
                    )
                    score_rows, scoring_scope, aggregate_row = _score_rows(
                        score_source, contexts
                    )
                    aggregate_rows.append(aggregate_row)
                    placeholder_records_removed = (
                        _placeholder_score_record_count(score_source)
                        if scoring_scope == "aggregate_only"
                        else 0
                    )
                    rows_by_config = {
                        "responses": response_rows,
                        "extractions": extraction_rows,
                        "scores": score_rows,
                    }
                    source_by_config = {
                        "responses": response_source,
                        "extractions": extraction_source,
                        "scores": score_source,
                    }
                    for config_name in CONFIG_NAMES:
                        rows = rows_by_config[config_name]
                        if not rows:
                            raise PublicExportError(f"empty public {config_name} slice")
                        identity = _part_identity(rows[0])
                        dataset_sha = str(rows[0]["dataset_sha256"])
                        parquet = _write_public_parquet(
                            temporary,
                            config_name=config_name,
                            identity=identity,
                            rows=rows,
                            private_run_id=export_plan.source_run_id,
                        )
                        part_manifest = _write_part_manifest(
                            temporary,
                            config_name=config_name,
                            identity=identity,
                            rows=len(rows),
                            parquet_path=parquet[0],
                            parquet_sha256=parquet[1],
                            parquet_size=parquet[2],
                            provenance=_public_provenance(
                                source_by_config[config_name], dataset_sha
                            ),
                            scoring_scope=(
                                scoring_scope if config_name == "scores" else None
                            ),
                            evaluated_rows=(
                                int(aggregate_row["evaluated_rows"])
                                if config_name == "scores"
                                else None
                            ),
                            placeholder_records_removed=(
                                placeholder_records_removed
                                if config_name == "scores"
                                else 0
                            ),
                        )
                        artifacts.append(
                            _artifact_entry(
                                config_name=config_name,
                                identity=identity,
                                rows=len(rows),
                                parquet=parquet,
                                part_manifest=part_manifest,
                                scoring_scope=(
                                    scoring_scope if config_name == "scores" else None
                                ),
                                evaluated_rows=(
                                    int(aggregate_row["evaluated_rows"])
                                    if config_name == "scores"
                                    else None
                                ),
                                placeholder_records_removed=(
                                    placeholder_records_removed
                                    if config_name == "scores"
                                    else 0
                                ),
                            )
                        )
                        config_stats[config_name]["rows"] += len(rows)
                        config_stats[config_name]["parts"] += 1
                        config_stats[config_name][
                            "placeholder_records_removed"
                        ] += (
                            placeholder_records_removed
                            if config_name == "scores"
                            else 0
                        )

        artifacts.sort(
            key=lambda item: (
                CONFIG_NAMES.index(item["config_name"]),
                item["model_id"],
                item["seed"],
                export_plan.benchmarks.index(item["benchmark_id"]),
            )
        )
        results = build_results(export_plan, aggregate_rows)
        metadata_files: list[dict[str, Any]] = []
        suite_metadata = {
            "schema_version": SUITE_METADATA_VERSION,
            "suite_id": export_plan.suite_id,
            "benchmark_count": len(export_plan.benchmarks),
            "benchmark_ids": list(export_plan.benchmarks),
            "categories": [
                {
                    "category_id": _slug(name),
                    "category_name": name,
                    "benchmark_ids": list(members),
                }
                for name, members in export_plan.categories
            ],
        }
        metadata_files.append(
            _metadata_file(
                temporary,
                f"metadata/suites/{export_plan.suite_id}.json",
                suite_metadata,
            )
        )
        for model in export_plan.models:
            metadata_files.append(
                _metadata_file(
                    temporary,
                    f"metadata/models/{model.model_id}.json",
                    {
                        "schema_version": MODEL_METADATA_VERSION,
                        "model_id": model.model_id,
                        "model_revision": model.model_revision,
                        "display_name": model.display_name,
                        "repository_id": model.repository_id,
                        "repository_revision": model.repository_revision,
                    },
                )
            )
        parsed_settings = [json.loads(item) for item in sorted(generation_settings)]
        if len(media_inputs_values) != 1:
            raise PublicExportIntegrityError(
                "source generation media settings differ across selected slices"
            )
        media_inputs = json.loads(next(iter(media_inputs_values)))
        run_metadata = {
            "schema_version": RUN_METADATA_VERSION,
            "run_id": export_plan.run_id,
            "suite_id": export_plan.suite_id,
            "model_ids": [model.model_id for model in export_plan.models],
            "seeds": list(export_plan.seeds),
            "generation_settings": parsed_settings,
            "media_inputs": media_inputs,
            "judge_model": {
                "model_id": export_plan.judge.model_id,
                "model_revision": export_plan.judge.model_revision,
            },
            "source_selection_sha256": export_plan.source_selection_sha256,
            "source_slice_set_sha256": export_plan.source_slice_set_sha256,
            "prompt_reconstruction": {
                "verification_supported": True,
                "dataset_reconstruction_status": "external_release_gate",
            },
        }
        metadata_files.append(
            _metadata_file(
                temporary,
                f"metadata/runs/{export_plan.run_id}.json",
                run_metadata,
            )
        )
        metadata_files.append(
            _metadata_file(
                temporary, "metadata/results/benchmark_scores.json", results
            )
        )
        readme_path = temporary / "README.md"
        readme_path.write_text(_readme(export_plan, config_stats), encoding="utf-8")
        release_files = [
            {
                "path": "README.md",
                "sha256": sha256_file(readme_path),
                "size": readme_path.stat().st_size,
            }
        ]
        manifest = {
            "schema_version": EXPORT_MANIFEST_VERSION,
            "suite_id": export_plan.suite_id,
            "run_ids": [export_plan.run_id],
            "neutralized": True,
            "source_selection_sha256": export_plan.source_selection_sha256,
            "source_slice_set_sha256": export_plan.source_slice_set_sha256,
            "configs": [
                {
                    "config_name": config,
                    "contract_version": PUBLIC_CONTRACTS[config],
                    **config_stats[config],
                }
                for config in CONFIG_NAMES
            ],
            "artifacts": artifacts,
            "metadata_files": sorted(metadata_files, key=lambda item: item["path"]),
            "release_files": release_files,
        }
        _assert_no_internal_markers(
            manifest,
            dynamic_markers=(export_plan.source_run_id,),
        )
        _write_json(temporary / "metadata" / "manifest.json", manifest)
        load_and_verify_public_export(
            temporary, expected_artifacts=len(export_plan.expected_identities)
        )
        if destination.exists():
            existing = load_and_verify_public_export(
                destination, expected_artifacts=len(export_plan.expected_identities)
            )
            if _tree_files(temporary) != _tree_files(destination):
                raise PublicExportIntegrityError(
                    "existing public export is valid but differs from deterministic rebuild"
                )
            shutil.rmtree(temporary)
            return existing
        temporary.replace(destination)
        return load_and_verify_public_export(
            destination, expected_artifacts=len(export_plan.expected_identities)
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicExportIntegrityError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PublicExportIntegrityError(f"{label} must be a JSON object: {path}")
    return value


def _artifact_file(root: Path, relative: Any, expected_sha: Any, expected_size: Any) -> Path:
    path = root / _safe_public_relative(_require_text(relative, "artifact path"))
    if not path.is_file():
        raise PublicExportIntegrityError(f"missing public artifact: {path}")
    digest = _require_sha256(expected_sha, f"digest for {relative}")
    if sha256_file(path) != digest:
        raise PublicExportIntegrityError(f"public artifact digest mismatch: {relative}")
    try:
        size = int(expected_size)
    except (TypeError, ValueError, OverflowError) as error:
        raise PublicExportIntegrityError(f"invalid size for {relative}") from error
    if path.stat().st_size != size:
        raise PublicExportIntegrityError(f"public artifact size mismatch: {relative}")
    return path


def _validate_response_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("schema_version") != SCHEMA_VERSION:
        raise PublicExportIntegrityError("response schema version mismatch")
    if row.get("contract_version") != RESPONSE_CONTRACT_VERSION:
        raise PublicExportIntegrityError("response contract mismatch")
    response = row.get("response_text")
    if not isinstance(response, str) or response_sha256(response) != row.get("response_sha256"):
        raise PublicExportIntegrityError("response text digest mismatch")
    ordered_media = row.get("ordered_media_sha256")
    if not isinstance(ordered_media, list):
        raise PublicExportIntegrityError("response ordered media must be a list")
    media_digest = media_set_sha256(ordered_media)
    if media_digest != row.get("media_set_sha256"):
        raise PublicExportIntegrityError("response media digest mismatch")
    source_record_digest = source_record_sha256(
        str(row.get("source_row_sha256")), media_digest
    )
    if source_record_digest != row.get("source_record_sha256"):
        raise PublicExportIntegrityError("response source-record digest mismatch")
    try:
        settings = json.loads(str(row.get("generation_settings_json")))
    except json.JSONDecodeError as error:
        raise PublicExportIntegrityError("invalid generation settings JSON") from error
    if not isinstance(settings, dict) or canonical_json(settings) != row.get(
        "generation_settings_json"
    ):
        raise PublicExportIntegrityError("generation settings are not canonical")
    request_digest = public_request_sha256_from_hashes(
        prompt_digest=str(row.get("prompt_sha256")),
        ordered_media_sha256=ordered_media,
        generation_settings=settings,
    )
    if request_digest != row.get("request_sha256"):
        raise PublicExportIntegrityError("response request digest mismatch")
    contract_digest = prompt_contract_sha256(
        dataset_sha256=str(row.get("dataset_sha256")),
        source_row_sha256=str(row.get("source_row_sha256")),
        source_record_digest=source_record_digest,
        media_digest=media_digest,
        prompt_digest=str(row.get("prompt_sha256")),
        request_digest=request_digest,
    )
    if contract_digest != row.get("prompt_contract_sha256"):
        raise PublicExportIntegrityError("response prompt contract mismatch")
    expected_sample = _sample_id(
        run_id=str(row.get("run_id")),
        model_id=str(row.get("model_id")),
        model_revision=str(row.get("model_revision")),
        seed=int(row.get("seed")),
        benchmark_id=str(row.get("benchmark_id")),
        source_index=str(row.get("source_index")),
        source_ordinal=int(row.get("source_ordinal")),
        source_row_sha256=str(row.get("source_row_sha256")),
    )
    if expected_sample != row.get("sample_id"):
        raise PublicExportIntegrityError("response sample identity mismatch")
    if _response_id(expected_sample, str(row.get("response_sha256"))) != row.get(
        "response_id"
    ):
        raise PublicExportIntegrityError("response identity mismatch")
    if _dataset_revision(str(row.get("dataset_sha256"))) != row.get("dataset_revision"):
        raise PublicExportIntegrityError("response dataset revision mismatch")
    _media_inputs_from_settings(settings)
    return settings


def _assert_linked_common(
    row: Mapping[str, Any], response: Mapping[str, Any], label: str
) -> None:
    for field in COMMON_SAMPLE_FIELDS:
        if row.get(field.name) != response.get(field.name):
            raise PublicExportIntegrityError(
                f"{label} does not match response field {field.name}"
            )


def _validate_extraction_row(
    row: Mapping[str, Any], responses: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str] | None:
    if row.get("schema_version") != SCHEMA_VERSION or row.get(
        "contract_version"
    ) != EXTRACTION_CONTRACT_VERSION:
        raise PublicExportIntegrityError("extraction contract mismatch")
    sample_id = str(row.get("sample_id"))
    response = responses.get(sample_id)
    if response is None:
        raise PublicExportIntegrityError("extraction has no linked response")
    _assert_linked_common(row, response, "extraction")
    try:
        value = json.loads(str(row.get("extraction_value_json")))
        candidates = json.loads(str(row.get("extraction_candidates_json")))
    except json.JSONDecodeError as error:
        raise PublicExportIntegrityError("invalid extraction JSON") from error
    if canonical_json(value) != row.get("extraction_value_json") or canonical_json(
        candidates
    ) != row.get("extraction_candidates_json"):
        raise PublicExportIntegrityError("extraction JSON is not canonical")
    if _json_value_type(value) != row.get("extraction_value_type"):
        raise PublicExportIntegrityError("extraction value type mismatch")
    material = {
        "contract_version": EXTRACTION_CONTRACT_VERSION,
        "sample_id": sample_id,
        "status": row.get("extraction_status"),
        "value": value,
        "candidates": candidates,
    }
    if canonical_sha256(material) != row.get("extraction_sha256"):
        raise PublicExportIntegrityError("extraction digest mismatch")
    judge_output = row.get("judge_output")
    judge_digest = row.get("judge_output_sha256")
    if (judge_output is None) != (judge_digest is None):
        raise PublicExportIntegrityError("judge output and digest nullability mismatch")
    if judge_output is not None and sha256_bytes(str(judge_output).encode("utf-8")) != judge_digest:
        raise PublicExportIntegrityError("judge output digest mismatch")
    if bool(row.get("used_judge")) != (judge_output is not None):
        raise PublicExportIntegrityError("judge usage is not represented by judge output")
    judge_id = row.get("judge_model_id")
    judge_revision = row.get("judge_model_revision")
    if row.get("used_judge"):
        return (
            _require_repository_id(judge_id, "judge model id"),
            _require_revision(judge_revision, "judge model revision"),
        )
    if judge_id is not None or judge_revision is not None:
        raise PublicExportIntegrityError(
            "deterministic extraction contains judge model metadata"
        )
    return None


def _validate_score_rows(
    rows: Sequence[Mapping[str, Any]],
    responses: Mapping[str, Mapping[str, Any]],
    scoring_scope: str,
) -> dict[str, Any]:
    aggregate = [row for row in rows if row.get("score_scope") == "aggregate"]
    if len(aggregate) != 1:
        raise PublicExportIntegrityError("score part must contain one aggregate record")
    row_scores = [row for row in rows if row.get("score_scope") == "row"]
    if scoring_scope == "aggregate_only" and row_scores:
        raise PublicExportIntegrityError("aggregate-only part contains row scores")
    if scoring_scope == "row_and_aggregate" and len(row_scores) != len(responses):
        raise PublicExportIntegrityError("row-score coverage does not match responses")
    if scoring_scope not in {"aggregate_only", "row_and_aggregate"}:
        raise PublicExportIntegrityError(f"invalid scoring scope: {scoring_scope!r}")
    if int(aggregate[0].get("evaluated_rows") or -1) != len(responses):
        raise PublicExportIntegrityError(
            "aggregate evaluated rows do not match response count"
        )
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION or row.get(
            "contract_version"
        ) != SCORE_CONTRACT_VERSION:
            raise PublicExportIntegrityError("score contract mismatch")
        scope = str(row.get("score_scope"))
        sample_id = row.get("sample_id")
        expected_id = _score_id(row, scope=scope, sample_id=sample_id)
        if expected_id != row.get("score_id") or expected_id in seen:
            raise PublicExportIntegrityError("score identity mismatch or duplicate")
        seen.add(expected_id)
        try:
            score_json = json.loads(str(row.get("score_value_json")))
        except json.JSONDecodeError as error:
            raise PublicExportIntegrityError("invalid score value JSON") from error
        if canonical_json(score_json) != row.get("score_value_json"):
            raise PublicExportIntegrityError("score value JSON is not canonical")
        if scope == "row":
            response = responses.get(str(sample_id))
            if response is None:
                raise PublicExportIntegrityError("row score has no linked response")
            for name in (
                "run_id", "model_id", "model_revision", "seed", "benchmark_id",
                "dataset_split", "dataset_revision", "response_id", "source_index",
                "source_ordinal", "dataset_sha256", "source_row_sha256",
                "source_record_sha256", "media_set_sha256", "prompt_sha256",
                "request_sha256", "prompt_contract_sha256", "response_sha256",
            ):
                if row.get(name) != response.get(name):
                    raise PublicExportIntegrityError(
                        f"row score does not match response field {name}"
                    )
            numeric = _score_numeric(score_json, excluded=bool(row.get("excluded")))
            if numeric != row.get("score_value") or row.get("score_unit") != "fraction":
                raise PublicExportIntegrityError("row score value mismatch")
        else:
            if any(
                row.get(name) is not None
                for name in (
                    "sample_id", "response_id", "source_index", "source_ordinal",
                    "source_row_sha256", "source_record_sha256", "media_set_sha256",
                    "prompt_sha256", "request_sha256", "prompt_contract_sha256",
                    "response_sha256",
                )
            ):
                raise PublicExportIntegrityError("aggregate score contains sample linkage")
            if row.get("score_unit") != "percent" or row.get("evaluated_rows") is None:
                raise PublicExportIntegrityError("aggregate score metadata mismatch")
            if not isinstance(score_json, (int, float)) or isinstance(score_json, bool):
                raise PublicExportIntegrityError("aggregate score JSON must be numeric")
            if float(score_json) != row.get("score_value"):
                raise PublicExportIntegrityError("aggregate score value mismatch")
    return dict(aggregate[0])


def _load_verified_part(
    root: Path,
    artifact: Mapping[str, Any],
    *,
    dynamic_markers: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any], tuple[PublicFile, PublicFile]]:
    config = _require_text(artifact.get("config_name"), "artifact config")
    if config not in CONFIG_NAMES:
        raise PublicExportIntegrityError(f"unknown artifact config: {config}")
    parquet_path = _artifact_file(
        root,
        artifact.get("parquet_path"),
        artifact.get("parquet_sha256"),
        artifact.get("parquet_size"),
    )
    part_path = _artifact_file(
        root,
        artifact.get("manifest_path"),
        artifact.get("manifest_sha256"),
        artifact.get("manifest_size"),
    )
    part = _load_json(part_path, "part manifest")
    _assert_no_internal_markers(part, dynamic_markers=dynamic_markers)
    for name in (
        "config_name", "rows", "parquet_path", "parquet_sha256", "parquet_size"
    ):
        if part.get(name) != artifact.get(name):
            raise PublicExportIntegrityError(f"part manifest mismatch for {name}")
    if part.get("schema_version") != PART_MANIFEST_VERSION or part.get(
        "contract_version"
    ) != PUBLIC_CONTRACTS[config]:
        raise PublicExportIntegrityError("part manifest contract mismatch")
    identity = _require_mapping(part.get("identity"), "part identity")
    for name in ("run_id", "model_id", "seed", "benchmark_id"):
        if identity.get(name) != artifact.get(name):
            raise PublicExportIntegrityError(f"artifact identity mismatch for {name}")
    pf = pq.ParquetFile(parquet_path)
    validate_public_schema(pf.schema_arrow, config)
    table = pf.read()
    if table.num_rows != artifact.get("rows"):
        raise PublicExportIntegrityError("public Parquet row count mismatch")
    rows = table.to_pylist()
    for row in rows:
        _assert_no_internal_markers(row, dynamic_markers=dynamic_markers)
        _assert_private_output_safe(row, private_markers=dynamic_markers)
        for name in ("run_id", "model_id", "seed", "benchmark_id"):
            if row.get(name) != artifact.get(name):
                raise PublicExportIntegrityError(f"public row identity mismatch for {name}")
    return (
        rows,
        part,
        (
            PublicFile(
                str(artifact["parquet_path"]),
                str(artifact["parquet_sha256"]),
                int(artifact["parquet_size"]),
            ),
            PublicFile(
                str(artifact["manifest_path"]),
                str(artifact["manifest_sha256"]),
                int(artifact["manifest_size"]),
            ),
        ),
    )


def _public_source_binding_record(
    artifact: Mapping[str, Any], part: Mapping[str, Any]
) -> dict[str, Any]:
    provenance = _require_mapping(part.get("provenance"), "part provenance")
    return {
        "config_name": artifact["config_name"],
        "model_id": artifact["model_id"],
        "seed": int(artifact["seed"]),
        "benchmark_id": artifact["benchmark_id"],
        "manifest": {
            "sha256": _require_sha256(
                provenance.get("source_archive_manifest_sha256"),
                "source archive manifest digest",
            ),
            "size": int(provenance.get("source_archive_manifest_size")),
        },
        "parquet": {
            "sha256": _require_sha256(
                provenance.get("source_archive_part_sha256"),
                "source archive part digest",
            ),
            "size": int(provenance.get("source_archive_part_size")),
        },
    }


def load_and_verify_public_export(
    root: Path | str,
    *,
    expected_artifacts: int | None = None,
    dynamic_markers: Sequence[str] = (),
) -> VerifiedPublicExport:
    """Verify a public export and return its exact upload allowlist."""

    export_root = Path(root).expanduser().resolve()
    manifest_path = export_root / "metadata" / "manifest.json"
    manifest = _load_json(manifest_path, "export manifest")
    _assert_no_internal_markers(manifest, dynamic_markers=dynamic_markers)
    if manifest.get("schema_version") != EXPORT_MANIFEST_VERSION:
        raise PublicExportIntegrityError("export manifest schema mismatch")
    if manifest.get("neutralized") is not True:
        raise PublicExportIntegrityError("export manifest is not marked neutralized")
    artifacts_raw = manifest.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise PublicExportIntegrityError("export manifest artifacts must be a list")
    if expected_artifacts is not None and len(artifacts_raw) != expected_artifacts:
        raise PublicExportIntegrityError(
            f"artifact count mismatch: {len(artifacts_raw)} != {expected_artifacts}"
        )
    artifacts: list[Mapping[str, Any]] = []
    seen_artifact_keys: set[tuple[str, str, int, str]] = set()
    grouped: dict[tuple[str, int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for ordinal, raw in enumerate(artifacts_raw):
        artifact = _require_mapping(raw, f"artifacts[{ordinal}]")
        key = (
            str(artifact.get("config_name")),
            str(artifact.get("model_id")),
            int(artifact.get("seed")),
            str(artifact.get("benchmark_id")),
        )
        if key in seen_artifact_keys:
            raise PublicExportIntegrityError(f"duplicate artifact identity: {key}")
        seen_artifact_keys.add(key)
        group_key = (key[1], key[2], key[3])
        grouped[group_key][key[0]] = artifact
        artifacts.append(artifact)
    for key, configs in grouped.items():
        if set(configs) != set(CONFIG_NAMES):
            raise PublicExportIntegrityError(
                f"artifact stage coverage mismatch for {key}: {sorted(configs)}"
            )

    files: list[PublicFile] = []
    aggregate_rows: list[dict[str, Any]] = []
    observed_judges: set[tuple[str, str]] = set()
    observed_generation_settings: set[str] = set()
    observed_media_inputs: set[str] = set()
    source_binding_records: list[dict[str, Any]] = []
    computed_stats = {
        config: {"rows": 0, "parts": 0, "placeholder_records_removed": 0}
        for config in CONFIG_NAMES
    }
    for key in sorted(grouped):
        configs = grouped[key]
        response_rows, response_part, response_files = _load_verified_part(
            export_root,
            configs["responses"],
            dynamic_markers=dynamic_markers,
        )
        files.extend(response_files)
        source_binding_records.append(
            _public_source_binding_record(configs["responses"], response_part)
        )
        responses: dict[str, Mapping[str, Any]] = {}
        for row in response_rows:
            settings = _validate_response_row(row)
            observed_generation_settings.add(canonical_json(settings))
            observed_media_inputs.add(
                canonical_json(_media_inputs_from_settings(settings))
            )
            sample_id = str(row["sample_id"])
            if sample_id in responses:
                raise PublicExportIntegrityError("duplicate response sample identity")
            responses[sample_id] = row

        extraction_rows, extraction_part, extraction_files = _load_verified_part(
            export_root,
            configs["extractions"],
            dynamic_markers=dynamic_markers,
        )
        files.extend(extraction_files)
        source_binding_records.append(
            _public_source_binding_record(configs["extractions"], extraction_part)
        )
        for row in extraction_rows:
            observed_judge = _validate_extraction_row(row, responses)
            if observed_judge is not None:
                observed_judges.add(observed_judge)
        if {str(row["sample_id"]) for row in extraction_rows} != set(responses):
            raise PublicExportIntegrityError("extraction coverage does not match responses")

        score_artifact = configs["scores"]
        score_rows, score_part, score_files = _load_verified_part(
            export_root,
            score_artifact,
            dynamic_markers=dynamic_markers,
        )
        files.extend(score_files)
        source_binding_records.append(
            _public_source_binding_record(score_artifact, score_part)
        )
        scoring_scope = _require_text(
            score_artifact.get("scoring_scope"), "score artifact scoring scope"
        )
        if score_part.get("scoring_scope") != scoring_scope:
            raise PublicExportIntegrityError("score part scope mismatch")
        aggregate = _validate_score_rows(score_rows, responses, scoring_scope)
        if int(score_artifact.get("evaluated_rows")) != int(
            aggregate["evaluated_rows"]
        ):
            raise PublicExportIntegrityError("score evaluated row count mismatch")
        aggregate_rows.append(aggregate)

        for config in CONFIG_NAMES:
            artifact = configs[config]
            computed_stats[config]["rows"] += int(artifact["rows"])
            computed_stats[config]["parts"] += 1
            if config == "scores":
                removed = int(artifact.get("placeholder_records_removed", 0))
                normalization = _require_mapping(
                    score_part.get("normalization"), "score normalization"
                )
                if int(normalization.get("placeholder_records_removed", -1)) != removed:
                    raise PublicExportIntegrityError(
                        "score placeholder normalization mismatch"
                    )
                computed_stats[config]["placeholder_records_removed"] += removed

    configs_raw = manifest.get("configs")
    if not isinstance(configs_raw, list) or len(configs_raw) != len(CONFIG_NAMES):
        raise PublicExportIntegrityError("export config summary is incomplete")
    by_config = {str(item.get("config_name")): item for item in configs_raw if isinstance(item, Mapping)}
    if set(by_config) != set(CONFIG_NAMES):
        raise PublicExportIntegrityError("export config summary names mismatch")
    for config in CONFIG_NAMES:
        expected = {
            "rows": computed_stats[config]["rows"],
            "parts": computed_stats[config]["parts"],
            "placeholder_records_removed": computed_stats[config][
                "placeholder_records_removed"
            ],
        }
        for name, value in expected.items():
            if int(by_config[config].get(name, -1)) != value:
                raise PublicExportIntegrityError(
                    f"export config {config} summary mismatch for {name}"
                )
        if by_config[config].get("contract_version") != PUBLIC_CONTRACTS[config]:
            raise PublicExportIntegrityError(f"export config {config} contract mismatch")
    actual_slice_set = source_slice_set_sha256(source_binding_records)
    manifest_slice_set = _require_sha256(
        manifest.get("source_slice_set_sha256"), "manifest source slice-set digest"
    )
    if actual_slice_set != manifest_slice_set:
        raise PublicExportIntegrityError(
            "public part provenance does not match the source slice-set digest"
        )

    metadata_raw = manifest.get("metadata_files")
    release_raw = manifest.get("release_files")
    if not isinstance(metadata_raw, list) or not isinstance(release_raw, list):
        raise PublicExportIntegrityError("manifest file allowlists must be lists")
    metadata_values: dict[str, dict[str, Any]] = {}
    for item in [*metadata_raw, *release_raw]:
        entry = _require_mapping(item, "allowlisted file")
        path = _artifact_file(
            export_root, entry.get("path"), entry.get("sha256"), entry.get("size")
        )
        relative = path.relative_to(export_root).as_posix()
        if relative in metadata_values:
            raise PublicExportIntegrityError(f"duplicate allowlisted file: {relative}")
        if path.suffix == ".json":
            metadata_values[relative] = _load_json(path, "metadata file")
            _assert_no_internal_markers(
                metadata_values[relative], dynamic_markers=dynamic_markers
            )
        else:
            text = path.read_text(encoding="utf-8")
            _assert_no_internal_markers(text, dynamic_markers=dynamic_markers)
            metadata_values[relative] = {"text": text}
        files.append(PublicFile(relative, str(entry["sha256"]), int(entry["size"])))

    suite_id = _require_public_id(manifest.get("suite_id"), "manifest suite id")
    run_ids = manifest.get("run_ids")
    if not isinstance(run_ids, list) or len(run_ids) != 1:
        raise PublicExportIntegrityError("this export contract requires exactly one run")
    run_id = _require_public_id(run_ids[0], "manifest run id")
    suite_path = f"metadata/suites/{suite_id}.json"
    run_path = f"metadata/runs/{run_id}.json"
    results_path = "metadata/results/benchmark_scores.json"
    for required in (suite_path, run_path, results_path):
        if required not in metadata_values:
            raise PublicExportIntegrityError(f"required metadata file is absent: {required}")
    suite = metadata_values[suite_path]
    run = metadata_values[run_path]
    results = metadata_values[results_path]
    if suite.get("schema_version") != SUITE_METADATA_VERSION or suite.get(
        "suite_id"
    ) != suite_id:
        raise PublicExportIntegrityError("suite metadata mismatch")
    if run.get("schema_version") != RUN_METADATA_VERSION or run.get("run_id") != run_id:
        raise PublicExportIntegrityError("run metadata mismatch")
    if run.get("suite_id") != suite_id:
        raise PublicExportIntegrityError("run suite identity mismatch")
    if run.get("source_slice_set_sha256") != manifest_slice_set:
        raise PublicExportIntegrityError("run source slice-set digest mismatch")
    if run.get("source_selection_sha256") != manifest.get("source_selection_sha256"):
        raise PublicExportIntegrityError("run source selection digest mismatch")
    run_settings = run.get("generation_settings")
    if not isinstance(run_settings, list) or {
        canonical_json(item) for item in run_settings
    } != observed_generation_settings:
        raise PublicExportIntegrityError("run generation settings mismatch")
    run_media_inputs = _require_mapping(run.get("media_inputs"), "run media inputs")
    if observed_media_inputs != {canonical_json(run_media_inputs)}:
        raise PublicExportIntegrityError("run media inputs mismatch")
    _media_inputs_from_settings(run_media_inputs)
    judge_metadata = _require_mapping(run.get("judge_model"), "run judge model")
    judge_mapping = JudgeMapping(
        source_model_id="private-judge",
        model_id=_require_repository_id(
            judge_metadata.get("model_id"), "run judge model id"
        ),
        model_revision=_require_revision(
            judge_metadata.get("model_revision"), "run judge model revision"
        ),
    )
    if observed_judges and observed_judges != {
        (judge_mapping.model_id, judge_mapping.model_revision)
    }:
        raise PublicExportIntegrityError(
            "extraction judge identities do not match run metadata"
        )
    model_ids = run.get("model_ids")
    seeds = run.get("seeds")
    if not isinstance(model_ids, list) or not isinstance(seeds, list):
        raise PublicExportIntegrityError("run model or seed coverage is invalid")
    models: list[ModelMapping] = []
    for model_id in model_ids:
        model_path = f"metadata/models/{model_id}.json"
        model = metadata_values.get(model_path)
        if model is None or model.get("schema_version") != MODEL_METADATA_VERSION:
            raise PublicExportIntegrityError(f"model metadata missing: {model_id}")
        models.append(
            ModelMapping(
                source_model_id=f"private-{model_id}",
                source_revision=str(model["model_revision"]),
                model_id=str(model_id),
                model_revision=_require_revision(
                    model.get("model_revision"), "model revision"
                ),
                display_name=_require_text(model.get("display_name"), "model display name"),
                repository_id=_require_repository_id(
                    model.get("repository_id"), "model repository id"
                ),
                repository_revision=_require_revision(
                    model.get("repository_revision"), "model repository revision"
                ),
            )
        )
    categories_raw = suite.get("categories")
    benchmarks_raw = suite.get("benchmark_ids")
    if not isinstance(categories_raw, list) or not isinstance(benchmarks_raw, list):
        raise PublicExportIntegrityError("suite coverage metadata is invalid")
    categories = tuple(
        (
            _require_text(item.get("category_name"), "category name"),
            tuple(str(value) for value in item.get("benchmark_ids", [])),
        )
        for item in categories_raw
        if isinstance(item, Mapping)
    )
    verification_plan = ExportPlan(
        source_run_id="private-input",
        source_selection_sha256=_require_sha256(
            manifest.get("source_selection_sha256"), "source selection digest"
        ),
        source_slice_set_sha256=manifest_slice_set,
        run_id=run_id,
        suite_id=suite_id,
        benchmarks=tuple(str(item) for item in benchmarks_raw),
        categories=categories,
        seeds=tuple(int(item) for item in seeds),
        models=tuple(models),
        judge=judge_mapping,
    )
    recomputed_results = build_results(verification_plan, aggregate_rows)
    if canonical_json(recomputed_results) != canonical_json(results):
        raise PublicExportIntegrityError(
            "published results do not match aggregate score Parquets"
        )

    manifest_digest = sha256_file(manifest_path)
    files.append(
        PublicFile(
            "metadata/manifest.json", manifest_digest, manifest_path.stat().st_size
        )
    )
    allowed = {item.path for item in files}
    if len(allowed) != len(files):
        raise PublicExportIntegrityError("upload allowlist contains duplicate paths")
    actual = set(_tree_files(export_root))
    if actual != allowed:
        raise PublicExportIntegrityError(
            f"public tree differs from upload allowlist: "
            f"unexpected={sorted(actual - allowed)[:5]} missing={sorted(allowed - actual)[:5]}"
        )
    return VerifiedPublicExport(
        manifest=dict(manifest),
        manifest_sha256=manifest_digest,
        files=tuple(sorted(files, key=lambda item: item.path)),
    )


def verify_rebuilt_prompt(
    root: Path | str,
    *,
    model_id: str,
    seed: int,
    benchmark_id: str,
    source_index: str,
    prompt_text: str,
    source_ordinal: int | None = None,
) -> dict[str, Any]:
    verified = load_and_verify_public_export(root)
    matches = [
        artifact
        for artifact in verified.manifest["artifacts"]
        if artifact["config_name"] == "responses"
        and artifact["model_id"] == model_id
        and int(artifact["seed"]) == seed
        and artifact["benchmark_id"] == benchmark_id
    ]
    if len(matches) != 1:
        raise PublicExportError("response artifact selection is not unique")
    parquet = Path(root) / matches[0]["parquet_path"]
    candidates = [
        row
        for row in pq.ParquetFile(parquet).read().to_pylist()
        if row["source_index"] == source_index
        and (source_ordinal is None or row["source_ordinal"] == source_ordinal)
    ]
    if len(candidates) != 1:
        raise PublicExportError(
            f"prompt sample selection matched {len(candidates)} rows; provide source ordinal"
        )
    row = candidates[0]
    verify_prompt_contract(row, prompt_text=prompt_text)
    return {
        "verified": True,
        "sample_id": row["sample_id"],
        "prompt_sha256": row["prompt_sha256"],
        "request_sha256": row["request_sha256"],
        "prompt_contract_sha256": row["prompt_contract_sha256"],
        "dataset_reconstruction_status": "external_release_gate",
    }


def inspect_source_selection(
    source_root: Path | str, plan: ExportPlan | Mapping[str, Any] | Path | str
) -> dict[str, Any]:
    """Inspect coverage and derive a digest without claiming a sealed match."""

    if isinstance(plan, ExportPlan):
        export_plan = plan
    elif isinstance(plan, Mapping):
        export_plan = ExportPlan.from_mapping(plan)
    else:
        export_plan = load_export_plan(plan)
    selected = _discover_source_slices(
        _resolve_source_root(source_root), export_plan, verify_slice_set=False
    )
    digest = _discovered_source_slice_set_sha256(selected)
    return {
        "inspection_only": True,
        "coverage": len(selected),
        "expected_coverage": len(export_plan.expected_identities),
        "source_slice_set_sha256": digest,
        "matches_plan": digest == export_plan.source_slice_set_sha256,
    }


def _canonical_suite_dimensions() -> dict[str, Any]:
    try:
        from scripts.trace_eval_suite import load_trace_eval_suite
    except ModuleNotFoundError:  # Supports direct ``python scripts/...`` invocation.
        from trace_eval_suite import load_trace_eval_suite

    suite = load_trace_eval_suite()
    return {
        "suite_id": suite.suite_id,
        "suite_sha256": suite.manifest_sha256,
        "benchmarks": list(suite.benchmark_keys),
        "categories": {
            name: list(members) for name, members in suite.categories.items()
        },
    }


def _model_mapping_document(model: ModelMapping) -> dict[str, str]:
    return {
        "source_model_id": model.source_model_id,
        "source_revision": model.source_revision,
        "model_id": model.model_id,
        "model_revision": model.model_revision,
        "display_name": model.display_name,
        "repository_id": model.repository_id,
        "repository_revision": model.repository_revision,
    }


def _judge_mapping_document(judge: JudgeMapping) -> dict[str, str]:
    return {
        "source_model_id": judge.source_model_id,
        "model_id": judge.model_id,
        "model_revision": judge.model_revision,
    }


def _export_plan_document(
    *,
    source_run_id: str,
    source_selection_sha256: str,
    source_slice_set_sha256: str,
    public_run_id: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
    models: Sequence[ModelMapping],
    judge: JudgeMapping,
) -> dict[str, Any]:
    return {
        "schema_version": EXPORT_PLAN_VERSION,
        "source": {
            "run_id": source_run_id,
            "selection_sha256": source_selection_sha256,
            "slice_set_sha256": source_slice_set_sha256,
        },
        "public": {
            "run_id": public_run_id,
            "suite_id": suite["suite_id"],
            "benchmarks": list(suite["benchmarks"]),
            "categories": dict(suite["categories"]),
            "seeds": list(seeds),
        },
        "models": [_model_mapping_document(model) for model in models],
        "judge": _judge_mapping_document(judge),
    }


def _assert_exact_source_run_coverage(source_root: Path, plan: ExportPlan) -> None:
    """Reject staged slices in the named run that the plan would silently omit."""

    expected = {
        (stage, model.source_model_id, seed, benchmark)
        for stage in SOURCE_STAGES.values()
        for model in plan.models
        for seed in plan.seeds
        for benchmark in plan.benchmarks
    }
    observed: list[tuple[str, str, int, str]] = []
    for path in sorted((source_root / "metadata" / "slices").rglob("*.manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PublicExportError(f"cannot read source manifest {path}: {error}") from error
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping) or identity.get("run_id") != plan.source_run_id:
            continue
        seed = identity.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PublicExportError(f"source manifest has an invalid seed: {path}")
        observed.append(
            (
                str(identity.get("stage") or ""),
                str(identity.get("model_slug") or ""),
                seed,
                str(identity.get("benchmark") or ""),
            )
        )
    extras = sorted(set(observed) - expected)
    if extras:
        raise PublicExportError(
            "source run contains slices outside the requested canonical selection: "
            f"count={len(extras)} first={extras[:3]}"
        )


def _write_private_plan(path: Path, document: Mapping[str, Any]) -> None:
    requested = path.expanduser().absolute()
    for component in (requested, *requested.parents):
        if component.is_symlink():
            raise PublicExportError(
                f"refusing to write a private plan through a symlink: {component}"
            )
    requested.parent.mkdir(parents=True, exist_ok=True)
    for component in (requested, *requested.parents):
        if component.is_symlink():
            raise PublicExportError(
                f"refusing to write a private plan through a symlink: {component}"
            )
    path = requested.resolve()
    # Category insertion order is part of the export-plan contract, so this
    # private control file must not alphabetize object keys on serialization.
    payload = (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise PublicExportError(
                f"refusing to replace a different export plan: {path}"
            )
        os.chmod(path, 0o600)
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        file_descriptor = -1
        temporary.replace(path)
        os.chmod(path, 0o600)
    except Exception:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temporary.unlink(missing_ok=True)
        raise


def build_private_export_plan(
    source_root: Path | str,
    *,
    source_run_id: str,
    public_run_id: str,
    seeds: Sequence[int],
    models: Sequence[Mapping[str, Any]],
    judge: Mapping[str, Any],
    output: Path | str,
) -> dict[str, Any]:
    """Seal one complete canonical campaign selection into a private export plan."""

    suite = _canonical_suite_dimensions()
    if not seeds or len(set(seeds)) != len(seeds):
        raise PublicExportError("build-plan seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise PublicExportError("build-plan seeds must be nonnegative integers")
    provisional_document = {
        "schema_version": EXPORT_PLAN_VERSION,
        "source": {
            "run_id": source_run_id,
            "selection_sha256": "0" * 64,
            "slice_set_sha256": "0" * 64,
        },
        "public": {
            "run_id": public_run_id,
            "suite_id": suite["suite_id"],
            "benchmarks": suite["benchmarks"],
            "categories": suite["categories"],
            "seeds": list(seeds),
        },
        "models": [dict(model) for model in models],
        "judge": dict(judge),
    }
    provisional = ExportPlan.from_mapping(provisional_document)
    for index, model in enumerate(provisional.models):
        _require_public_id(model.source_model_id, f"models[{index}].source_model_id")
    resolved_source = _resolve_source_root(source_root)
    _assert_exact_source_run_coverage(resolved_source, provisional)
    selected = _discover_source_slices(
        resolved_source, provisional, verify_slice_set=False
    )
    slice_set_digest = _discovered_source_slice_set_sha256(selected)
    selection_digest = canonical_sha256(
        {
            "schema_version": SOURCE_SELECTION_CONTRACT_VERSION,
            "canonical_suite": suite,
            "source_run_id": provisional.source_run_id,
            "source_slice_set_sha256": slice_set_digest,
            "public_run_id": provisional.run_id,
            "seeds": list(provisional.seeds),
            "models": [
                _model_mapping_document(model) for model in provisional.models
            ],
            "judge": _judge_mapping_document(provisional.judge),
        }
    )
    document = _export_plan_document(
        source_run_id=provisional.source_run_id,
        source_selection_sha256=selection_digest,
        source_slice_set_sha256=slice_set_digest,
        public_run_id=provisional.run_id,
        suite=suite,
        seeds=provisional.seeds,
        models=provisional.models,
        judge=provisional.judge,
    )
    final_plan = ExportPlan.from_mapping(document)
    if len(selected) != len(final_plan.expected_identities):
        raise PublicExportIntegrityError(
            "sealed source coverage changed during plan construction"
        )
    output_path = Path(output).expanduser()
    _write_private_plan(output_path, document)
    if load_export_plan(output_path) != final_plan:
        raise PublicExportIntegrityError("written export plan did not round-trip")
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser(
        "build-plan", help="seal a canonical local campaign into a private export plan"
    )
    plan_parser.add_argument("--source-root", type=Path, required=True)
    plan_parser.add_argument("--source-run-id", required=True)
    plan_parser.add_argument("--public-run-id", required=True)
    plan_parser.add_argument("--seed", type=int, action="append", required=True)
    plan_parser.add_argument(
        "--model",
        nargs=7,
        action="append",
        required=True,
        metavar=(
            "SOURCE_ID",
            "SOURCE_REV",
            "MODEL_ID",
            "MODEL_REV",
            "DISPLAY_NAME",
            "REPOSITORY_ID",
            "REPOSITORY_REV",
        ),
    )
    plan_parser.add_argument(
        "--judge",
        nargs=3,
        required=True,
        metavar=("SOURCE_ID", "REPOSITORY_ID", "REPOSITORY_REV"),
    )
    plan_parser.add_argument("--output", type=Path, required=True)
    export_parser = subparsers.add_parser("export", help="build a neutral public export")
    export_parser.add_argument("--source-root", type=Path, required=True)
    export_parser.add_argument("--plan", type=Path, required=True)
    export_parser.add_argument("--output-root", type=Path, required=True)
    inspect_parser = subparsers.add_parser(
        "inspect-source", help="derive source coverage and slice-set digest"
    )
    inspect_parser.add_argument("--source-root", type=Path, required=True)
    inspect_parser.add_argument("--plan", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify", help="verify a public export")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--expected-artifacts", type=int)
    prompt_parser = subparsers.add_parser(
        "verify-prompt", help="verify one privately rebuilt prompt"
    )
    prompt_parser.add_argument("--root", type=Path, required=True)
    prompt_parser.add_argument("--model-id", required=True)
    prompt_parser.add_argument("--seed", type=int, required=True)
    prompt_parser.add_argument("--benchmark-id", required=True)
    prompt_parser.add_argument("--source-index", required=True)
    prompt_parser.add_argument("--source-ordinal", type=int)
    prompt_parser.add_argument("--prompt-file", type=Path, required=True)
    prompt_parser.add_argument("--emit-prompt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build-plan":
        model_keys = (
            "source_model_id",
            "source_revision",
            "model_id",
            "model_revision",
            "display_name",
            "repository_id",
            "repository_revision",
        )
        document = build_private_export_plan(
            args.source_root,
            source_run_id=args.source_run_id,
            public_run_id=args.public_run_id,
            seeds=args.seed,
            models=[dict(zip(model_keys, values)) for values in args.model],
            judge={
                "source_model_id": args.judge[0],
                "model_id": args.judge[1],
                "model_revision": args.judge[2],
            },
            output=args.output,
        )
        output = {
            "written": True,
            "path": str(args.output.expanduser().resolve()),
            "schema_version": document["schema_version"],
            "suite_id": document["public"]["suite_id"],
            "coverage": (
                len(CONFIG_NAMES)
                * len(document["models"])
                * len(document["public"]["seeds"])
                * len(document["public"]["benchmarks"])
            ),
            "source_selection_sha256": document["source"]["selection_sha256"],
            "source_slice_set_sha256": document["source"]["slice_set_sha256"],
        }
    elif args.command == "export":
        result = export_public_artifacts(args.source_root, args.plan, args.output_root)
        output = {
            "verified": True,
            "root": str(args.output_root.resolve()),
            "manifest_sha256": result.manifest_sha256,
            "files": len(result.files),
            "artifacts": len(result.manifest["artifacts"]),
        }
    elif args.command == "inspect-source":
        output = inspect_source_selection(args.source_root, args.plan)
    elif args.command == "verify":
        result = load_and_verify_public_export(
            args.root, expected_artifacts=args.expected_artifacts
        )
        output = {
            "verified": True,
            "root": str(args.root.resolve()),
            "manifest_sha256": result.manifest_sha256,
            "files": len(result.files),
            "artifacts": len(result.manifest["artifacts"]),
        }
    else:
        prompt_bytes = args.prompt_file.read_bytes()
        try:
            prompt_text = prompt_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicExportError("rebuilt prompt file must be UTF-8") from error
        output = verify_rebuilt_prompt(
            args.root,
            model_id=args.model_id,
            seed=args.seed,
            benchmark_id=args.benchmark_id,
            source_index=args.source_index,
            source_ordinal=args.source_ordinal,
            prompt_text=prompt_text,
        )
        if args.emit_prompt is not None:
            args.emit_prompt.write_text(prompt_text, encoding="utf-8")
    print(canonical_json(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
