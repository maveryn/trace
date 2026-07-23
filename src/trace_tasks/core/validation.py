"""Pre-finalize validation for Trace dataset builds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import re
import stat
from typing import Any, Dict, List, Mapping

import zstandard as zstd

from . import error_codes
from .canonical import canonical_json_bytes
from .hash_utils import blake3_file, blake3_hex
from .identity import compute_instance_id
from .prompts import load_prompt_bundle, load_scene_prompt_bundle
from .prompts.schema import REQUIRED_PROMPT_VARIANTS
from .reward_contracts import validate_reward_contract_payload
from .source_layout_policy import uses_current_source_layout
from .trace_store import read_trace_shard


@dataclass(frozen=True)
class _ValidationError:
    """Structured validation error for machine/human reporting."""

    error_code: str
    message: str
    category: str
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category,
        }
        out.update(self.context)
        return out


def _category_for_code(code: str) -> str:
    """Derive top-level error category prefix from an error code string."""
    return code.split("_", 1)[0]


def _err(code: str, message: str, **context: Any) -> _ValidationError:
    """Build a structured validation error with derived category metadata."""
    return _ValidationError(
        error_code=code,
        message=message,
        category=_category_for_code(code),
        context=context,
    )


def is_safe_trace_shard_id(value: Any) -> bool:
    """Return whether a trace shard id is one portable filename component."""

    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if value in {".", ".."} or "/" in value or "\\" in value:
        return False
    return Path(value).name == value


def _resolve_trace_shard_path(root: Path, shard_id: Any) -> Path | None:
    """Resolve a shard without allowing reads outside ``<root>/traces``."""

    if not is_safe_trace_shard_id(shard_id):
        return None
    try:
        resolved_root = root.resolve()
        traces_path = resolved_root / "traces"
        # Reject a symlinked traces directory and shard symlinks that leave it.
        if traces_path.resolve() != traces_path:
            return None
        shard_path = traces_path / shard_id
        resolved_shard = shard_path.resolve()
    except (OSError, RuntimeError):
        return None
    if resolved_shard.parent != traces_path:
        return None
    return resolved_shard


def _read_trace_shard_for_validation(
    shard_path: Path,
) -> tuple[List[Dict[str, Any]] | None, str | None]:
    """Read one shard and convert malformed input into a validation result."""

    try:
        records = read_trace_shard(shard_path)
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("trace shard records must be JSON objects")
        for record in records:
            canonical_json_bytes(record)
    except (OSError, EOFError, UnicodeError, ValueError, zstd.ZstdError) as exc:
        return None, type(exc).__name__
    return records, None


def _resolve_dataset_regular_file(
    root: Path,
    relative_path: str,
) -> tuple[Path | None, str]:
    """Resolve one root-relative, non-symlink regular file without escaping.

    The component-by-component ``lstat`` checks deliberately reject symlinks,
    including links whose targets happen to remain inside the dataset.  This
    keeps validation from becoming an arbitrary-file hash oracle when it is
    run over an untrusted JSONL file.
    """

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None, "unsafe"
    try:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            return None, "missing"
        candidate = resolved_root.joinpath(*relative.parts)
        candidate.relative_to(resolved_root)
        current = resolved_root
        for part in relative.parts:
            if part in {"", "."}:
                continue
            current = current / part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                return None, "unsafe"
        if current != candidate or not stat.S_ISREG(current.lstat().st_mode):
            return None, "missing"
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError, ValueError):
        return None, "missing"
    return candidate, "ok"


_REQUIRED_INSTANCE_FIELDS = [
    "instance_version",
    "instance_id",
    "instance_seed",
    "domain",
    "task",
    "scene_id",
    "query_id",
    "prompt",
    "prompt_variants",
    "images",
    "answer_gt",
    "annotation_gt",
    "reward_contract",
    "trace_ref",
    "versions",
]

_PROMPT_PLACEHOLDER_PATTERN = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _to_int(value: Any) -> int | None:
    """Best-effort integer coercion used for metadata validation checks."""
    try:
        return int(value)
    except Exception:
        return None


def _validate_prompt_contract(
    instance: Mapping[str, Any], trace_record: Mapping[str, Any]
) -> List[_ValidationError]:
    """Validate prompt metadata/bundle conformance for one train/trace pair."""
    errors: List[_ValidationError] = []
    iid = instance.get("instance_id", "<missing>")
    prompt_text = instance.get("prompt")

    if isinstance(prompt_text, str):
        unresolved_tokens = sorted(
            {
                match.group(0)
                for match in _PROMPT_PLACEHOLDER_PATTERN.finditer(prompt_text)
            }
        )
        if unresolved_tokens:
            errors.append(
                _err(
                    error_codes.PROMPT_UNRESOLVED_PLACEHOLDER,
                    "prompt contains unresolved template placeholder(s)",
                    instance_id=iid,
                    field_path="prompt",
                    unresolved_tokens=unresolved_tokens,
                )
            )

    prompt_variants = instance.get("prompt_variants")
    if prompt_variants is not None:
        if not isinstance(prompt_variants, Mapping):
            errors.append(
                _err(
                    error_codes.SCHEMA_TYPE_MISMATCH,
                    "prompt_variants must be a mapping when present",
                    instance_id=iid,
                    field_path="prompt_variants",
                )
            )
        else:
            for variant_key, variant_prompt in prompt_variants.items():
                if not isinstance(variant_prompt, str):
                    errors.append(
                        _err(
                            error_codes.SCHEMA_TYPE_MISMATCH,
                            "prompt_variants values must be strings",
                            instance_id=iid,
                            field_path=f"prompt_variants.{variant_key}",
                        )
                    )
                    continue
                unresolved_tokens = sorted(
                    {
                        match.group(0)
                        for match in _PROMPT_PLACEHOLDER_PATTERN.finditer(
                            variant_prompt
                        )
                    }
                )
                if unresolved_tokens:
                    errors.append(
                        _err(
                            error_codes.PROMPT_UNRESOLVED_PLACEHOLDER,
                            "prompt_variants contains unresolved template placeholder(s)",
                            instance_id=iid,
                            field_path=f"prompt_variants.{variant_key}",
                            unresolved_tokens=unresolved_tokens,
                        )
                    )

    query_spec = trace_record.get("query_spec")
    if not isinstance(query_spec, Mapping):
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "trace query_spec is missing or invalid for prompt validation",
                instance_id=iid,
                field_path="query_spec",
            )
        )
        return errors

    prompt_variant = query_spec.get("prompt_variant")
    if not isinstance(prompt_variant, Mapping):
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "trace query_spec.prompt_variant is missing or invalid",
                instance_id=iid,
                field_path="query_spec.prompt_variant",
            )
        )
        return errors

    required_meta_fields = (
        "prompt_bundle_id",
        "scene_key",
        "task_key",
        "scene_template_index",
        "task_template_index",
        "variant_count_by_key",
    )
    for field in required_meta_fields:
        if field not in prompt_variant:
            errors.append(
                _err(
                    error_codes.PROMPT_METADATA_MISSING,
                    f"missing required prompt metadata field '{field}'",
                    instance_id=iid,
                    field_path=f"query_spec.prompt_variant.{field}",
                )
            )

    bundle_id = str(prompt_variant.get("prompt_bundle_id", "")).strip()
    scene_key = str(prompt_variant.get("scene_key", "")).strip()
    task_key = str(prompt_variant.get("task_key", "")).strip()
    query_key_raw = prompt_variant.get("query_key")
    query_key = str(query_key_raw).strip() if query_key_raw not in (None, "") else ""
    variant_count_by_key = prompt_variant.get("variant_count_by_key")

    if not bundle_id:
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "prompt bundle id is empty",
                instance_id=iid,
                field_path="query_spec.prompt_variant.prompt_bundle_id",
            )
        )
    if not scene_key:
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "scene_key is empty",
                instance_id=iid,
                field_path="query_spec.prompt_variant.scene_key",
            )
        )
    if not task_key:
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "task_key is empty",
                instance_id=iid,
                field_path="query_spec.prompt_variant.task_key",
            )
        )
    if not isinstance(variant_count_by_key, Mapping):
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "variant_count_by_key must be a mapping",
                instance_id=iid,
                field_path="query_spec.prompt_variant.variant_count_by_key",
            )
        )

    if errors:
        return errors

    trace_taxonomy = (
        trace_record.get("taxonomy")
        if isinstance(trace_record.get("taxonomy"), Mapping)
        else {}
    )
    taxonomy_source = (
        trace_taxonomy.get("source")
        if isinstance(trace_taxonomy.get("source"), Mapping)
        else {}
    )
    domain = str(
        prompt_variant.get("prompt_domain")
        or taxonomy_source.get("prompt_domain")
        or taxonomy_source.get("implementation_domain")
        or instance.get("domain", "")
    )
    prompt_scene_id = str(
        prompt_variant.get("prompt_scene_id")
        or taxonomy_source.get("prompt_scene_id")
        or taxonomy_source.get("config_scene_id")
        or taxonomy_source.get("implementation_scene_id")
        or instance.get("scene_id", "")
        or ""
    )
    scene_id = str(
        prompt_variant.get("prompt_scene_id")
        or taxonomy_source.get("prompt_scene_id")
        or taxonomy_source.get("implementation_scene_id")
        or instance.get("scene_id", "")
    )
    try:
        if prompt_scene_id.strip() and uses_current_source_layout(
            str(instance.get("task", "")), domain=domain
        ):
            bundle = load_scene_prompt_bundle(
                domain=domain, scene_id=prompt_scene_id, bundle_id=bundle_id
            )
        else:
            bundle = load_prompt_bundle(
                domain=domain, scene_id=scene_id, bundle_id=bundle_id
            )
    except FileNotFoundError as exc:
        errors.append(
            _err(
                error_codes.PROMPT_BUNDLE_NOT_FOUND,
                str(exc),
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                domain=domain,
                scene_id=scene_id,
            )
        )
        return errors
    except Exception as exc:
        errors.append(
            _err(
                error_codes.PROMPT_BUNDLE_INVALID,
                f"invalid prompt bundle: {exc}",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                domain=domain,
                scene_id=scene_id,
            )
        )
        return errors

    if scene_key not in bundle.scene_templates:
        errors.append(
            _err(
                error_codes.PROMPT_KEY_MISSING,
                "prompt scene key not found in bundle",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                scene_key=scene_key,
            )
        )
    if task_key not in bundle.task_templates:
        errors.append(
            _err(
                error_codes.PROMPT_KEY_MISSING,
                "prompt task key not found in bundle",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                task_key=task_key,
            )
        )
    if query_key and query_key not in bundle.query_templates:
        errors.append(
            _err(
                error_codes.PROMPT_KEY_MISSING,
                "prompt query key not found in bundle",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                query_key=query_key,
            )
        )
    if errors:
        return errors

    scene_templates = bundle.scene_templates[scene_key]
    task_templates = bundle.task_templates[task_key]
    query_templates = bundle.query_templates[query_key] if query_key else ()
    mode_templates = dict(bundle.answer_or_annotation_templates)
    if len(scene_templates) != REQUIRED_PROMPT_VARIANTS:
        errors.append(
            _err(
                error_codes.PROMPT_BUNDLE_INVALID,
                "scene template query-id count does not match required count",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                scene_key=scene_key,
                required_count=REQUIRED_PROMPT_VARIANTS,
                actual_count=len(scene_templates),
            )
        )
    if len(task_templates) != REQUIRED_PROMPT_VARIANTS:
        errors.append(
            _err(
                error_codes.PROMPT_BUNDLE_INVALID,
                "task template query-id count does not match required count",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                task_key=task_key,
                required_count=REQUIRED_PROMPT_VARIANTS,
                actual_count=len(task_templates),
            )
        )
    if query_key and len(query_templates) != REQUIRED_PROMPT_VARIANTS:
        errors.append(
            _err(
                error_codes.PROMPT_BUNDLE_INVALID,
                "query template query-id count does not match required count",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                query_key=query_key,
                required_count=REQUIRED_PROMPT_VARIANTS,
                actual_count=len(query_templates),
            )
        )

    scene_count_key = f"scene:{scene_key}"
    task_count_key = f"task:{task_key}"
    query_count_key = f"query:{query_key}" if query_key else None
    observed_scene_count = _to_int(variant_count_by_key.get(scene_count_key))
    observed_task_count = _to_int(variant_count_by_key.get(task_count_key))
    observed_query_count = (
        _to_int(variant_count_by_key.get(query_count_key))
        if query_count_key is not None
        else None
    )
    expected_scene_count = len(scene_templates)
    expected_task_count = len(task_templates)
    expected_query_count = len(query_templates) if query_key else None
    mode_key = (
        str(prompt_variant.get("answer_or_annotation_key", "")).strip()
        if mode_templates
        else ""
    )
    mode_query_id_index = (
        _to_int(prompt_variant.get("answer_or_annotation_query_id_index"))
        if mode_templates
        else None
    )
    expected_mode_count = (
        len(mode_templates[mode_key]) if mode_key in mode_templates else None
    )

    if observed_scene_count is None:
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "missing scene query-id count in prompt metadata",
                instance_id=iid,
                field_path=f"query_spec.prompt_variant.variant_count_by_key.{scene_count_key}",
            )
        )
    elif observed_scene_count != expected_scene_count:
        errors.append(
            _err(
                error_codes.PROMPT_VARIANT_COUNT_MISMATCH,
                "scene query-id count mismatch between metadata and bundle",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                scene_key=scene_key,
                expected_count=expected_scene_count,
                actual_count=observed_scene_count,
            )
        )

    if observed_task_count is None:
        errors.append(
            _err(
                error_codes.PROMPT_METADATA_MISSING,
                "missing query id count in prompt metadata",
                instance_id=iid,
                field_path=f"query_spec.prompt_variant.variant_count_by_key.{task_count_key}",
            )
        )
    elif observed_task_count != expected_task_count:
        errors.append(
            _err(
                error_codes.PROMPT_VARIANT_COUNT_MISMATCH,
                "query id count mismatch between metadata and bundle",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                task_key=task_key,
                expected_count=expected_task_count,
                actual_count=observed_task_count,
            )
        )
    if query_count_key is not None:
        if observed_query_count is None:
            errors.append(
                _err(
                    error_codes.PROMPT_METADATA_MISSING,
                    "missing query id count in prompt metadata",
                    instance_id=iid,
                    field_path=f"query_spec.prompt_variant.variant_count_by_key.{query_count_key}",
                )
            )
        elif observed_query_count != expected_query_count:
            errors.append(
                _err(
                    error_codes.PROMPT_VARIANT_COUNT_MISMATCH,
                    "query id count mismatch between metadata and bundle",
                    instance_id=iid,
                    prompt_bundle_id=bundle_id,
                    query_key=query_key,
                    expected_count=expected_query_count,
                    actual_count=observed_query_count,
                )
            )

    if mode_templates:
        if not mode_key:
            errors.append(
                _err(
                    error_codes.PROMPT_METADATA_MISSING,
                    "missing answer_or_annotation key in prompt metadata",
                    instance_id=iid,
                    field_path="query_spec.prompt_variant.answer_or_annotation_key",
                )
            )
        elif mode_key not in mode_templates:
            errors.append(
                _err(
                    error_codes.PROMPT_KEY_MISSING,
                    "prompt answer_or_annotation key not found in bundle",
                    instance_id=iid,
                    prompt_bundle_id=bundle_id,
                    answer_or_annotation_key=mode_key,
                )
            )
        else:
            mode_count_key = f"answer_or_annotation:{mode_key}"
            observed_mode_count = _to_int(variant_count_by_key.get(mode_count_key))
            if observed_mode_count is None:
                errors.append(
                    _err(
                        error_codes.PROMPT_METADATA_MISSING,
                        "missing answer_or_annotation query-id count in prompt metadata",
                        instance_id=iid,
                        field_path=f"query_spec.prompt_variant.variant_count_by_key.{mode_count_key}",
                    )
                )
            elif observed_mode_count != expected_mode_count:
                errors.append(
                    _err(
                        error_codes.PROMPT_VARIANT_COUNT_MISMATCH,
                        "answer_or_annotation query-id count mismatch between metadata and bundle",
                        instance_id=iid,
                        prompt_bundle_id=bundle_id,
                        answer_or_annotation_key=mode_key,
                        expected_count=expected_mode_count,
                        actual_count=observed_mode_count,
                    )
                )

    scene_template_index = _to_int(prompt_variant.get("scene_template_index"))
    task_template_index = _to_int(prompt_variant.get("task_template_index"))
    query_template_index = (
        _to_int(prompt_variant.get("query_template_index")) if query_key else None
    )
    if (
        scene_template_index is None
        or scene_template_index < 0
        or scene_template_index >= expected_scene_count
    ):
        errors.append(
            _err(
                error_codes.PROMPT_VARIANT_INDEX_OUT_OF_RANGE,
                "scene variant index out of range",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                scene_key=scene_key,
                query_id_index=prompt_variant.get("scene_template_index"),
                variant_count=expected_scene_count,
            )
        )
    if (
        task_template_index is None
        or task_template_index < 0
        or task_template_index >= expected_task_count
    ):
        errors.append(
            _err(
                error_codes.PROMPT_VARIANT_INDEX_OUT_OF_RANGE,
                "query id index out of range",
                instance_id=iid,
                prompt_bundle_id=bundle_id,
                task_key=task_key,
                query_id_index=prompt_variant.get("task_template_index"),
                variant_count=expected_task_count,
            )
        )
    if query_key and expected_query_count is not None:
        if (
            query_template_index is None
            or query_template_index < 0
            or query_template_index >= expected_query_count
        ):
            errors.append(
                _err(
                    error_codes.PROMPT_VARIANT_INDEX_OUT_OF_RANGE,
                    "query template index out of range",
                    instance_id=iid,
                    prompt_bundle_id=bundle_id,
                    query_key=query_key,
                    query_id_index=prompt_variant.get("query_template_index"),
                    variant_count=expected_query_count,
                )
            )
    if (
        mode_templates
        and mode_key in mode_templates
        and expected_mode_count is not None
    ):
        if (
            mode_query_id_index is None
            or mode_query_id_index < 0
            or mode_query_id_index >= expected_mode_count
        ):
            errors.append(
                _err(
                    error_codes.PROMPT_VARIANT_INDEX_OUT_OF_RANGE,
                    "answer_or_annotation variant index out of range",
                    instance_id=iid,
                    prompt_bundle_id=bundle_id,
                    answer_or_annotation_key=mode_key,
                    query_id_index=prompt_variant.get(
                        "answer_or_annotation_query_id_index"
                    ),
                    variant_count=expected_mode_count,
                )
            )

    required_slots = list(bundle.required_slots_by_key.get(f"scene:{scene_key}", ()))
    required_slots.extend(bundle.required_slots_by_key.get(f"task:{task_key}", ()))
    if query_key:
        required_slots.extend(
            bundle.required_slots_by_key.get(f"query:{query_key}", ())
        )
    if mode_templates and mode_key:
        required_slots.extend(
            bundle.required_slots_by_key.get(f"answer_or_annotation:{mode_key}", ())
        )
    if required_slots:
        slot_values = prompt_variant.get("slot_values")
        if not isinstance(slot_values, Mapping):
            errors.append(
                _err(
                    error_codes.PROMPT_REQUIRED_SLOT_MISSING,
                    "prompt metadata is missing slot_values for required slots",
                    instance_id=iid,
                    field_path="query_spec.prompt_variant.slot_values",
                    required_slots=sorted(set(str(slot) for slot in required_slots)),
                )
            )
        else:
            missing_slots = sorted(
                {
                    str(slot)
                    for slot in required_slots
                    if str(slot) not in slot_values
                    or slot_values.get(str(slot)) in (None, "")
                }
            )
            if missing_slots:
                errors.append(
                    _err(
                        error_codes.PROMPT_REQUIRED_SLOT_MISSING,
                        "required prompt slot values are missing in metadata",
                        instance_id=iid,
                        field_path="query_spec.prompt_variant.slot_values",
                        missing_slots=missing_slots,
                    )
                )

    if mode_templates:
        if not isinstance(prompt_variants, Mapping):
            errors.append(
                _err(
                    error_codes.PROMPT_METADATA_MISSING,
                    "prompt_variants is required when bundle defines answer_or_annotation templates",
                    instance_id=iid,
                    field_path="prompt_variants",
                )
            )
        else:
            missing_modes = sorted(
                [
                    mode_name
                    for mode_name in mode_templates.keys()
                    if str(prompt_variants.get(mode_name, "")).strip() == ""
                ]
            )
            if missing_modes:
                errors.append(
                    _err(
                        error_codes.PROMPT_METADATA_MISSING,
                        "prompt_variants is missing required answer_or_annotation prompts",
                        instance_id=iid,
                        field_path="prompt_variants",
                        missing_modes=missing_modes,
                    )
                )
            elif (
                mode_key
                and isinstance(prompt_text, str)
                and prompt_text.strip()
                and str(prompt_variants.get(mode_key, "")).strip()
                != prompt_text.strip()
            ):
                errors.append(
                    _err(
                        error_codes.PROMPT_METADATA_MISSING,
                        "prompt text does not match active prompt_variants entry",
                        instance_id=iid,
                        field_path=f"prompt_variants.{mode_key}",
                    )
                )

    return errors


def _validate_schema(instance: Mapping[str, Any]) -> List[_ValidationError]:
    """Validate required TrainInstance fields and envelope-schema invariants."""
    errors: List[_ValidationError] = []
    iid = instance.get("instance_id", "<missing>")
    for field in _REQUIRED_INSTANCE_FIELDS:
        if field not in instance:
            errors.append(
                _err(
                    error_codes.SCHEMA_MISSING_FIELD,
                    f"missing required field '{field}'",
                    instance_id=iid,
                    field_path=field,
                )
            )

    if "scene_id" in instance:
        scene_id = instance["scene_id"]
        if not isinstance(scene_id, str):
            errors.append(
                _err(
                    error_codes.SCHEMA_TYPE_MISMATCH,
                    "scene_id must be a string",
                    instance_id=iid,
                    field_path="scene_id",
                )
            )
        elif not scene_id.strip():
            errors.append(
                _err(
                    error_codes.SCHEMA_INVALID_VALUE,
                    "scene_id must be non-empty",
                    instance_id=iid,
                    field_path="scene_id",
                )
            )

    if "answer_gt" in instance:
        answer = instance["answer_gt"]
        if (
            not isinstance(answer, dict)
            or "type" not in answer
            or "value" not in answer
        ):
            errors.append(
                _err(
                    error_codes.SCHEMA_TYPE_MISMATCH,
                    "answer_gt must be an object with keys {type, value}",
                    instance_id=iid,
                    field_path="answer_gt",
                )
            )

    if "annotation_gt" in instance:
        annotation = instance["annotation_gt"]
        if (
            not isinstance(annotation, dict)
            or "type" not in annotation
            or "value" not in annotation
        ):
            errors.append(
                _err(
                    error_codes.SCHEMA_TYPE_MISMATCH,
                    "annotation_gt must be an object with keys {type, value}",
                    instance_id=iid,
                    field_path="annotation_gt",
                )
            )

    if "reward_contract" in instance:
        reward_contract = instance["reward_contract"]
        reward_contract_error = validate_reward_contract_payload(
            reward_contract,
            answer_type=(
                instance.get("answer_gt", {}).get("type")
                if isinstance(instance.get("answer_gt"), dict)
                else None
            ),
            annotation_type=(
                instance.get("annotation_gt", {}).get("type")
                if isinstance(instance.get("annotation_gt"), dict)
                else None
            ),
        )
        if reward_contract_error is not None:
            errors.append(
                _err(
                    error_codes.SCHEMA_INVALID_VALUE,
                    reward_contract_error,
                    instance_id=iid,
                    field_path="reward_contract",
                )
            )

    if "images" in instance:
        images = instance["images"]
        if not isinstance(images, list):
            errors.append(
                _err(
                    error_codes.SCHEMA_TYPE_MISMATCH,
                    "images must be a list",
                    instance_id=iid,
                    field_path="images",
                )
            )
        else:
            for index, image in enumerate(images):
                if not isinstance(image, Mapping):
                    errors.append(
                        _err(
                            error_codes.SCHEMA_TYPE_MISMATCH,
                            "images entries must be mappings",
                            instance_id=iid,
                            field_path=f"images[{index}]",
                        )
                    )

    # Catch canonicalization failures early (non-string key/non-finite/unsupported types).
    try:
        canonical_json_bytes(dict(instance))
    except Exception as exc:
        code = getattr(exc, "code", error_codes.SCHEMA_CANONICALIZATION_FAILED)
        errors.append(
            _err(
                code,
                str(exc),
                instance_id=iid,
                field_path="<instance>",
            )
        )

    return errors


def _validate_instance_identity(
    instance: Mapping[str, Any],
) -> List[_ValidationError]:
    """Recompute identity without letting malformed envelopes abort validation."""

    iid = instance.get("instance_id", "<missing>")
    try:
        recomputed_id = compute_instance_id(dict(instance))
    except Exception as exc:
        return [
            _err(
                getattr(exc, "code", error_codes.SCHEMA_CANONICALIZATION_FAILED),
                "instance identity payload could not be constructed and canonicalized",
                instance_id=iid,
                field_path="<identity_payload>",
                failure_type=type(exc).__name__,
            )
        ]
    if recomputed_id == iid:
        return []
    return [
        _err(
            error_codes.IDENTITY_INSTANCE_ID_MISMATCH,
            "instance_id does not match canonical identity payload",
            instance_id=iid,
            recomputed_instance_id=recomputed_id,
        )
    ]


def _validate_image_files(
    instance: Mapping[str, Any],
    *,
    root: Path,
) -> List[_ValidationError]:
    """Validate image records using only root-contained non-symlink files."""

    errors: List[_ValidationError] = []
    iid = instance.get("instance_id", "<missing>")
    images = instance.get("images")
    if not isinstance(images, list):
        return errors

    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            # ``_validate_schema`` owns the structured type error.
            continue
        rel_path = image.get("path")
        image_hash = image.get("image_hash")
        field_prefix = f"images[{index}]"
        if not isinstance(rel_path, str):
            errors.append(
                _err(
                    error_codes.IMAGE_FILE_NOT_FOUND,
                    "image path missing or non-string",
                    instance_id=iid,
                    field_path=f"{field_prefix}.path",
                )
            )
            continue

        full_path, resolution = _resolve_dataset_regular_file(root, rel_path)
        if resolution == "unsafe":
            errors.append(
                _err(
                    error_codes.IMAGE_PATH_NOT_RELATIVE,
                    "image path must name a root-contained, non-symlink regular file",
                    instance_id=iid,
                    field_path=f"{field_prefix}.path",
                    image_path=rel_path,
                )
            )
            continue
        if full_path is None:
            errors.append(
                _err(
                    error_codes.IMAGE_FILE_NOT_FOUND,
                    "image path does not name a regular file",
                    instance_id=iid,
                    field_path=f"{field_prefix}.path",
                    image_path=rel_path,
                )
            )
            continue
        if not isinstance(image_hash, str) or not image_hash:
            errors.append(
                _err(
                    error_codes.IMAGE_HASH_MISSING,
                    "image_hash is missing or non-string",
                    instance_id=iid,
                    field_path=f"{field_prefix}.image_hash",
                )
            )
            continue
        try:
            actual_image_hash = blake3_file(full_path)
        except OSError as exc:
            errors.append(
                _err(
                    error_codes.IMAGE_FILE_NOT_FOUND,
                    "image file could not be read as a regular file",
                    instance_id=iid,
                    field_path=f"{field_prefix}.path",
                    image_path=rel_path,
                    failure_type=type(exc).__name__,
                )
            )
            continue
        if actual_image_hash != image_hash:
            errors.append(
                _err(
                    error_codes.IMAGE_HASH_MISMATCH,
                    "image hash mismatch",
                    instance_id=iid,
                    image_path=rel_path,
                    expected_image_hash=image_hash,
                    actual_image_hash=actual_image_hash,
                )
            )
    return errors


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _iter_text_legibility_blocks(
    value: Any, *, field_path: str
) -> List[tuple[str, Any]]:
    """Find text_legibility metadata blocks inside render metadata."""

    found: List[tuple[str, Any]] = []
    if not isinstance(value, Mapping):
        return found
    if "text_legibility" in value:
        found.append((f"{field_path}.text_legibility", value.get("text_legibility")))
    for key, child in value.items():
        if key == "text_legibility":
            continue
        if isinstance(child, Mapping):
            found.extend(
                _iter_text_legibility_blocks(child, field_path=f"{field_path}.{key}")
            )
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    found.extend(
                        _iter_text_legibility_blocks(
                            item, field_path=f"{field_path}.{key}[{index}]"
                        )
                    )
    return found


def _validate_one_text_legibility_block(
    text_legibility: Any,
    *,
    instance_id: str,
    field_path_root: str,
) -> List[_ValidationError]:
    """Validate one rendered text-legibility metadata block."""

    errors: List[_ValidationError] = []
    if not isinstance(text_legibility, Mapping):
        return [
            _err(
                error_codes.TEXT_LEGIBILITY_INVALID,
                "text_legibility metadata must be a mapping when present",
                instance_id=instance_id,
                field_path=field_path_root,
            )
        ]
    if not bool(text_legibility.get("enabled", True)):
        return []

    failure_count = _to_int(text_legibility.get("failure_count"))
    if failure_count is None or int(failure_count) < 0:
        errors.append(
            _err(
                error_codes.TEXT_LEGIBILITY_INVALID,
                "text_legibility.failure_count must be a non-negative integer",
                instance_id=instance_id,
                field_path=f"{field_path_root}.failure_count",
            )
        )
    elif int(failure_count) > 0:
        errors.append(
            _err(
                error_codes.TEXT_LEGIBILITY_CONTRAST_FAILED,
                "required/read-off text legibility metadata reports failures",
                instance_id=instance_id,
                field_path=f"{field_path_root}.failure_count",
                failure_count=int(failure_count),
            )
        )

    records = text_legibility.get("records")
    if records is None:
        records = []
    if not isinstance(records, list):
        errors.append(
            _err(
                error_codes.TEXT_LEGIBILITY_INVALID,
                "text_legibility.records must be an array",
                instance_id=instance_id,
                field_path=f"{field_path_root}.records",
            )
        )
        return errors

    for index, record in enumerate(records):
        field_prefix = f"{field_path_root}.records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(
                _err(
                    error_codes.TEXT_LEGIBILITY_INVALID,
                    "text legibility record must be a mapping",
                    instance_id=instance_id,
                    field_path=field_prefix,
                )
            )
            continue
        required = bool(record.get("required", True))
        if not required:
            continue
        passes = record.get("passes")
        if passes is not True:
            errors.append(
                _err(
                    error_codes.TEXT_LEGIBILITY_CONTRAST_FAILED,
                    "required/read-off text record does not pass legibility thresholds",
                    instance_id=instance_id,
                    field_path=f"{field_prefix}.passes",
                    role=record.get("role"),
                )
            )
        contrast = _finite_float(record.get("min_contrast_ratio"))
        contrast_required = _finite_float(record.get("min_contrast_required"))
        if (
            contrast is None
            or contrast_required is None
            or contrast < contrast_required
        ):
            errors.append(
                _err(
                    error_codes.TEXT_LEGIBILITY_CONTRAST_FAILED,
                    "required/read-off text contrast is below threshold",
                    instance_id=instance_id,
                    field_path=f"{field_prefix}.min_contrast_ratio",
                    role=record.get("role"),
                    min_contrast_ratio=record.get("min_contrast_ratio"),
                    min_contrast_required=record.get("min_contrast_required"),
                )
            )
        lab_distance = _finite_float(record.get("min_lab_distance"))
        lab_distance_required = _finite_float(record.get("min_lab_distance_required"))
        if (
            lab_distance is None
            or lab_distance_required is None
            or lab_distance < lab_distance_required
        ):
            errors.append(
                _err(
                    error_codes.TEXT_LEGIBILITY_CONTRAST_FAILED,
                    "required/read-off text color distance is below threshold",
                    instance_id=instance_id,
                    field_path=f"{field_prefix}.min_lab_distance",
                    role=record.get("role"),
                    min_lab_distance=record.get("min_lab_distance"),
                    min_lab_distance_required=record.get("min_lab_distance_required"),
                )
            )
        bbox = record.get("bbox_px")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(
                    _err(
                        error_codes.TEXT_LEGIBILITY_INVALID,
                        "drawn text bbox_px must be an array of four finite numbers",
                        instance_id=instance_id,
                        field_path=f"{field_prefix}.bbox_px",
                        role=record.get("role"),
                    )
                )
                continue
            coords = [_finite_float(value) for value in bbox]
            if any(value is None for value in coords):
                errors.append(
                    _err(
                        error_codes.TEXT_LEGIBILITY_INVALID,
                        "drawn text bbox_px must contain only finite numbers",
                        instance_id=instance_id,
                        field_path=f"{field_prefix}.bbox_px",
                        role=record.get("role"),
                    )
                )
                continue
            x0, y0, x1, y1 = [float(value) for value in coords if value is not None]
            if x1 <= x0 or y1 <= y0:
                errors.append(
                    _err(
                        error_codes.TEXT_LEGIBILITY_INVALID,
                        "drawn text bbox_px must have positive width and height",
                        instance_id=instance_id,
                        field_path=f"{field_prefix}.bbox_px",
                        role=record.get("role"),
                        bbox_px=bbox,
                    )
                )

    return errors


def _validate_text_legibility_contract(
    trace_record: Mapping[str, Any],
    *,
    instance_id: str,
) -> List[_ValidationError]:
    """Validate rendered text-legibility metadata when a renderer records it.

    The first rollout is intentionally compatibility-safe: absence of
    text_legibility metadata is handled by static source checks, while
    malformed or failing metadata on source-layout renderers fails dataset
    validation here.
    """

    render_spec = trace_record.get("render_spec")
    if not isinstance(render_spec, Mapping):
        return []
    errors: List[_ValidationError] = []
    for field_path_root, text_legibility in _iter_text_legibility_blocks(
        render_spec, field_path="trace.render_spec"
    ):
        errors.extend(
            _validate_one_text_legibility_block(
                text_legibility,
                instance_id=str(instance_id),
                field_path_root=str(field_path_root),
            )
        )
    return errors


def _iter_marker_legibility_blocks(
    value: Any, *, field_path: str
) -> List[tuple[str, Any]]:
    """Find marker_legibility metadata blocks inside render metadata."""

    found: List[tuple[str, Any]] = []
    if not isinstance(value, Mapping):
        return found
    if "marker_legibility" in value:
        found.append(
            (f"{field_path}.marker_legibility", value.get("marker_legibility"))
        )
    for key, child in value.items():
        if key == "marker_legibility":
            continue
        if isinstance(child, Mapping):
            found.extend(
                _iter_marker_legibility_blocks(child, field_path=f"{field_path}.{key}")
            )
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, Mapping):
                    found.extend(
                        _iter_marker_legibility_blocks(
                            item, field_path=f"{field_path}.{key}[{index}]"
                        )
                    )
    return found


def _validate_one_marker_legibility_block(
    marker_legibility: Any,
    *,
    instance_id: str,
    field_path_root: str,
) -> List[_ValidationError]:
    """Validate one rendered semantic-marker metadata block."""

    errors: List[_ValidationError] = []
    if not isinstance(marker_legibility, Mapping):
        return [
            _err(
                error_codes.MARKER_LEGIBILITY_INVALID,
                "marker_legibility metadata must be a mapping when present",
                instance_id=instance_id,
                field_path=field_path_root,
            )
        ]
    if not bool(marker_legibility.get("enabled", True)):
        return []

    failure_count = _to_int(marker_legibility.get("failure_count"))
    if failure_count is None or int(failure_count) < 0:
        errors.append(
            _err(
                error_codes.MARKER_LEGIBILITY_INVALID,
                "marker_legibility.failure_count must be a non-negative integer",
                instance_id=instance_id,
                field_path=f"{field_path_root}.failure_count",
            )
        )
    elif int(failure_count) > 0:
        errors.append(
            _err(
                error_codes.MARKER_LEGIBILITY_CONTRAST_FAILED,
                "required semantic marker legibility metadata reports failures",
                instance_id=instance_id,
                field_path=f"{field_path_root}.failure_count",
                failure_count=int(failure_count),
            )
        )

    records = marker_legibility.get("records")
    if records is None:
        records = []
    if not isinstance(records, list):
        errors.append(
            _err(
                error_codes.MARKER_LEGIBILITY_INVALID,
                "marker_legibility.records must be an array",
                instance_id=instance_id,
                field_path=f"{field_path_root}.records",
            )
        )
        return errors

    for index, record in enumerate(records):
        field_prefix = f"{field_path_root}.records[{index}]"
        if not isinstance(record, Mapping):
            errors.append(
                _err(
                    error_codes.MARKER_LEGIBILITY_INVALID,
                    "semantic marker legibility record must be a mapping",
                    instance_id=instance_id,
                    field_path=field_prefix,
                )
            )
            continue
        required = bool(record.get("required", True))
        if not required:
            continue
        if record.get("passes") is not True:
            errors.append(
                _err(
                    error_codes.MARKER_LEGIBILITY_CONTRAST_FAILED,
                    "required semantic marker record does not pass legibility thresholds",
                    instance_id=instance_id,
                    field_path=f"{field_prefix}.passes",
                    role=record.get("role"),
                )
            )
        contrast = _finite_float(record.get("min_effective_contrast_ratio"))
        contrast_required = _finite_float(record.get("min_contrast_required"))
        if (
            contrast is None
            or contrast_required is None
            or contrast < contrast_required
        ):
            errors.append(
                _err(
                    error_codes.MARKER_LEGIBILITY_CONTRAST_FAILED,
                    "required semantic marker contrast is below threshold",
                    instance_id=instance_id,
                    field_path=f"{field_prefix}.min_effective_contrast_ratio",
                    role=record.get("role"),
                    min_effective_contrast_ratio=record.get(
                        "min_effective_contrast_ratio"
                    ),
                    min_contrast_required=record.get("min_contrast_required"),
                )
            )
        lab_distance = _finite_float(record.get("min_effective_lab_distance"))
        lab_distance_required = _finite_float(record.get("min_lab_distance_required"))
        if (
            lab_distance is None
            or lab_distance_required is None
            or lab_distance < lab_distance_required
        ):
            errors.append(
                _err(
                    error_codes.MARKER_LEGIBILITY_CONTRAST_FAILED,
                    "required semantic marker color distance is below threshold",
                    instance_id=instance_id,
                    field_path=f"{field_prefix}.min_effective_lab_distance",
                    role=record.get("role"),
                    min_effective_lab_distance=record.get("min_effective_lab_distance"),
                    min_lab_distance_required=record.get("min_lab_distance_required"),
                )
            )
        bbox = record.get("bbox_px")
        if bbox is not None:
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(
                    _err(
                        error_codes.MARKER_LEGIBILITY_INVALID,
                        "semantic marker bbox_px must be an array of four finite numbers",
                        instance_id=instance_id,
                        field_path=f"{field_prefix}.bbox_px",
                        role=record.get("role"),
                    )
                )
                continue
            coords = [_finite_float(value) for value in bbox]
            if any(value is None for value in coords):
                errors.append(
                    _err(
                        error_codes.MARKER_LEGIBILITY_INVALID,
                        "semantic marker bbox_px must contain only finite numbers",
                        instance_id=instance_id,
                        field_path=f"{field_prefix}.bbox_px",
                        role=record.get("role"),
                    )
                )
                continue
            x0, y0, x1, y1 = [float(value) for value in coords if value is not None]
            if x1 <= x0 or y1 <= y0:
                errors.append(
                    _err(
                        error_codes.MARKER_LEGIBILITY_INVALID,
                        "semantic marker bbox_px must have positive width and height",
                        instance_id=instance_id,
                        field_path=f"{field_prefix}.bbox_px",
                        role=record.get("role"),
                        bbox_px=bbox,
                    )
                )
    return errors


def _validate_marker_legibility_contract(
    trace_record: Mapping[str, Any],
    *,
    instance_id: str,
) -> List[_ValidationError]:
    """Validate rendered semantic-marker metadata when a renderer records it."""

    render_spec = trace_record.get("render_spec")
    if not isinstance(render_spec, Mapping):
        return []
    errors: List[_ValidationError] = []
    for field_path_root, marker_legibility in _iter_marker_legibility_blocks(
        render_spec, field_path="trace.render_spec"
    ):
        errors.extend(
            _validate_one_marker_legibility_block(
                marker_legibility,
                instance_id=str(instance_id),
                field_path_root=str(field_path_root),
            )
        )
    return errors


def validate_dataset(
    instances: List[Dict[str, Any]],
    *,
    staging_root: str | Path,
    expected_task_counts: Mapping[str, int],
    dataset_id: str,
    expected_instance_version: str,
    expected_trace_shard_counts: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Run required pre-finalize checks and return a full validation report."""
    root = Path(staging_root)
    errors: List[_ValidationError] = []

    for inst in instances:
        errors.extend(_validate_schema(inst))

    versions = sorted({str(inst.get("instance_version")) for inst in instances})
    if len(versions) > 1:
        errors.append(
            _err(
                error_codes.VERSION_MIXED_INSTANCE_VERSION,
                f"mixed instance versions detected: {versions}",
                field_path="instance_version",
            )
        )
    for inst in instances:
        iid = inst.get("instance_id", "<missing>")
        version = inst.get("instance_version")
        if version != expected_instance_version:
            errors.append(
                _err(
                    error_codes.VERSION_UNSUPPORTED_INSTANCE_VERSION,
                    f"unexpected instance_version {version!r}, expected {expected_instance_version!r}",
                    instance_id=iid,
                    field_path="instance_version",
                )
            )

    trace_cache: Dict[str, List[Dict[str, Any]]] = {}
    trace_read_failures: dict[str, str] = {}
    trace_ref_counts: Counter[str] = Counter()
    trace_ref_indices: dict[str, set[int]] = {}
    expected_shard_ids = (
        set(expected_trace_shard_counts)
        if expected_trace_shard_counts is not None
        else None
    )

    def load_trace_shard(
        shard_id: str,
        shard_path: Path,
        *,
        instance_id: Any | None = None,
    ) -> bool:
        if shard_id in trace_cache:
            return True
        if shard_id in trace_read_failures:
            return False
        records, failure_type = _read_trace_shard_for_validation(shard_path)
        if records is None:
            trace_read_failures[shard_id] = str(failure_type)
            context: dict[str, Any] = {
                "shard_id": shard_id,
                "failure_type": str(failure_type),
            }
            if instance_id is not None:
                context["instance_id"] = instance_id
            errors.append(
                _err(
                    error_codes.TRACE_SHARD_READ_FAILED,
                    "trace shard could not be decoded and validated",
                    **context,
                )
            )
            return False
        trace_cache[shard_id] = records
        return True

    for inst in instances:
        iid = inst.get("instance_id", "<missing>")

        errors.extend(_validate_instance_identity(inst))

        trace_ref = inst.get("trace_ref")
        if not isinstance(trace_ref, dict):
            errors.append(
                _err(
                    error_codes.TRACE_REF_MISSING,
                    "trace_ref is missing or invalid",
                    instance_id=iid,
                    field_path="trace_ref",
                )
            )
            continue

        shard_id = trace_ref.get("shard_id")
        line_index = trace_ref.get("line_index")
        trace_hash = trace_ref.get("trace_record_hash")

        if isinstance(shard_id, str) and shard_id:
            trace_ref_counts[shard_id] += 1
            if isinstance(line_index, int) and not isinstance(line_index, bool):
                trace_ref_indices.setdefault(shard_id, set()).add(line_index)

        if not is_safe_trace_shard_id(shard_id):
            errors.append(
                _err(
                    error_codes.TRACE_REF_NOT_FOUND,
                    "trace_ref shard_id must be one filename component beneath traces/",
                    instance_id=iid,
                    trace_ref=trace_ref,
                )
            )
            continue

        # Canonical manifests are authoritative. Do not touch a path named only
        # by an unexpected training-row reference.
        if expected_shard_ids is not None and shard_id not in expected_shard_ids:
            continue

        shard_path = _resolve_trace_shard_path(root, shard_id)
        if shard_path is None or not shard_path.is_file():
            errors.append(
                _err(
                    error_codes.TRACE_REF_NOT_FOUND,
                    "trace shard not found or is not a safe regular file",
                    instance_id=iid,
                    trace_ref=trace_ref,
                )
            )
            continue

        if not load_trace_shard(shard_id, shard_path, instance_id=iid):
            continue

        records = trace_cache[shard_id]
        if (
            isinstance(line_index, bool)
            or not isinstance(line_index, int)
            or line_index < 0
            or line_index >= len(records)
        ):
            errors.append(
                _err(
                    error_codes.TRACE_REF_INDEX_OUT_OF_RANGE,
                    "trace_ref line_index out of range",
                    instance_id=iid,
                    trace_ref=trace_ref,
                )
            )
            continue

        record = records[line_index]
        actual_hash = blake3_hex(canonical_json_bytes(record))
        if actual_hash != trace_hash:
            errors.append(
                _err(
                    error_codes.TRACE_REF_HASH_MISMATCH,
                    "trace_ref hash mismatch",
                    instance_id=iid,
                    expected_trace_hash=trace_hash,
                    actual_trace_hash=actual_hash,
                )
            )

        if record.get("instance_id") != iid:
            errors.append(
                _err(
                    error_codes.TRACE_REF_HASH_MISMATCH,
                    "trace record instance_id mismatch",
                    instance_id=iid,
                    trace_instance_id=record.get("instance_id"),
                )
            )

        errors.extend(_validate_prompt_contract(inst, record))
        errors.extend(_validate_text_legibility_contract(record, instance_id=str(iid)))
        errors.extend(
            _validate_marker_legibility_contract(record, instance_id=str(iid))
        )

        trace_reward_contract = record.get("reward_contract")
        if trace_reward_contract is None:
            errors.append(
                _err(
                    error_codes.SCHEMA_MISSING_FIELD,
                    "trace record is missing reward_contract",
                    instance_id=iid,
                    field_path="trace.reward_contract",
                )
            )
        else:
            reward_contract_error = validate_reward_contract_payload(
                trace_reward_contract,
                answer_type=(
                    inst.get("answer_gt", {}).get("type")
                    if isinstance(inst.get("answer_gt"), dict)
                    else None
                ),
                annotation_type=(
                    inst.get("annotation_gt", {}).get("type")
                    if isinstance(inst.get("annotation_gt"), dict)
                    else None
                ),
            )
            if reward_contract_error is not None:
                errors.append(
                    _err(
                        error_codes.SCHEMA_INVALID_VALUE,
                        reward_contract_error,
                        instance_id=iid,
                        field_path="trace.reward_contract",
                    )
                )
            elif trace_reward_contract != inst.get("reward_contract"):
                errors.append(
                    _err(
                        error_codes.SCHEMA_INVALID_VALUE,
                        "trace reward_contract must match the training record reward_contract",
                        instance_id=iid,
                        field_path="trace.reward_contract",
                    )
                )

        errors.extend(_validate_image_files(inst, root=root))

    observed_counts = Counter(str(inst.get("task")) for inst in instances)
    for task_id, expected_count in expected_task_counts.items():
        have = int(observed_counts.get(task_id, 0))
        if have < int(expected_count):
            errors.append(
                _err(
                    error_codes.COUNT_PER_TASK_SHORTFALL,
                    "accepted count below expectation",
                    task=task_id,
                    expected_count=int(expected_count),
                    actual_count=have,
                )
            )
    expected_set = set(expected_task_counts)
    for task_id, have in observed_counts.items():
        if task_id not in expected_set:
            errors.append(
                _err(
                    error_codes.COUNT_UNEXPECTED_TASK_PRESENT,
                    "unexpected task appears in build output",
                    task=task_id,
                    actual_count=int(have),
                )
            )

    if expected_trace_shard_counts is not None:
        observed_shard_ids = set(trace_ref_counts)

        for shard_id, expected_count in expected_trace_shard_counts.items():
            actual_count: int | None = None
            shard_path = _resolve_trace_shard_path(root, shard_id)
            if shard_path is not None and shard_path.is_file():
                load_trace_shard(shard_id, shard_path)
                if shard_id in trace_cache:
                    actual_count = len(trace_cache[shard_id])

            referenced_count = int(trace_ref_counts.get(shard_id, 0))
            unique_reference_count = len(trace_ref_indices.get(shard_id, set()))
            if (
                actual_count != int(expected_count)
                or referenced_count != int(expected_count)
                or unique_reference_count != int(expected_count)
            ):
                errors.append(
                    _err(
                        error_codes.TRACE_SHARD_MANIFEST_MISMATCH,
                        "trace shard does not match the build-report manifest",
                        shard_id=shard_id,
                        expected_count=int(expected_count),
                        actual_count=actual_count,
                        referenced_count=referenced_count,
                        unique_reference_count=unique_reference_count,
                    )
                )

        for shard_id in sorted(observed_shard_ids - expected_shard_ids):
            errors.append(
                _err(
                    error_codes.TRACE_SHARD_MANIFEST_MISMATCH,
                    "trace reference names a shard absent from the build-report manifest",
                    shard_id=shard_id,
                    referenced_count=int(trace_ref_counts[shard_id]),
                )
            )

        try:
            traces_path = root.resolve() / "traces"
            physical_shard_ids = (
                {path.name for path in traces_path.iterdir() if path.is_file()}
                if traces_path.is_dir() and traces_path.resolve() == traces_path
                else set()
            )
        except (OSError, RuntimeError):
            physical_shard_ids = set()
        for shard_id in sorted(physical_shard_ids - expected_shard_ids):
            errors.append(
                _err(
                    error_codes.TRACE_SHARD_MANIFEST_MISMATCH,
                    "physical trace shard is absent from the build-report manifest",
                    shard_id=shard_id,
                )
            )

    ordered = sorted(
        errors,
        key=lambda err: (
            err.error_code,
            str(err.context.get("instance_id", "")),
            str(err.context.get("field_path", "")),
            err.message,
        ),
    )

    error_counts_by_code = Counter(err.error_code for err in ordered)
    error_counts_by_category = Counter(err.category for err in ordered)

    report = {
        "total_errors": int(len(ordered)),
        "error_counts_by_code": dict(sorted(error_counts_by_code.items())),
        "error_counts_by_category": dict(sorted(error_counts_by_category.items())),
        "errors": [err.to_dict() for err in ordered],
        "build_context": {
            "dataset_id": dataset_id,
            "temp_path": str(root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    return report


def validate_candidate_instance(
    instance: Mapping[str, Any],
    trace_record: Mapping[str, Any],
    *,
    staging_root: str | Path,
    expected_instance_version: str,
    dataset_id: str = "candidate",
) -> Dict[str, Any]:
    """Validate one candidate before it is accepted into a dataset build.

    This intentionally mirrors the per-instance portions of ``validate_dataset``
    but uses the in-memory trace record so builders do not have to append
    rejected candidates to trace shards.
    """

    root = Path(staging_root)
    errors: List[_ValidationError] = []
    inst = dict(instance)
    record = dict(trace_record)
    iid = inst.get("instance_id", "<missing>")

    errors.extend(_validate_schema(inst))
    version = inst.get("instance_version")
    if version != expected_instance_version:
        errors.append(
            _err(
                error_codes.VERSION_UNSUPPORTED_INSTANCE_VERSION,
                f"unexpected instance_version {version!r}, expected {expected_instance_version!r}",
                instance_id=iid,
                field_path="instance_version",
            )
        )

    errors.extend(_validate_instance_identity(inst))

    trace_ref = inst.get("trace_ref")
    if not isinstance(trace_ref, Mapping):
        errors.append(
            _err(
                error_codes.TRACE_REF_MISSING,
                "trace_ref is missing or invalid",
                instance_id=iid,
                field_path="trace_ref",
            )
        )
    else:
        actual_hash: str | None
        try:
            actual_hash = blake3_hex(canonical_json_bytes(record))
        except Exception as exc:
            actual_hash = None
            errors.append(
                _err(
                    getattr(exc, "code", error_codes.SCHEMA_CANONICALIZATION_FAILED),
                    "trace record could not be canonicalized",
                    instance_id=iid,
                    field_path="<trace_record>",
                    failure_type=type(exc).__name__,
                )
            )
        trace_hash = trace_ref.get("trace_record_hash")
        if actual_hash is not None and actual_hash != trace_hash:
            errors.append(
                _err(
                    error_codes.TRACE_REF_HASH_MISMATCH,
                    "trace_ref hash mismatch",
                    instance_id=iid,
                    expected_trace_hash=trace_hash,
                    actual_trace_hash=actual_hash,
                )
            )
        line_index = trace_ref.get("line_index")
        if (
            isinstance(line_index, bool)
            or not isinstance(line_index, int)
            or line_index < 0
        ):
            errors.append(
                _err(
                    error_codes.TRACE_REF_INDEX_OUT_OF_RANGE,
                    "trace_ref line_index is invalid",
                    instance_id=iid,
                    trace_ref=dict(trace_ref),
                )
            )
        if not is_safe_trace_shard_id(trace_ref.get("shard_id")):
            errors.append(
                _err(
                    error_codes.TRACE_REF_MISSING,
                    "trace_ref shard_id is missing or invalid",
                    instance_id=iid,
                    trace_ref=dict(trace_ref),
                )
            )

    if record.get("instance_id") != iid:
        errors.append(
            _err(
                error_codes.TRACE_REF_HASH_MISMATCH,
                "trace record instance_id mismatch",
                instance_id=iid,
                trace_instance_id=record.get("instance_id"),
            )
        )

    errors.extend(_validate_prompt_contract(inst, record))
    errors.extend(_validate_text_legibility_contract(record, instance_id=str(iid)))
    errors.extend(_validate_marker_legibility_contract(record, instance_id=str(iid)))

    trace_reward_contract = record.get("reward_contract")
    if trace_reward_contract is None:
        errors.append(
            _err(
                error_codes.SCHEMA_MISSING_FIELD,
                "trace record is missing reward_contract",
                instance_id=iid,
                field_path="trace.reward_contract",
            )
        )
    else:
        reward_contract_error = validate_reward_contract_payload(
            trace_reward_contract,
            answer_type=(
                inst.get("answer_gt", {}).get("type")
                if isinstance(inst.get("answer_gt"), dict)
                else None
            ),
            annotation_type=(
                inst.get("annotation_gt", {}).get("type")
                if isinstance(inst.get("annotation_gt"), dict)
                else None
            ),
        )
        if reward_contract_error is not None:
            errors.append(
                _err(
                    error_codes.SCHEMA_INVALID_VALUE,
                    reward_contract_error,
                    instance_id=iid,
                    field_path="trace.reward_contract",
                )
            )
        elif trace_reward_contract != inst.get("reward_contract"):
            errors.append(
                _err(
                    error_codes.SCHEMA_INVALID_VALUE,
                    "trace reward_contract must match the training record reward_contract",
                    instance_id=iid,
                    field_path="trace.reward_contract",
                )
            )

    errors.extend(_validate_image_files(inst, root=root))

    ordered = sorted(
        errors,
        key=lambda err: (
            err.error_code,
            str(err.context.get("instance_id", "")),
            str(err.context.get("field_path", "")),
            err.message,
        ),
    )
    error_counts_by_code = Counter(err.error_code for err in ordered)
    error_counts_by_category = Counter(err.category for err in ordered)
    return {
        "total_errors": int(len(ordered)),
        "error_counts_by_code": dict(sorted(error_counts_by_code.items())),
        "error_counts_by_category": dict(sorted(error_counts_by_category.items())),
        "errors": [err.to_dict() for err in ordered],
        "build_context": {
            "dataset_id": dataset_id,
            "temp_path": str(root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_scope": "candidate",
        },
    }
