"""Prompt bundle schema parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import string
from typing import Any, Dict, Mapping, Tuple

REQUIRED_PROMPT_VARIANTS = 5
PROMPT_SCHEMA_V1 = "v1"
ALLOWED_DYNAMIC_SLOT_TYPES = frozenset(
    {
        "string",
        "integer",
        "number",
        "boolean",
        "label",
        "json",
        "string_list",
        "label_list",
    }
)


@dataclass(frozen=True)
class PromptBundle:
    """Validated prompt bundle payload."""

    bundle_id: str
    schema_version: str
    allow_empty_task_templates: bool
    scene_templates: Dict[str, Tuple[str, ...]]
    task_templates: Dict[str, Tuple[str, ...]]
    query_templates: Dict[str, Tuple[str, ...]]
    answer_or_annotation_templates: Dict[str, Tuple[str, ...]]
    required_slots_by_key: Dict[str, Tuple[str, ...]]
    source_path: str
    source_hash: str = ""
    static_slots_by_key: Dict[str, Dict[str, Any]] | None = None
    dynamic_slots: Dict[str, Dict[str, Any]] | None = None


def _parse_template_map(
    raw: Any,
    *,
    field_name: str,
    allow_empty_templates: bool = False,
) -> Dict[str, Tuple[str, ...]]:
    """Validate and normalize a template-map field from bundle JSON."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    parsed: Dict[str, Tuple[str, ...]] = {}
    for key, values in raw.items():
        entry_key = str(key)
        if not isinstance(values, list):
            raise ValueError(f"{field_name}.{entry_key} must be a list")
        if bool(allow_empty_templates):
            templates = tuple(str(item) for item in values)
        else:
            templates = tuple(str(item).strip() for item in values if str(item).strip())
        if len(templates) != REQUIRED_PROMPT_VARIANTS:
            raise ValueError(
                f"{field_name}.{entry_key} must contain exactly {REQUIRED_PROMPT_VARIANTS} prompt variants"
            )
        parsed[entry_key] = templates
    return parsed


def _parse_required_slots(raw: Any) -> Dict[str, Tuple[str, ...]]:
    """Validate and normalize required-slot declarations by template key."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("required_slots_by_key must be a mapping")
    parsed: Dict[str, Tuple[str, ...]] = {}
    for key, values in raw.items():
        entry_key = str(key)
        if not isinstance(values, list):
            raise ValueError(f"required_slots_by_key.{entry_key} must be a list")
        slot_names = tuple(str(item).strip() for item in values if str(item).strip())
        parsed[entry_key] = slot_names
    return parsed


def _parse_static_slots(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Validate and normalize v1 static-slot declarations."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("static_slots_by_key must be a mapping")
    parsed: Dict[str, Dict[str, Any]] = {}
    for key, values in raw.items():
        entry_key = str(key)
        if not isinstance(values, Mapping):
            raise ValueError(f"static_slots_by_key.{entry_key} must be a mapping")
        slot_values: Dict[str, Any] = {}
        for slot_name, slot_value in values.items():
            name = str(slot_name).strip()
            if not name:
                raise ValueError(f"static_slots_by_key.{entry_key} contains an empty slot name")
            if name.startswith("json_example"):
                _validate_json_example_slot(name, slot_value)
            elif isinstance(slot_value, str) and _template_placeholders(slot_value):
                raise ValueError(f"static slot {entry_key}.{name} must not contain nested placeholders")
            slot_values[name] = slot_value
        parsed[entry_key] = slot_values
    return parsed


def _parse_dynamic_slots(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Validate and normalize v1 dynamic-slot declarations."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("dynamic_slots must be a mapping")
    parsed: Dict[str, Dict[str, Any]] = {}
    for slot_name, spec in raw.items():
        name = str(slot_name).strip()
        if not name:
            raise ValueError("dynamic_slots contains an empty slot name")
        if not isinstance(spec, Mapping):
            raise ValueError(f"dynamic_slots.{name} must be a mapping")
        slot_type = str(spec.get("type", "")).strip()
        if slot_type not in ALLOWED_DYNAMIC_SLOT_TYPES:
            raise ValueError(
                f"dynamic_slots.{name}.type must be one of {sorted(ALLOWED_DYNAMIC_SLOT_TYPES)}"
            )
        parsed[name] = {str(key): value for key, value in spec.items()}
    return parsed


def _template_placeholders(template: str) -> set[str]:
    """Return placeholder names used by one Python-format template."""

    placeholders: set[str] = set()
    for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(str(template)):
        if field_name is None:
            continue
        name = str(field_name).split(".", 1)[0].split("[", 1)[0].strip()
        if name:
            placeholders.add(name)
    return placeholders


def _all_template_placeholders(*template_maps: Mapping[str, Tuple[str, ...]]) -> set[str]:
    placeholders: set[str] = set()
    for template_map in template_maps:
        for templates in template_map.values():
            for template in templates:
                placeholders.update(_template_placeholders(str(template)))
    return placeholders


def _validate_json_example_slot(slot_name: str, value: Any) -> None:
    """Validate basic JSON-example shape for prompt assets."""

    if not isinstance(value, str):
        raise ValueError(f"{slot_name} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{slot_name} must be valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{slot_name} must be a JSON object")
    if "answer" not in parsed:
        raise ValueError(f"{slot_name} must include an answer key")
    if not slot_name.endswith("answer_only") and "annotation" not in parsed:
        raise ValueError(f"{slot_name} must include an annotation key")
    if _json_value_contains_placeholder(parsed):
        raise ValueError(f"{slot_name} must not contain nested placeholders")


def _json_value_contains_placeholder(value: Any) -> bool:
    """Return whether a parsed JSON example hides a template placeholder."""

    if isinstance(value, str):
        return bool(_template_placeholders(value))
    if isinstance(value, Mapping):
        return any(_json_value_contains_placeholder(child) for child in value.values())
    if isinstance(value, list):
        return any(_json_value_contains_placeholder(child) for child in value)
    return False


def _parse_v1_templates(raw: Mapping[str, Any]) -> tuple[
    Dict[str, Tuple[str, ...]],
    Dict[str, Tuple[str, ...]],
    Dict[str, Tuple[str, ...]],
    Dict[str, Tuple[str, ...]],
]:
    templates = raw.get("templates")
    if not isinstance(templates, Mapping):
        raise ValueError("v1 prompt bundles require templates")
    scene_templates = _parse_template_map(templates.get("scene"), field_name="templates.scene")
    task_templates = _parse_template_map(
        templates.get("task"),
        field_name="templates.task",
        allow_empty_templates=bool(raw.get("allow_empty_task_templates", False)),
    )
    query_raw = templates.get("query")
    query_templates = (
        _parse_template_map(query_raw, field_name="templates.query")
        if query_raw is not None
        else {}
    )
    output_templates = _parse_template_map(
        templates.get("output"),
        field_name="templates.output",
        allow_empty_templates=True,
    )
    for required_output_key in ("answer_only", "answer_and_annotation"):
        if required_output_key not in output_templates:
            raise ValueError(f"templates.output must include {required_output_key}")
    return scene_templates, task_templates, query_templates, output_templates


def _validate_v1_slot_declarations(
    *,
    scene_templates: Mapping[str, Tuple[str, ...]],
    task_templates: Mapping[str, Tuple[str, ...]],
    query_templates: Mapping[str, Tuple[str, ...]],
    output_templates: Mapping[str, Tuple[str, ...]],
    static_slots_by_key: Mapping[str, Mapping[str, Any]],
    dynamic_slots: Mapping[str, Mapping[str, Any]],
    required_slots_by_key: Mapping[str, Tuple[str, ...]],
) -> None:
    static_slot_names = {
        str(slot_name)
        for slot_values in static_slots_by_key.values()
        for slot_name in slot_values.keys()
    }
    dynamic_slot_names = set(dynamic_slots.keys())
    collisions = sorted(static_slot_names & dynamic_slot_names)
    if collisions:
        raise ValueError(f"static/dynamic prompt slot collision: {collisions}")

    declared_slots = static_slot_names | dynamic_slot_names
    placeholders = _all_template_placeholders(scene_templates, task_templates, query_templates, output_templates)
    undeclared = sorted(placeholders - declared_slots)
    if undeclared:
        raise ValueError(f"template placeholders are not declared as static or dynamic slots: {undeclared}")

    for scope, slot_names in required_slots_by_key.items():
        unknown = sorted(str(slot_name) for slot_name in slot_names if str(slot_name) not in declared_slots)
        if unknown:
            raise ValueError(f"required_slots_by_key.{scope} references undeclared slots: {unknown}")


def parse_prompt_bundle(raw: Mapping[str, Any], *, source_path: str, source_hash: str = "") -> PromptBundle:
    """Parse and validate a prompt bundle mapping."""
    bundle_id = str(raw.get("bundle_id", "")).strip()
    schema_version = str(raw.get("schema_version", "")).strip()
    if not bundle_id:
        raise ValueError("bundle_id is required")
    if not schema_version:
        raise ValueError("schema_version is required")

    if schema_version == PROMPT_SCHEMA_V1:
        (
            scene_templates,
            task_templates,
            query_templates,
            answer_or_annotation_templates,
        ) = _parse_v1_templates(raw)
        required_slots_by_key = _parse_required_slots(raw.get("required_slots_by_key"))
        static_slots_by_key = _parse_static_slots(raw.get("static_slots_by_key"))
        dynamic_slots = _parse_dynamic_slots(raw.get("dynamic_slots"))
        _validate_v1_slot_declarations(
            scene_templates=scene_templates,
            task_templates=task_templates,
            query_templates=query_templates,
            output_templates=answer_or_annotation_templates,
            static_slots_by_key=static_slots_by_key,
            dynamic_slots=dynamic_slots,
            required_slots_by_key=required_slots_by_key,
        )
        return PromptBundle(
            bundle_id=bundle_id,
            schema_version=schema_version,
            allow_empty_task_templates=bool(raw.get("allow_empty_task_templates", False)),
            scene_templates=scene_templates,
            task_templates=task_templates,
            query_templates=query_templates,
            answer_or_annotation_templates=answer_or_annotation_templates,
            required_slots_by_key=required_slots_by_key,
            source_path=str(source_path),
            source_hash=str(source_hash),
            static_slots_by_key=static_slots_by_key,
            dynamic_slots=dynamic_slots,
        )

    scene_templates = _parse_template_map(
        raw.get("scene_templates"),
        field_name="scene_templates",
    )
    allow_empty_task_templates = bool(raw.get("allow_empty_task_templates", False))
    task_templates = _parse_template_map(
        raw.get("task_templates"),
        field_name="task_templates",
        allow_empty_templates=allow_empty_task_templates,
    )
    query_raw = raw.get("query_templates")
    query_templates = (
        _parse_template_map(
            query_raw,
            field_name="query_templates",
        )
        if query_raw is not None
        else {}
    )
    answer_or_annotation_raw = raw.get("answer_or_annotation_templates")
    answer_or_annotation_templates = (
        _parse_template_map(
            answer_or_annotation_raw,
            field_name="answer_or_annotation_templates",
            allow_empty_templates=True,
        )
        if answer_or_annotation_raw is not None
        else {}
    )
    required_slots_by_key = _parse_required_slots(raw.get("required_slots_by_key"))

    return PromptBundle(
        bundle_id=bundle_id,
        schema_version=schema_version,
        allow_empty_task_templates=allow_empty_task_templates,
        scene_templates=scene_templates,
        task_templates=task_templates,
        query_templates=query_templates,
        answer_or_annotation_templates=answer_or_annotation_templates,
        required_slots_by_key=required_slots_by_key,
        source_path=str(source_path),
        source_hash=str(source_hash),
        static_slots_by_key={},
        dynamic_slots={},
    )
