"""Prompt rendering entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, Mapping, Sequence

from .assets import load_prompt_bundle, load_scene_prompt_bundle
from .schema import PROMPT_SCHEMA_V1
from .select import choose_variant


@dataclass(frozen=True)
class PromptRenderResult:
    """Rendered prompt and deterministic variant metadata."""

    prompt: str
    metadata: Dict[str, Any]


class _StrictSlotMap(dict):
    """Format-map dict that raises on missing placeholder keys."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - exercised via caller exception path
        raise KeyError(f"missing prompt slot: {key}")


_ANSWER_ONLY_SCHEMA_LINE_RE = re.compile(
    r'^Use a valid JSON object with key "answer" for the final answer\.\s*$',
    re.IGNORECASE,
)
_ANSWER_AND_ANNOTATION_SCHEMA_LINE_RE = re.compile(
    r'^Use a valid JSON object with keys (?:"annotation" and "answer" in that order|"answer" and "annotation") for the final answer\.\s*$',
    re.IGNORECASE,
)


def _render_template(
    template: str,
    slots: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> str:
    """Render one prompt template with strict placeholder requirements."""
    rendered = str(template).format_map(_StrictSlotMap({str(k): v for k, v in dict(slots).items()})).strip()
    if not rendered and not bool(allow_empty):
        raise ValueError("rendered prompt template is empty")
    return rendered


def _strip_generic_output_contract_line(rendered_mode_text: str, *, answer_or_annotation_key: str | None) -> str:
    """Remove generic schema boilerplate while preserving task-specific format guidance."""
    if not rendered_mode_text or answer_or_annotation_key is None:
        return rendered_mode_text

    schema_line_re = (
        _ANSWER_ONLY_SCHEMA_LINE_RE
        if answer_or_annotation_key == "answer_only"
        else _ANSWER_AND_ANNOTATION_SCHEMA_LINE_RE
        if answer_or_annotation_key == "answer_and_annotation"
        else None
    )
    if schema_line_re is None:
        return rendered_mode_text

    filtered_lines = [line for line in rendered_mode_text.splitlines() if not schema_line_re.match(line.strip())]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in filtered_lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line.rstrip())
        previous_blank = is_blank
    return "\n".join(cleaned_lines).strip()


def _validate_required_slots(
    required_slots_by_key: Mapping[str, tuple[str, ...]],
    *,
    scene_key: str,
    task_key: str,
    query_key: str | None,
    answer_or_annotation_key: str | None,
    slots: Mapping[str, Any],
) -> None:
    """Ensure all slots declared by the selected scene/task/query keys are present."""
    required_scene = required_slots_by_key.get(f"scene:{scene_key}", ())
    required_task = required_slots_by_key.get(f"task:{task_key}", ())
    required_query = (
        required_slots_by_key.get(f"query:{query_key}", ())
        if query_key
        else ()
    )
    required_mode = (
        required_slots_by_key.get(f"answer_or_annotation:{answer_or_annotation_key}", ())
        if answer_or_annotation_key
        else ()
    )
    missing = [
        name
        for name in list(required_scene) + list(required_task) + list(required_query) + list(required_mode)
        if str(name) not in slots
    ]
    if missing:
        raise ValueError(f"missing required prompt slots: {sorted(set(missing))}")


def _required_slots_for_selected_keys(
    required_slots_by_key: Mapping[str, tuple[str, ...]],
    *,
    scene_key: str,
    task_key: str,
    query_key: str | None,
    output_key: str | None,
) -> tuple[str, ...]:
    """Return required slots for one selected prompt key path."""

    required: list[str] = []
    for scope in ("global", f"scene:{scene_key}", f"task:{task_key}"):
        required.extend(required_slots_by_key.get(scope, ()))
    if query_key:
        required.extend(required_slots_by_key.get(f"query:{query_key}", ()))
    if output_key:
        required.extend(required_slots_by_key.get(f"output:{output_key}", ()))
        required.extend(required_slots_by_key.get(f"answer_or_annotation:{output_key}", ()))
    return tuple(required)


def _resolve_answer_or_annotation_key(bundle, requested_key: str | None) -> str | None:
    """Resolve optional answer/annotation mode key for one bundle."""
    mode_templates = bundle.answer_or_annotation_templates
    if not mode_templates:
        return None
    if requested_key is not None:
        key = str(requested_key).strip()
        if key not in mode_templates:
            raise ValueError(f"missing answer_or_annotation key in bundle: {key}")
        return key
    if "answer_and_annotation" in mode_templates:
        return "answer_and_annotation"
    return sorted(mode_templates.keys())[0]


def _render_slot_value(value: Any, *, slot_type: str | None = None) -> str:
    """Render one raw slot value for template substitution."""

    if slot_type == "json" or isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if slot_type in {"string_list", "label_list"} and isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _validate_dynamic_slot_type(slot_name: str, slot_value: Any, slot_type: str) -> None:
    """Validate one v1 dynamic slot value against its declared primitive type."""

    if slot_type in {"string", "label"} and not isinstance(slot_value, str):
        raise ValueError(f"dynamic slot {slot_name} must be a string")
    if slot_type == "integer" and (isinstance(slot_value, bool) or not isinstance(slot_value, int)):
        raise ValueError(f"dynamic slot {slot_name} must be an integer")
    if slot_type == "number" and (isinstance(slot_value, bool) or not isinstance(slot_value, (int, float))):
        raise ValueError(f"dynamic slot {slot_name} must be a number")
    if slot_type == "boolean" and not isinstance(slot_value, bool):
        raise ValueError(f"dynamic slot {slot_name} must be a boolean")
    if slot_type == "json":
        try:
            json.dumps(slot_value)
        except TypeError as exc:
            raise ValueError(f"dynamic slot {slot_name} must be JSON-serializable") from exc
    if slot_type in {"string_list", "label_list"}:
        if not isinstance(slot_value, (list, tuple)) or not all(isinstance(item, str) for item in slot_value):
            raise ValueError(f"dynamic slot {slot_name} must be a list of strings")


def _resolve_v1_slots(
    bundle,
    *,
    scene_key: str,
    task_key: str,
    query_key: str | None,
    output_key: str | None,
    dynamic_slots: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    """Resolve v1 static and dynamic slots for rendering and metadata."""

    static_slots_by_key = bundle.static_slots_by_key or {}
    dynamic_slot_specs = bundle.dynamic_slots or {}
    dynamic_slots = {str(key): value for key, value in dict(dynamic_slots).items()}

    unknown_dynamic = sorted(set(dynamic_slots) - set(dynamic_slot_specs))
    if unknown_dynamic:
        raise ValueError(f"dynamic slots are not declared by prompt asset: {unknown_dynamic}")

    selected_static_scopes = ["global", f"scene:{scene_key}", f"task:{task_key}"]
    if query_key:
        selected_static_scopes.append(f"query:{query_key}")
    if output_key:
        selected_static_scopes.extend([f"output:{output_key}", f"answer_or_annotation:{output_key}"])

    raw_slot_values: dict[str, Any] = {}
    rendered_slots: dict[str, str] = {}
    slot_sources: dict[str, str] = {}
    for scope in selected_static_scopes:
        for slot_name, slot_value in dict(static_slots_by_key.get(scope, {})).items():
            name = str(slot_name)
            raw_slot_values[name] = slot_value
            rendered_slots[name] = _render_slot_value(slot_value)
            slot_sources[name] = f"static:{scope}"

    static_overrides = sorted(set(dynamic_slots) & set(raw_slot_values))
    if static_overrides:
        conflicting = [
            slot_name
            for slot_name in static_overrides
            if _render_slot_value(dynamic_slots[slot_name]) != rendered_slots[slot_name]
        ]
        if conflicting:
            raise ValueError(f"dynamic slots must not override static prompt slots: {conflicting}")
        dynamic_slots = {
            slot_name: slot_value
            for slot_name, slot_value in dynamic_slots.items()
            if slot_name not in set(static_overrides)
        }

    for slot_name, slot_value in dynamic_slots.items():
        spec = dict(dynamic_slot_specs[slot_name])
        slot_type = str(spec.get("type", "")).strip()
        _validate_dynamic_slot_type(slot_name, slot_value, slot_type)
        raw_slot_values[slot_name] = slot_value
        rendered_slots[slot_name] = _render_slot_value(slot_value, slot_type=slot_type)
        slot_sources[slot_name] = f"dynamic:{slot_name}"

    required = _required_slots_for_selected_keys(
        bundle.required_slots_by_key,
        scene_key=scene_key,
        task_key=task_key,
        query_key=query_key,
        output_key=output_key,
    )
    missing = [slot_name for slot_name in required if str(slot_name) not in raw_slot_values]
    if missing:
        raise ValueError(f"missing required prompt slots: {sorted(set(missing))}")
    return rendered_slots, raw_slot_values, slot_sources


def render_prompt(
    *,
    domain: str,
    scene_id: str | None = None,
    bundle_id: str,
    scene_key: str,
    task_key: str,
    query_key: str | None = None,
    answer_or_annotation_key: str | None = None,
    slots: Mapping[str, Any] | None = None,
    dynamic_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
) -> PromptRenderResult:
    """Render one prompt using deterministic scene/task/query templates."""
    if scene_id is None or not str(scene_id).strip():
        raise ValueError("render_prompt requires scene_id")
    bundle = load_scene_prompt_bundle(domain=domain, scene_id=str(scene_id), bundle_id=bundle_id)

    if scene_key not in bundle.scene_templates:
        raise ValueError(f"missing scene key in bundle: {scene_key}")
    if task_key not in bundle.task_templates:
        raise ValueError(f"missing task key in bundle: {task_key}")
    resolved_query_key = str(query_key).strip() if query_key is not None else None
    if resolved_query_key:
        if resolved_query_key not in bundle.query_templates:
            raise ValueError(f"missing query key in bundle: {resolved_query_key}")
    else:
        resolved_query_key = None

    resolved_mode_key = _resolve_answer_or_annotation_key(bundle, answer_or_annotation_key)
    if bundle.schema_version == PROMPT_SCHEMA_V1:
        if slots:
            raise ValueError("v1 prompt rendering requires dynamic_slots, not arbitrary slots")
        render_slots, raw_slot_values, slot_sources = _resolve_v1_slots(
            bundle,
            scene_key=str(scene_key),
            task_key=str(task_key),
            query_key=resolved_query_key,
            output_key=resolved_mode_key,
            dynamic_slots=dynamic_slots or {},
        )
    else:
        if dynamic_slots:
            raise ValueError("dynamic_slots are supported only by v1 prompt bundles")
        if slots is None:
            raise ValueError("v0 prompt rendering requires slots")
        render_slots = {str(key): value for key, value in dict(slots).items()}
        raw_slot_values = dict(render_slots)
        slot_sources = {str(key): "runtime" for key in render_slots}
        _validate_required_slots(
            bundle.required_slots_by_key,
            scene_key=scene_key,
            task_key=task_key,
            query_key=resolved_query_key,
            answer_or_annotation_key=resolved_mode_key,
            slots=render_slots,
        )

    scene_template, scene_idx, scene_count = choose_variant(
        bundle.scene_templates[scene_key],
        instance_seed=instance_seed,
        namespace=f"prompt.scene.{scene_key}",
    )
    task_template, task_idx, task_count = choose_variant(
        bundle.task_templates[task_key],
        instance_seed=instance_seed,
        namespace=f"prompt.task.{task_key}",
    )
    query_text = ""
    query_idx = None
    query_count = None
    if resolved_query_key is not None:
        query_template, query_idx, query_count = choose_variant(
            bundle.query_templates[resolved_query_key],
            instance_seed=instance_seed,
            namespace=f"prompt.query.{resolved_query_key}",
        )
        query_text = _render_template(query_template, render_slots)
    mode_text = ""
    mode_idx = None
    mode_count = None
    if resolved_mode_key is not None:
        mode_template, mode_idx, mode_count = choose_variant(
            bundle.answer_or_annotation_templates[resolved_mode_key],
            instance_seed=instance_seed,
            namespace=f"prompt.answer_or_annotation.{resolved_mode_key}",
        )
        mode_text = _render_template(mode_template, render_slots, allow_empty=True)
        mode_text = _strip_generic_output_contract_line(mode_text, answer_or_annotation_key=resolved_mode_key)

    scene_text = _render_template(scene_template, render_slots)
    task_text = _render_template(task_template, render_slots, allow_empty=bool(bundle.allow_empty_task_templates))
    prompt = " ".join(text for text in (scene_text, task_text, query_text) if text).strip()
    if mode_text:
        prompt = f"{prompt}\n{mode_text}".strip()

    metadata = {
        "prompt_bundle_id": bundle.bundle_id,
        "schema_version": bundle.schema_version,
        "prompt_schema_version": bundle.schema_version,
        "prompt_bundle_path": bundle.source_path,
        "prompt_bundle_hash": bundle.source_hash,
        "prompt_domain": str(domain),
        "scene_key": str(scene_key),
        "task_key": str(task_key),
        "query_key": (str(resolved_query_key) if resolved_query_key else None),
        "selected_keys": {
            "scene": str(scene_key),
            "task": str(task_key),
            "query": (str(resolved_query_key) if resolved_query_key else None),
            "output": (str(resolved_mode_key) if resolved_mode_key else None),
        },
        "scene_template_index": int(scene_idx),
        "task_template_index": int(task_idx),
        "query_template_index": (int(query_idx) if query_idx is not None else None),
        "selected_indices": {
            "scene": int(scene_idx),
            "task": int(task_idx),
            "query": (int(query_idx) if query_idx is not None else None),
            "output": (int(mode_idx) if mode_idx is not None else None),
        },
        "variant_count_by_key": {
            f"scene:{scene_key}": int(scene_count),
            f"task:{task_key}": int(task_count),
        },
        "variant_counts": {
            "scene": int(scene_count),
            "task": int(task_count),
            "query": (int(query_count) if query_count is not None else None),
            "output": (int(mode_count) if mode_count is not None else None),
        },
        "slot_values": {str(key): raw_slot_values[key] for key in sorted(raw_slot_values.keys(), key=str)},
        "slot_values_rendered": {
            str(key): render_slots[key]
            for key in sorted(render_slots.keys(), key=str)
        },
        "slot_sources": {
            str(key): slot_sources[key]
            for key in sorted(slot_sources.keys(), key=str)
        },
        "template_paths": [bundle.source_path],
    }
    metadata["prompt_scene_id"] = str(scene_id)
    if resolved_query_key is not None and query_count is not None:
        metadata["variant_count_by_key"][f"query:{resolved_query_key}"] = int(query_count)
    if resolved_mode_key is not None and mode_idx is not None and mode_count is not None:
        metadata["answer_or_annotation_key"] = str(resolved_mode_key)
        metadata["answer_or_annotation_query_id_index"] = int(mode_idx)
        metadata["variant_count_by_key"][f"answer_or_annotation:{resolved_mode_key}"] = int(mode_count)
    return PromptRenderResult(prompt=prompt, metadata=metadata)


def render_prompt_variants(
    *,
    domain: str,
    scene_id: str | None = None,
    bundle_id: str,
    scene_key: str,
    task_key: str,
    query_key: str | None = None,
    answer_or_annotation_keys: Sequence[str],
    slots: Mapping[str, Any] | None = None,
    dynamic_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
) -> Dict[str, PromptRenderResult]:
    """Render multiple answer/annotation prompt modes deterministically for one instance."""
    rendered: Dict[str, PromptRenderResult] = {}
    for key in [str(item).strip() for item in answer_or_annotation_keys]:
        if not key:
            raise ValueError("answer_or_annotation_keys must not contain empty values")
        rendered[key] = render_prompt(
            domain=domain,
            scene_id=scene_id,
            bundle_id=bundle_id,
            scene_key=scene_key,
            task_key=task_key,
            query_key=query_key,
            answer_or_annotation_key=key,
            slots=slots,
            dynamic_slots=dynamic_slots,
            instance_seed=instance_seed,
        )
    return rendered
