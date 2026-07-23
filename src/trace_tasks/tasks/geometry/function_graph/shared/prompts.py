"""Prompt-slot helpers for function-graph scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.prompts import load_scene_prompt_bundle
from trace_tasks.tasks.shared.config_defaults import required_group_default, required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, PROMPT_DEFAULTS, SCENE_ID


def prompt_defaults(keys: Sequence[str], *, context: str) -> dict[str, Any]:
    """Return required prompt defaults from the scene prompt bundle."""

    return required_group_defaults(PROMPT_DEFAULTS, tuple(str(key) for key in keys), context=str(context))


def prompt_asset_slot(defaults: Mapping[str, Any], slot_key: str) -> str:
    """Read one static slot from the scene prompt asset."""

    bundle = load_scene_prompt_bundle(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
    )
    static_slots = dict(bundle.static_slots_by_key or {}).get("global", {})
    value = static_slots.get(str(slot_key))
    if value is None or not str(value).strip():
        raise ValueError(f"missing function_graph prompt asset slot: {slot_key}")
    return str(value)


def function_object_description(*, defaults: Mapping[str, Any], family: str, has_guide_line: bool) -> str:
    """Resolve a prompt-facing function description."""

    suffix = "_with_guide_line" if bool(has_guide_line) else ""
    key = f"object_description_{str(family).strip().lower()}{suffix}"
    if key in defaults:
        return str(defaults[key])
    return prompt_asset_slot(defaults, key)


def prompt_artifacts_for_scene(
    *,
    defaults: Mapping[str, Any],
    prompt_template_key: str,
    slots: Mapping[str, Any],
    instance_seed: int,
):
    """Render prompt variants for one task-selected prompt template key."""

    selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_template_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={str(key): value for key, value in slots.items()},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(selection)


def integer_json_examples(annotation_value: Any) -> tuple[str, str]:
    """Build generic integer answer+annotation JSON examples."""

    return build_prompt_json_examples(annotation_value=annotation_value, answer_type="integer")
