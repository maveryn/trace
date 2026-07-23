"""Prompt assembly helpers for circle-theorem tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.prompts import load_scene_prompt_bundle
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID

_OBJECT_DESCRIPTION_ASSET_KEYS = {
    "object_description": "object_description_default",
    "object_description_chord_length": "object_description_chord_length",
    "object_description_tangent_radius": "object_description_tangent_radius",
}


def _keyed_point_prompt_examples(
    annotation_keys: Sequence[str], *, answer: int | float
) -> tuple[str, str]:
    """Build JSON examples using the requested visible point-label keys."""

    example_points = {
        str(key): [120 + (37 * index), 180 + (23 * index)]
        for index, key in enumerate(annotation_keys)
    }
    return dump_prompt_json_examples(
        annotation=example_points,
        answer=answer,
        ensure_ascii=False,
    )


def _global_prompt_asset_slot(*, bundle_id: str, slot_key: str) -> str:
    """Read one scene prompt prose slot from the v1 prompt asset."""

    bundle = load_scene_prompt_bundle(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
    )
    static_slots = dict(bundle.static_slots_by_key or {}).get("global", {})
    value = static_slots.get(str(slot_key))
    if value is None or not str(value).strip():
        raise ValueError(f"missing circle_theorem prompt asset slot: {slot_key}")
    return str(value)


def _object_description_slot_key(object_description_key: str) -> str:
    """Map historical object-description selectors to v1 asset slot names."""

    try:
        return _OBJECT_DESCRIPTION_ASSET_KEYS[str(object_description_key)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported circle_theorem object description key: {object_description_key}"
        ) from exc


def build_circle_theorem_prompt_artifacts(
    *,
    prompt_query_key: str,
    prompt_slots: Mapping[str, Any],
    annotation_keys: Sequence[str],
    answer_hint_key: str,
    answer_example: int | float,
    annotation_hint_key: str = "annotation_hint_circle_points",
    object_description_key: str = "object_description",
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts for one task-owned circle-theorem query."""

    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="circle_theorem prompt wiring defaults",
    )
    bundle_id = str(prompt_defaults["bundle_id"])
    object_description = _global_prompt_asset_slot(
        bundle_id=bundle_id,
        slot_key=_object_description_slot_key(object_description_key),
    )
    answer_hint = _global_prompt_asset_slot(
        bundle_id=bundle_id,
        slot_key=str(answer_hint_key),
    )
    key_text = ", ".join(f'"{key}"' for key in annotation_keys)
    json_example, json_example_answer_only = _keyed_point_prompt_examples(
        annotation_keys,
        answer=answer_example,
    )
    dynamic_slots = {
        "object_description": object_description,
        "annotation_keys": key_text,
        "answer_hint": answer_hint,
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
        **{str(key): str(value) for key, value in dict(prompt_slots).items()},
    }
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=bundle_id,
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_circle_theorem_prompt_artifacts"]
