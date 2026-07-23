"""Prompt assembly for conveyor tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID, semantic_color_label


DOMAIN = "three_d"
PROMPT_BUNDLE_ID = "three_d_carousel_v1"
PROMPT_WIRING_KEYS = ("bundle_id", "scene_key", "task_key")

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)


def dynamic_slots_for_conveyor(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Return dynamic prompt slots bound to one conveyor dataset."""

    target_color_name = str(dataset.get("target_color_name", ""))
    anchor_records = [dict(record) for record in dataset.get("marked_anchor_records", [])]
    start_anchor = anchor_records[0] if len(anchor_records) > 0 else {}
    end_anchor = anchor_records[1] if len(anchor_records) > 1 else {}
    slots = {
        "belt_label": str(dataset.get("target_belt_label", "")),
        "target_object_plural": str(dataset.get("target_object_plural", "")),
        "target_color_label": semantic_color_label(target_color_name) if target_color_name else "",
        "first_target_color_label": str(dataset.get("target_color_label", "")),
        "second_target_color_label": str(dataset.get("second_target_color_label", "")),
        "first_target_object_plural": (
            str(dataset.get("target_object_plural_pair", [""])[0])
            if dataset.get("target_object_plural_pair")
            else str(dataset.get("target_object_plural", ""))
        ),
        "second_target_object_plural": (
            str(dataset.get("target_object_plural_pair", ["", ""])[1])
            if len(dataset.get("target_object_plural_pair", [])) > 1
            else ""
        ),
        "source_belt_label": str(dataset.get("source_belt_label", "")),
        "destination_belt_label": str(dataset.get("destination_belt_label", "")),
        "start_anchor_label": str(start_anchor.get("anchor_label", "A")),
        "end_anchor_label": str(end_anchor.get("anchor_label", "B")),
        "start_anchor_object_name": str(start_anchor.get("object_name", "")),
        "end_anchor_object_name": str(end_anchor.get("object_name", "")),
    }
    if "arithmetic_operation" in dataset:
        annotation_keys = [str(dataset["annotation_key_by_scope"][scope]) for scope in dataset.get("scope_keys", [])]
        slots.update(
            {
                "first_belt_label": "INNER",
                "second_belt_label": "OUTER",
                "first_annotation_key": str(annotation_keys[0]),
                "second_annotation_key": str(annotation_keys[1]),
            }
        )
    return slots


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Build prompt artifacts for one conveyor objective."""

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        PROMPT_WIRING_KEYS,
        context="carousel prompt wiring defaults",
    )
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots_for_conveyor"]
