"""Prompt assembly for surface-fixture tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID, semantic_color_label


DOMAIN = "three_d"
PROMPT_BUNDLE_ID = "three_d_surface_fixture_v1"
PROMPT_WIRING_KEYS = ("bundle_id", "scene_key", "task_key")

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)


def dynamic_slots_for_surface(dataset: Mapping[str, Any], *, object_description: str) -> dict[str, Any]:
    """Return dynamic prompt slots bound to one rendered fixture dataset."""

    target_color_name = str(dataset.get("target_color_name", ""))
    return {
        "object_description": str(object_description),
        "target_element_name": str(dataset.get("target_element_name", "")),
        "target_element_plural": str(dataset.get("target_element_plural", "")),
        "target_color_label": semantic_color_label(target_color_name) if target_color_name else "",
        "operation_phrase": str(dataset.get("operation_phrase", "")),
        "recolor_phrase": str(dataset.get("recolor_phrase", "")),
        "source_color_label": semantic_color_label(str(dataset.get("source_color_name", ""))) if dataset.get("source_color_name") else "",
        "destination_color_label": (
            semantic_color_label(str(dataset.get("destination_color_name", ""))) if dataset.get("destination_color_name") else ""
        ),
        "scope_phrase": str(dataset.get("scope_phrase", "")),
        "fixture_display_name": str(dataset.get("fixture_display_name", "fixture surface")),
    }


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Build prompt artifacts for one public surface-fixture objective."""

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        PROMPT_WIRING_KEYS,
        context="surface_fixture prompt wiring defaults",
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


__all__ = ["build_prompt_artifacts", "dynamic_slots_for_surface"]
