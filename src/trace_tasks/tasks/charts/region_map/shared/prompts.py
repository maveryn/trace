"""Prompt assembly for region-map chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


DOMAIN = "charts"
SCENE_ID = "region_map"
PROMPT_BUNDLE_ID = "charts_region_map_v1"
SCENE_PROMPT_KEY = "region_map_scene"
TASK_PROMPT_KEY = "region_map_query"

_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _DEFAULTS if isinstance(_DEFAULTS, Mapping) else {},
    task_id=f"{SCENE_ID}_prompt",
)


def dynamic_slots(dataset: Mapping[str, Any]) -> dict[str, Any]:
    question_params = dict(dataset.get("question_params", {}))
    scene_variant = str(dataset.get("scene_variant") or "")
    if "marker_render_variant" in dataset or "marker_region_ids" in question_params:
        object_description = str(dataset.get("map_object_description") or "a map with marker bubbles")
    elif bool(dataset.get("show_region_value_labels")):
        object_description = "a map with colored regions, visible region labels, visible integer values, and a legend"
        if str(scene_variant) == "geographic_region_map":
            object_description = "a world map with selected countries colored by value, visible integer values, and a legend"
    elif bool(dataset.get("show_region_reference_labels")):
        object_description = "a synthetic map with colored regions, short region labels, and a legend"
    elif str(scene_variant) == "geographic_region_map":
        object_description = str(dataset.get("map_object_description") or "a geographic map with selected colored regions and a legend")
        if bool(dataset.get("categorical")):
            object_description = object_description.replace(
                "colored by value and a color legend",
                "colored by category and a category legend",
            )
    else:
        object_description = "a synthetic map with colored regions and a legend"
    return {
        "object_description": str(object_description),
        "region_noun": str(dataset.get("map_region_noun") or ("countries" if str(scene_variant) == "geographic_region_map" else "regions")),
        "continent_label": str(question_params.get("continent_label", "")),
        "threshold_phrase": str(question_params.get("threshold_phrase", "")),
        "extremum_word": str(question_params.get("extremum_word", "")),
        "interval_phrase": str(question_params.get("interval_phrase", "")),
        "category_label": str(question_params.get("category_label", "")),
        "reference_region_label": str(question_params.get("reference_region_label", "")),
        "region_set_name": str(question_params.get("region_set_name", "")),
        "region_set_label_list": str(question_params.get("region_set_label_list", "")),
    }


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(_PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
