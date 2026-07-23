"""Prompt assembly for radar chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import prompt_bundle_id
from .state import RadarDataset, SCENE_ID


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_radar_v1"
SCENE_PROMPT_KEY = "radar_profile_charts"
TASK_PROMPT_KEY = "radar_profile_query"


OBJECT_DESCRIPTION_BY_VARIANT = {
    "small_multiple_radar": (
        "small multiple radar charts. Each panel has the same metric spokes and one colored profile polygon; "
        "panel labels identify the separate radar charts"
    ),
    "single_radar_multi_profile": (
        "one radar chart with the same metric spokes for two colored profile polygons. The legend names the two profiles"
    ),
}


def dynamic_slots(*, dataset: RadarDataset) -> dict[str, Any]:
    return {
        "object_description": str(OBJECT_DESCRIPTION_BY_VARIANT[str(dataset.scene_variant)]),
        "metric_label": str(dataset.query.metric_label),
        "panel_label": str(dataset.query.panel_label),
        "profile_a_label": str(dataset.query.profile_a_label),
        "profile_b_label": str(dataset.query.profile_b_label),
        "threshold_value": int(dataset.query.threshold_value),
        "minimum_metric_count": int(dataset.query.minimum_metric_count),
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
        bundle_id=str(prompt_bundle_id() or PROMPT_BUNDLE_ID),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
