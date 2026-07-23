"""Prompt assembly for scatter-cluster chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import resolved_prompt_bundle_id
from .state import AREA_ENVELOPE_SCATTER, DOMAIN, SCENE_ID, ScatterClusterDataset


def dynamic_slots(*, dataset: ScatterClusterDataset) -> dict[str, Any]:
    trace = dict(dataset.question.params)
    return {
        "object_description": (
            "a scatter plot with several colored point clusters, shaded cluster footprints, and a matching legend"
            if str(dataset.scene_variant) == AREA_ENVELOPE_SCATTER
            else "a scatter plot with several colored point clusters and a matching legend"
        ),
        "trend_direction_phrase": str(trace.get("trend_direction", "")),
        "spread_axis_phrase": str(trace.get("spread_axis", "")),
        "spread_extremum_phrase": str(trace.get("spread_extremum", "")),
        "area_rank_phrase": str(trace.get("area_rank_phrase", "")),
        "target_cluster_label": str(trace.get("target_cluster_label", "")),
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
        bundle_id=resolved_prompt_bundle_id(),
        scene_key="scatter_cluster",
        task_key="scatter_cluster_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)
