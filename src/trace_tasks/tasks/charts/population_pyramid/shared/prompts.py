"""Prompt assembly for population-pyramid tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_trace_artifacts, render_scene_prompt_variants

from .defaults import prompt_bundle_id
from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID, PopulationPyramidDataset


def dynamic_slots(*, dataset: PopulationPyramidDataset) -> dict[str, Any]:
    qparams = dict(dataset.query.params)
    return {
        "object_description": (
            "a mirrored horizontal bar chart with one row per age group. "
            "The left and right bars show the two legend series on the same positive scale"
        ),
        "rank_phrase": str(qparams.get("rank_phrase", "")),
        "metric_phrase": str(qparams.get("metric_phrase", "")),
        "side_series_label": str(qparams.get("side_series_label", "")),
        "other_series_label": str(qparams.get("other_series_label", "")),
        "extremum_phrase": str(qparams.get("extremum_phrase", "")),
        "threshold_relation_phrase": str(qparams.get("threshold_relation_phrase", "")),
        "threshold_value": int(qparams.get("threshold_value", 0)),
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
        scene_key="population_pyramid_scene",
        task_key="population_pyramid_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
