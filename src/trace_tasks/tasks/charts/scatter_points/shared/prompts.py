"""Prompt assembly for scatter-point chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import Dataset, SCENE_ID


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_scatter_points_v1"


def axis_phrase(axis: str) -> str:
    return "x" if str(axis) == "x" else "y"


def direction_phrase(direction: str) -> str:
    return "greater than" if str(direction) == "above" else "less than"


def extremum_phrase(extremum: str) -> str:
    return "largest" if str(extremum) == "largest" else "smallest"


def dynamic_slots(*, dataset: Dataset) -> dict[str, Any]:
    trace = dict(dataset.query.trace)
    if dataset.categories:
        object_description = "a categorized scatter plot with individual data points, numeric x- and y-axes, and a legend"
    else:
        object_description = "a scatter plot of individual data points with numeric x- and y-axes"
    return {
        "object_description": str(object_description),
        "axis_phrase": axis_phrase(str(trace.get("threshold_axis", trace.get("mean_axis", "x")))),
        "threshold_direction_phrase": direction_phrase(str(trace.get("threshold_direction", "above"))),
        "threshold_value": str(trace.get("threshold_value", "")),
        "mean_extremum_phrase": extremum_phrase(str(trace.get("mean_extremum", "largest"))),
        "target_category_label": str(trace.get("target_category_label", "")),
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
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="scatter_points",
        task_key="scatter_points_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = [
    "axis_phrase",
    "build_prompt_artifacts",
    "direction_phrase",
    "dynamic_slots",
    "extremum_phrase",
]
