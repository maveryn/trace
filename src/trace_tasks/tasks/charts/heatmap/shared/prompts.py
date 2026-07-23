"""Prompt assembly for heatmap chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.heatmap.shared.defaults import PROMPT_DEFAULTS
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


DOMAIN = "charts"
SCENE_ID = "heatmap"
PROMPT_BUNDLE_ID = "charts_heatmap_v1"


def dynamic_slots(dataset: Mapping[str, Any], *, supports_unanswerable: bool) -> dict[str, Any]:
    question_params = dict(dataset.get("question_params", {}))
    scene_variant = str(dataset["scene_variant"])
    slots = {
        "object_description": {
            "intensity_heatmap": "labeled grid cells where darker colors indicate higher intensity and lighter colors indicate lower intensity",
            "signed_change_heatmap": "labeled grid cells where blue colors indicate decreases, neutral colors indicate no clear change, and orange-red colors indicate increases",
            "calendar_heatmap": "a week-by-week calendar grid where darker colors indicate higher activity and lighter colors indicate lower activity",
            "continuous_colorbar_heatmap": "labeled grid cells colored by a continuous numeric colorbar scale",
        }.get(str(scene_variant), "a labeled heatmap grid"),
        "condition_phrase": str(question_params.get("condition_phrase", "")),
        "threshold_value": str(question_params.get("threshold_value", "")),
        "lower_bound": str(question_params.get("lower_bound", "")),
        "upper_bound": str(question_params.get("upper_bound", "")),
        "query_axis": str(question_params.get("query_axis", "")),
        "answer_axis": str(question_params.get("answer_axis", "")),
        "axis_label": str(question_params.get("axis_label", "")),
        "column_label": str(question_params.get("column_label", "")),
        "row_label": str(question_params.get("row_label", "")),
        "extremum_phrase": str(question_params.get("extremum_phrase", "")),
    }
    if bool(supports_unanswerable):
        slots["unanswerable_instruction"] = (
            'If the requested label or color condition is not visible, answer exactly "unanswerable".'
        )
    else:
        slots["unanswerable_instruction"] = ""
    return slots


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
        scene_key="heatmap_scene",
        task_key="heatmap_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = [
    "build_prompt_artifacts",
    "dynamic_slots",
]
