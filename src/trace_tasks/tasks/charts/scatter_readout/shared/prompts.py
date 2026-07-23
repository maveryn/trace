"""Prompt assembly for scatter-readout chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.unanswerable import UNANSWERABLE_ANSWER
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID, QueryBinding


PROMPT_BUNDLE_ID = "charts_scatter_readout_v1"


def dynamic_slots(
    *,
    binding: QueryBinding,
    include_unanswerable_instruction: bool,
) -> dict[str, Any]:
    trace = dict(binding.trace)
    unanswerable_instruction = ""
    if include_unanswerable_instruction:
        unanswerable_instruction = (
            f'If the requested series is not visible in the legend, answer exactly "{UNANSWERABLE_ANSWER}".'
        )
    return {
        "object_description": "a multi-series scatter plot with labeled points and a legend",
        "series_label": str(trace.get("target_series_label", "")),
        "comparison_series_label": str(trace.get("comparison_series_label", "")),
        "extremum_phrase": str(trace.get("extremum", "")),
        "target_y_value": str(trace.get("target_y_value", "")),
        "target_x_label": str(trace.get("target_x_label", "")),
        "unanswerable_instruction": str(unanswerable_instruction),
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
        scene_key="scatter_readout",
        task_key="scatter_series_readout_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
