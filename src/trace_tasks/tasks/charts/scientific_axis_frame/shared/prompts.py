"""Prompt assembly for scientific axis-frame chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.scientific_axis_frame.shared.defaults import PROMPT_DEFAULTS
from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


def dynamic_slots(*, axis_name: str) -> dict[str, Any]:
    return {
        "object_description": "a scientific plot frame with numeric x-axis and y-axis tick labels",
        "axis_name": str(axis_name),
    }


def build_prompt_artifacts(
    *,
    prompt_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_ID,
        task_key="axis_frame_query",
        query_key=str(prompt_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
