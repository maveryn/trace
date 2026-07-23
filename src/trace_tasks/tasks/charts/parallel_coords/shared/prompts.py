"""Prompt assembly for parallel-coordinates chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID


SCENE_PROMPT_KEY = "parallel_coords_chart"
TASK_PROMPT_KEY = "parallel_coords_query"
OBJECT_DESCRIPTION = (
    "a parallel-coordinates chart. Each colored profile line is labeled at the ends and crosses the "
    "vertical metric axes; larger metric values are higher on each axis"
)


def dynamic_slots(dataset: Any) -> dict[str, Any]:
    slots: dict[str, Any] = {
        "object_description": OBJECT_DESCRIPTION,
        "axis_i": str(dataset.metrics[int(dataset.query.axis_i)]),
        "axis_j": str(dataset.metrics[int(dataset.query.axis_j)]),
    }
    if dataset.query.threshold is not None:
        slots["threshold"] = int(dataset.query.threshold)
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
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
