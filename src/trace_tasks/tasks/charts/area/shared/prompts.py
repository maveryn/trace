"""Prompt assembly for the area chart scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.area.shared.defaults import PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


TASK_PROMPT_KEY = "area_query"
SCENE_PROMPT_KEY_BY_VARIANT = {
    "single_area": "single_area",
    "stacked_area": "stacked_area",
}


def build_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    scene_key = SCENE_PROMPT_KEY_BY_VARIANT.get(str(scene_variant), "single_area")
    rendered_prompt = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=scene_key,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)
