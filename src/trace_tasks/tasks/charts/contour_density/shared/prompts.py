"""Prompt assembly for contour-density chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.contour_density.shared.defaults import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


SCENE_PROMPT_KEY = "contour_density_scene"
TASK_PROMPT_KEY = "contour_density_query"


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts"]
