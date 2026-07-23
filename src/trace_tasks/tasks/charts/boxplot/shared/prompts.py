"""Prompt assembly for the boxplot chart scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.boxplot.shared.defaults import (
    DOMAIN,
    PROMPT_BUNDLE_ID,
    PROMPT_DEFAULTS,
    SCENE_ID,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


TASK_PROMPT_KEY = "boxplot_query"
SINGLE_SCENE_PROMPT_KEY = "boxplot"
PAIRED_SCENE_PROMPT_KEY = "paired_boxplot"


def build_prompt_artifacts(
    *,
    scene_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(scene_key),
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = [
    "PAIRED_SCENE_PROMPT_KEY",
    "SINGLE_SCENE_PROMPT_KEY",
    "TASK_PROMPT_KEY",
    "build_prompt_artifacts",
]
