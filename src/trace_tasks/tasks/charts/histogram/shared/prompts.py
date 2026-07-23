"""Prompt assembly for histogram chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.histogram.shared.defaults import (
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


SCENE_PROMPT_KEY = "histogram_scene"
TASK_PROMPT_KEY = "histogram_query"
OBJECT_DESCRIPTION = "a histogram with labeled x-axis values and bar heights showing counts"


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render the histogram prompt from external v1 prompt assets."""

    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots={
            "object_description": OBJECT_DESCRIPTION,
            "query_interval_label": "",
            "query_bin_label": "",
            "target_rank": "",
            "interval_relation_phrase": "",
            **dict(dynamic_slots),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = [
    "OBJECT_DESCRIPTION",
    "SCENE_PROMPT_KEY",
    "TASK_PROMPT_KEY",
    "build_prompt_artifacts",
]
