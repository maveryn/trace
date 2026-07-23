"""Prompt assembly for waterfall chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.waterfall.shared.defaults import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


SCENE_PROMPT_KEY = "waterfall"
TASK_PROMPT_KEY = "waterfall_query"


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render the external waterfall prompt template."""

    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


__all__ = ["SCENE_PROMPT_KEY", "TASK_PROMPT_KEY", "build_prompt_artifacts"]
