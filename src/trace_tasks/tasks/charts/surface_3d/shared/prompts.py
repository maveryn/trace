"""Prompt rendering helpers for 3D chart tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.charts.surface_3d.shared.defaults import DOMAIN, SCENE_ID, prompt_bundle_id
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


SCENE_PROMPT_KEY = "surface_3d"
TASK_PROMPT_KEY = "three_d_chart_query"


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render the external 3D chart prompt bundle."""

    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=prompt_bundle_id(),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


__all__ = ["SCENE_PROMPT_KEY", "TASK_PROMPT_KEY", "build_prompt_artifacts"]
