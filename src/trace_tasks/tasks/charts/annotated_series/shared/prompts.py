"""Prompt assembly for annotated-series chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.annotated_series.shared.defaults import PROMPT_BUNDLE_ID, PROMPT_DEFAULTS
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


SCENE_PROMPT_KEY_BY_VARIANT = {
    "line": "annotated_series_line",
    "bar": "annotated_series_bar",
    "area": "annotated_series_area",
    "dot_plot": "annotated_series_dot_plot",
    "lollipop": "annotated_series_lollipop",
}

TASK_PROMPT_KEY = "annotated_series_query"


def build_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    scene_key = SCENE_PROMPT_KEY_BY_VARIANT.get(scene_variant, "annotated_series_line")
    bundle_id = str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID))
    rendered_prompt = render_scene_prompt_variants(
        domain=domain,
        scene_id="annotated_series",
        bundle_id=bundle_id,
        scene_key=scene_key,
        task_key=prompt_task_key,
        query_key=prompt_query_key,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=instance_seed,
    )
    return build_prompt_trace_artifacts(rendered_prompt)
