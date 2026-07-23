"""Prompt assembly for error-bar series chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.errorbar_series.shared.defaults import PROMPT_DEFAULTS
from trace_tasks.tasks.charts.errorbar_series.shared.state import DOMAIN, SCENE_ID, ErrorbarDataset
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


PROMPT_BUNDLE_ID = "charts_errorbar_series_v1"


def dynamic_slots(
    dataset: ErrorbarDataset,
    *,
    bound_phrase: str = "",
    extremum_phrase: str = "",
) -> dict[str, Any]:
    """Return prompt slots from task-owned semantic arguments."""

    return {
        "object_description": "a scientific chart with labeled series, ordered x-axis labels, central point markers, and vertical error bars",
        "target_series_label": str(dataset.query.params.get("target_series_label", "")),
        "target_x_label": str(dataset.query.params.get("target_x_label", "")),
        "bound_phrase": str(bound_phrase),
        "extremum_phrase": str(extremum_phrase),
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
        scene_key="errorbar_series_scene",
        task_key="errorbar_series_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
