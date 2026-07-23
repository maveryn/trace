"""Prompt assembly for dumbbell chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.dumbbell.shared.defaults import SCENE_ID
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_dumbbell_v1"
_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _DEFAULTS if isinstance(_DEFAULTS, Mapping) else {},
    task_id="charts_dumbbell_prompt",
)


DUMBBELL_OBJECT_DESCRIPTION = (
    "a horizontal dumbbell chart. Each row is a category label, and the two colored dots on that row "
    "give the values for the two legend series on the shared horizontal numeric axis. The gray "
    "connector shows the gap between the two dots"
)


def base_dynamic_slots(**overrides: Any) -> dict[str, Any]:
    """Return prompt slots shared by dumbbell query templates."""

    slots: dict[str, Any] = {
        "object_description": DUMBBELL_OBJECT_DESCRIPTION,
        "rank_phrase": "second largest",
        "winner_series": "",
        "loser_series": "",
        "threshold_value": 0,
        "gap_relation_phrase": "at least",
        "gap_threshold_value": 0,
    }
    slots.update(dict(overrides))
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
        bundle_id=str(_PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="dumbbell_pairwise_chart",
        task_key="dumbbell_pairwise_comparison_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["DUMBBELL_OBJECT_DESCRIPTION", "base_dynamic_slots", "build_prompt_artifacts"]
