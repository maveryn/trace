"""Prompt assembly for density-curve chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.density_curve.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.density_curve.shared.state import DensityCurveDataset
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_density_curve_v1"
_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _DEFAULTS if isinstance(_DEFAULTS, Mapping) else {},
)


def dynamic_slots(dataset: DensityCurveDataset) -> dict[str, str]:
    return {
        "object_description": "several smooth labeled density curves with a legend and exact x-axis reference marks when needed",
        "interval_label": str(dataset.query.trace["interval_label"]),
        "reference_x_label": str(dataset.query.trace["reference_x_label"]),
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
        bundle_id=str(_PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="density_curve",
        task_key="density_curve_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
