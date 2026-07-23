"""Prompt assembly for curve-panel chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.curve_panels.shared.defaults import SCENE_ID
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_curve_panels_v1"
_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_CONFIG_CONTEXT_KEY = "_".join(("task", "id"))
GENERATION_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = (
    split_scene_generation_rendering_prompt_defaults(
        _DEFAULTS if isinstance(_DEFAULTS, Mapping) else {},
        **{_CONFIG_CONTEXT_KEY: "charts_curve_panels_prompt"},
    )
)


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
        scene_key="curve_panels_subplot",
        task_key="multipanel_subplot_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts"]
