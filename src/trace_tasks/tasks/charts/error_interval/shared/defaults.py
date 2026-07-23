"""Defaults and scene-level constants for error-interval charts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


DOMAIN = "charts"
SCENE_NAMESPACE = "charts_error_interval_base"
SCENE_ID = "error_interval"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    "horizontal_forest",
    "vertical_dot_whisker",
    "bar_with_error",
)

_TASK_GROUP_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def support_probability_map(values: Sequence[int | str]) -> Dict[str, float]:
    """Return a uniform probability map over one finite support."""

    support = [str(value) for value in values]
    if not support:
        return {}
    weight = 1.0 / float(len(support))
    return {str(value): float(weight) for value in support}


__all__ = [
    "DOMAIN",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "_GEN_DEFAULTS",
    "_PROMPT_DEFAULTS",
    "_RENDER_DEFAULTS",
    "support_probability_map",
]
