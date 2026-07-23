"""Config/default helpers for the styled table chart scene."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)

from .state import DOMAIN, SCENE_ID, SCENE_NAMESPACE


SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve one table generation value with task params taking precedence."""

    return params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))


def rendering_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve one table rendering value with task params taking precedence."""

    return params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), fallback))


__all__ = [
    "BACKGROUND_DEFAULTS",
    "GENERATION_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "generation_value",
    "rendering_value",
]
