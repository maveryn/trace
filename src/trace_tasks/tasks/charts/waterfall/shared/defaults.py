"""Shared defaults for waterfall chart scene primitives."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
)


DOMAIN = "charts"
SCENE_ID = "waterfall"
SCENE_NAMESPACE = "charts.waterfall"
SCENE_VARIANT = "waterfall"
PROMPT_BUNDLE_ID = "charts_waterfall_v1"

GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID)
)
POST_IMAGE_BACKGROUND_DEFAULTS: dict[str, Any] = load_chart_scene_background_defaults(
    scene_id=SCENE_ID
)
POST_IMAGE_NOISE_DEFAULTS: dict[str, Any] = load_chart_scene_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.0,
)


def generation_default(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve one waterfall generation default."""

    return params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))


def rendering_default(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve one waterfall rendering default."""

    return params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), fallback))


__all__ = [
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "generation_default",
    "rendering_default",
]
