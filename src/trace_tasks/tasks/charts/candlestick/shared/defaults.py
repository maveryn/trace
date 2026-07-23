"""Scene constants and config defaults for candlestick charts."""

from __future__ import annotations

from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


DOMAIN = "charts"
SCENE_ID = "candlestick"
SCENE_NAMESPACE = "charts.candlestick"
PROMPT_BUNDLE_ID = "charts_candlestick_v1"

GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID)
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


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
]
