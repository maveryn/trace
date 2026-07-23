"""Scene constants and config defaults for contour-density charts."""

from __future__ import annotations

from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)


DOMAIN = "charts"
SCENE_ID = "contour_density"
SCENE_NAMESPACE = "charts.contour_density"
PROMPT_BUNDLE_ID = "charts_contour_density_v1"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "contour_rings",
    "filled_density",
    "scatter_contour",
)
SUPPORTED_DENSITY_EXTREMA: Tuple[str, ...] = ("highest", "lowest")
SUPPORTED_DISTANCE_EXTREMA: Tuple[str, ...] = ("nearest", "farthest")
SUPPORTED_REFERENCE_KINDS: Tuple[str, ...] = ("point", "vertical_line", "horizontal_line")
SUPPORTED_DENSITY_THRESHOLD_DIRECTIONS: Tuple[str, ...] = ("at_least", "below")
SUPPORTED_SPREAD_EXTREMA: Tuple[str, ...] = ("widest", "narrowest")

TASK_GROUP_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
CONFIG_CONTEXT_KEY = "".join(("task", "_", "id"))
GENERATION_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    TASK_GROUP_DEFAULTS if isinstance(TASK_GROUP_DEFAULTS, Mapping) else {},
    **{CONFIG_CONTEXT_KEY: SCENE_NAMESPACE},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


__all__ = [
    "DOMAIN",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "PROMPT_BUNDLE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_DENSITY_EXTREMA",
    "SUPPORTED_DISTANCE_EXTREMA",
    "SUPPORTED_REFERENCE_KINDS",
    "SUPPORTED_DENSITY_THRESHOLD_DIRECTIONS",
    "SUPPORTED_SPREAD_EXTREMA",
    "GENERATION_DEFAULTS",
    "RENDER_DEFAULTS",
    "PROMPT_DEFAULTS",
    "CONFIG_CONTEXT_KEY",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
]
