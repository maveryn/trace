"""Scene defaults for synthetic 3D chart panels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
)


DOMAIN = "charts"
SCENE_ID = "surface_3d"
RENDER_NAMESPACE = "charts.surface_3d"
PROMPT_BUNDLE_ID = "charts_surface_3d_v1"

SCATTER_VARIANT = "three_d_scatter"
SURFACE_VARIANT = "three_d_surface"
PANEL_VARIANT = "three_d_small_multiples"
SCENE_VARIANTS = (SCATTER_VARIANT, SURFACE_VARIANT, PANEL_VARIANT)

GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer generation knob from params or scene defaults."""

    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def prompt_bundle_id() -> str:
    """Return the configured v1 prompt bundle identifier."""

    return str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID))


__all__ = [
    "DOMAIN",
    "GEN_DEFAULTS",
    "PANEL_VARIANT",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "RENDER_NAMESPACE",
    "SCATTER_VARIANT",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SURFACE_VARIANT",
    "generation_int",
    "prompt_bundle_id",
]
