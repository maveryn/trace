"""Scene defaults for combo-mark chart tasks."""

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
SCENE_ID = "combo_mark"
SCENE_NAMESPACE = "charts.combo_mark"
PROMPT_BUNDLE_ID = "charts_combo_mark_v1"

GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID)
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

SCENE_VARIANTS: tuple[str, ...] = (
    "bar_line_shared_axis",
    "bar_line_dual_axis",
    "stacked_bar_line",
    "area_line_overlay",
)


def scene_default(defaults: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Return one scene default from a loaded generation/rendering section."""

    return group_default(defaults, str(key), fallback)


def int_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    low_key: str,
    high_key: str,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """Resolve inclusive integer bounds from params over scene defaults."""

    low = int(params.get(low_key, scene_default(defaults, low_key, int(fallback[0]))))
    high = int(params.get(high_key, scene_default(defaults, high_key, int(fallback[1]))))
    if low > high:
        raise ValueError(f"{low_key} must be <= {high_key}")
    return low, high


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
    "SCENE_VARIANTS",
    "int_bounds",
    "scene_default",
]
