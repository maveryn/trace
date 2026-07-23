"""Shared defaults for the area chart scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    group_default as _config_group_default,
    load_scene_generation_rendering_prompt_defaults,
    resolve_required_int_bounds,
)
from trace_tasks.tasks.charts.shared.visual_defaults import load_chart_scene_noise_defaults


DOMAIN = "charts"
SCENE_ID = "area"
SCENE_NAMESPACE = "charts.area"
PROMPT_BUNDLE_ID = "charts_area_v1"

DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (72, 132, 204),
    (225, 127, 72),
    (92, 166, 106),
    (166, 114, 190),
    (214, 177, 69),
    (86, 170, 176),
)

GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID)
)
POST_IMAGE_NOISE_DEFAULTS: dict[str, Any] = load_chart_scene_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.0,
)


def scene_default(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Return an explicit param value, then scene defaults, then fallback."""

    return params.get(str(key), _config_group_default(defaults, str(key), fallback))


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(scene_default(params, GENERATION_DEFAULTS, key, fallback))


def generation_bounds(
    params: Mapping[str, Any],
    lower_key: str,
    upper_key: str,
    fallback_lower: int,
    fallback_upper: int,
) -> tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key=str(lower_key),
        max_key=str(upper_key),
        fallback_min=int(fallback_lower),
        fallback_max=int(fallback_upper),
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )


def rendering_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return scene_default(params, RENDERING_DEFAULTS, key, fallback)


def rendering_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(rendering_value(params, key, fallback))
