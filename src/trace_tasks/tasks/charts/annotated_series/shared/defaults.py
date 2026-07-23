"""Shared defaults for the annotated-series chart scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.labeled_chart_defaults import LabeledChartDefaults
from trace_tasks.tasks.shared.config_defaults import (
    group_default as _config_group_default,
    load_scene_generation_rendering_prompt_defaults,
    resolve_required_int_bounds,
)


DOMAIN = "charts"
SCENE_ID = "annotated_series"
SCENE_NAMESPACE = "charts.annotated_series"

SUPPORTED_SCENE_VARIANTS = ("line", "bar", "area", "dot_plot", "lollipop")

PROMPT_BUNDLE_ID = "charts_annotated_series_v1"

FALLBACK_CHART_DEFAULTS = LabeledChartDefaults(
    mark_count_min=5,
    mark_count_max=9,
    value_min=10,
    value_max=90,
)

GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID)
)

POST_IMAGE_NOISE_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "shadow_probability": 0.35,
    "paper_texture_probability": 0.30,
    "blur_probability": 0.12,
    "blur_radius_min": 0.15,
    "blur_radius_max": 0.55,
    "noise_probability": 0.18,
    "noise_std_min": 1.5,
    "noise_std_max": 4.5,
    "jpeg_probability": 0.12,
    "jpeg_quality_min": 70,
    "jpeg_quality_max": 92,
}

CONTEXT_PARAM_KEYS = (
    "chart_context_profile",
    "context_text_profile",
    "chart_context_mode",
    "chart_context_mode_weights",
    "context_text_mode",
    "context_text_mode_weights",
    "context_text_enabled",
    "context_text_top_reserved_px",
    "context_text_bottom_reserved_px",
    "context_text_left_margin_px",
    "context_text_right_margin_px",
    "context_text_sidebar_width_min_px",
    "context_text_sidebar_width_max_px",
    "context_text_sidebar_gap_px",
    "context_text_bottom_band_height_min_px",
    "context_text_bottom_band_height_max_px",
    "context_text_bottom_band_gap_px",
    "context_text_box_count_min",
    "context_text_box_count_max",
    "context_text_placement_weights",
    "context_text_font_family_weights",
    "context_text_light_font_family",
)


def group_default(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Return an explicit param value, then scene defaults, then fallback."""

    return params.get(str(key), _config_group_default(defaults, str(key), fallback))


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(group_default(params, GENERATION_DEFAULTS, key, fallback))


def rendering_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return group_default(params, RENDERING_DEFAULTS, key, fallback)


def rendering_bool(params: Mapping[str, Any], key: str, fallback: bool) -> bool:
    return bool(rendering_value(params, key, fallback))


def rendering_float(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(rendering_value(params, key, fallback))


def rendering_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(rendering_value(params, key, fallback))


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
        min_key=lower_key,
        max_key=upper_key,
        fallback_min=fallback_lower,
        fallback_max=fallback_upper,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
