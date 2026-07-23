"""Defaults and neutral constants for the multiseries chart scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import resolve_required_int_bounds, split_scene_generation_rendering_prompt_defaults
from ...shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    sample_chart_font_family as sample_shared_chart_font_family,
)
from .state import MultiseriesChartDefaults


DOMAIN = "charts"
SCENE_ID = "multiseries"
SCENE_NAMESPACE = "charts_multiseries"
PROMPT_BUNDLE_ID = "charts_multiseries_v1"


SUPPORTED_CHANGE_MEASURES: Tuple[str, ...] = ("directional_change", "absolute_gap")
SUPPORTED_RATIO_MEASURES: Tuple[str, ...] = ("series_share", "pair_ratio")
SUPPORTED_CHANGE_DIRECTIONS: Tuple[str, ...] = ("increase", "decrease")
SUPPORTED_EXTREMUM_DIRECTIONS: Tuple[str, ...] = ("largest", "smallest")
SUPPORTED_COMPARISONS: Tuple[str, ...] = ("greater_than", "less_than")

DEFAULTS = MultiseriesChartDefaults()
_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

SCENE_VARIANT_LOADS: Dict[str, float] = {
    "grouped_bar": 0.0,
    "grouped_horizontal_bar": 0.50,
    "grouped_lollipop": 0.55,
    "multi_line": 1.0,
}
FAMILY_RANGE_KEYS: Tuple[str, ...] = (
    "category_count_min",
    "category_count_max",
    "series_count_min",
    "series_count_max",
    "value_min",
    "value_max",
)


def sample_chart_font_family(instance_seed: int, params: Mapping[str, Any]) -> str:
    """Sample the shared chart text font for this multiseries render."""

    return sample_shared_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )


def resolve_category_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    context_label: str,
) -> Tuple[int, int]:
    """Resolve inclusive category-count bounds."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="category_count_min",
        max_key="category_count_max",
        fallback_min=int(defaults.category_count_min),
        fallback_max=int(defaults.category_count_max),
        context=f"generation defaults for {context_label}",
    )


def resolve_series_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    context_label: str,
) -> Tuple[int, int]:
    """Resolve inclusive series-count bounds."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="series_count_min",
        max_key="series_count_max",
        fallback_min=int(defaults.series_count_min),
        fallback_max=int(defaults.series_count_max),
        context=f"generation defaults for {context_label}",
    )


__all__ = [
    "DEFAULTS",
    "DOMAIN",
    "GEN_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT_LOADS",
    "SUPPORTED_CHANGE_DIRECTIONS",
    "SUPPORTED_CHANGE_MEASURES",
    "SUPPORTED_COMPARISONS",
    "SUPPORTED_EXTREMUM_DIRECTIONS",
    "SUPPORTED_RATIO_MEASURES",
    "resolve_category_count_bounds",
    "resolve_series_count_bounds",
    "sample_chart_font_family",
]
