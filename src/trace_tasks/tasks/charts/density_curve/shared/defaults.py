"""Defaults and constants for density-curve chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.distribution.config import DistributionChartDefaults
from trace_tasks.tasks.charts.shared.labeled_chart_defaults import LabeledChartDefaults
from trace_tasks.tasks.charts.shared.labeled_chart_render_params import resolve_chart_render_params_for_task
from trace_tasks.tasks.charts.shared.visual_defaults import load_chart_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.color_distance import VISIBILITY_SAFE_FALLBACK_COLORS


SCENE_NAMESPACE = "charts.density_curve"
SCENE_ID = "density_curve"
SCENE_VARIANT = "density_curve"

SUPPORTED_DENSITY_FAMILIES: Tuple[str, ...] = (
    "gaussian",
    "gaussian_mixture_2",
    "gaussian_mixture_3",
    "skewed_unimodal",
    "lognormal_like",
    "student_t_like",
    "asymmetric_bimodal",
)
SUPPORTED_CURVE_LINE_STYLES: Tuple[str, ...] = ("solid", "dash", "dot")
TRACE_SAFE_DENSITY_CURVE_PALETTE_RGB: Tuple[Tuple[int, int, int], ...] = tuple(
    tuple(int(channel) for channel in color)
    for color in VISIBILITY_SAFE_FALLBACK_COLORS
    if tuple(int(channel) for channel in color) not in {(255, 255, 255), (0, 0, 0)}
)
DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E = 50.0

_SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
_DISTRIBUTION_DEFAULTS = DistributionChartDefaults()
_RENDER_DEFAULTS_FALLBACK = LabeledChartDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.0,
)


def gen_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer generation default."""

    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def gen_float(params: Mapping[str, Any], key: str, fallback: float) -> float:
    """Resolve one float generation default."""

    return float(params.get(str(key), group_default(GEN_DEFAULTS, str(key), float(fallback))))


def render_default(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve one rendering default."""

    return params.get(str(key), group_default(RENDER_DEFAULTS, str(key), fallback))


def resolve_density_curve_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> Any:
    """Resolve chart render parameters for density-curve scenes."""

    return resolve_chart_render_params_for_task(
        params,
        render_defaults=RENDER_DEFAULTS,
        defaults=_RENDER_DEFAULTS_FALLBACK,
        instance_seed=int(instance_seed),
    )


def density_curve_count_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Return configured density-curve count bounds."""

    return (
        int(gen_int(params, "density_curve_count_min", 4)),
        int(gen_int(params, "density_curve_count_max", 6)),
    )
