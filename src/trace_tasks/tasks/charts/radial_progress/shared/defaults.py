"""Configuration defaults for radial-progress chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    sample_chart_font_family as sample_shared_chart_font_family,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import resolve_render_rgb

from .state import RGB, SCENE_ID, SCENE_NAMESPACE


BASE_CONFIG_ID = "charts_radial_progress_base"

SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=BASE_CONFIG_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.2)


def generation_default(key: str, fallback: Any) -> Any:
    return group_default(GEN_DEFAULTS, str(key), fallback)


def rendering_default(key: str, fallback: Any) -> Any:
    return group_default(RENDER_DEFAULTS, str(key), fallback)


def prompt_bundle_id() -> str:
    return str(PROMPT_DEFAULTS.get("bundle_id", "charts_radial_progress_v1"))


def int_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="generation defaults for radial_progress",
    )


def render_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), rendering_default(str(key), int(fallback))))


def render_rgb(
    params: Mapping[str, Any],
    key: str,
    fallback: list[int],
    *,
    instance_seed: int,
) -> RGB:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        list(fallback),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )


def support_probability_map(values: list[int] | tuple[int | str, ...] | range) -> dict[str, float]:
    support = [str(value) for value in values]
    if not support:
        return {}
    weight = 1.0 / float(len(support))
    return {str(value): float(weight) for value in support}


def sample_chart_font_family(instance_seed: int, params: Mapping[str, Any]) -> str:
    return sample_shared_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )


def font_assets_payload(*, chart_font_family: str) -> dict[str, str]:
    return chart_font_asset_metadata(str(chart_font_family))
