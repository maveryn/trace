"""Configuration/default helpers for treemap charts."""

from __future__ import annotations

from collections.abc import Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb

from .state import PROMPT_BUNDLE_ID, RGB, SCENE_ID, SCENE_NAMESPACE


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_range(
    params: Mapping[str, object],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context="treemap generation defaults",
    )


def render_int(params: Mapping[str, object], key: str, fallback: int, *, instance_seed: int) -> int:
    return int(
        resolve_render_int(
            params,
            RENDERING_DEFAULTS,
            str(key),
            int(fallback),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        )
    )


def render_rgb(params: Mapping[str, object], key: str, fallback: RGB, *, instance_seed: int) -> RGB:
    return resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        str(key),
        tuple(int(value) for value in fallback),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )


def render_default(params: Mapping[str, object], key: str, fallback: object) -> object:
    return params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), fallback))


def prompt_bundle_id() -> str:
    return str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID))


def select_chart_font_family(instance_seed: int, params: Mapping[str, object]) -> str:
    return str(
        sample_chart_font_family(
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.chart_font",
            params=params,
        )
    )


__all__ = [
    "BACKGROUND_DEFAULTS",
    "GENERATION_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "SCENE_DEFAULTS",
    "chart_font_asset_metadata",
    "generation_range",
    "prompt_bundle_id",
    "render_default",
    "render_int",
    "render_rgb",
    "select_chart_font_family",
]
