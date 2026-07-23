"""Defaults and resolvers for composition-panel charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb

from .state import SCENE_ID, SCENE_NAMESPACE, SUPPORTED_SCENE_VARIANTS


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

SCENE_VARIANT_LOADS: dict[str, float] = {
    "composition_pie_panels": 0.48,
    "composition_donut_panels": 0.55,
}


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        task_id=SCENE_NAMESPACE,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_count_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    min_value = int(params.get(str(min_key), group_default(GEN_DEFAULTS, str(min_key), int(fallback_min))))
    max_value = int(params.get(str(max_key), group_default(GEN_DEFAULTS, str(max_key), int(fallback_max))))
    if int(min_value) > int(max_value):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(min_value), int(max_value)


def resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def render_int(params: Mapping[str, Any], key: str, fallback: int, *, instance_seed: int) -> int:
    return resolve_render_int(
        params,
        RENDER_DEFAULTS,
        str(key),
        int(fallback),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )


def render_rgb(
    params: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    *,
    instance_seed: int,
) -> tuple[int, int, int]:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
