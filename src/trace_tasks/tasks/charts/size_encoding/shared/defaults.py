"""Configuration defaults and resolvers for the size-encoding scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults
from ...shared.labeled_chart_variants import resolve_chart_axis_variant
from ...shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    resolve_chart_render_rgb,
)
from .state import RGB, SCENE_ID, SCENE_NAMESPACE, SUPPORTED_SCENE_VARIANTS


_SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

SCENE_VARIANT_LOADS = {
    "rect_word_cloud": 0.22,
    "circle_word_cloud": 0.34,
    "packed_bubble_cloud": 0.40,
    "small_multiple_bubble_cloud": 0.72,
}


def resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    supported_variants: Sequence[str] | None = None,
) -> tuple[str, dict[str, float]]:
    """Resolve the scene's visual grammar variant from scene-level defaults."""

    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=tuple(supported_variants or SUPPORTED_SCENE_VARIANTS),
        task_id=SCENE_NAMESPACE,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def resolve_rgb(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> RGB:
    return resolve_chart_render_rgb(params, RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)


def category_palette(params: Mapping[str, Any], category_count: int) -> tuple[RGB, ...]:
    raw = params.get("category_palette_rgb", group_default(RENDER_DEFAULTS, "category_palette_rgb", []))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < int(category_count):
        raise ValueError("category_palette_rgb must contain enough RGB colors")
    colors: list[RGB] = []
    for value in raw[: int(category_count)]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
            raise ValueError("category_palette_rgb entries must be RGB sequences")
        colors.append((int(value[0]), int(value[1]), int(value[2])))
    return tuple(colors)
