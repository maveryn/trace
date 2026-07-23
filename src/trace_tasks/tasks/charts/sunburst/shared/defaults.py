"""Config/default helpers for the sunburst chart scene."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import resolve_render_int, resolve_render_rgb

from .state import DOMAIN, RGB, SCENE_ID, SCENE_NAMESPACE


SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))


def rendering_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), fallback))


def resolve_bounds(
    params: Mapping[str, Any],
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
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )


def int_sequence(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> tuple[int, ...]:
    raw = generation_value(params, str(key), list(fallback))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return tuple(int(value) for value in fallback)
    values = tuple(int(value) for value in raw)
    return values if values else tuple(int(value) for value in fallback)


def render_int(params: Mapping[str, Any], key: str, fallback: int, *, instance_seed: int) -> int:
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


def render_rgb(params: Mapping[str, Any], key: str, fallback: RGB, *, instance_seed: int) -> RGB:
    return resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        str(key),
        tuple(int(channel) for channel in fallback),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )


def canvas_size(params: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int(rendering_value(params, "canvas_width", 1224)),
        int(rendering_value(params, "canvas_height", 1032)),
    )


__all__ = [
    "BACKGROUND_DEFAULTS",
    "GENERATION_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "canvas_size",
    "generation_value",
    "int_sequence",
    "render_int",
    "render_rgb",
    "rendering_value",
    "resolve_bounds",
]
