"""Config defaults for scatter-readout chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    render_style_seed,
    resolve_chart_render_float,
    resolve_chart_render_int,
    resolve_chart_render_rgb,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)

from .state import RGB, SCENE_ID, SCENE_NAMESPACE


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return resolve_chart_render_int(params, RENDER_DEFAULTS, str(key), int(fallback), namespace=SCENE_NAMESPACE)


def resolve_float(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return resolve_chart_render_float(params, RENDER_DEFAULTS, str(key), float(fallback), namespace=SCENE_NAMESPACE)


def resolve_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_chart_render_rgb(params, RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)


def gen_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def render_sequence(params: Mapping[str, Any], key: str, fallback: Sequence[Any]) -> tuple[Any, ...]:
    raw = params.get(str(key), group_default(RENDER_DEFAULTS, str(key), tuple(fallback)))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple(raw)
    return tuple(fallback)


__all__ = [
    "BACKGROUND_DEFAULTS",
    "GEN_DEFAULTS",
    "NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "gen_int",
    "group_default",
    "render_sequence",
    "render_style_seed",
    "resolve_float",
    "resolve_int",
    "resolve_rgb",
]
