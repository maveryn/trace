"""Configuration and deterministic default helpers for radar charts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults
from ...shared.visual_defaults import (
    chart_font_asset_metadata,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    render_style_seed,
    sample_chart_font_family as sample_shared_chart_font_family,
    resolve_chart_render_rgb,
)

from .state import PROMPT_BUNDLE_ID, RGB, SCENE_ID, SCENE_NAMESPACE


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)
PANEL_LABEL_SUPPORT: tuple[str, ...] = tuple("ABCDEFGH")


def sample_chart_font_family(instance_seed: int, params: Mapping[str, Any]) -> str:
    return sample_shared_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )


def font_assets_payload(*, chart_font_family: str) -> dict[str, str]:
    return chart_font_asset_metadata(str(chart_font_family))


def prompt_bundle_id() -> str:
    return str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID))


def resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), RENDER_DEFAULTS.get(str(key), int(fallback))))


def resolve_gen_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def resolve_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_chart_render_rgb(params, RENDER_DEFAULTS, str(key), fallback, namespace=SCENE_NAMESPACE)


def profile_palette(params: Mapping[str, Any]) -> tuple[RGB, ...]:
    raw = params.get("profile_palette_rgb", RENDER_DEFAULTS.get("profile_palette_rgb", ()))
    colors: list[RGB] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 3:
                colors.append(tuple(max(0, min(255, int(channel))) for channel in item[:3]))  # type: ignore[index]
    if colors:
        return tuple(colors)
    return (
        (41, 108, 179),
        (205, 82, 74),
        (55, 148, 104),
        (139, 92, 186),
        (214, 139, 44),
        (54, 148, 168),
    )


def bbox(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values]
