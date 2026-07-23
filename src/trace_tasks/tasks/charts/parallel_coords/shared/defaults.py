"""Defaults and neutral constants for the parallel-coordinates chart scene."""

from __future__ import annotations

from typing import Any, Mapping

from .....core.scene_config import get_scene_defaults
from ....shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from ...shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    render_style_seed,
    sample_chart_font_family as sample_shared_chart_font_family,
)
from .state import ParallelRenderParams, RGB


DOMAIN = "charts"
SCENE_ID = "parallel_coords"
SCENE_NAMESPACE = "charts.parallel_coords"
PROMPT_BUNDLE_ID = "charts_parallel_coords_v1"

SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = ("vertical_parallel_coordinates",)

SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15)

PROFILE_PALETTE: tuple[RGB, ...] = (
    (38, 101, 176),
    (216, 95, 2),
    (27, 158, 119),
    (117, 112, 179),
    (231, 41, 138),
    (102, 166, 30),
    (230, 171, 2),
    (166, 118, 29),
)


def generation_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Return generation param, scene default, then fallback."""

    return params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(generation_value(params, key, int(fallback)))


def rendering_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), int(fallback))))


def sample_chart_font_family(instance_seed: int, params: Mapping[str, Any]) -> str:
    """Sample the shared chart text font for one render."""

    return sample_shared_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )


def resolve_render_params(params: Mapping[str, Any]) -> ParallelRenderParams:
    """Resolve renderer params after layout jitter and color defaults."""

    from ....shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb

    left = rendering_int(params, "plot_margin_left_px", 168)
    right = rendering_int(params, "plot_margin_right_px", 168)
    top = rendering_int(params, "plot_margin_top_px", 116)
    bottom = rendering_int(params, "plot_margin_bottom_px", 118)
    left, right, top, bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(left),
        right_px=int(right),
        top_px=int(top),
        bottom_px=int(bottom),
        params=params,
        defaults=RENDERING_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )

    def rgb(key: str, fallback: RGB) -> RGB:
        return resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            str(key),
            fallback,
            instance_seed=render_style_seed(params),
            namespace=SCENE_NAMESPACE,
        )

    return ParallelRenderParams(
        canvas_width=rendering_int(params, "canvas_width", 1180),
        canvas_height=rendering_int(params, "canvas_height", 760),
        plot_margin_left_px=int(left),
        plot_margin_right_px=int(right),
        plot_margin_top_px=int(top),
        plot_margin_bottom_px=int(bottom),
        panel_fill_rgb=rgb("panel_fill_rgb", (255, 255, 255)),
        panel_border_rgb=rgb("panel_border_rgb", (194, 202, 214)),
        plot_fill_rgb=rgb("plot_fill_rgb", (252, 253, 255)),
        axis_rgb=rgb("axis_rgb", (78, 88, 102)),
        selected_axis_rgb=rgb("selected_axis_rgb", (26, 66, 118)),
        grid_rgb=rgb("grid_rgb", (226, 231, 238)),
        threshold_rgb=rgb("threshold_rgb", (204, 56, 60)),
        text_rgb=rgb("text_rgb", (32, 38, 48)),
        muted_text_rgb=rgb("muted_text_rgb", (89, 100, 116)),
        text_stroke_rgb=rgb("text_stroke_rgb", (255, 255, 255)),
        line_width_px=rendering_int(params, "line_width_px", 4),
        point_radius_px=rendering_int(params, "point_radius_px", 5),
        axis_line_width_px=rendering_int(params, "axis_line_width_px", 2),
        selected_axis_line_width_px=rendering_int(params, "selected_axis_line_width_px", 4),
        grid_line_width_px=rendering_int(params, "grid_line_width_px", 1),
        label_font_size_px=rendering_int(params, "label_font_size_px", 18),
        tick_font_size_px=rendering_int(params, "tick_font_size_px", 15),
        title_font_size_px=rendering_int(params, "title_font_size_px", 28),
        threshold_font_size_px=rendering_int(params, "threshold_font_size_px", 15),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


__all__ = [
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROFILE_PALETTE",
    "PROMPT_BUNDLE_ID",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "generation_int",
    "generation_value",
    "render_style_seed",
    "resolve_render_params",
    "sample_chart_font_family",
]
