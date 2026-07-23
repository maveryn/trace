"""Configuration defaults and render parameter resolution for Sankey charts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
    sample_chart_font_family,
)
from trace_tasks.tasks.charts.shared.flow import resolve_flow_required_int_bounds
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb

from .state import RGB, SCENE_ID, SCENE_NAMESPACE, FlowRenderParams


SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {}
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)

TITLE_OPTIONS: tuple[str, ...] = (
    "Program Flow Summary",
    "Budget Transfer Sankey",
    "Pipeline Flow Paths",
    "Resource Routing Diagram",
    "Channel Flow Map",
)

FLOW_PALETTE_RGB: tuple[RGB, ...] = (
    (51, 113, 176),
    (204, 103, 79),
    (68, 153, 112),
    (139, 111, 190),
    (208, 151, 57),
    (67, 139, 160),
    (176, 89, 136),
    (95, 127, 66),
    (62, 102, 148),
    (190, 126, 84),
    (105, 151, 190),
    (160, 111, 74),
)


def render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def rgb_param(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )


def int_param(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), RENDER_DEFAULTS.get(str(key), int(fallback))))


def float_param(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), RENDER_DEFAULTS.get(str(key), float(fallback))))


def gen_int_param(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), GEN_DEFAULTS.get(str(key), int(fallback))))


def required_int_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    context: str,
) -> tuple[int, int]:
    return resolve_flow_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )


def resolve_render_params(params: Mapping[str, Any]) -> FlowRenderParams:
    """Resolve visual parameters for the single Sankey panel and its value labels."""

    outer = int_param(params, "outer_margin_px", 36)
    jitter_left, _jitter_right, jitter_top, _jitter_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(outer),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    return FlowRenderParams(
        canvas_width=int_param(params, "canvas_width", 1180),
        canvas_height=int_param(params, "canvas_height", 760),
        outer_margin_px=int(outer),
        panel_padding_px=int_param(params, "panel_padding_px", 28),
        title_band_height_px=int_param(params, "title_band_height_px", 62),
        node_width_px=int_param(params, "node_width_px", 96),
        node_height_px=int_param(params, "node_height_px", 44),
        node_border_width_px=int_param(params, "node_border_width_px", 2),
        port_separation_px=max(14, int_param(params, "port_separation_px", 34)),
        shared_pair_lane_gap_px=max(0, int_param(params, "shared_pair_lane_gap_px", 72)),
        min_flow_width_px=int_param(params, "min_flow_width_px", 6),
        max_flow_width_px=int_param(params, "max_flow_width_px", 18),
        value_label_font_size_px=int_param(params, "value_label_font_size_px", 19),
        value_label_gap_px=max(0, int_param(params, "value_label_gap_px", 8)),
        source_middle_label_t=max(0.15, min(0.85, float_param(params, "source_middle_label_t", 0.34))),
        middle_target_label_t=max(0.15, min(0.85, float_param(params, "middle_target_label_t", 0.66))),
        node_label_font_size_px=int_param(params, "node_label_font_size_px", 24),
        title_font_size_px=int_param(params, "title_font_size_px", 29),
        panel_fill_rgb=rgb_param(params, "panel_fill_rgb", (252, 253, 251)),
        panel_border_rgb=rgb_param(params, "panel_border_rgb", (70, 80, 90)),
        plot_fill_rgb=rgb_param(params, "plot_fill_rgb", (255, 255, 255)),
        node_fill_rgb=rgb_param(params, "node_fill_rgb", (54, 63, 74)),
        node_border_rgb=rgb_param(params, "node_border_rgb", (30, 38, 46)),
        node_text_rgb=rgb_param(params, "node_text_rgb", (255, 255, 255)),
        value_label_fill_rgb=rgb_param(params, "value_label_fill_rgb", (255, 255, 255)),
        value_label_border_rgb=rgb_param(params, "value_label_border_rgb", (82, 88, 96)),
        value_label_text_rgb=rgb_param(params, "value_label_text_rgb", (28, 34, 42)),
        title_color_rgb=rgb_param(params, "title_color_rgb", (32, 38, 46)),
        flow_alpha=max(40, min(220, int_param(params, "flow_alpha", 138))),
        layout_offset_x_px=int(jitter_left) - int(outer),
        layout_offset_y_px=int(jitter_top) - int(outer),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def sample_font_family(instance_seed: int, params: Mapping[str, Any]) -> str:
    return str(
        sample_chart_font_family(
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.chart_font",
            params=params,
        )
    )


def font_assets_payload(*, chart_font_family: str) -> dict[str, str]:
    return chart_font_asset_metadata(str(chart_font_family))
