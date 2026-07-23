"""Config and render defaults for dashboard chart tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.color_distance import coerce_rgb as _rgb
from trace_tasks.tasks.shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb
from trace_tasks.tasks.charts.shared.visual_defaults import load_chart_scene_background_defaults, load_chart_scene_noise_defaults

from .state import RGB, SCENE_ID, SCENE_NAMESPACE, RenderParams


_SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    **{"task" + "_" + "id": SCENE_NAMESPACE},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_default(key: str, fallback: Any) -> Any:
    return group_default(_GEN_DEFAULTS, str(key), fallback)


def render_default(key: str, fallback: Any) -> Any:
    return group_default(_RENDER_DEFAULTS, str(key), fallback)


def prompt_default(key: str, fallback: Any) -> Any:
    return group_default(_PROMPT_DEFAULTS, str(key), fallback)


def render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def render_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        _RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )


def resolve_context_text_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve dashboard context-text render defaults plus caller overrides."""

    keys = (
        "chart_context_profile",
        "context_text_profile",
        "context_text_enabled",
        "chart_context_mode",
        "chart_context_mode_weights",
        "context_text_mode",
        "context_text_mode_weights",
        "context_text_layout_mode",
        "context_text_placement",
        "context_text_placement_weights",
        "context_text_box_count_min",
        "context_text_box_count_max",
        "context_text_top_reserved_px",
        "context_text_bottom_reserved_px",
        "context_text_left_margin_px",
        "context_text_right_margin_px",
        "context_text_sidebar_width_px",
        "context_text_sidebar_width_min_px",
        "context_text_sidebar_width_max_px",
        "context_text_sidebar_gap_px",
        "context_text_bottom_band_height_min_px",
        "context_text_bottom_band_height_max_px",
        "context_text_bottom_band_gap_px",
        "dashboard_title_enabled",
        "dashboard_title_drop_probability",
    )
    resolved: Dict[str, Any] = {}
    for key in keys:
        if str(key) in params:
            resolved[str(key)] = params[str(key)]
        else:
            try:
                resolved[str(key)] = group_default(_RENDER_DEFAULTS, str(key), None)
            except Exception:
                resolved[str(key)] = None
    return {str(key): value for key, value in resolved.items() if value is not None}


def _rgb_sequence(value: Any, fallback: Sequence[RGB]) -> Tuple[RGB, ...]:
    if not isinstance(value, Sequence):
        return tuple(tuple(item) for item in fallback)
    colors: list[RGB] = []
    for item in value:
        if isinstance(item, Sequence) and len(item) >= 3:
            colors.append(_rgb(item, (0, 0, 0)))
    return tuple(colors or [tuple(item) for item in fallback])


def resolve_render_params(params: Mapping[str, Any]) -> RenderParams:
    """Resolve dashboard rendering knobs without public task identity."""

    dashboard_margin = int(params.get("dashboard_margin_px", render_default("dashboard_margin_px", 24)))
    jitter_left, _jitter_right, jitter_top, _jitter_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(dashboard_margin),
        right_px=int(dashboard_margin),
        top_px=int(dashboard_margin),
        bottom_px=int(dashboard_margin),
        params=params,
        defaults=_RENDER_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    palette_fallback: Tuple[RGB, ...] = (
        (35, 99, 180), (218, 88, 72), (54, 145, 95), (140, 88, 184),
        (218, 143, 44), (72, 156, 190), (187, 84, 130), (106, 122, 58),
        (98, 112, 214), (35, 157, 170), (197, 104, 38), (82, 121, 111),
        (165, 78, 168), (126, 133, 39), (186, 70, 98), (69, 132, 204),
    )
    return RenderParams(
        canvas_width=int(params.get("canvas_width", render_default("canvas_width", 1260))),
        canvas_height=int(params.get("canvas_height", render_default("canvas_height", 1000))),
        panel_gap_px=int(params.get("panel_gap_px", render_default("panel_gap_px", 18))),
        dashboard_margin_px=int(dashboard_margin),
        title_height_px=int(params.get("title_height_px", render_default("title_height_px", 44))),
        panel_padding_px=int(params.get("panel_padding_px", render_default("panel_padding_px", 16))),
        panel_border_width_px=int(params.get("panel_border_width_px", render_default("panel_border_width_px", 2))),
        axis_line_width_px=int(params.get("axis_line_width_px", render_default("axis_line_width_px", 2))),
        grid_line_width_px=int(params.get("grid_line_width_px", render_default("grid_line_width_px", 1))),
        bar_min_height_px=int(params.get("bar_min_height_px", render_default("bar_min_height_px", 12))),
        point_radius_px=int(params.get("point_radius_px", render_default("point_radius_px", 5))),
        line_width_px=int(params.get("line_width_px", render_default("line_width_px", 3))),
        title_font_size_px=int(params.get("title_font_size_px", render_default("title_font_size_px", 24))),
        panel_title_font_size_px=int(params.get("panel_title_font_size_px", render_default("panel_title_font_size_px", 18))),
        label_font_size_px=int(params.get("label_font_size_px", render_default("label_font_size_px", 12))),
        value_font_size_px=int(params.get("value_font_size_px", render_default("value_font_size_px", 12))),
        tick_font_size_px=int(params.get("tick_font_size_px", render_default("tick_font_size_px", 11))),
        panel_fill_rgb=render_rgb(params, "panel_fill_rgb", (255, 255, 255)),
        panel_border_rgb=render_rgb(params, "panel_border_rgb", (200, 207, 216)),
        axis_color_rgb=render_rgb(params, "axis_color_rgb", (72, 78, 88)),
        grid_color_rgb=render_rgb(params, "grid_color_rgb", (228, 232, 238)),
        text_color_rgb=render_rgb(params, "text_color_rgb", (35, 40, 48)),
        muted_text_color_rgb=render_rgb(params, "muted_text_color_rgb", (90, 96, 108)),
        connector_color_rgb=render_rgb(params, "connector_color_rgb", (80, 87, 99)),
        donut_hole_fill_rgb=render_rgb(params, "donut_hole_fill_rgb", (255, 255, 255)),
        category_palette_rgb=_rgb_sequence(
            params.get("category_palette_rgb", render_default("category_palette_rgb", palette_fallback)),
            palette_fallback,
        ),
        font_family=sample_font_family(
            role="readout",
            instance_seed=render_style_seed(params),
            namespace=f"{SCENE_NAMESPACE}.chart_font",
            params=params,
            exclude_tags=("display",),
        ),
        layout_offset_x_px=int(jitter_left) - int(dashboard_margin),
        layout_offset_y_px=int(jitter_top) - int(dashboard_margin),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


__all__ = [
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "generation_default",
    "prompt_default",
    "render_default",
    "render_rgb",
    "render_style_seed",
    "resolve_context_text_params",
    "resolve_render_params",
]
