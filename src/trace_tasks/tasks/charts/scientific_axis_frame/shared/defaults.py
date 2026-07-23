"""Config defaults for scientific axis-frame chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)
from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import (
    AxisFrameRenderParams,
    RGB,
    SCENE_ID,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import (
    apply_layout_jitter_to_margins,
    resolve_render_int,
    resolve_render_rgb,
)


_SCENE_DEFAULTS = get_scene_defaults("charts", SCENE_ID)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {}
)
BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def generation_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), fallback))


def rendering_value(params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    return params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), fallback))


def render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def render_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(
        resolve_render_int(
            params,
            RENDERING_DEFAULTS,
            str(key),
            int(fallback),
            instance_seed=render_style_seed(params),
            namespace=SCENE_NAMESPACE,
        )
    )


def render_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        str(key),
        fallback,
        instance_seed=render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )


def tick_count_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="axis_frame_tick_count_min",
        max_key="axis_frame_tick_count_max",
        fallback_min=4,
        fallback_max=8,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return max(3, int(low)), max(max(3, int(low)), int(high))


def tick_step_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="axis_frame_tick_step_min",
        max_key="axis_frame_tick_step_max",
        fallback_min=2,
        fallback_max=12,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return max(1, int(low)), max(max(1, int(low)), int(high))


def axis_start_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="axis_frame_start_min",
        max_key="axis_frame_start_max",
        fallback_min=-24,
        fallback_max=24,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return int(low), int(high)


def resolve_render_params(params: Mapping[str, Any], *, chart_font_family: str) -> AxisFrameRenderParams:
    """Resolve scene-level render knobs once for the numeric frame renderer.

    Invariant: this helper is visual-only and does not inspect public task or
    query identity; callers pass semantic render inputs separately.
    """

    canvas_width = render_int(params, "axis_frame_canvas_width", render_int(params, "canvas_width", 1120))
    canvas_height = render_int(params, "axis_frame_canvas_height", render_int(params, "canvas_height", 760))
    margins = {
        "left": render_int(params, "axis_frame_margin_left_px", 108),
        "right": render_int(params, "axis_frame_margin_right_px", 68),
        "top": render_int(params, "axis_frame_margin_top_px", 90),
        "bottom": render_int(params, "axis_frame_margin_bottom_px", 112),
    }
    left_px, right_px, top_px, bottom_px, jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margins["left"]),
        right_px=int(margins["right"]),
        top_px=int(margins["top"]),
        bottom_px=int(margins["bottom"]),
        params=params,
        defaults=RENDERING_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )
    return AxisFrameRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        margin_left_px=int(left_px),
        margin_right_px=int(right_px),
        margin_top_px=int(top_px),
        margin_bottom_px=int(bottom_px),
        axis_label_font_size_px=render_int(params, "axis_frame_axis_label_font_size_px", 17),
        tick_font_size_px=render_int(params, "axis_frame_tick_font_size_px", 16),
        axis_line_width_px=render_int(params, "axis_frame_axis_line_width_px", 2),
        grid_line_width_px=render_int(params, "axis_frame_grid_line_width_px", 1),
        line_width_px=render_int(params, "axis_frame_line_width_px", 3),
        point_radius_px=render_int(params, "axis_frame_point_radius_px", 4),
        text_rgb=render_rgb(params, "axis_frame_text_rgb", (38, 44, 54)),
        muted_text_rgb=render_rgb(params, "axis_frame_muted_text_rgb", (86, 94, 108)),
        text_stroke_rgb=render_rgb(params, "axis_frame_text_stroke_rgb", (255, 255, 255)),
        axis_rgb=render_rgb(params, "axis_frame_axis_rgb", (55, 62, 72)),
        grid_rgb=render_rgb(params, "axis_frame_grid_rgb", (218, 224, 232)),
        panel_fill_rgb=render_rgb(params, "axis_frame_panel_fill_rgb", (255, 255, 255)),
        panel_outline_rgb=render_rgb(params, "axis_frame_panel_outline_rgb", (188, 198, 212)),
        series_rgb=render_rgb(params, "axis_frame_series_rgb", (62, 102, 156)),
        marker_rgb=render_rgb(params, "axis_frame_marker_rgb", (203, 91, 60)),
        font_family=str(chart_font_family),
        layout_jitter_meta=dict(jitter_meta),
    )


__all__ = [
    "BACKGROUND_DEFAULTS",
    "GENERATION_DEFAULTS",
    "NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "axis_start_bounds",
    "generation_value",
    "render_int",
    "render_rgb",
    "render_style_seed",
    "rendering_value",
    "resolve_render_params",
    "tick_count_bounds",
    "tick_step_bounds",
]
