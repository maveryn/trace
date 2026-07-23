"""Shared chart-scene histogram renderers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ...shared.text_rendering import draw_text_centered, load_font
from .chart_scene_types import (
    BoxPlotSpec,
    ChartColor,
    ChartMarkSpec,
    ChartRenderParams,
    HistogramBinSpec,
    MultiSeriesChartMarkSpec,
    RenderedChartScene,
    SUPPORTED_CHART_SCENE_VARIANTS,
    SUPPORTED_COMPOSITION_CHART_SCENE_VARIANTS,
    SUPPORTED_DISTRIBUTION_CHART_SCENE_VARIANTS,
    SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS,
    ViolinPlotSpec,
)
from .chart_scene_primitives import (
    _axis_slot_centers,
    _axis_ticks,
    _bbox_from_points,
    _clamp_text_center_to_canvas,
    _darken_rgb,
    _draw_styled_line,
    _draw_violin_polygon,
    _guide_lines_enabled,
    _label_center_for_variant,
    _radar_polygon_points,
    _resolve_plot_bbox,
    _resolve_value_axis,
    _scatter_slot_centers,
    _slot_centers,
    _text_bbox,
    _text_size,
    _tick_x,
    _tick_y,
    _union_bboxes,
    _violin_palette_color,
    resolve_chart_render_params,
    value_axis_render_metadata,
)

def render_histogram_scene(
    background: Image.Image,
    *,
    bins: Sequence[HistogramBinSpec],
    render_params: ChartRenderParams,
) -> RenderedChartScene:
    """Render one contiguous-bin histogram scene."""

    if len(bins) < 2:
        raise ValueError("histograms require at least two bins")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = _resolve_plot_bbox(render_params)
    plot_bbox = (int(plot_left), int(plot_top), int(plot_right), int(plot_bottom))
    draw.rectangle(plot_bbox, fill=render_params.plot_fill_rgb)

    y_axis_min, y_axis_max, y_ticks, y_minor_ticks, value_axis_window_enabled = _resolve_value_axis(
        [int(bin_spec.count) for bin_spec in bins],
        render_params=render_params,
    )
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_stroke_width = max(0, int(render_params.label_stroke_width_px))
    axis_color = tuple(int(value) for value in render_params.axis_color_rgb)
    grid_color = tuple(int(value) for value in render_params.grid_color_rgb)

    major_tick_values = set(int(value) for value in y_ticks)
    for tick_value in y_minor_ticks:
        y_px = _tick_y(
            int(tick_value),
            y_axis_min=int(y_axis_min),
            y_axis_max=int(y_axis_max),
            plot_top=int(plot_top),
            plot_bottom=int(plot_bottom),
        )
        draw.line(
            [(float(plot_left), float(y_px)), (float(plot_right), float(y_px))],
            fill=grid_color,
            width=int(render_params.grid_line_width_px),
        )
        if int(tick_value) in major_tick_values:
            draw.line(
                [
                    (float(plot_left) - float(render_params.tick_length_px), float(y_px)),
                    (float(plot_left), float(y_px)),
                ],
                fill=axis_color,
                width=int(render_params.axis_line_width_px),
            )
            draw_text_centered(
                draw,
                text=str(tick_value),
                center=(
                    float(plot_left) - float(render_params.tick_length_px) - 18.0,
                    float(y_px),
                ),
                font=tick_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=label_stroke_width,
            )

    draw.line(
        [(float(plot_left), float(plot_top)), (float(plot_left), float(plot_bottom))],
        fill=axis_color,
        width=int(render_params.axis_line_width_px),
    )
    draw.line(
        [(float(plot_left), float(plot_bottom)), (float(plot_right), float(plot_bottom))],
        fill=axis_color,
        width=int(render_params.axis_line_width_px),
    )

    plot_width = float(max(1, int(plot_right) - int(plot_left)))
    slot_width = float(plot_width / max(1, len(bins)))
    mark_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    guide_lines: List[Dict[str, Any]] = []
    draw_guides = _guide_lines_enabled(render_params)
    for index, bin_spec in enumerate(bins):
        left = float(plot_left) + float(index) * float(slot_width)
        right = float(plot_left) + float(index + 1) * float(slot_width)
        top = _tick_y(
            int(bin_spec.count),
            y_axis_min=int(y_axis_min),
            y_axis_max=int(y_axis_max),
            plot_top=int(plot_top),
            plot_bottom=int(plot_bottom),
        )
        bar_bbox = (float(left), float(top), float(right), float(plot_bottom))
        fill_rgb = (
            tuple(int(channel) for channel in bin_spec.fill_rgb)
            if isinstance(bin_spec.fill_rgb, tuple)
            else tuple(int(value) for value in render_params.mark_fill_rgb)
        )
        outline_rgb = (
            tuple(int(channel) for channel in bin_spec.outline_rgb)
            if isinstance(bin_spec.outline_rgb, tuple)
            else tuple(int(value) for value in render_params.mark_outline_rgb)
        )
        draw.rectangle(
            bar_bbox,
            fill=fill_rgb,
            outline=outline_rgb,
            width=int(render_params.mark_outline_width_px),
        )
        x_center = 0.5 * float(left + right)
        if bool(draw_guides):
            guide_points = [(float(plot_left), float(top)), (float(x_center), float(top))]
            _draw_styled_line(
                draw,
                guide_points,
                fill=render_params.guide_line_color_rgb,
                width=int(render_params.guide_line_width_px),
                style=str(render_params.guide_line_style),
            )
            guide_lines.append(
                {
                    "entity_id": f"bin_{index}",
                    "label": str(bin_spec.label),
                    "value": int(bin_spec.count),
                    "orientation": "horizontal",
                    "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in guide_points],
                    "style": str(render_params.guide_line_style),
                }
            )
        label_center = (
            float(x_center),
            float(plot_bottom) + float(max(18, int(render_params.label_font_size_px) + 6)),
        )
        draw_text_centered(
            draw,
            text=str(bin_spec.label),
            center=label_center,
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=label_stroke_width,
        )
        label_bbox = _text_bbox(draw, text=str(bin_spec.label), center=label_center, font=tick_font)
        mark_center = (float(x_center), float(0.5 * (float(top) + float(plot_bottom))))
        mark_trace = {
            "entity_id": f"bin_{index}",
            "label": str(bin_spec.label),
            "value": int(bin_spec.count),
            "x_rank": int(index),
            "interval_start": int(bin_spec.interval_start),
            "interval_end": int(bin_spec.interval_end),
            "mark_center_px": [round(float(mark_center[0]), 3), round(float(mark_center[1]), 3)],
            "mark_bbox_px": [round(float(value), 3) for value in bar_bbox],
            "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
            "label_bbox_px": [round(float(value), 3) for value in label_bbox],
            "mark_fill_rgb": [int(channel) for channel in fill_rgb],
            "mark_outline_rgb": [int(channel) for channel in outline_rgb],
        }
        mark_traces.append(mark_trace)
        entities.append(
            {
                "entity_id": str(mark_trace["entity_id"]),
                "entity_type": "bin",
                "attrs": {
                    "label": str(bin_spec.label),
                    "count": int(bin_spec.count),
                    "x_rank": int(index),
                    "interval_start": int(bin_spec.interval_start),
                    "interval_end": int(bin_spec.interval_end),
                    "scene_variant": "histogram",
                    "mark_center_px": list(mark_trace["mark_center_px"]),
                    "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                    "label_center_px": list(mark_trace["label_center_px"]),
                    "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                    "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                },
            }
        )

    return RenderedChartScene(
        image=image,
        mark_traces=tuple(dict(item) for item in mark_traces),
        entities=tuple(dict(item) for item in entities),
        plot_bbox_px=tuple(int(value) for value in plot_bbox),
        y_axis_max=int(y_axis_max),
        y_ticks=tuple(int(value) for value in y_ticks),
        scene_variant="histogram",
        value_axis_min=int(y_axis_min),
        value_axis_max=int(y_axis_max),
        value_axis_span=int(y_axis_max) - int(y_axis_min),
        value_axis_major_ticks=tuple(int(value) for value in y_ticks),
        value_axis_minor_ticks=tuple(int(value) for value in y_minor_ticks),
        value_axis_window_enabled=bool(value_axis_window_enabled),
        guide_line_style=str(render_params.guide_line_style if guide_lines else "none"),
        guide_lines=tuple(dict(item) for item in guide_lines),
    )


__all__ = [
    'render_histogram_scene',
]
