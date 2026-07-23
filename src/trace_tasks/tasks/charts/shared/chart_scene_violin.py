"""Shared chart-scene violin renderers."""

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

def render_violin_scene(
    background: Image.Image,
    *,
    violins: Sequence[ViolinPlotSpec],
    render_params: ChartRenderParams,
) -> RenderedChartScene:
    """Render one categorical violin-plot scene."""

    if len(violins) < 2:
        raise ValueError("violin scenes require at least two categories")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = _resolve_plot_bbox(render_params)
    plot_bbox = (int(plot_left), int(plot_top), int(plot_right), int(plot_bottom))
    draw.rectangle(plot_bbox, fill=render_params.plot_fill_rgb)

    max_value = max(int(spec.support_max) for spec in violins)
    y_axis_max = max(4, int(max_value) + 1)
    y_ticks = _axis_ticks(y_axis_max)
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    axis_color = tuple(int(value) for value in render_params.axis_color_rgb)
    grid_color = tuple(int(value) for value in render_params.grid_color_rgb)

    for tick_value in y_ticks:
        y_px = _tick_y(
            int(tick_value),
            y_axis_max=int(y_axis_max),
            plot_top=int(plot_top),
            plot_bottom=int(plot_bottom),
        )
        draw.line(
            [(float(plot_left), float(y_px)), (float(plot_right), float(y_px))],
            fill=grid_color,
            width=int(render_params.grid_line_width_px),
        )
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
            stroke_width=max(1, int(round(0.06 * float(render_params.tick_font_size_px)))),
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

    centers = _slot_centers(count=len(violins), plot_left=int(plot_left), plot_right=int(plot_right))
    slot_width = float(max(1.0, (float(plot_right) - float(plot_left)) / max(1, len(violins))))
    width_scale = max(0.72, min(1.25, float(render_params.violin_width_scale)))
    smoothing_scale = max(0.70, min(1.35, float(render_params.violin_smoothing_scale)))
    fill_style = str(render_params.violin_fill_style).strip().lower()
    if fill_style not in {"solid", "light", "outline", "hatch"}:
        fill_style = "solid"
    mode_line_style = str(render_params.violin_mode_line_style).strip().lower()
    if mode_line_style not in {"full", "short", "dot", "none"}:
        mode_line_style = "full"
    palette_mode = str(render_params.violin_palette_mode).strip().lower()
    if palette_mode not in {"single", "per_violin_muted"}:
        palette_mode = "single"
    half_width = float(max(16.0, 0.44 * float(slot_width) * float(width_scale)))

    mark_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    for index, spec in enumerate(violins):
        x_center = float(centers[index])
        support_min = int(spec.support_min)
        support_max = int(spec.support_max)
        mode_values = [int(value) for value in spec.mode_values]
        support_span = max(2, int(support_max) - int(support_min))

        fill_rgb = (
            tuple(int(channel) for channel in spec.fill_rgb)
            if isinstance(spec.fill_rgb, tuple)
            else tuple(int(value) for value in render_params.mark_fill_rgb)
        )
        outline_rgb = (
            tuple(int(channel) for channel in spec.outline_rgb)
            if isinstance(spec.outline_rgb, tuple)
            else tuple(int(value) for value in render_params.mark_outline_rgb)
        )
        if palette_mode == "per_violin_muted":
            fill_rgb = _violin_palette_color(
                tuple(int(value) for value in render_params.mark_fill_rgb),
                index=int(index),
                offset=int(render_params.violin_palette_offset),
                plot_fill=tuple(int(value) for value in render_params.plot_fill_rgb),
            )
            outline_rgb = _darken_rgb(fill_rgb, 0.54)

        def _width_fraction(value: float) -> float:
            total = 0.16
            for mode in mode_values:
                sigma = max(0.9, 0.16 * float(support_span) * float(smoothing_scale))
                delta = (float(value) - float(mode)) / float(sigma)
                total += 0.52 * math.exp(-0.5 * float(delta * delta))
            if len(mode_values) >= 2:
                valley_center = 0.5 * float(mode_values[0] + mode_values[-1])
                valley_sigma = max(0.8, 0.12 * float(support_span) * float(smoothing_scale))
                valley_delta = (float(value) - float(valley_center)) / float(valley_sigma)
                total -= 0.20 * math.exp(-0.5 * float(valley_delta * valley_delta))
            return max(0.10, min(1.0, float(total)))

        sample_values = [float(support_min) + (float(step) / 40.0) * float(support_max - support_min) for step in range(41)]
        right_points: List[Tuple[float, float]] = []
        left_points: List[Tuple[float, float]] = []
        for sample_value in sample_values:
            y_px = _tick_y(
                int(round(sample_value)),
                y_axis_max=int(y_axis_max),
                plot_top=int(plot_top),
                plot_bottom=int(plot_bottom),
            )
            width = float(half_width) * float(_width_fraction(sample_value))
            right_points.append((float(x_center + width), float(y_px)))
            left_points.append((float(x_center - width), float(y_px)))
        polygon_points = right_points + list(reversed(left_points))
        _draw_violin_polygon(
            image,
            points=polygon_points,
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
            fill_style=str(fill_style),
            outline_width=int(render_params.mark_outline_width_px),
            plot_fill_rgb=tuple(int(value) for value in render_params.plot_fill_rgb),
        )
        draw.line(
            [(float(x_center), float(_tick_y(support_min, y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom)))),
             (float(x_center), float(_tick_y(support_max, y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))))],
            fill=outline_rgb,
            width=max(1, int(render_params.line_width_px) - 2),
        )
        mode_center_points: List[List[float]] = []
        for mode in mode_values:
            mode_y = _tick_y(int(mode), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
            mode_width = float(half_width) * float(_width_fraction(float(mode)))
            if mode_line_style == "short":
                draw.line(
                    [(float(x_center - 0.34 * mode_width), float(mode_y)), (float(x_center + 0.34 * mode_width), float(mode_y))],
                    fill=outline_rgb,
                    width=max(1, int(render_params.line_width_px) - 2),
                )
            elif mode_line_style == "dot":
                radius = max(2.0, 0.38 * float(render_params.point_radius_px))
                draw.ellipse(
                    [
                        float(x_center - radius),
                        float(mode_y - radius),
                        float(x_center + radius),
                        float(mode_y + radius),
                    ],
                    fill=outline_rgb,
                    outline=outline_rgb,
                )
            elif mode_line_style == "full":
                draw.line(
                    [(float(x_center - 0.78 * mode_width), float(mode_y)), (float(x_center + 0.78 * mode_width), float(mode_y))],
                    fill=outline_rgb,
                    width=max(1, int(render_params.line_width_px) - 1),
                )
            mode_center_points.append([round(float(x_center), 3), round(float(mode_y), 3)])

        label_center = (
            float(x_center),
            float(plot_bottom) + float(max(18, int(render_params.label_font_size_px) + 6)),
        )
        draw_text_centered(
            draw,
            text=str(spec.label),
            center=label_center,
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )
        label_bbox = _text_bbox(draw, text=str(spec.label), center=label_center, font=label_font)
        violin_bbox = _bbox_from_points(polygon_points)
        mark_center = [round(float(x_center), 3), round(float(0.5 * (violin_bbox[1] + violin_bbox[3])), 3)]
        mark_trace = {
            "entity_id": f"violin_{str(spec.label)}",
            "label": str(spec.label),
            "value": int(mode_values[-1]),
            "x_rank": int(index),
            "support_min": int(support_min),
            "support_max": int(support_max),
            "mode_values": [int(value) for value in mode_values],
            "mark_center_px": list(mark_center),
            "mark_bbox_px": [round(float(value), 3) for value in violin_bbox],
            "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
            "label_bbox_px": [round(float(value), 3) for value in label_bbox],
            "mode_center_points_px": [list(point) for point in mode_center_points],
            "mark_fill_rgb": [int(channel) for channel in fill_rgb],
            "mark_outline_rgb": [int(channel) for channel in outline_rgb],
            "violin_fill_style": str(fill_style),
            "violin_mode_line_style": str(mode_line_style),
            "violin_width_scale": round(float(width_scale), 4),
            "violin_smoothing_scale": round(float(smoothing_scale), 4),
            "violin_palette_mode": str(palette_mode),
            "violin_palette_offset": int(render_params.violin_palette_offset),
        }
        mark_traces.append(mark_trace)
        entities.append(
            {
                "entity_id": str(mark_trace["entity_id"]),
                "entity_type": "violin",
                "attrs": {
                    "label": str(spec.label),
                    "x_rank": int(index),
                    "scene_variant": "violin",
                    "support_min": int(support_min),
                    "support_max": int(support_max),
                    "mode_values": [int(value) for value in mode_values],
                    "mark_center_px": list(mark_trace["mark_center_px"]),
                    "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                    "label_center_px": list(mark_trace["label_center_px"]),
                    "mode_center_points_px": [list(point) for point in mark_trace["mode_center_points_px"]],
                    "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                    "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                    "violin_fill_style": str(mark_trace["violin_fill_style"]),
                    "violin_mode_line_style": str(mark_trace["violin_mode_line_style"]),
                    "violin_width_scale": float(mark_trace["violin_width_scale"]),
                    "violin_smoothing_scale": float(mark_trace["violin_smoothing_scale"]),
                    "violin_palette_mode": str(mark_trace["violin_palette_mode"]),
                    "violin_palette_offset": int(mark_trace["violin_palette_offset"]),
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
        scene_variant="violin",
    )


__all__ = [
    'render_violin_scene',
]
