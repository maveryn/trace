"""Shared chart-scene labeled single-series renderers."""

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

def render_labeled_chart_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    marks: Sequence[ChartMarkSpec],
    render_params: ChartRenderParams,
    instance_seed: int,
) -> RenderedChartScene:
    """Render one labeled chart scene onto `background`."""

    selected_variant = str(scene_variant)
    if selected_variant not in set(SUPPORTED_CHART_SCENE_VARIANTS):
        raise ValueError(f"unsupported chart scene_variant: {selected_variant}")
    if len(marks) < 2:
        raise ValueError("charts require at least two marks")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = _resolve_plot_bbox(render_params)
    plot_bbox = (int(plot_left), int(plot_top), int(plot_right), int(plot_bottom))
    draw.rectangle(plot_bbox, fill=render_params.plot_fill_rgb)

    visible_marks = [mark for mark in marks if bool(getattr(mark, "visible", True))]
    if not visible_marks:
        raise ValueError("charts require at least one visible mark")
    max_value = max(int(mark.value) for mark in visible_marks)
    y_axis_min, y_axis_max, y_ticks, y_minor_ticks, value_axis_window_enabled = _resolve_value_axis(
        [int(mark.value) for mark in visible_marks],
        render_params=render_params,
    )
    if selected_variant in {"pie", "donut", "radar"}:
        y_axis_min = 0
        y_axis_max = max(4, int(max_value) + 1)
        y_ticks = _axis_ticks(y_axis_max)
        y_minor_ticks = tuple(int(value) for value in y_ticks)
        value_axis_window_enabled = False

    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=bool(render_params.label_bold))
    axis_color = tuple(int(value) for value in render_params.axis_color_rgb)
    grid_color = tuple(int(value) for value in render_params.grid_color_rgb)

    if selected_variant in {"pie", "donut"}:
        plot_width = float(max(1, int(plot_right) - int(plot_left)))
        plot_height = float(max(1, int(plot_bottom) - int(plot_top)))
        legend_width = float(min(max(150.0, plot_width * 0.28), plot_width * 0.38))
        legend_gap = float(max(18.0, float(render_params.label_font_size_px)))
        chart_right = float(plot_right) - float(legend_width) - float(legend_gap)
        chart_width = float(max(140.0, chart_right - float(plot_left)))
        side = float(min(chart_width, plot_height))
        pie_margin = float(max(28.0, int(render_params.label_font_size_px) * 1.8))
        diameter = float(max(140.0, float(side) - float(pie_margin)))
        radius = 0.5 * float(diameter)
        hole_radius = float(radius * 0.42) if selected_variant == "donut" else 0.0
        center_x = float(plot_left) + (0.5 * float(chart_width))
        center_y = 0.5 * float(plot_top + plot_bottom)
        pie_bbox = (
            float(center_x - radius),
            float(center_y - radius),
            float(center_x + radius),
            float(center_y + radius),
        )
        total_value = int(sum(int(mark.value) for mark in marks))
        if int(total_value) <= 0:
            raise ValueError("pie charts require a positive total value")
        mark_traces: List[Dict[str, Any]] = []
        entities: List[Dict[str, Any]] = []
        start_angle = -90.0
        percentage_radius = (
            float(0.5 * (float(radius) + float(hole_radius)))
            if hole_radius > 0.0
            else float(radius) * 0.60
        )
        legend_left = float(chart_right + float(legend_gap))
        legend_row_height = float(max(render_params.label_font_size_px + 16, 42))
        legend_top = float(plot_top) + float(max(12.0, 0.5 * (plot_height - (len(marks) * legend_row_height))))
        legend_swatch_side = float(max(28, int(round(render_params.label_font_size_px * 1.05))))
        legend_frame_pad = float(max(3, int(round(render_params.mark_outline_width_px * 1.5))))
        legend_text_gap = float(max(16, int(render_params.label_font_size_px * 0.8)))
        legend_frame_fill = tuple(int(value) for value in render_params.plot_fill_rgb)
        for index, mark in enumerate(marks):
            sweep = 360.0 * (float(int(mark.value)) / float(total_value))
            end_angle = float(start_angle + sweep)
            fill_rgb = (
                tuple(int(channel) for channel in mark.fill_rgb)
                if isinstance(mark.fill_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_fill_rgb)
            )
            outline_rgb = (
                tuple(int(channel) for channel in mark.outline_rgb)
                if isinstance(mark.outline_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_outline_rgb)
            )
            draw.pieslice(
                pie_bbox,
                start=float(start_angle),
                end=float(end_angle),
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
            mid_angle = float(start_angle + (0.5 * sweep))
            theta = math.radians(float(mid_angle))
            mark_center = (
                float(center_x + (percentage_radius * math.cos(theta))),
                float(center_y + (percentage_radius * math.sin(theta))),
            )
            percent_radius = float(percentage_radius)
            if float(abs(sweep)) < 18.0:
                percent_radius = float(radius) + 16.0
            percent_center = (
                float(center_x + (percent_radius * math.cos(theta))),
                float(center_y + (percent_radius * math.sin(theta))),
            )
            percent_text = f"{int(mark.value)}"
            percent_center = _clamp_text_center_to_canvas(
                draw,
                text=str(percent_text),
                center=percent_center,
                font=label_font,
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
            )
            draw_text_centered(
                draw,
                text=str(percent_text),
                center=percent_center,
                font=label_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=int(render_params.label_stroke_width_px),
            )
            percent_bbox = _text_bbox(draw, text=str(percent_text), center=percent_center, font=label_font)

            legend_row_y = float(legend_top) + float(index) * float(legend_row_height)
            legend_swatch_bbox = (
                float(legend_left),
                float(legend_row_y),
                float(legend_left + legend_swatch_side),
                float(legend_row_y + legend_swatch_side),
            )
            legend_frame_bbox = (
                float(legend_swatch_bbox[0] - legend_frame_pad),
                float(legend_swatch_bbox[1] - legend_frame_pad),
                float(legend_swatch_bbox[2] + legend_frame_pad),
                float(legend_swatch_bbox[3] + legend_frame_pad),
            )
            draw.rectangle(
                legend_frame_bbox,
                fill=legend_frame_fill,
                outline=axis_color,
                width=max(1, int(render_params.mark_outline_width_px)),
            )
            draw.rectangle(
                legend_swatch_bbox,
                fill=fill_rgb,
                outline=outline_rgb,
                width=max(2, int(render_params.mark_outline_width_px)),
            )
            label_width, _ = _text_size(draw, text=str(mark.label), font=label_font)
            label_left = float(legend_frame_bbox[2]) + float(legend_text_gap)
            label_center = (
                float(label_left + (0.5 * label_width)),
                float(0.5 * (legend_swatch_bbox[1] + legend_swatch_bbox[3])),
            )
            draw_text_centered(
                draw,
                text=str(mark.label),
                center=label_center,
                font=label_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=int(render_params.label_stroke_width_px),
            )
            label_bbox = _text_bbox(draw, text=str(mark.label), center=label_center, font=label_font)

            step_count = max(2, int(math.ceil(abs(float(sweep)) / 18.0)))
            sector_points: List[Tuple[float, float]] = [(float(center_x), float(center_y))]
            for step in range(int(step_count) + 1):
                angle = float(start_angle + (float(step) / float(step_count)) * float(sweep))
                angle_rad = math.radians(float(angle))
                sector_points.append(
                    (
                        float(center_x + (radius * math.cos(angle_rad))),
                        float(center_y + (radius * math.sin(angle_rad))),
                    )
                )
            mark_bbox = _bbox_from_points(sector_points)
            mark_trace = {
                "entity_id": f"mark_{str(mark.label)}",
                "label": str(mark.label),
                "value": int(mark.value),
                "x_rank": int(index),
                "mark_center_px": [round(float(mark_center[0]), 3), round(float(mark_center[1]), 3)],
                "mark_bbox_px": [round(float(value), 3) for value in mark_bbox],
                "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
                "label_bbox_px": [round(float(value), 3) for value in label_bbox],
                "percentage_center_px": [round(float(percent_center[0]), 3), round(float(percent_center[1]), 3)],
                "percentage_bbox_px": [round(float(value), 3) for value in percent_bbox],
                "legend_swatch_bbox_px": [round(float(value), 3) for value in legend_swatch_bbox],
                "mark_fill_rgb": [int(channel) for channel in fill_rgb],
                "mark_outline_rgb": [int(channel) for channel in outline_rgb],
                "start_angle_deg": round(float(start_angle), 3),
                "end_angle_deg": round(float(end_angle), 3),
            }
            mark_traces.append(mark_trace)
            entities.append(
                {
                    "entity_id": str(mark_trace["entity_id"]),
                    "entity_type": "slice",
                    "attrs": {
                        "label": str(mark.label),
                        "value": int(mark.value),
                        "x_rank": int(index),
                        "scene_variant": str(selected_variant),
                        "mark_center_px": list(mark_trace["mark_center_px"]),
                        "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                        "label_center_px": list(mark_trace["label_center_px"]),
                        "label_bbox_px": list(mark_trace["label_bbox_px"]),
                        "legend_swatch_bbox_px": list(mark_trace["legend_swatch_bbox_px"]),
                        "percentage_center_px": list(mark_trace["percentage_center_px"]),
                        "percentage_bbox_px": list(mark_trace["percentage_bbox_px"]),
                        "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                        "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                        "start_angle_deg": float(mark_trace["start_angle_deg"]),
                        "end_angle_deg": float(mark_trace["end_angle_deg"]),
                    },
                }
            )
            start_angle = float(end_angle)

        if hole_radius > 0.0:
            inner_bbox = (
                float(center_x - hole_radius),
                float(center_y - hole_radius),
                float(center_x + hole_radius),
                float(center_y + hole_radius),
            )
            draw.ellipse(
                inner_bbox,
                fill=render_params.plot_fill_rgb,
                outline=render_params.mark_outline_rgb,
                width=max(1, int(render_params.mark_outline_width_px)),
            )

        return RenderedChartScene(
            image=image,
            mark_traces=tuple(dict(item) for item in mark_traces),
            entities=tuple(dict(item) for item in entities),
            plot_bbox_px=tuple(int(value) for value in plot_bbox),
            y_axis_max=int(y_axis_max),
            y_ticks=tuple(int(value) for value in y_ticks),
            scene_variant=str(selected_variant),
            value_axis_min=0,
            value_axis_max=int(y_axis_max),
            value_axis_span=int(y_axis_max),
            value_axis_major_ticks=tuple(int(value) for value in y_ticks),
            value_axis_minor_ticks=tuple(int(value) for value in y_ticks),
            value_axis_window_enabled=False,
        )

    if selected_variant == "radar":
        plot_width = float(max(1, int(plot_right) - int(plot_left)))
        plot_height = float(max(1, int(plot_bottom) - int(plot_top)))
        center_x = 0.5 * float(plot_left + plot_right)
        center_y = 0.5 * float(plot_top + plot_bottom)
        outer_padding = float(max(72.0, float(render_params.label_font_size_px) * 2.2))
        radius = 0.5 * float(min(plot_width, plot_height)) - float(outer_padding)
        radius = float(max(120.0, radius))
        count = len(marks)
        angles_rad = tuple(
            float((-math.pi / 2.0) + (2.0 * math.pi * float(index) / float(count)))
            for index in range(int(count))
        )

        radar_ring_ticks = sorted(
            {
                int(max(1, int(round(float(y_axis_max) * ratio))))
                for ratio in (1.0 / 3.0, 2.0 / 3.0, 1.0)
            }
        )

        for ring_value in radar_ring_ticks:
            ring_radius = float(radius) * (float(ring_value) / float(max(1, int(y_axis_max))))
            ring_points = _radar_polygon_points(
                center_x=float(center_x),
                center_y=float(center_y),
                radius=float(ring_radius),
                angles_rad=angles_rad,
            )
            draw.polygon(
                ring_points,
                outline=grid_color,
                width=max(1, int(render_params.grid_line_width_px)),
            )
            tick_center = (
                float(center_x),
                float(center_y - float(ring_radius)),
            )
            tick_center = _clamp_text_center_to_canvas(
                draw,
                text=str(ring_value),
                center=(float(tick_center[0]), float(tick_center[1]) - 12.0),
                font=tick_font,
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
            )
            draw_text_centered(
                draw,
                text=str(ring_value),
                center=tick_center,
                font=tick_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=max(1, int(round(0.06 * float(render_params.tick_font_size_px)))),
            )

        for angle in angles_rad:
            spoke_end = (
                float(center_x + (float(radius) * math.cos(float(angle)))),
                float(center_y + (float(radius) * math.sin(float(angle)))),
            )
            draw.line(
                [(float(center_x), float(center_y)), (float(spoke_end[0]), float(spoke_end[1]))],
                fill=grid_color,
                width=max(1, int(render_params.grid_line_width_px)),
            )

        polygon_points: List[Tuple[float, float]] = []
        mark_traces: List[Dict[str, Any]] = []
        entities: List[Dict[str, Any]] = []
        for index, (mark, angle) in enumerate(zip(marks, angles_rad)):
            fill_rgb = (
                tuple(int(channel) for channel in mark.fill_rgb)
                if isinstance(mark.fill_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_fill_rgb)
            )
            outline_rgb = (
                tuple(int(channel) for channel in mark.outline_rgb)
                if isinstance(mark.outline_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_outline_rgb)
            )
            radial_fraction = float(int(mark.value)) / float(max(1, int(y_axis_max)))
            point_radius = float(radius) * float(radial_fraction)
            x_center = float(center_x + (float(point_radius) * math.cos(float(angle))))
            y_center = float(center_y + (float(point_radius) * math.sin(float(angle))))
            polygon_points.append((float(x_center), float(y_center)))

            point_marker_radius = float(render_params.point_radius_px)
            ellipse_box = (
                float(x_center - point_marker_radius),
                float(y_center - point_marker_radius),
                float(x_center + point_marker_radius),
                float(y_center + point_marker_radius),
            )
            draw.ellipse(
                ellipse_box,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )

            value_center = (
                float(center_x + (max(18.0, float(point_radius) - 18.0) * math.cos(float(angle)))),
                float(center_y + (max(18.0, float(point_radius) - 18.0) * math.sin(float(angle)))),
            )
            value_center = _clamp_text_center_to_canvas(
                draw,
                text=str(mark.value),
                center=value_center,
                font=tick_font,
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
            )
            draw_text_centered(
                draw,
                text=str(mark.value),
                center=value_center,
                font=tick_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=max(1, int(round(0.06 * float(render_params.tick_font_size_px)))),
            )
            value_bbox = _text_bbox(draw, text=str(mark.value), center=value_center, font=tick_font)

            label_radius = float(radius) + float(max(28, int(render_params.label_font_size_px) + 8))
            label_center = (
                float(center_x + (float(label_radius) * math.cos(float(angle)))),
                float(center_y + (float(label_radius) * math.sin(float(angle)))),
            )
            label_center = _clamp_text_center_to_canvas(
                draw,
                text=str(mark.label),
                center=label_center,
                font=label_font,
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
            )
            draw_text_centered(
                draw,
                text=str(mark.label),
                center=label_center,
                font=label_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=int(render_params.label_stroke_width_px),
            )
            label_bbox = _text_bbox(draw, text=str(mark.label), center=label_center, font=label_font)

            mark_bbox = [float(value) for value in ellipse_box]
            mark_trace = {
                "entity_id": f"mark_{str(mark.label)}",
                "label": str(mark.label),
                "value": int(mark.value),
                "x_rank": int(index),
                "mark_center_px": [round(float(x_center), 3), round(float(y_center), 3)],
                "mark_bbox_px": [round(float(value), 3) for value in mark_bbox],
                "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
                "label_bbox_px": [round(float(value), 3) for value in label_bbox],
                "value_center_px": [round(float(value_center[0]), 3), round(float(value_center[1]), 3)],
                "value_bbox_px": [round(float(value), 3) for value in value_bbox],
                "mark_fill_rgb": [int(channel) for channel in fill_rgb],
                "mark_outline_rgb": [int(channel) for channel in outline_rgb],
            }
            mark_traces.append(mark_trace)
            entities.append(
                {
                    "entity_id": str(mark_trace["entity_id"]),
                    "entity_type": "point",
                    "attrs": {
                        "label": str(mark.label),
                        "value": int(mark.value),
                        "x_rank": int(index),
                        "scene_variant": str(selected_variant),
                        "mark_center_px": list(mark_trace["mark_center_px"]),
                        "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                        "label_center_px": list(mark_trace["label_center_px"]),
                        "value_center_px": list(mark_trace["value_center_px"]),
                        "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                        "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                    },
                }
            )

        if len(polygon_points) >= 2:
            draw.line(
                [*polygon_points, polygon_points[0]],
                fill=tuple(int(value) for value in render_params.mark_outline_rgb),
                width=max(1, int(render_params.line_width_px) - 1),
            )

        return RenderedChartScene(
            image=image,
            mark_traces=tuple(dict(item) for item in mark_traces),
            entities=tuple(dict(item) for item in entities),
            plot_bbox_px=tuple(int(value) for value in plot_bbox),
            y_axis_max=int(y_axis_max),
            y_ticks=tuple(int(value) for value in y_ticks),
            scene_variant=str(selected_variant),
            value_axis_min=0,
            value_axis_max=int(y_axis_max),
            value_axis_span=int(y_axis_max),
            value_axis_major_ticks=tuple(int(value) for value in y_ticks),
            value_axis_minor_ticks=tuple(int(value) for value in y_ticks),
            value_axis_window_enabled=False,
        )

    if selected_variant == "horizontal_bar":
        for tick_value in y_minor_ticks:
            x_px = _tick_x(
                int(tick_value),
                x_axis_min=int(y_axis_min),
                x_axis_max=int(y_axis_max),
                plot_left=int(plot_left),
                plot_right=int(plot_right),
            )
            draw.line(
                [(float(x_px), float(plot_top)), (float(x_px), float(plot_bottom))],
                fill=grid_color,
                width=int(render_params.grid_line_width_px),
            )
            if int(tick_value) in set(int(value) for value in y_ticks):
                draw.line(
                    [
                        (float(x_px), float(plot_bottom)),
                        (float(x_px), float(plot_bottom) + float(render_params.tick_length_px)),
                    ],
                    fill=axis_color,
                    width=int(render_params.axis_line_width_px),
                )
                tick_center = (
                    float(x_px),
                    float(plot_bottom) + float(render_params.tick_length_px) + 18.0,
                )
                draw_text_centered(
                    draw,
                    text=str(tick_value),
                    center=tick_center,
                    font=tick_font,
                    fill=render_params.text_color_rgb,
                    stroke_fill=render_params.text_stroke_rgb,
                    stroke_width=max(1, int(round(0.06 * float(render_params.tick_font_size_px)))),
                )
    else:
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
            if int(tick_value) in set(int(value) for value in y_ticks):
                draw.line(
                    [
                        (float(plot_left) - float(render_params.tick_length_px), float(y_px)),
                        (float(plot_left), float(y_px)),
                    ],
                    fill=axis_color,
                    width=int(render_params.axis_line_width_px),
                )
                tick_center = (
                    float(plot_left) - float(render_params.tick_length_px) - 18.0,
                    float(y_px),
                )
                draw_text_centered(
                    draw,
                    text=str(tick_value),
                    center=tick_center,
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

    base_centers = _slot_centers(count=len(marks), plot_left=int(plot_left), plot_right=int(plot_right))
    if selected_variant == "scatter":
        scatter_rng = spawn_rng(int(instance_seed), f"charts.scatter_slots.{selected_variant}")
        x_centers = _scatter_slot_centers(
            scatter_rng,
            count=len(marks),
            plot_left=int(plot_left),
            plot_right=int(plot_right),
        )
    else:
        x_centers = base_centers
    y_slot_centers = _axis_slot_centers(count=len(marks), start_px=int(plot_top), end_px=int(plot_bottom))

    line_points: List[Tuple[float, float]] = []
    mark_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    guide_lines: List[Dict[str, Any]] = []
    guide_color = tuple(int(value) for value in render_params.guide_line_color_rgb)
    draw_guides = _guide_lines_enabled(render_params)
    slot_width = float(max(1.0, (float(plot_right) - float(plot_left)) / max(1, len(marks))))
    bar_width = float(max(12.0, float(render_params.bar_width_fraction) * float(slot_width)))
    slot_height = float(max(1.0, (float(plot_bottom) - float(plot_top)) / max(1, len(marks))))
    horizontal_bar_height = float(max(12.0, float(render_params.bar_width_fraction) * float(slot_height)))

    for index, mark in enumerate(marks):
        mark_visible = bool(getattr(mark, "visible", True))
        bar_bbox: Tuple[float, float, float, float] | None = None
        if selected_variant == "horizontal_bar":
            y_center = float(y_slot_centers[index])
            x_extent = _tick_x(
                int(mark.value),
                x_axis_min=int(y_axis_min),
                x_axis_max=int(y_axis_max),
                plot_left=int(plot_left),
                plot_right=int(plot_right),
            )
            x_center = float(plot_left) + 0.5 * float(x_extent - float(plot_left))
            mark_point = (float(x_extent), float(y_center))
            top = float(y_center - 0.5 * float(horizontal_bar_height))
            bottom = float(y_center + 0.5 * float(horizontal_bar_height))
            bar_bbox = (float(plot_left), float(top), float(x_extent), float(bottom))
            if mark_visible:
                fill_rgb = (
                    tuple(int(channel) for channel in mark.fill_rgb)
                    if isinstance(mark.fill_rgb, tuple)
                    else tuple(int(value) for value in render_params.mark_fill_rgb)
                )
                outline_rgb = (
                    tuple(int(channel) for channel in mark.outline_rgb)
                    if isinstance(mark.outline_rgb, tuple)
                    else tuple(int(value) for value in render_params.mark_outline_rgb)
                )
                draw.rectangle(
                    bar_bbox,
                    fill=fill_rgb,
                    outline=outline_rgb,
                    width=int(render_params.mark_outline_width_px),
                )
        else:
            x_center = float(x_centers[index])
            if mark_visible:
                y_center = _tick_y(
                    int(mark.value),
                    y_axis_min=int(y_axis_min),
                    y_axis_max=int(y_axis_max),
                    plot_top=int(plot_top),
                    plot_bottom=int(plot_bottom),
                )
                line_points.append((float(x_center), float(y_center)))
                mark_point = (float(x_center), float(y_center))
            else:
                y_center = float(plot_bottom)
                mark_point = (float(x_center), float(y_center))

        if bool(draw_guides) and bool(mark_visible):
            if selected_variant == "horizontal_bar":
                guide_points = [(float(mark_point[0]), float(plot_bottom)), (float(mark_point[0]), float(mark_point[1]))]
                guide_orientation = "vertical"
            else:
                guide_points = [(float(plot_left), float(mark_point[1])), (float(mark_point[0]), float(mark_point[1]))]
                guide_orientation = "horizontal"
            _draw_styled_line(
                draw,
                guide_points,
                fill=guide_color,
                width=int(render_params.guide_line_width_px),
                style=str(render_params.guide_line_style),
            )
            guide_lines.append(
                {
                    "label": str(mark.label),
                    "value": int(mark.value),
                    "orientation": str(guide_orientation),
                    "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in guide_points],
                    "style": str(render_params.guide_line_style),
                }
            )

        if mark_visible and selected_variant == "bar":
            resolved_bar_width = float(bar_width)
            left = float(x_center - 0.5 * float(bar_width))
            right = float(x_center + 0.5 * float(bar_width))
            left = float(x_center - 0.5 * float(resolved_bar_width))
            right = float(x_center + 0.5 * float(resolved_bar_width))
            bar_bbox = (float(left), float(y_center), float(right), float(plot_bottom))
            fill_rgb = (
                tuple(int(channel) for channel in mark.fill_rgb)
                if isinstance(mark.fill_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_fill_rgb)
            )
            outline_rgb = (
                tuple(int(channel) for channel in mark.outline_rgb)
                if isinstance(mark.outline_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_outline_rgb)
            )
            draw.rectangle(
                bar_bbox,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
        elif mark_visible and selected_variant == "lollipop":
            fill_rgb = (
                tuple(int(channel) for channel in mark.fill_rgb)
                if isinstance(mark.fill_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_fill_rgb)
            )
            outline_rgb = (
                tuple(int(channel) for channel in mark.outline_rgb)
                if isinstance(mark.outline_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_outline_rgb)
            )
            draw.line(
                [(float(x_center), float(plot_bottom)), (float(x_center), float(y_center))],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )
            radius = float(render_params.point_radius_px)
            ellipse_box = (
                float(x_center - radius),
                float(y_center - radius),
                float(x_center + radius),
                float(y_center + radius),
            )
            draw.ellipse(
                ellipse_box,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
        elif mark_visible and selected_variant in {"scatter", "dot_plot"}:
            fill_rgb = (
                tuple(int(channel) for channel in mark.fill_rgb)
                if isinstance(mark.fill_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_fill_rgb)
            )
            outline_rgb = (
                tuple(int(channel) for channel in mark.outline_rgb)
                if isinstance(mark.outline_rgb, tuple)
                else tuple(int(value) for value in render_params.mark_outline_rgb)
            )
            radius = float(render_params.point_radius_px)
            ellipse_box = (
                float(x_center - radius),
                float(y_center - radius),
                float(x_center + radius),
                float(y_center + radius),
            )
            draw.ellipse(
                ellipse_box,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )

        if mark_visible:
            label_center = _label_center_for_variant(
                scene_variant=selected_variant,
                mark_center=(float(x_center), float(y_center)),
                bar_bbox=bar_bbox,
                point_radius_px=int(render_params.point_radius_px),
                label_font_size_px=int(render_params.label_font_size_px),
            )
        else:
            draw.line(
                [
                    (float(x_center), float(plot_bottom)),
                    (float(x_center), float(plot_bottom) + float(render_params.tick_length_px)),
                ],
                fill=axis_color,
                width=max(1, int(render_params.axis_line_width_px)),
            )
            label_center = (
                float(x_center),
                float(plot_bottom) + float(render_params.tick_length_px) + float(render_params.label_font_size_px),
            )
        draw_text_centered(
            draw,
            text=str(mark.label),
            center=label_center,
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )

        label_bbox = _text_bbox(draw, text=str(mark.label), center=label_center, font=label_font)
        if mark_visible:
            mark_bbox = (
                list(bar_bbox)
                if bar_bbox is not None
                else [
                    float(x_center - float(render_params.point_radius_px)),
                    float(y_center - float(render_params.point_radius_px)),
                    float(x_center + float(render_params.point_radius_px)),
                    float(y_center + float(render_params.point_radius_px)),
                ]
            )
            entity_type = "bar" if selected_variant in {"bar", "horizontal_bar"} else "point"
        else:
            mark_bbox = list(label_bbox)
            entity_type = "future_label_slot"
        mark_trace = {
            "entity_id": f"mark_{str(mark.label)}",
            "label": str(mark.label),
            "value": int(mark.value),
            "visible": bool(mark_visible),
            "x_rank": int(index),
            "mark_center_px": [
                round(float(mark_point[0] if mark_visible else label_center[0]), 3),
                round(float(mark_point[1] if mark_visible else label_center[1]), 3),
            ],
            "mark_bbox_px": [round(float(value), 3) for value in mark_bbox],
            "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
            "label_bbox_px": [round(float(value), 3) for value in label_bbox],
            "mark_fill_rgb": [
                int(channel)
                for channel in (
                    tuple(int(channel) for channel in mark.fill_rgb)
                    if isinstance(mark.fill_rgb, tuple)
                    else tuple(int(value) for value in render_params.mark_fill_rgb)
                )
            ],
            "mark_outline_rgb": [
                int(channel)
                for channel in (
                    tuple(int(channel) for channel in mark.outline_rgb)
                    if isinstance(mark.outline_rgb, tuple)
                    else tuple(int(value) for value in render_params.mark_outline_rgb)
                )
            ],
        }
        mark_traces.append(mark_trace)
        entities.append(
            {
                "entity_id": str(mark_trace["entity_id"]),
                "entity_type": str(entity_type),
                "attrs": {
                    "label": str(mark.label),
                    "x_rank": int(index),
                    "scene_variant": str(selected_variant),
                    "mark_center_px": list(mark_trace["mark_center_px"]),
                    "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                    "label_center_px": list(mark_trace["label_center_px"]),
                    "visible": bool(mark_visible),
                    "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                    "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                    **({"value": int(mark.value)} if mark_visible else {}),
                },
            }
        )

    if selected_variant == "area" and len(line_points) >= 2:
        polygon_points = list(line_points)
        polygon_points.extend(
            [
                (float(line_points[-1][0]), float(plot_bottom)),
                (float(line_points[0][0]), float(plot_bottom)),
            ]
        )
        draw.polygon(
            polygon_points,
            fill=render_params.mark_fill_rgb,
            outline=None,
        )
        for guide in guide_lines:
            guide_points = [
                (float(point[0]), float(point[1]))
                for point in guide.get("points_px", [])
                if isinstance(point, list) and len(point) == 2
            ]
            _draw_styled_line(
                draw,
                guide_points,
                fill=guide_color,
                width=int(render_params.guide_line_width_px),
                style=str(render_params.guide_line_style),
            )
        draw.line(
            line_points,
            fill=render_params.mark_outline_rgb,
            width=int(render_params.line_width_px),
            joint="curve",
        )
        for trace in mark_traces:
            if not bool(trace.get("visible", True)):
                continue
            center_x, center_y = trace["mark_center_px"]
            fill_rgb = tuple(int(value) for value in trace.get("mark_fill_rgb", render_params.mark_fill_rgb))
            outline_rgb = tuple(int(value) for value in trace.get("mark_outline_rgb", render_params.mark_outline_rgb))
            radius = float(render_params.point_radius_px)
            ellipse_box = (
                float(center_x - radius),
                float(center_y - radius),
                float(center_x + radius),
                float(center_y + radius),
            )
            draw.ellipse(
                ellipse_box,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
            draw_text_centered(
                draw,
                text=str(trace["label"]),
                center=(float(trace["label_center_px"][0]), float(trace["label_center_px"][1])),
                font=label_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=int(render_params.label_stroke_width_px),
            )

    if selected_variant == "line" and len(line_points) >= 2:
        draw.line(
            line_points,
            fill=render_params.mark_outline_rgb,
            width=int(render_params.line_width_px),
            joint="curve",
        )
        for trace in mark_traces:
            if not bool(trace.get("visible", True)):
                continue
            center_x, center_y = trace["mark_center_px"]
            fill_rgb = tuple(int(value) for value in trace.get("mark_fill_rgb", render_params.mark_fill_rgb))
            outline_rgb = tuple(int(value) for value in trace.get("mark_outline_rgb", render_params.mark_outline_rgb))
            radius = float(render_params.point_radius_px)
            ellipse_box = (
                float(center_x - radius),
                float(center_y - radius),
                float(center_x + radius),
                float(center_y + radius),
            )
            draw.ellipse(
                ellipse_box,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )

    return RenderedChartScene(
        image=image,
        mark_traces=tuple(dict(item) for item in mark_traces),
        entities=tuple(dict(item) for item in entities),
        plot_bbox_px=tuple(int(value) for value in plot_bbox),
        y_axis_max=int(y_axis_max),
        y_ticks=tuple(int(value) for value in y_ticks),
        scene_variant=str(selected_variant),
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
    'render_labeled_chart_scene',
]
