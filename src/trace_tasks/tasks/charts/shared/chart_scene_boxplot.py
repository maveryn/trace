"""Shared chart-scene boxplot renderers."""

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

def render_boxplot_scene(
    background: Image.Image,
    *,
    boxplots: Sequence[BoxPlotSpec],
    render_params: ChartRenderParams,
) -> RenderedChartScene:
    """Render one categorical boxplot scene."""

    if len(boxplots) < 2:
        raise ValueError("boxplot scenes require at least two categories")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = _resolve_plot_bbox(render_params)
    plot_bbox = (int(plot_left), int(plot_top), int(plot_right), int(plot_bottom))
    draw.rectangle(plot_bbox, fill=render_params.plot_fill_rgb)

    boxplot_values: List[int] = []
    for spec in boxplots:
        boxplot_values.extend([int(spec.whisker_min), int(spec.q1), int(spec.median), int(spec.q3), int(spec.whisker_max)])
    y_axis_min, y_axis_max, y_ticks, y_minor_ticks, value_axis_window_enabled = _resolve_value_axis(
        boxplot_values,
        render_params=render_params,
    )
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
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

    centers = _slot_centers(count=len(boxplots), plot_left=int(plot_left), plot_right=int(plot_right))
    slot_width = float(max(1.0, (float(plot_right) - float(plot_left)) / max(1, len(boxplots))))
    box_width = float(max(18.0, float(render_params.bar_width_fraction) * float(slot_width)))
    whisker_cap = float(max(14.0, 0.55 * float(box_width)))

    mark_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    guide_lines: List[Dict[str, Any]] = []
    draw_guides = _guide_lines_enabled(render_params)
    for index, spec in enumerate(boxplots):
        x_center = float(centers[index])
        y_whisker_min = _tick_y(int(spec.whisker_min), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
        y_q1 = _tick_y(int(spec.q1), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
        y_median = _tick_y(int(spec.median), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
        y_q3 = _tick_y(int(spec.q3), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
        y_whisker_max = _tick_y(int(spec.whisker_max), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))

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

        draw.line(
            [(float(x_center), float(y_whisker_max)), (float(x_center), float(y_whisker_min))],
            fill=outline_rgb,
            width=max(1, int(render_params.line_width_px) - 1),
        )
        draw.line(
            [
                (float(x_center - 0.5 * whisker_cap), float(y_whisker_max)),
                (float(x_center + 0.5 * whisker_cap), float(y_whisker_max)),
            ],
            fill=outline_rgb,
            width=max(1, int(render_params.line_width_px) - 1),
        )
        draw.line(
            [
                (float(x_center - 0.5 * whisker_cap), float(y_whisker_min)),
                (float(x_center + 0.5 * whisker_cap), float(y_whisker_min)),
            ],
            fill=outline_rgb,
            width=max(1, int(render_params.line_width_px) - 1),
        )

        box_bbox = (
            float(x_center - 0.5 * float(box_width)),
            float(y_q3),
            float(x_center + 0.5 * float(box_width)),
            float(y_q1),
        )
        draw.rectangle(
            box_bbox,
            fill=fill_rgb,
            outline=outline_rgb,
            width=int(render_params.mark_outline_width_px),
        )
        draw.line(
            [(float(box_bbox[0]), float(y_median)), (float(box_bbox[2]), float(y_median))],
            fill=outline_rgb,
            width=max(1, int(render_params.line_width_px) - 1),
        )
        if bool(draw_guides):
            guide_points = [(float(plot_left), float(y_median)), (float(x_center), float(y_median))]
            _draw_styled_line(
                draw,
                guide_points,
                fill=render_params.guide_line_color_rgb,
                width=int(render_params.guide_line_width_px),
                style=str(render_params.guide_line_style),
            )
            guide_lines.append(
                {
                    "entity_id": f"boxplot_{str(spec.label)}",
                    "label": str(spec.label),
                    "value": int(spec.median),
                    "stat": "median",
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
            text=str(spec.label),
            center=label_center,
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )
        label_bbox = _text_bbox(draw, text=str(spec.label), center=label_center, font=label_font)
        mark_bbox = (
            float(box_bbox[0]),
            float(min(y_whisker_max, y_whisker_min)),
            float(box_bbox[2]),
            float(max(y_whisker_max, y_whisker_min)),
        )
        mark_center = (float(x_center), float(0.5 * (float(box_bbox[1]) + float(box_bbox[3]))))
        mark_trace = {
            "entity_id": f"boxplot_{str(spec.label)}",
            "label": str(spec.label),
            "value": int(spec.median),
            "x_rank": int(index),
            "whisker_min": int(spec.whisker_min),
            "q1": int(spec.q1),
            "median": int(spec.median),
            "q3": int(spec.q3),
            "whisker_max": int(spec.whisker_max),
            "mark_center_px": [round(float(mark_center[0]), 3), round(float(mark_center[1]), 3)],
            "box_bbox_px": [round(float(value), 3) for value in box_bbox],
            "mark_bbox_px": [round(float(value), 3) for value in mark_bbox],
            "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
            "label_bbox_px": [round(float(value), 3) for value in label_bbox],
            "mark_fill_rgb": [int(channel) for channel in fill_rgb],
            "mark_outline_rgb": [int(channel) for channel in outline_rgb],
        }
        mark_traces.append(mark_trace)
        entities.append(
            {
                "entity_id": str(mark_trace["entity_id"]),
                "entity_type": "boxplot",
                "attrs": {
                    "label": str(spec.label),
                    "x_rank": int(index),
                    "scene_variant": "boxplot",
                    "whisker_min": int(spec.whisker_min),
                    "q1": int(spec.q1),
                    "median": int(spec.median),
                    "q3": int(spec.q3),
                    "whisker_max": int(spec.whisker_max),
                    "mark_center_px": list(mark_trace["mark_center_px"]),
                    "box_bbox_px": list(mark_trace["box_bbox_px"]),
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
        scene_variant="boxplot",
        value_axis_min=int(y_axis_min),
        value_axis_max=int(y_axis_max),
        value_axis_span=int(y_axis_max) - int(y_axis_min),
        value_axis_major_ticks=tuple(int(value) for value in y_ticks),
        value_axis_minor_ticks=tuple(int(value) for value in y_minor_ticks),
        value_axis_window_enabled=bool(value_axis_window_enabled),
        guide_line_style=str(render_params.guide_line_style if guide_lines else "none"),
        guide_lines=tuple(dict(item) for item in guide_lines),
    )

def render_paired_boxplot_scene(
    background: Image.Image,
    *,
    before_boxplots: Sequence[BoxPlotSpec],
    after_boxplots: Sequence[BoxPlotSpec],
    render_params: ChartRenderParams,
    before_title: str = "Before",
    after_title: str = "After",
) -> RenderedChartScene:
    """Render matched before/after boxplots as two aligned panels."""

    if len(before_boxplots) < 2 or len(after_boxplots) < 2:
        raise ValueError("paired boxplot scenes require at least two categories per panel")
    if len(before_boxplots) != len(after_boxplots):
        raise ValueError("paired boxplot scenes require equal before/after category counts")
    before_labels = [str(spec.label) for spec in before_boxplots]
    after_labels = [str(spec.label) for spec in after_boxplots]
    if before_labels != after_labels:
        raise ValueError("paired boxplot panels must use the same labels in the same order")

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = _resolve_plot_bbox(render_params)
    plot_bbox = (int(plot_left), int(plot_top), int(plot_right), int(plot_bottom))
    draw.rectangle(plot_bbox, fill=render_params.plot_fill_rgb)

    plot_width = float(max(1, int(plot_right) - int(plot_left)))
    panel_gap = float(max(42.0, min(82.0, 0.075 * float(plot_width))))
    panel_width = float(max(1.0, (float(plot_width) - float(panel_gap)) / 2.0))
    left_panel = (
        int(plot_left),
        int(plot_top),
        int(round(float(plot_left) + float(panel_width))),
        int(plot_bottom),
    )
    right_panel = (
        int(round(float(plot_left) + float(panel_width) + float(panel_gap))),
        int(plot_top),
        int(plot_right),
        int(plot_bottom),
    )

    boxplot_values: List[int] = []
    for spec in [*before_boxplots, *after_boxplots]:
        boxplot_values.extend([int(spec.whisker_min), int(spec.q1), int(spec.median), int(spec.q3), int(spec.whisker_max)])
    y_axis_min, y_axis_max, y_ticks, y_minor_ticks, value_axis_window_enabled = _resolve_value_axis(
        boxplot_values,
        render_params=render_params,
    )
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    axis_color = tuple(int(value) for value in render_params.axis_color_rgb)
    grid_color = tuple(int(value) for value in render_params.grid_color_rgb)

    title_y = max(12.0, float(plot_top) - float(max(18, int(render_params.label_font_size_px))))
    for panel_bbox, title in ((left_panel, before_title), (right_panel, after_title)):
        panel_left, _, panel_right, _ = panel_bbox
        draw_text_centered(
            draw,
            text=str(title),
            center=(0.5 * (float(panel_left) + float(panel_right)), float(title_y)),
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )

    major_tick_values = set(int(value) for value in y_ticks)
    for tick_value in y_minor_ticks:
        y_px = _tick_y(
            int(tick_value),
            y_axis_min=int(y_axis_min),
            y_axis_max=int(y_axis_max),
            plot_top=int(plot_top),
            plot_bottom=int(plot_bottom),
        )
        for panel_bbox in (left_panel, right_panel):
            panel_left, _, panel_right, _ = panel_bbox
            draw.line(
                [(float(panel_left), float(y_px)), (float(panel_right), float(y_px))],
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
                stroke_width=max(1, int(round(0.06 * float(render_params.tick_font_size_px)))),
            )

    for panel_bbox in (left_panel, right_panel):
        panel_left, panel_top, panel_right, panel_bottom = panel_bbox
        draw.line(
            [(float(panel_left), float(panel_top)), (float(panel_left), float(panel_bottom))],
            fill=axis_color,
            width=int(render_params.axis_line_width_px),
        )
        draw.line(
            [(float(panel_left), float(panel_bottom)), (float(panel_right), float(panel_bottom))],
            fill=axis_color,
            width=int(render_params.axis_line_width_px),
        )

    mark_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    guide_lines: List[Dict[str, Any]] = []
    draw_guides = _guide_lines_enabled(render_params)

    def _draw_panel_boxplots(
        *,
        panel_id: str,
        panel_title: str,
        panel_rank: int,
        panel_bbox: Tuple[int, int, int, int],
        specs: Sequence[BoxPlotSpec],
    ) -> None:
        panel_left, panel_top, panel_right, panel_bottom = panel_bbox
        centers = _slot_centers(count=len(specs), plot_left=int(panel_left), plot_right=int(panel_right))
        slot_width = float(max(1.0, (float(panel_right) - float(panel_left)) / max(1, len(specs))))
        box_width = float(max(18.0, float(render_params.bar_width_fraction) * float(slot_width)))
        whisker_cap = float(max(14.0, 0.55 * float(box_width)))
        for index, spec in enumerate(specs):
            display_label = str(spec.label)
            trace_label = f"{display_label}__{str(panel_id)}"
            x_center = float(centers[index])
            y_whisker_min = _tick_y(int(spec.whisker_min), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
            y_q1 = _tick_y(int(spec.q1), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
            y_median = _tick_y(int(spec.median), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
            y_q3 = _tick_y(int(spec.q3), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))
            y_whisker_max = _tick_y(int(spec.whisker_max), y_axis_min=int(y_axis_min), y_axis_max=int(y_axis_max), plot_top=int(plot_top), plot_bottom=int(plot_bottom))

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

            draw.line(
                [(float(x_center), float(y_whisker_max)), (float(x_center), float(y_whisker_min))],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )
            draw.line(
                [
                    (float(x_center - 0.5 * whisker_cap), float(y_whisker_max)),
                    (float(x_center + 0.5 * whisker_cap), float(y_whisker_max)),
                ],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )
            draw.line(
                [
                    (float(x_center - 0.5 * whisker_cap), float(y_whisker_min)),
                    (float(x_center + 0.5 * whisker_cap), float(y_whisker_min)),
                ],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )

            box_bbox = (
                float(x_center - 0.5 * float(box_width)),
                float(y_q3),
                float(x_center + 0.5 * float(box_width)),
                float(y_q1),
            )
            draw.rectangle(
                box_bbox,
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
            draw.line(
                [(float(box_bbox[0]), float(y_median)), (float(box_bbox[2]), float(y_median))],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )
            if bool(draw_guides):
                guide_points = [(float(panel_left), float(y_median)), (float(x_center), float(y_median))]
                _draw_styled_line(
                    draw,
                    guide_points,
                    fill=render_params.guide_line_color_rgb,
                    width=int(render_params.guide_line_width_px),
                    style=str(render_params.guide_line_style),
                )
                guide_lines.append(
                    {
                        "entity_id": f"boxplot_{trace_label}",
                        "label": str(trace_label),
                        "display_label": str(display_label),
                        "panel": str(panel_id),
                        "panel_title": str(panel_title),
                        "value": int(spec.median),
                        "stat": "median",
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
                text=str(display_label),
                center=label_center,
                font=label_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=int(render_params.label_stroke_width_px),
            )
            label_bbox = _text_bbox(draw, text=str(display_label), center=label_center, font=label_font)
            mark_bbox = (
                float(box_bbox[0]),
                float(min(y_whisker_max, y_whisker_min)),
                float(box_bbox[2]),
                float(max(y_whisker_max, y_whisker_min)),
            )
            mark_center = (float(x_center), float(0.5 * (float(box_bbox[1]) + float(box_bbox[3]))))
            mark_trace = {
                "entity_id": f"boxplot_{trace_label}",
                "label": str(trace_label),
                "display_label": str(display_label),
                "panel": str(panel_id),
                "panel_title": str(panel_title),
                "panel_rank": int(panel_rank),
                "value": int(spec.median),
                "x_rank": int(index),
                "whisker_min": int(spec.whisker_min),
                "q1": int(spec.q1),
                "median": int(spec.median),
                "q3": int(spec.q3),
                "whisker_max": int(spec.whisker_max),
                "panel_bbox_px": [int(value) for value in panel_bbox],
                "mark_center_px": [round(float(mark_center[0]), 3), round(float(mark_center[1]), 3)],
                "box_bbox_px": [round(float(value), 3) for value in box_bbox],
                "mark_bbox_px": [round(float(value), 3) for value in mark_bbox],
                "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
                "label_bbox_px": [round(float(value), 3) for value in label_bbox],
                "mark_fill_rgb": [int(channel) for channel in fill_rgb],
                "mark_outline_rgb": [int(channel) for channel in outline_rgb],
            }
            mark_traces.append(mark_trace)
            entities.append(
                {
                    "entity_id": str(mark_trace["entity_id"]),
                    "entity_type": "boxplot",
                    "attrs": {
                        "label": str(trace_label),
                        "display_label": str(display_label),
                        "panel": str(panel_id),
                        "panel_title": str(panel_title),
                        "panel_rank": int(panel_rank),
                        "x_rank": int(index),
                        "scene_variant": "boxplot",
                        "whisker_min": int(spec.whisker_min),
                        "q1": int(spec.q1),
                        "median": int(spec.median),
                        "q3": int(spec.q3),
                        "whisker_max": int(spec.whisker_max),
                        "panel_bbox_px": list(mark_trace["panel_bbox_px"]),
                        "mark_center_px": list(mark_trace["mark_center_px"]),
                        "box_bbox_px": list(mark_trace["box_bbox_px"]),
                        "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                        "label_center_px": list(mark_trace["label_center_px"]),
                        "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                        "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                    },
                }
            )

    _draw_panel_boxplots(
        panel_id="before",
        panel_title=str(before_title),
        panel_rank=0,
        panel_bbox=left_panel,
        specs=before_boxplots,
    )
    _draw_panel_boxplots(
        panel_id="after",
        panel_title=str(after_title),
        panel_rank=1,
        panel_bbox=right_panel,
        specs=after_boxplots,
    )

    return RenderedChartScene(
        image=image,
        mark_traces=tuple(dict(item) for item in mark_traces),
        entities=tuple(dict(item) for item in entities),
        plot_bbox_px=tuple(int(value) for value in plot_bbox),
        y_axis_max=int(y_axis_max),
        y_ticks=tuple(int(value) for value in y_ticks),
        scene_variant="boxplot",
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
    'render_boxplot_scene',
    'render_paired_boxplot_scene',
]
