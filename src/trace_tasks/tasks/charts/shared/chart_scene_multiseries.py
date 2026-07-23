"""Shared chart-scene multiseries renderers."""

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

_MULTISERIES_GROUP_WIDTH_FRACTION = 0.60
_MULTISERIES_LEGEND_WIDTH_FRACTION = 0.18
_MULTISERIES_LEGEND_WIDTH_MAX_FRACTION = 0.24
_MULTISERIES_LEGEND_MIN_WIDTH_PX = 140.0

def render_multiseries_chart_scene(
    background: Image.Image,
    *,
    scene_variant: str,
    marks: Sequence[MultiSeriesChartMarkSpec],
    render_params: ChartRenderParams,
    instance_seed: int,
) -> RenderedChartScene:
    """Render one multiseries chart scene onto `background`."""

    del instance_seed
    selected_variant = str(scene_variant)
    if selected_variant not in set(SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS):
        raise ValueError(f"unsupported multiseries chart scene_variant: {selected_variant}")
    if len(marks) < 4:
        raise ValueError("multiseries charts require at least four marks")

    category_specs: Dict[int, str] = {}
    series_specs: Dict[int, str] = {}
    marks_by_key: Dict[Tuple[int, int], MultiSeriesChartMarkSpec] = {}
    for mark in marks:
        category_specs[int(mark.category_rank)] = str(mark.category_label)
        series_specs[int(mark.series_rank)] = str(mark.series_label)
        key = (int(mark.category_rank), int(mark.series_rank))
        if key in marks_by_key:
            raise ValueError("duplicate multiseries mark for category/series pair")
        marks_by_key[key] = mark
    category_ranks = sorted(category_specs.keys())
    series_ranks = sorted(series_specs.keys())
    if not category_ranks or not series_ranks:
        raise ValueError("multiseries charts require at least one category and one series")
    expected_keys = {
        (int(category_rank), int(series_rank))
        for category_rank in category_ranks
        for series_rank in series_ranks
    }
    if set(marks_by_key.keys()) != expected_keys:
        raise ValueError("multiseries charts require one mark per category/series pair")

    categories = [(int(rank), str(category_specs[int(rank)])) for rank in category_ranks]
    series_list = [(int(rank), str(series_specs[int(rank)])) for rank in series_ranks]

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left, plot_top, plot_right, plot_bottom = _resolve_plot_bbox(render_params)
    draw.rectangle((int(plot_left), int(plot_top), int(plot_right), int(plot_bottom)), fill=render_params.plot_fill_rgb)

    plot_width = float(max(1, int(plot_right) - int(plot_left)))
    legend_width = float(
        min(
            max(_MULTISERIES_LEGEND_MIN_WIDTH_PX, plot_width * _MULTISERIES_LEGEND_WIDTH_FRACTION),
            plot_width * _MULTISERIES_LEGEND_WIDTH_MAX_FRACTION,
        )
    )
    legend_gap = float(max(18.0, float(render_params.label_font_size_px)))
    chart_right = int(max(int(plot_left) + 180, int(round(float(plot_right) - float(legend_width) - float(legend_gap)))))
    chart_bbox = (int(plot_left), int(plot_top), int(chart_right), int(plot_bottom))

    y_axis_min, y_axis_max, y_ticks, y_minor_ticks, value_axis_window_enabled = _resolve_value_axis(
        [int(mark.value) for mark in marks],
        render_params=render_params,
    )

    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    axis_color = tuple(int(value) for value in render_params.axis_color_rgb)
    grid_color = tuple(int(value) for value in render_params.grid_color_rgb)

    if selected_variant == "grouped_horizontal_bar":
        major_tick_values = set(int(value) for value in y_ticks)
        for tick_value in y_minor_ticks:
            x_px = _tick_x(
                int(tick_value),
                x_axis_min=int(y_axis_min),
                x_axis_max=int(y_axis_max),
                plot_left=int(plot_left),
                plot_right=int(chart_right),
            )
            draw.line(
                [(float(x_px), float(plot_top)), (float(x_px), float(plot_bottom))],
                fill=grid_color,
                width=int(render_params.grid_line_width_px),
            )
            if int(tick_value) in major_tick_values:
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
                [(float(plot_left), float(y_px)), (float(chart_right), float(y_px))],
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
        [(float(plot_left), float(plot_bottom)), (float(chart_right), float(plot_bottom))],
        fill=axis_color,
        width=int(render_params.axis_line_width_px),
    )

    if selected_variant == "grouped_horizontal_bar":
        category_centers = _axis_slot_centers(
            count=len(categories),
            start_px=int(plot_top),
            end_px=int(plot_bottom),
        )
        slot_height = float(max(1.0, (float(plot_bottom) - float(plot_top)) / max(1, len(categories))))
        group_inner_height = float(slot_height * _MULTISERIES_GROUP_WIDTH_FRACTION)
        subgroup_height = float(group_inner_height / max(1, len(series_list)))
        bar_height = float(max(8.0, float(render_params.bar_width_fraction) * float(subgroup_height)))
    else:
        category_centers = _slot_centers(
            count=len(categories),
            plot_left=int(plot_left),
            plot_right=int(chart_right),
        )
        slot_width = float(max(1.0, (float(chart_right) - float(plot_left)) / max(1, len(categories))))
        # Leave more whitespace between adjacent category groups so dense
        # multiseries charts remain readable at the upper category-count range.
        group_inner_width = float(slot_width * _MULTISERIES_GROUP_WIDTH_FRACTION)
        subgroup_width = float(group_inner_width / max(1, len(series_list)))
        bar_width = float(max(8.0, float(render_params.bar_width_fraction) * float(subgroup_width)))

    category_label_meta: Dict[str, Dict[str, List[float]]] = {}
    for category_rank, category_label in categories:
        if selected_variant == "grouped_horizontal_bar":
            category_center_y = float(category_centers[int(category_rank)])
            label_center = (
                float(plot_left) - float(max(20, int(render_params.label_font_size_px) + 8)),
                float(category_center_y),
            )
        else:
            category_center_x = float(category_centers[int(category_rank)])
            label_center = (
                float(category_center_x),
                float(plot_bottom) + float(render_params.tick_length_px) + 18.0,
            )
        draw_text_centered(
            draw,
            text=str(category_label),
            center=label_center,
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )
        label_bbox = _text_bbox(draw, text=str(category_label), center=label_center, font=label_font)
        category_label_meta[str(category_label)] = {
            "center": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
            "bbox": [round(float(value), 3) for value in label_bbox],
        }

    mark_records: List[Dict[str, Any]] = []
    series_points: Dict[str, List[Tuple[int, Tuple[float, float]]]] = {
        str(series_label): []
        for _, series_label in series_list
    }
    mark_bboxes_by_category: Dict[str, List[Tuple[float, float, float, float]]] = {
        str(category_label): []
        for _, category_label in categories
    }
    guide_lines: List[Dict[str, Any]] = []
    guide_color = tuple(int(value) for value in render_params.guide_line_color_rgb)
    draw_guides = _guide_lines_enabled(render_params)

    for category_rank, category_label in categories:
        if selected_variant == "grouped_horizontal_bar":
            category_center_y = float(category_centers[int(category_rank)])
            group_top = float(category_center_y) - 0.5 * float(group_inner_height)
        else:
            category_center_x = float(category_centers[int(category_rank)])
            group_left = float(category_center_x) - 0.5 * float(group_inner_width)
        for series_rank, series_label in series_list:
            mark = marks_by_key[(int(category_rank), int(series_rank))]
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
            if selected_variant == "multi_line":
                x_center = float(category_center_x)
                y_center = _tick_y(
                    int(mark.value),
                    y_axis_min=int(y_axis_min),
                    y_axis_max=int(y_axis_max),
                    plot_top=int(plot_top),
                    plot_bottom=int(plot_bottom),
                )
                mark_point = (float(x_center), float(y_center))
            elif selected_variant == "grouped_horizontal_bar":
                y_center = float(group_top) + (float(series_rank) + 0.5) * float(subgroup_height)
                x_extent = _tick_x(
                    int(mark.value),
                    x_axis_min=int(y_axis_min),
                    x_axis_max=int(y_axis_max),
                    plot_left=int(plot_left),
                    plot_right=int(chart_right),
                )
                x_center = float(plot_left) + 0.5 * float(x_extent - float(plot_left))
                mark_point = (float(x_extent), float(y_center))
            else:
                x_center = float(group_left) + (float(series_rank) + 0.5) * float(subgroup_width)
                y_center = _tick_y(
                    int(mark.value),
                    y_axis_min=int(y_axis_min),
                    y_axis_max=int(y_axis_max),
                    plot_top=int(plot_top),
                    plot_bottom=int(plot_bottom),
                )
                mark_point = (float(x_center), float(y_center))

            if selected_variant == "grouped_bar":
                mark_bbox = (
                    float(x_center - 0.5 * float(bar_width)),
                    float(y_center),
                    float(x_center + 0.5 * float(bar_width)),
                    float(plot_bottom),
                )
            elif selected_variant == "grouped_horizontal_bar":
                mark_bbox = (
                    float(plot_left),
                    float(y_center - 0.5 * float(bar_height)),
                    float(x_extent),
                    float(y_center + 0.5 * float(bar_height)),
                )
            else:
                point_radius = float(render_params.point_radius_px)
                mark_bbox = (
                    float(x_center - point_radius),
                    float(y_center - point_radius),
                    float(x_center + point_radius),
                    float(y_center + point_radius),
                )
            mark_bboxes_by_category[str(category_label)].append(tuple(float(value) for value in mark_bbox))
            series_points[str(series_label)].append((int(category_rank), (float(x_center), float(y_center))))
            if bool(draw_guides):
                if selected_variant == "grouped_horizontal_bar":
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
                        "category_label": str(category_label),
                        "series_label": str(series_label),
                        "value": int(mark.value),
                        "orientation": str(guide_orientation),
                        "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in guide_points],
                        "style": str(render_params.guide_line_style),
                    }
                )
            mark_records.append(
                {
                    "category_label": str(category_label),
                    "series_label": str(series_label),
                    "category_rank": int(category_rank),
                    "series_rank": int(series_rank),
                    "value": int(mark.value),
                    "fill_rgb": [int(channel) for channel in fill_rgb],
                    "outline_rgb": [int(channel) for channel in outline_rgb],
                    "mark_center_px": [round(float(mark_point[0]), 3), round(float(mark_point[1]), 3)],
                    "mark_bbox_px": [round(float(value), 3) for value in mark_bbox],
                }
            )

    if selected_variant == "multi_line":
        for _, series_label in series_list:
            records = sorted(series_points[str(series_label)], key=lambda item: int(item[0]))
            points = [tuple(float(value) for value in point) for _, point in records]
            sample_record = next(record for record in mark_records if str(record["series_label"]) == str(series_label))
            draw.line(
                points,
                fill=tuple(int(channel) for channel in sample_record["outline_rgb"]),
                width=int(render_params.line_width_px),
                joint="curve",
            )

    for record in mark_records:
        fill_rgb = tuple(int(channel) for channel in record["fill_rgb"])
        outline_rgb = tuple(int(channel) for channel in record["outline_rgb"])
        x_center, y_center = record["mark_center_px"]
        if selected_variant in {"grouped_bar", "grouped_horizontal_bar"}:
            draw.rectangle(
                record["mark_bbox_px"],
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
        elif selected_variant == "grouped_lollipop":
            draw.line(
                [(float(x_center), float(plot_bottom)), (float(x_center), float(y_center))],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )
            draw.ellipse(
                record["mark_bbox_px"],
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )
        else:
            draw.ellipse(
                record["mark_bbox_px"],
                fill=fill_rgb,
                outline=outline_rgb,
                width=int(render_params.mark_outline_width_px),
            )

    legend_left = float(chart_right) + float(legend_gap)
    legend_top = float(plot_top) + 16.0
    legend_row_height = float(max(render_params.label_font_size_px + 16, 42))
    legend_swatch_side = float(max(28, int(round(render_params.label_font_size_px * 1.05))))
    legend_frame_pad = float(max(3, int(round(render_params.mark_outline_width_px * 1.5))))
    legend_text_gap = float(max(16, int(render_params.label_font_size_px * 0.8)))
    legend_frame_fill = tuple(int(value) for value in render_params.plot_fill_rgb)
    legend_meta_by_series: Dict[str, Dict[str, List[float]]] = {}
    legend_boxes: List[Tuple[float, float, float, float]] = []
    for index, (_, series_label) in enumerate(series_list):
        sample_record = next(record for record in mark_records if str(record["series_label"]) == str(series_label))
        fill_rgb = tuple(int(channel) for channel in sample_record["fill_rgb"])
        outline_rgb = tuple(int(channel) for channel in sample_record["outline_rgb"])
        row_y = float(legend_top) + float(index) * float(legend_row_height)
        swatch_bbox = (
            float(legend_left),
            float(row_y),
            float(legend_left + legend_swatch_side),
            float(row_y + legend_swatch_side),
        )
        legend_frame_bbox = (
            float(swatch_bbox[0]) - float(legend_frame_pad),
            float(swatch_bbox[1]) - float(legend_frame_pad),
            float(swatch_bbox[2]) + float(legend_frame_pad),
            float(swatch_bbox[3]) + float(legend_frame_pad),
        )
        if selected_variant == "multi_line":
            center_y = 0.5 * float(swatch_bbox[1] + swatch_bbox[3])
            draw.line(
                [(float(swatch_bbox[0]), float(center_y)), (float(swatch_bbox[2]), float(center_y))],
                fill=outline_rgb,
                width=max(1, int(render_params.line_width_px) - 1),
            )
            point_radius = float(max(4, int(round(0.6 * float(render_params.point_radius_px)))))
            draw.ellipse(
                [
                    float(0.5 * (swatch_bbox[0] + swatch_bbox[2]) - point_radius),
                    float(center_y - point_radius),
                    float(0.5 * (swatch_bbox[0] + swatch_bbox[2]) + point_radius),
                    float(center_y + point_radius),
                ],
                fill=fill_rgb,
                outline=outline_rgb,
                width=max(1, int(render_params.mark_outline_width_px)),
            )
        else:
            draw.rectangle(
                swatch_bbox,
                fill=fill_rgb,
                outline=outline_rgb,
                width=max(1, int(render_params.mark_outline_width_px)),
            )
        label_width, _ = _text_size(draw, text=str(series_label), font=label_font)
        label_left = float(legend_frame_bbox[2]) + float(legend_text_gap)
        label_center = (
            float(label_left) + 0.5 * float(label_width),
            float(0.5 * (swatch_bbox[1] + swatch_bbox[3])),
        )
        draw_text_centered(
            draw,
            text=str(series_label),
            center=label_center,
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )
        label_bbox = _text_bbox(
            draw,
            text=str(series_label),
            center=label_center,
            font=label_font,
        )
        row_bbox = _union_bboxes((legend_frame_bbox, label_bbox))
        legend_meta = {
            "legend_swatch_bbox_px": [round(float(value), 3) for value in legend_frame_bbox],
            "legend_label_bbox_px": [round(float(value), 3) for value in label_bbox],
            "legend_row_bbox_px": [round(float(value), 3) for value in row_bbox],
        }
        legend_meta_by_series[str(series_label)] = legend_meta
        legend_boxes.append(tuple(float(value) for value in row_bbox))

    category_group_bboxes = {}
    for _, category_label in categories:
        label_bbox = tuple(float(value) for value in category_label_meta[str(category_label)]["bbox"])
        union_inputs = [label_bbox] + [
            tuple(float(value) for value in bbox)
            for bbox in mark_bboxes_by_category[str(category_label)]
        ]
        category_group_bboxes[str(category_label)] = [
            round(float(value), 3) for value in _union_bboxes(union_inputs)
        ]

    mark_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    for record in mark_records:
        category_label = str(record["category_label"])
        category_center = list(category_label_meta[category_label]["center"])
        category_bbox = list(category_label_meta[category_label]["bbox"])
        category_group_bbox = list(category_group_bboxes[category_label])
        legend_meta = legend_meta_by_series.get(str(record["series_label"]), {})
        mark_trace = {
            "entity_id": f"mark_{category_label}_{str(record['series_label'])}",
            "category_label": str(category_label),
            "series_label": str(record["series_label"]),
            "category_rank": int(record["category_rank"]),
            "series_rank": int(record["series_rank"]),
            "value": int(record["value"]),
            "mark_center_px": list(record["mark_center_px"]),
            "mark_bbox_px": list(record["mark_bbox_px"]),
            "category_label_center_px": list(category_center),
            "category_label_bbox_px": list(category_bbox),
            "category_group_bbox_px": list(category_group_bbox),
            "mark_fill_rgb": list(record["fill_rgb"]),
            "mark_outline_rgb": list(record["outline_rgb"]),
            **{str(key): list(value) for key, value in legend_meta.items()},
        }
        mark_traces.append(mark_trace)
        entities.append(
            {
                "entity_id": str(mark_trace["entity_id"]),
                "entity_type": "bar" if selected_variant in {"grouped_bar", "grouped_horizontal_bar"} else "point",
                "attrs": {
                    "category_label": str(category_label),
                    "series_label": str(record["series_label"]),
                    "category_rank": int(record["category_rank"]),
                    "series_rank": int(record["series_rank"]),
                    "value": int(record["value"]),
                    "scene_variant": str(selected_variant),
                    "mark_center_px": list(mark_trace["mark_center_px"]),
                    "mark_bbox_px": list(mark_trace["mark_bbox_px"]),
                        "category_label_center_px": list(mark_trace["category_label_center_px"]),
                        "category_group_bbox_px": list(mark_trace["category_group_bbox_px"]),
                        "mark_fill_rgb": list(mark_trace["mark_fill_rgb"]),
                        "mark_outline_rgb": list(mark_trace["mark_outline_rgb"]),
                    },
                }
        )

    return RenderedChartScene(
        image=image,
        mark_traces=tuple(dict(item) for item in mark_traces),
        entities=tuple(dict(item) for item in entities),
        plot_bbox_px=tuple(int(value) for value in chart_bbox),
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
        legend_bbox_px=(
            tuple(round(float(value), 3) for value in _union_bboxes(legend_boxes))
            if legend_boxes
            else ()
        ),
        legend_item_bboxes_px={
            str(series_label): tuple(float(value) for value in legend_meta["legend_row_bbox_px"])
            for series_label, legend_meta in legend_meta_by_series.items()
        },
    )


__all__ = [
    'render_multiseries_chart_scene',
]
