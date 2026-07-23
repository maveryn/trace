"""Rendering helpers for mixed-dashboard chart tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width, dense_text_style_meta
from ....shared.font_assets import sample_font_family
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font, resolve_text_stroke_fill
from ....shared.visual_style.context_layer import (
    draw_dashboard_reserved_margin_context,
    resolve_dashboard_context_layout,
    sample_dashboard_title,
)
from trace_tasks.tasks.charts.shared.panel.grid_layout import layout_panel_grid_int, panel_row_lengths
from .defaults import render_default, render_rgb, resolve_context_text_params
from .state import (
    SCENE_NAMESPACE,
    BBox,
    Point,
    RGB,
    OPTION_LETTERS,
    Category,
    DashboardDataset,
    Panel,
    RenderParams,
    RenderedDashboard,
)


_CHART_CONTEXT_MODE_WEIGHTS = {"clean": 0.3, "minimal": 0.4, "paragraph_box": 0.3}


def _normalize_chart_context_mode(value: str) -> str:
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"light", "light_context"}:
        return "minimal"
    if normalized in {"large", "large_distractor", "paragraph", "right_sidebar", "left_sidebar", "bottom_band", "sidebar"}:
        return "paragraph_box"
    return str(normalized)


def _resolve_chart_context_mode(*, context_params: Mapping[str, Any], instance_seed: int, namespace: str) -> tuple[str, dict[str, float]]:
    supported = ("clean", "minimal", "paragraph_box")
    explicit = context_params.get("chart_context_mode", context_params.get("context_text_mode"))
    if explicit is not None:
        mode = _normalize_chart_context_mode(str(explicit))
        if mode not in set(supported):
            raise ValueError(f"unsupported dashboard chart context mode: {explicit!r}")
        return str(mode), {key: 1.0 if key == mode else 0.0 for key in supported}
    raw_weights = context_params.get("chart_context_mode_weights", context_params.get("context_text_mode_weights", _CHART_CONTEXT_MODE_WEIGHTS))
    weights = {key: 0.0 for key in supported}
    if isinstance(raw_weights, Mapping):
        for raw_key, raw_value in raw_weights.items():
            mode = _normalize_chart_context_mode(str(raw_key))
            if mode not in weights:
                continue
            weights[str(mode)] += max(0.0, float(raw_value))
    if sum(weights.values()) <= 0.0:
        weights = dict(_CHART_CONTEXT_MODE_WEIGHTS)
    rng = spawn_rng(int(instance_seed), f"{namespace}.chart_context_mode")
    cursor = rng.random() * sum(weights.values())
    running = 0.0
    selected = str(supported[-1])
    for mode in supported:
        running += float(weights[str(mode)])
        if cursor <= running:
            selected = str(mode)
            break
    total = sum(weights.values())
    normalized = {mode: float(weights[mode]) / float(total) for mode in supported}
    return str(selected), dict(normalized)


def _dashboard_context_layer_mode(context_layout: Mapping[str, Any]) -> str:
    mode = str(context_layout.get("chart_context_mode", ""))
    if mode in {"clean", "minimal", "paragraph_box"}:
        return f"chart_context:{mode}"
    return f"{context_layout.get('layout_mode', 'reserved_context')}:{context_layout.get('placement', 'none')}"


def _bbox_tuple(box: Sequence[float]) -> BBox:
    return (
        int(math.floor(float(box[0]))),
        int(math.floor(float(box[1]))),
        int(math.ceil(float(box[2]))),
        int(math.ceil(float(box[3]))),
    )


def _union_bboxes(boxes: Sequence[Sequence[float]]) -> BBox:
    valid = [tuple(float(v) for v in box[:4]) for box in boxes if len(box) >= 4]
    if not valid:
        return (0, 0, 0, 0)
    return _bbox_tuple(
        (
            min(box[0] for box in valid),
            min(box[1] for box in valid),
            max(box[2] for box in valid),
            max(box[3] for box in valid),
        )
    )


def _pad_bbox(box: Sequence[float], padding: int) -> BBox:
    return _bbox_tuple((float(box[0]) - int(padding), float(box[1]) - int(padding), float(box[2]) + int(padding), float(box[3]) + int(padding)))


def _clip_bbox_to_canvas(box: Sequence[int], *, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = [int(value) for value in box[:4]]
    clipped_x0 = min(max(0, x0), max(0, int(width) - 1))
    clipped_y0 = min(max(0, y0), max(0, int(height) - 1))
    clipped_x1 = min(max(clipped_x0 + 1, x1), int(width))
    clipped_y1 = min(max(clipped_y0 + 1, y1), int(height))
    return (int(clipped_x0), int(clipped_y0), int(clipped_x1), int(clipped_y1))


def _clip_bbox_map_to_canvas(mapping: Mapping[str, BBox], *, width: int, height: int) -> Dict[str, BBox]:
    return {
        str(key): _clip_bbox_to_canvas(bbox, width=int(width), height=int(height))
        for key, bbox in mapping.items()
    }


def _bbox_map_to_json(mapping: Mapping[str, Sequence[int]]) -> Dict[str, List[int]]:
    return {str(key): [int(value) for value in bbox[:4]] for key, bbox in mapping.items()}


def _nested_bbox_map_to_json(mapping: Mapping[str, Mapping[str, Sequence[int]]]) -> Dict[str, Dict[str, List[int]]]:
    return {str(key): _bbox_map_to_json(value) for key, value in mapping.items()}


def _clip_point_to_canvas(point: Sequence[float], *, width: int, height: int) -> Point:
    x = min(max(0, int(round(float(point[0])))), max(0, int(width) - 1))
    y = min(max(0, int(round(float(point[1])))), max(0, int(height) - 1))
    return (int(x), int(y))


def _clip_point_map_to_canvas(mapping: Mapping[str, Sequence[float]], *, width: int, height: int) -> Dict[str, Point]:
    return {
        str(key): _clip_point_to_canvas(point, width=int(width), height=int(height))
        for key, point in mapping.items()
    }


def _point_map_to_json(mapping: Mapping[str, Sequence[float]]) -> Dict[str, List[int]]:
    return {
        str(key): [int(round(float(point[0]))), int(round(float(point[1])))]
        for key, point in mapping.items()
    }


def _nested_point_map_to_json(mapping: Mapping[str, Mapping[str, Sequence[float]]]) -> Dict[str, Dict[str, List[int]]]:
    return {str(key): _point_map_to_json(value) for key, value in mapping.items()}


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    *,
    font,
    fill: RGB,
    anchor: str = "mm",
    stroke_width: int = 0,
) -> BBox:
    stroke_fill = resolve_text_stroke_fill(fill)
    bbox = draw.textbbox(tuple(xy), str(text), font=font, anchor=str(anchor), stroke_width=int(stroke_width))
    draw_text_traced(draw,
        tuple(xy),
        str(text),
        font=font,
        anchor=str(anchor),
        fill=tuple(fill),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(stroke_fill),
     role="readout", required=False,)
    return _bbox_tuple(bbox)


def _panel_layout(render_params: RenderParams, panel_count: int) -> Tuple[BBox, ...]:
    """Reserve context/option bands, then place dashboard panels with shared centered-row rules."""

    margin = int(render_params.dashboard_margin_px)
    offset_x = int(render_params.layout_offset_x_px)
    offset_y = int(render_params.layout_offset_y_px)
    gap = int(render_params.panel_gap_px)
    top = int(margin + offset_y + render_params.title_height_px)
    context_placement = str(render_params.layout_jitter_meta.get("context_text_placement", "none"))
    sidebar_width = int(render_params.layout_jitter_meta.get("context_text_sidebar_width_px", 0))
    sidebar_gap = int(render_params.layout_jitter_meta.get("context_text_sidebar_gap_px", 14))
    bottom_band_height = int(render_params.layout_jitter_meta.get("context_text_bottom_band_height_px", 0))
    bottom_band_gap = int(render_params.layout_jitter_meta.get("context_text_bottom_band_gap_px", 14))
    option_panel_height = int(render_params.layout_jitter_meta.get("option_panel_height_px", 0))
    option_panel_gap = int(render_params.layout_jitter_meta.get("option_panel_gap_px", 0))
    left_reserved = int(sidebar_width + sidebar_gap) if context_placement == "left_sidebar" else 0
    right_reserved = int(sidebar_width + sidebar_gap) if context_placement == "right_sidebar" else 0
    bottom_reserved = int(bottom_band_height + bottom_band_gap) if context_placement == "bottom_band" else 0
    if int(option_panel_height) > 0:
        bottom_reserved += int(option_panel_height) + int(option_panel_gap)
    row_lengths = panel_row_lengths(int(panel_count))
    cols = max(row_lengths)
    rows = len(row_lengths)
    usable_width = int(render_params.canvas_width) - (2 * margin) - int(left_reserved) - int(right_reserved)
    usable_height = int(render_params.canvas_height) - top - margin - int(bottom_reserved)
    if usable_width < (cols * 180) + (gap * (cols - 1)):
        usable_width = int(render_params.canvas_width) - (2 * margin)
        left_reserved = 0
        right_reserved = 0
    if usable_height < (rows * 150) + (gap * (rows - 1)):
        usable_height = int(render_params.canvas_height) - top - margin
    return layout_panel_grid_int(
        (
            margin + offset_x + int(left_reserved),
            top,
            margin + offset_x + int(left_reserved) + int(usable_width),
            top + int(usable_height),
        ),
        panel_count=int(panel_count),
        gap_x=float(gap),
        gap_y=float(gap),
        row_lengths=row_lengths,
    )


def _option_panel_render_defaults(params: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "height_px": int(params.get("option_panel_height_px", render_default("option_panel_height_px", 238))),
        "gap_px": int(params.get("option_panel_gap_px", render_default("option_panel_gap_px", 16))),
        "padding_px": int(params.get("option_panel_padding_px", render_default("option_panel_padding_px", 16))),
        "column_gap_px": int(params.get("option_panel_column_gap_px", render_default("option_panel_column_gap_px", 22))),
        "column_count": int(params.get("option_panel_column_count", render_default("option_panel_column_count", 2))),
        "font_size_px": int(params.get("option_panel_font_size_px", render_default("option_panel_font_size_px", 15))),
        "letter_font_size_px": int(params.get("option_panel_letter_font_size_px", render_default("option_panel_letter_font_size_px", 16))),
    }


def _scale_y(value: int, plot_bbox: BBox) -> float:
    x0, y0, x1, y1 = plot_bbox
    del x0, x1
    return float(y1) - (float(value) / 100.0) * float(y1 - y0)


def _draw_panel_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Panel,
    panel_bbox: BBox,
    render_params: RenderParams,
) -> BBox:
    draw.rounded_rectangle(
        panel_bbox,
        radius=10,
        fill=tuple(render_params.panel_fill_rgb),
        outline=tuple(render_params.panel_border_rgb),
        width=int(render_params.panel_border_width_px),
    )
    title_font = load_font(int(render_params.panel_title_font_size_px), bold=False, font_family=render_params.font_family)
    title_bbox = _draw_text(
        draw,
        ((panel_bbox[0] + panel_bbox[2]) / 2.0, panel_bbox[1] + 24),
        str(panel.name),
        font=title_font,
        fill=tuple(render_params.text_color_rgb),
        anchor="mm",
        stroke_width=0,
    )
    return title_bbox


def _draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Panel,
    panel_bbox: BBox,
    categories: Sequence[Category],
    render_params: RenderParams,
) -> Tuple[Dict[str, BBox], Dict[str, BBox], Dict[str, Point], List[Dict[str, Any]]]:
    """Draw one bar panel and record one support point per category mark."""

    pad = int(render_params.panel_padding_px)
    label_size = 10 if len(categories) > 12 else int(render_params.label_font_size_px)
    value_size = 10 if len(categories) > 12 else int(render_params.value_font_size_px)
    label_font = load_font(int(label_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    value_font = load_font(int(value_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False, font_family=render_params.font_family)
    plot_bbox = (
        panel_bbox[0] + pad + 28,
        panel_bbox[1] + pad + 56,
        panel_bbox[2] - pad - 10,
        panel_bbox[3] - pad - 50,
    )
    draw.line((plot_bbox[0], plot_bbox[3], plot_bbox[2], plot_bbox[3]), fill=tuple(render_params.axis_color_rgb), width=int(render_params.axis_line_width_px))
    draw.line((plot_bbox[0], plot_bbox[1], plot_bbox[0], plot_bbox[3]), fill=tuple(render_params.axis_color_rgb), width=int(render_params.axis_line_width_px))
    for tick in (25, 50, 75):
        y = _scale_y(tick, plot_bbox)
        draw.line((plot_bbox[0], y, plot_bbox[2], y), fill=tuple(render_params.grid_color_rgb), width=int(render_params.grid_line_width_px))
        _draw_text(draw, (plot_bbox[0] - 10, y), str(tick), font=tick_font, fill=tuple(render_params.muted_text_color_rgb), anchor="rm")
    slot = float(plot_bbox[2] - plot_bbox[0]) / float(len(categories))
    bar_width = max(18.0, slot * 0.58)
    support: Dict[str, BBox] = {}
    values: Dict[str, BBox] = {}
    points: Dict[str, Point] = {}
    entities: List[Dict[str, Any]] = []
    for index, category in enumerate(categories):
        if str(category.category_id) not in panel.values_by_category_id:
            continue
        cx = float(plot_bbox[0]) + slot * (float(index) + 0.5)
        value = int(panel.values_by_category_id[str(category.category_id)])
        y = _scale_y(value, plot_bbox)
        bar_box = _bbox_tuple((cx - bar_width / 2.0, y, cx + bar_width / 2.0, plot_bbox[3]))
        draw.rounded_rectangle(bar_box, radius=4, fill=tuple(category.color_rgb), outline=(55, 62, 72), width=1)
        value_bbox = _draw_text(draw, (cx, y - 10), str(value), font=value_font, fill=tuple(render_params.text_color_rgb), anchor="mb", stroke_width=dense_stroke_width())
        label_bbox = _draw_text(draw, (cx, plot_bbox[3] + 16), str(category.label), font=label_font, fill=tuple(render_params.text_color_rgb), anchor="mt", stroke_width=dense_stroke_width())
        support[str(category.category_id)] = _pad_bbox(_union_bboxes([bar_box, value_bbox, label_bbox]), 4)
        values[str(category.category_id)] = value_bbox
        points[str(category.category_id)] = (int(round(cx)), int(round(y)))
        entities.append(
            {
                "entity_id": f"{panel.panel_id}:{category.category_id}",
                "entity_type": "dashboard_bar_mark",
                "bbox_xyxy": list(support[str(category.category_id)]),
                "point_xy": list(points[str(category.category_id)]),
                "attrs": {
                    "panel_id": str(panel.panel_id),
                    "panel_name": str(panel.name),
                    "category_id": str(category.category_id),
                    "category_label": str(category.label),
                    "value": int(value),
                },
            }
        )
    return support, values, points, entities


def _draw_line_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Panel,
    panel_bbox: BBox,
    categories: Sequence[Category],
    render_params: RenderParams,
) -> Tuple[Dict[str, BBox], Dict[str, BBox], Dict[str, Point], List[Dict[str, Any]]]:
    """Draw one line panel with category points as annotation witnesses."""

    pad = int(render_params.panel_padding_px)
    label_size = 10 if len(categories) > 12 else int(render_params.label_font_size_px)
    value_size = 10 if len(categories) > 12 else int(render_params.value_font_size_px)
    label_font = load_font(int(label_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    value_font = load_font(int(value_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False, font_family=render_params.font_family)
    plot_bbox = (
        panel_bbox[0] + pad + 28,
        panel_bbox[1] + pad + 58,
        panel_bbox[2] - pad - 12,
        panel_bbox[3] - pad - 50,
    )
    draw.line((plot_bbox[0], plot_bbox[3], plot_bbox[2], plot_bbox[3]), fill=tuple(render_params.axis_color_rgb), width=int(render_params.axis_line_width_px))
    draw.line((plot_bbox[0], plot_bbox[1], plot_bbox[0], plot_bbox[3]), fill=tuple(render_params.axis_color_rgb), width=int(render_params.axis_line_width_px))
    for tick in (25, 50, 75):
        y = _scale_y(tick, plot_bbox)
        draw.line((plot_bbox[0], y, plot_bbox[2], y), fill=tuple(render_params.grid_color_rgb), width=int(render_params.grid_line_width_px))
        _draw_text(draw, (plot_bbox[0] - 10, y), str(tick), font=tick_font, fill=tuple(render_params.muted_text_color_rgb), anchor="rm")
    if len(categories) == 1:
        x_positions = [(plot_bbox[0] + plot_bbox[2]) / 2.0]
    else:
        slot = float(plot_bbox[2] - plot_bbox[0]) / float(len(categories))
        x_positions = [
            float(plot_bbox[0]) + slot * (float(index) + 0.5)
            for index in range(len(categories))
        ]
    points_by_category_id: Dict[str, Tuple[float, float]] = {}
    for x, category in zip(x_positions, categories):
        if str(category.category_id) not in panel.values_by_category_id:
            continue
        points_by_category_id[str(category.category_id)] = (
            float(x),
            _scale_y(int(panel.values_by_category_id[str(category.category_id)]), plot_bbox),
        )
    for left_category, right_category in zip(categories, categories[1:]):
        left_point = points_by_category_id.get(str(left_category.category_id))
        right_point = points_by_category_id.get(str(right_category.category_id))
        if left_point is not None and right_point is not None:
            draw.line(
                (left_point, right_point),
                fill=tuple(render_params.connector_color_rgb),
                width=int(render_params.line_width_px),
            )
    support: Dict[str, BBox] = {}
    values: Dict[str, BBox] = {}
    mark_points: Dict[str, Point] = {}
    entities: List[Dict[str, Any]] = []
    radius = int(render_params.point_radius_px)
    for category in categories:
        point = points_by_category_id.get(str(category.category_id))
        if point is None:
            continue
        x, y = point
        value = int(panel.values_by_category_id[str(category.category_id)])
        point_box = _bbox_tuple((x - radius, y - radius, x + radius, y + radius))
        draw.ellipse(point_box, fill=tuple(category.color_rgb), outline=(42, 48, 58), width=2)
        value_anchor_y = y - 10 if y - 10 > plot_bbox[1] + 14 else y + 16
        value_anchor = "mb" if value_anchor_y < y else "mt"
        value_bbox = _draw_text(draw, (x, value_anchor_y), str(value), font=value_font, fill=tuple(render_params.text_color_rgb), anchor=value_anchor, stroke_width=dense_stroke_width())
        label_bbox = _draw_text(draw, (x, plot_bbox[3] + 16), str(category.label), font=label_font, fill=tuple(render_params.text_color_rgb), anchor="mt", stroke_width=dense_stroke_width())
        support[str(category.category_id)] = _pad_bbox(_union_bboxes([point_box, value_bbox, label_bbox]), 4)
        values[str(category.category_id)] = value_bbox
        mark_points[str(category.category_id)] = (int(round(x)), int(round(y)))
        entities.append(
            {
                "entity_id": f"{panel.panel_id}:{category.category_id}",
                "entity_type": "dashboard_line_point",
                "bbox_xyxy": list(support[str(category.category_id)]),
                "point_xy": list(mark_points[str(category.category_id)]),
                "attrs": {
                    "panel_id": str(panel.panel_id),
                    "panel_name": str(panel.name),
                    "category_id": str(category.category_id),
                    "category_label": str(category.label),
                    "value": int(value),
                },
            }
        )
    return support, values, mark_points, entities


def _draw_donut_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Panel,
    panel_bbox: BBox,
    categories: Sequence[Category],
    render_params: RenderParams,
) -> Tuple[Dict[str, BBox], Dict[str, BBox], Dict[str, Point], List[Dict[str, Any]]]:
    """Draw one donut panel and use segment centers as category witness points."""

    label_size = 9 if len(categories) > 12 else int(render_params.label_font_size_px)
    value_size = 9 if len(categories) > 12 else int(render_params.value_font_size_px)
    label_font = load_font(int(label_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    value_font = load_font(int(value_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    x0, y0, x1, y1 = panel_bbox
    panel_width = int(x1 - x0)
    panel_height = int(y1 - y0)
    donut_center = (x0 + max(86, int(panel_width * 0.29)), y0 + max(142, int(panel_height * 0.58)))
    radius = int(min(86, panel_width * 0.22, panel_height * 0.30))
    donut_box = (donut_center[0] - radius, donut_center[1] - radius, donut_center[0] + radius, donut_center[1] + radius)
    visible_categories = [
        category
        for category in categories
        if str(category.category_id) in panel.values_by_category_id
    ]
    total = sum(
        int(panel.values_by_category_id[str(category.category_id)])
        for category in visible_categories
    )
    start_angle = -90.0
    segment_points: Dict[str, Point] = {}
    for category in visible_categories:
        value = int(panel.values_by_category_id[str(category.category_id)])
        sweep = 360.0 * (float(value) / float(max(1, total)))
        draw.pieslice(donut_box, start=start_angle, end=start_angle + sweep, fill=tuple(category.color_rgb), outline=(255, 255, 255), width=2)
        mid_angle = math.radians(start_angle + sweep / 2.0)
        mid_radius = float(radius) * 0.72
        segment_points[str(category.category_id)] = (
            int(round(float(donut_center[0]) + math.cos(mid_angle) * mid_radius)),
            int(round(float(donut_center[1]) + math.sin(mid_angle) * mid_radius)),
        )
        start_angle += sweep
    inner_radius = max(34, int(radius * 0.46))
    inner_box = (donut_center[0] - inner_radius, donut_center[1] - inner_radius, donut_center[0] + inner_radius, donut_center[1] + inner_radius)
    draw.ellipse(inner_box, fill=tuple(render_params.donut_hole_fill_rgb), outline=tuple(render_params.panel_border_rgb), width=1)
    _draw_text(draw, donut_center, "Value", font=value_font, fill=tuple(render_params.muted_text_color_rgb), anchor="mm")
    support: Dict[str, BBox] = {}
    values: Dict[str, BBox] = {}
    entities: List[Dict[str, Any]] = []
    legend_x = x0 + max(190, int(panel_width * 0.54))
    legend_y = y0 + 58
    row_h = max(12, int((y1 - legend_y - 18) / max(1, len(visible_categories))))
    for index, category in enumerate(visible_categories):
        cy = legend_y + index * row_h
        swatch_size = min(12, max(7, int(row_h - 3)))
        row_mid = cy + row_h / 2.0
        swatch_box = (legend_x, int(row_mid - swatch_size / 2), legend_x + swatch_size, int(row_mid + swatch_size / 2))
        draw.rounded_rectangle(swatch_box, radius=3, fill=tuple(category.color_rgb), outline=(70, 76, 86), width=1)
        label_bbox = _draw_text(draw, (legend_x + swatch_size + 8, row_mid), str(category.label), font=label_font, fill=tuple(render_params.text_color_rgb), anchor="lm")
        value = int(panel.values_by_category_id[str(category.category_id)])
        value_bbox = _draw_text(draw, (x1 - 14, row_mid), str(value), font=value_font, fill=tuple(render_params.text_color_rgb), anchor="rm")
        row_box = _union_bboxes([swatch_box, label_bbox, value_bbox])
        support[str(category.category_id)] = _pad_bbox(row_box, 5)
        values[str(category.category_id)] = value_bbox
        entities.append(
            {
                "entity_id": f"{panel.panel_id}:{category.category_id}",
                "entity_type": "dashboard_donut_legend_row",
                "bbox_xyxy": list(support[str(category.category_id)]),
                "point_xy": list(segment_points[str(category.category_id)]),
                "attrs": {
                    "panel_id": str(panel.panel_id),
                    "panel_name": str(panel.name),
                    "category_id": str(category.category_id),
                    "category_label": str(category.label),
                    "value": int(value),
                },
            }
        )
    return support, values, segment_points, entities


def _draw_radar_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Panel,
    panel_bbox: BBox,
    categories: Sequence[Category],
    render_params: RenderParams,
) -> Tuple[Dict[str, BBox], Dict[str, BBox], Dict[str, Point], List[Dict[str, Any]]]:
    """Draw one radar panel and preserve each vertex point for annotations."""

    label_size = 9 if len(categories) > 12 else int(render_params.label_font_size_px)
    value_size = 9 if len(categories) > 12 else int(render_params.value_font_size_px)
    label_font = load_font(int(label_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    value_font = load_font(int(value_size), bold=dense_fit_bold(), font_family=render_params.font_family)
    x0, y0, x1, y1 = panel_bbox
    panel_width = float(x1 - x0)
    panel_height = float(y1 - y0)
    center = ((x0 + x1) / 2.0, y0 + panel_height * 0.60)
    radius = min(92.0, panel_width * 0.22, panel_height * 0.24)
    for frac in (0.25, 0.5, 0.75, 1.0):
        points = []
        for index in range(len(categories)):
            angle = -math.pi / 2.0 + (2.0 * math.pi * float(index) / float(len(categories)))
            points.append((center[0] + math.cos(angle) * radius * frac, center[1] + math.sin(angle) * radius * frac))
        draw.polygon(points, outline=tuple(render_params.grid_color_rgb))
    vertex_by_category_id: Dict[str, Tuple[float, float]] = {}
    support: Dict[str, BBox] = {}
    values: Dict[str, BBox] = {}
    mark_points: Dict[str, Point] = {}
    entities: List[Dict[str, Any]] = []
    point_radius = int(render_params.point_radius_px)
    for index, category in enumerate(categories):
        angle = -math.pi / 2.0 + (2.0 * math.pi * float(index) / float(len(categories)))
        axis_end = (center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius)
        draw.line((center[0], center[1], axis_end[0], axis_end[1]), fill=tuple(render_params.grid_color_rgb), width=1)
        if str(category.category_id) not in panel.values_by_category_id:
            continue
        value = int(panel.values_by_category_id[str(category.category_id)])
        r_value = radius * (float(value) / 100.0)
        vx = center[0] + math.cos(angle) * r_value
        vy = center[1] + math.sin(angle) * r_value
        vertex_by_category_id[str(category.category_id)] = (vx, vy)
    category_ring = tuple(categories)
    for left_category, right_category in zip(category_ring, category_ring[1:] + category_ring[:1]):
        left_point = vertex_by_category_id.get(str(left_category.category_id))
        right_point = vertex_by_category_id.get(str(right_category.category_id))
        if left_point is not None and right_point is not None:
            draw.line(
                (left_point, right_point),
                fill=tuple(render_params.connector_color_rgb),
                width=int(render_params.line_width_px),
            )
    for index, category in enumerate(categories):
        point = vertex_by_category_id.get(str(category.category_id))
        if point is None:
            continue
        angle = -math.pi / 2.0 + (2.0 * math.pi * float(index) / float(len(categories)))
        value = int(panel.values_by_category_id[str(category.category_id)])
        label_radius = radius + 17
        label_x = center[0] + math.cos(angle) * label_radius
        label_y = center[1] + math.sin(angle) * label_radius
        label_bbox = _draw_text(draw, (label_x, label_y), str(category.label), font=label_font, fill=tuple(render_params.text_color_rgb), anchor="mm", stroke_width=dense_stroke_width())
        point_box = _bbox_tuple((point[0] - point_radius, point[1] - point_radius, point[0] + point_radius, point[1] + point_radius))
        draw.ellipse(point_box, fill=tuple(category.color_rgb), outline=(42, 48, 58), width=2)
        value_offset = -20 if int(value) >= 70 else 17
        value_x = point[0] + math.cos(angle) * value_offset
        value_y = point[1] + math.sin(angle) * value_offset
        value_bbox = _draw_text(draw, (value_x, value_y), str(value), font=value_font, fill=tuple(render_params.text_color_rgb), anchor="mm", stroke_width=dense_stroke_width())
        support[str(category.category_id)] = _pad_bbox(_union_bboxes([label_bbox, point_box, value_bbox]), 4)
        values[str(category.category_id)] = value_bbox
        mark_points[str(category.category_id)] = (int(round(point[0])), int(round(point[1])))
        entities.append(
            {
                "entity_id": f"{panel.panel_id}:{category.category_id}",
                "entity_type": "dashboard_radar_vertex",
                "bbox_xyxy": list(support[str(category.category_id)]),
                "point_xy": list(mark_points[str(category.category_id)]),
                "attrs": {
                    "panel_id": str(panel.panel_id),
                    "panel_name": str(panel.name),
                    "category_id": str(category.category_id),
                    "category_label": str(category.label),
                    "value": int(value),
                },
            }
        )
    return support, values, mark_points, entities


def _fit_font_for_width(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font_family: str,
    max_width_px: int,
    start_size_px: int,
    min_size_px: int = 11,
):
    for size in range(int(start_size_px), int(min_size_px) - 1, -1):
        font = load_font(int(size), bold=False, font_family=str(font_family))
        if draw.textbbox((0, 0), str(text), font=font)[2] <= int(max_width_px):
            return font
    return load_font(int(min_size_px), bold=False, font_family=str(font_family))


def _draw_statement_option_panel(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: DashboardDataset,
    render_params: RenderParams,
    params: Mapping[str, Any],
) -> Tuple[Dict[str, BBox], List[Dict[str, Any]]]:
    """Render visual answer options without making option boxes annotation targets."""

    raw_options = dataset.query.params.get("statement_options", ())
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)) or not raw_options:
        return {}, []
    options = [dict(option) for option in raw_options if isinstance(option, Mapping)]
    if not options:
        return {}, []

    defaults = _option_panel_render_defaults(params)
    height_px = int(defaults["height_px"])
    pad = int(defaults["padding_px"])
    column_count = max(1, min(2, int(defaults["column_count"])))
    column_gap = max(0, int(defaults["column_gap_px"]))
    x0 = int(render_params.dashboard_margin_px)
    x1 = int(render_params.canvas_width) - int(render_params.dashboard_margin_px)
    y1 = int(render_params.canvas_height) - int(render_params.dashboard_margin_px)
    y0 = int(y1) - int(height_px)
    panel_bbox = (int(x0), int(y0), int(x1), int(y1))
    fill_rgb = render_rgb(params, "option_panel_fill_rgb", render_params.panel_fill_rgb)
    border_rgb = render_rgb(params, "option_panel_border_rgb", render_params.panel_border_rgb)
    draw.rounded_rectangle(panel_bbox, radius=10, fill=tuple(fill_rgb), outline=tuple(border_rgb), width=2)

    title_font = load_font(16, bold=False, font_family=render_params.font_family)
    title_bbox = _draw_text(
        draw,
        (x0 + pad, y0 + pad + 2),
        "Statement options",
        font=title_font,
        fill=tuple(render_params.muted_text_color_rgb),
        anchor="la",
    )
    content_top = int(max(title_bbox[3] + 8, y0 + pad + 24))
    row_count = int(math.ceil(float(len(options)) / float(column_count)))
    row_height = max(24, int((y1 - pad - content_top) / max(1, row_count)))
    letter_font = load_font(int(defaults["letter_font_size_px"]), bold=False, font_family=render_params.font_family)
    usable_width = max(1, int(x1 - x0 - (2 * pad)))
    column_width = int((usable_width - (column_gap * (column_count - 1))) / column_count)
    max_text_width = max(80, int(column_width - 46))
    bboxes: Dict[str, BBox] = {}
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "statement_option_panel",
            "entity_type": "dashboard_statement_option_panel",
            "bbox_xyxy": list(panel_bbox),
            "attrs": {
                "option_count": int(len(options)),
                "column_count": int(column_count),
                "row_count": int(row_count),
                "excluded_from_annotation": True,
            },
        }
    ]
    for index, option in enumerate(options):
        option_label = str(option.get("option_label", OPTION_LETTERS[int(index)]))
        option_id = str(option.get("option_id", f"option_{option_label}"))
        statement_text = str(option.get("text", ""))
        row_index = int(index) // int(column_count)
        column_index = int(index) % int(column_count)
        column_x0 = int(x0 + pad + (column_index * (column_width + column_gap)))
        row_y0 = int(content_top + row_index * row_height)
        row_mid_y = int(row_y0 + row_height / 2)
        letter_bbox = _draw_text(
            draw,
            (column_x0 + 14, row_mid_y),
            f"{option_label}.",
            font=letter_font,
            fill=tuple(render_params.text_color_rgb),
            anchor="mm",
        )
        statement_font = _fit_font_for_width(
            draw,
            text=statement_text,
            font_family=str(render_params.font_family),
            max_width_px=int(max_text_width),
            start_size_px=int(defaults["font_size_px"]),
        )
        text_bbox = _draw_text(
            draw,
            (column_x0 + 40, row_mid_y),
            statement_text,
            font=statement_font,
            fill=tuple(render_params.text_color_rgb),
            anchor="lm",
        )
        option_bbox = _clip_bbox_to_canvas(
            _pad_bbox(_union_bboxes([letter_bbox, text_bbox]), 4),
            width=int(render_params.canvas_width),
            height=int(render_params.canvas_height),
        )
        bboxes[str(option_id)] = option_bbox
        entities.append(
            {
                "entity_id": str(option_id),
                "entity_type": "dashboard_statement_option",
                "bbox_xyxy": list(option_bbox),
                "attrs": {
                    "option_label": str(option_label),
                    "statement_text": str(statement_text),
                    "truth_value": bool(option.get("truth_value", False)),
                    "statement_kind": str(option.get("statement_kind", "")),
                    "comparison": str(option.get("comparison", "")),
                    "first_panel_id": str(option.get("first_panel_id", "")),
                    "first_category_id": str(option.get("first_category_id", "")),
                    "second_panel_id": str(option.get("second_panel_id", "")),
                    "second_category_id": str(option.get("second_category_id", "")),
                    "excluded_from_annotation": True,
                },
            }
        )
    return bboxes, entities


def render_dashboard(
    background: Image.Image,
    *,
    dataset: DashboardDataset,
    render_params: RenderParams,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedDashboard:
    """Render the dashboard scene after the public task has fixed answer semantics."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    context_params = resolve_context_text_params(params)
    has_option_panel = bool(dataset.query.params.get("statement_options"))
    chart_context_mode, chart_context_mode_weights = _resolve_chart_context_mode(
        context_params=context_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{dataset.scene_variant}",
    )
    if bool(has_option_panel):
        chart_context_mode = "clean"
        chart_context_mode_weights = {"clean": 1.0, "minimal": 0.0, "paragraph_box": 0.0}
    context_params["dashboard_title_enabled"] = bool(chart_context_mode in {"minimal", "paragraph_box"})
    if chart_context_mode != "paragraph_box":
        context_params["context_text_enabled"] = False
    context_layout = resolve_dashboard_context_layout(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{dataset.scene_variant}",
        params=context_params,
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        top_reserved_px=int(render_params.title_height_px + render_params.dashboard_margin_px),
        bottom_reserved_px=int(render_params.dashboard_margin_px),
        left_margin_px=int(render_params.dashboard_margin_px),
        right_margin_px=int(render_params.dashboard_margin_px),
    )
    context_layout = {
        **dict(context_layout),
        "context_profile": str(context_params.get("chart_context_profile", context_params.get("context_text_profile", "report_paragraph"))),
        "chart_context_mode": str(chart_context_mode),
        "chart_context_mode_weights": dict(chart_context_mode_weights),
    }
    context_layout_mode = _dashboard_context_layer_mode(context_layout)
    option_layout_meta: Dict[str, Any] = {}
    if bool(has_option_panel):
        option_defaults = _option_panel_render_defaults(params)
        option_layout_meta = {
            "option_panel_height_px": int(option_defaults["height_px"]),
            "option_panel_gap_px": int(option_defaults["gap_px"]),
        }
    render_params = RenderParams(
        **{
            **render_params.__dict__,
            "layout_jitter_meta": {
                **dict(render_params.layout_jitter_meta),
                "context_text_layout_mode": str(context_layout_mode),
                "context_text_placement": str(context_layout.get("placement", "none")),
                "context_text_box_count": int(context_layout.get("box_count", 0)),
                "context_text_sidebar_width_px": int(context_layout.get("sidebar_width_px", 0)),
                "context_text_sidebar_gap_px": int(context_layout.get("sidebar_gap_px", 0)),
                "context_text_bottom_band_height_px": int(context_layout.get("bottom_band_height_px", 0)),
                "context_text_bottom_band_gap_px": int(context_layout.get("bottom_band_gap_px", 0)),
                **dict(option_layout_meta),
            },
        }
    )
    context_elements = draw_dashboard_reserved_margin_context(
        image,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{dataset.scene_variant}",
        params=context_params,
        text_rgb=tuple(render_params.text_color_rgb),
        muted_text_rgb=tuple(render_params.muted_text_color_rgb),
        panel_fill_rgb=tuple(render_params.panel_fill_rgb),
        panel_border_rgb=tuple(render_params.panel_border_rgb),
        accent_rgb=tuple(render_params.connector_color_rgb),
        top_reserved_px=int(render_params.title_height_px + render_params.dashboard_margin_px),
        bottom_reserved_px=int(render_params.dashboard_margin_px),
        left_margin_px=int(render_params.dashboard_margin_px),
        right_margin_px=int(render_params.dashboard_margin_px),
        layout_spec=context_layout,
    )
    layout = _panel_layout(render_params, panel_count=len(dataset.panels))
    title_record = sample_dashboard_title(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{dataset.scene_variant}",
        params=context_params,
    )
    if bool(title_record.get("enabled", False)):
        title_font_family = sample_font_family(
            role="context",
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{dataset.scene_variant}.main_title_font",
            params=context_params,
            exclude_tags=("mono", "display"),
            explicit_key="dashboard_title_font_family",
            weights_key="context_text_font_family_weights",
        )
        title_font = load_font(int(render_params.title_font_size_px), bold=False, font_family=title_font_family)
        title_min_x = min(panel_bbox[0] for panel_bbox in layout) if layout else int(render_params.dashboard_margin_px)
        title_max_x = max(panel_bbox[2] for panel_bbox in layout) if layout else int(render_params.canvas_width)
        title_text = str(title_record.get("text", ""))
        title_available_width = max(180, int(title_max_x) - int(title_min_x) - 60)
        if draw.textbbox((0, 0), title_text, font=title_font)[2] > int(title_available_width):
            title_words = title_text.split()
            while len(title_words) > 1 and draw.textbbox((0, 0), f"{' '.join(title_words)}...", font=title_font)[2] > int(title_available_width):
                title_words.pop()
            title_text = f"{' '.join(title_words)}..." if title_words else "Dashboard"
        title_bbox = _draw_text(
            draw,
            (
                (float(title_min_x) + float(title_max_x)) / 2.0,
                35 + float(render_params.layout_offset_y_px),
            ),
            title_text,
            font=title_font,
            fill=tuple(render_params.text_color_rgb),
            anchor="mm",
        )
        title_element = {
            "context_id": f"context_{len(context_elements):02d}",
            "role": "main_title",
            "text": str(title_text),
            "bbox_xyxy": list(title_bbox),
            "manifest_path": str(title_record.get("manifest_path", "")),
            "source_ids": list(title_record.get("source_ids", [])),
            "row_index": int(title_record.get("row_index", -1)),
            "layout_mode": str(context_layout_mode),
            "font_family": str(title_font_family),
            "excluded_from_answer": True,
        }
        context_element_records = [element.to_trace() for element in context_elements] + [dict(title_element)]
    else:
        context_element_records = [element.to_trace() for element in context_elements]
    panel_bboxes: Dict[str, BBox] = {}
    support_bboxes: Dict[str, Dict[str, BBox]] = {}
    support_points: Dict[str, Dict[str, Point]] = {}
    value_label_bboxes: Dict[str, Dict[str, BBox]] = {}
    option_statement_bboxes: Dict[str, BBox] = {}
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": str(element["context_id"]),
            "entity_type": "non_answer_context_text",
            "bbox_xyxy": list(element["bbox_xyxy"]),
            "attrs": {
                "role": str(element["role"]),
                "text": str(element["text"]),
                "manifest_path": str(element["manifest_path"]),
                "source_ids": list(element["source_ids"]),
                "row_index": int(element["row_index"]),
                "layout_mode": str(element["layout_mode"]),
                "excluded_from_answer": bool(element["excluded_from_answer"]),
            },
        }
        for element in context_element_records
    ]
    for index, panel in enumerate(dataset.panels):
        panel_bbox = layout[int(index)]
        panel_bboxes[str(panel.panel_id)] = panel_bbox
        title_bbox = _draw_panel_chrome(draw, panel=panel, panel_bbox=panel_bbox, render_params=render_params)
        entities.append(
            {
                "entity_id": f"panel:{panel.panel_id}",
                "entity_type": "dashboard_chart_panel",
                "bbox_xyxy": list(panel_bbox),
                "attrs": {"panel_id": str(panel.panel_id), "panel_name": str(panel.name), "panel_kind": str(panel.kind), "title_bbox": list(title_bbox)},
            }
        )
        if str(panel.kind) == "bar":
            support, value_bboxes, mark_points, mark_entities = _draw_bar_panel(draw, panel=panel, panel_bbox=panel_bbox, categories=dataset.categories, render_params=render_params)
        elif str(panel.kind) == "line":
            support, value_bboxes, mark_points, mark_entities = _draw_line_panel(draw, panel=panel, panel_bbox=panel_bbox, categories=dataset.categories, render_params=render_params)
        elif str(panel.kind) == "donut":
            support, value_bboxes, mark_points, mark_entities = _draw_donut_panel(draw, panel=panel, panel_bbox=panel_bbox, categories=dataset.categories, render_params=render_params)
        elif str(panel.kind) == "radar":
            support, value_bboxes, mark_points, mark_entities = _draw_radar_panel(draw, panel=panel, panel_bbox=panel_bbox, categories=dataset.categories, render_params=render_params)
        else:
            raise ValueError(f"unsupported panel kind: {panel.kind}")
        support_bboxes[str(panel.panel_id)] = _clip_bbox_map_to_canvas(
            support,
            width=int(render_params.canvas_width),
            height=int(render_params.canvas_height),
        )
        support_points[str(panel.panel_id)] = _clip_point_map_to_canvas(
            mark_points,
            width=int(render_params.canvas_width),
            height=int(render_params.canvas_height),
        )
        value_label_bboxes[str(panel.panel_id)] = _clip_bbox_map_to_canvas(
            value_bboxes,
            width=int(render_params.canvas_width),
            height=int(render_params.canvas_height),
        )
        entities.extend(mark_entities)
    if bool(has_option_panel):
        option_statement_bboxes, option_entities = _draw_statement_option_panel(
            draw,
            dataset=dataset,
            render_params=render_params,
            params=params,
        )
        entities.extend(option_entities)
    return RenderedDashboard(
        image=image,
        entities=tuple(entities),
        panel_bboxes_px=dict(panel_bboxes),
        support_bboxes_px=dict(support_bboxes),
        support_points_px=dict(support_points),
        value_label_bboxes_px=dict(value_label_bboxes),
        option_statement_bboxes_px=dict(option_statement_bboxes),
        context_text_elements=tuple(context_element_records),
        context_text_layout=dict(context_layout),
    )
