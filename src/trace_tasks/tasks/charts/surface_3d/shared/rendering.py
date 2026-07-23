"""Rendering primitives for synthetic 3D chart scenes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.tasks.charts.shared.three_d.color import blend_rgb as _blend
from trace_tasks.tasks.charts.shared.three_d.geometry import point_bbox as _point_bbox
from trace_tasks.tasks.charts.shared.three_d.geometry import round_bbox as _bbox
from trace_tasks.tasks.charts.shared.three_d.projection import axis_line_position as _axis_line_position
from trace_tasks.tasks.charts.shared.three_d.projection import project_ranged_point_3d as _project_3d
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.visual_defaults import sample_chart_font_family
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import (
    PANEL_VARIANT,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    RENDER_NAMESPACE,
    SCATTER_VARIANT,
    SURFACE_VARIANT,
)
from .state import (
    BBox,
    Panel3D,
    Point3D,
    RenderedSurface3D,
    RGB,
    Surface3DDataset,
    Surface3DRenderArtifacts,
    Surface3DRenderParams,
)


def _render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def _resolve_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=_render_style_seed(params),
        namespace=RENDER_NAMESPACE,
    )


def _resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(
        resolve_render_int(
            params,
            RENDER_DEFAULTS,
            str(key),
            int(fallback),
            instance_seed=_render_style_seed(params),
            namespace=RENDER_NAMESPACE,
        )
    )


def resolve_render_params(params: Mapping[str, Any]) -> Surface3DRenderParams:
    """Resolve scene render settings and layout jitter from config."""

    left = int(params.get("plot_margin_left_px", group_default(RENDER_DEFAULTS, "plot_margin_left_px", 132)))
    right = int(params.get("plot_margin_right_px", group_default(RENDER_DEFAULTS, "plot_margin_right_px", 118)))
    top = int(params.get("plot_margin_top_px", group_default(RENDER_DEFAULTS, "plot_margin_top_px", 98)))
    bottom = int(params.get("plot_margin_bottom_px", group_default(RENDER_DEFAULTS, "plot_margin_bottom_px", 132)))
    left, right, top, bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(left),
        right_px=int(right),
        top_px=int(top),
        bottom_px=int(bottom),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=_render_style_seed(params),
        namespace=f"{RENDER_NAMESPACE}.layout",
    )
    return Surface3DRenderParams(
        canvas_width=int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", 1280))),
        canvas_height=int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", 860))),
        plot_margin_left_px=int(left),
        plot_margin_right_px=int(right),
        plot_margin_top_px=int(top),
        plot_margin_bottom_px=int(bottom),
        panel_gap_px=int(params.get("panel_gap_px", group_default(RENDER_DEFAULTS, "panel_gap_px", 30))),
        point_radius_px=_resolve_int(params, "point_radius_px", 7),
        line_width_px=_resolve_int(params, "line_width_px", 4),
        axis_line_width_px=_resolve_int(params, "axis_line_width_px", 2),
        grid_line_width_px=_resolve_int(params, "grid_line_width_px", 1),
        tick_font_size_px=int(params.get("tick_font_size_px", group_default(RENDER_DEFAULTS, "tick_font_size_px", 18))),
        label_font_size_px=int(params.get("label_font_size_px", group_default(RENDER_DEFAULTS, "label_font_size_px", 24))),
        title_font_size_px=int(params.get("title_font_size_px", group_default(RENDER_DEFAULTS, "title_font_size_px", 30))),
        panel_title_font_size_px=int(params.get("panel_title_font_size_px", group_default(RENDER_DEFAULTS, "panel_title_font_size_px", 20))),
        plot_fill_rgb=_resolve_rgb(params, "plot_fill_rgb", (255, 255, 255)),
        panel_fill_rgb=_resolve_rgb(params, "panel_fill_rgb", (255, 255, 255)),
        panel_border_rgb=_resolve_rgb(params, "panel_border_rgb", (156, 168, 184)),
        axis_color_rgb=_resolve_rgb(params, "axis_color_rgb", (32, 38, 48)),
        grid_color_rgb=_resolve_rgb(params, "grid_color_rgb", (180, 190, 204)),
        text_color_rgb=_resolve_rgb(params, "text_color_rgb", (38, 41, 48)),
        text_stroke_rgb=_resolve_rgb(params, "text_stroke_rgb", (255, 255, 255)),
        surface_low_rgb=_resolve_rgb(params, "surface_low_rgb", (71, 135, 206)),
        surface_high_rgb=_resolve_rgb(params, "surface_high_rgb", (218, 71, 61)),
        surface_edge_rgb=_resolve_rgb(params, "surface_edge_rgb", (82, 91, 76)),
        marker_outline_rgb=_resolve_rgb(params, "marker_outline_rgb", (31, 38, 49)),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
    anchor: str | None = None,
) -> BBox:
    try:
        box = draw.textbbox(
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            stroke_width=max(0, int(stroke_width)),
            anchor=anchor,
        )
        return _bbox(box)
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return _bbox([float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)])


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int = 0,
    anchor: str | None = None,
) -> BBox:
    kwargs: dict[str, Any] = {}
    if anchor is not None:
        kwargs["anchor"] = str(anchor)
    draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=max(0, int(stroke_width)),
        role="readout",
        required=False,
        **kwargs,
    )
    return _text_bbox(draw, xy, str(text), font, stroke_width=max(0, int(stroke_width)), anchor=anchor)


def _draw_3d_axes(
    draw: ImageDraw.ImageDraw,
    *,
    plot_bbox: Sequence[float],
    dataset: Surface3DDataset,
    params: Surface3DRenderParams,
    tick_values: Sequence[float] = (0.0, 0.5, 1.0),
    label_xy_ticks: bool = True,
    label_y_ticks: bool = True,
    label_z_ticks: bool = True,
) -> None:
    """Draw the shared projected axes for all surface_3d variants."""

    plot_width = float(plot_bbox[2]) - float(plot_bbox[0])
    plot_height = float(plot_bbox[3]) - float(plot_bbox[1])
    compact_axis_labels = bool(plot_width < 320.0 or plot_height < 260.0)
    axis_font = load_font(max(12, int(params.tick_font_size_px)), bold=True)
    label_size = max(12, int(params.label_font_size_px) - (7 if compact_axis_labels else 0))
    label_font = load_font(max(14, int(label_size)), bold=True)
    x0, y0 = _project_3d(
        dataset.x_range[0],
        dataset.y_range[0],
        dataset.z_range[0],
        plot_bbox=plot_bbox,
        x_range=dataset.x_range,
        y_range=dataset.y_range,
        z_range=dataset.z_range,
    )
    x_tip = _project_3d(dataset.x_range[1], dataset.y_range[0], dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
    y_tip = _project_3d(dataset.x_range[0], dataset.y_range[1], dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
    z_tip = _project_3d(dataset.x_range[0], dataset.y_range[0], dataset.z_range[1], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
    for start, end in [((x0, y0), x_tip), ((x0, y0), y_tip), ((x0, y0), z_tip)]:
        draw.line([start, end], fill=params.axis_color_rgb, width=max(2, int(params.axis_line_width_px)))

    for tick in tick_values:
        xv = float(dataset.x_range[0]) + float(tick) * (float(dataset.x_range[1]) - float(dataset.x_range[0]))
        yv = float(dataset.y_range[0]) + float(tick) * (float(dataset.y_range[1]) - float(dataset.y_range[0]))
        zv = float(dataset.z_range[0]) + float(tick) * (float(dataset.z_range[1]) - float(dataset.z_range[0]))
        x_base = _project_3d(xv, dataset.y_range[0], dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        x_grid = _project_3d(xv, dataset.y_range[1], dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        y_base = _project_3d(dataset.x_range[0], yv, dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        y_grid = _project_3d(dataset.x_range[1], yv, dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        z_base = _project_3d(dataset.x_range[0], dataset.y_range[0], zv, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        z_grid_x = _project_3d(dataset.x_range[1], dataset.y_range[0], zv, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        z_grid_y = _project_3d(dataset.x_range[0], dataset.y_range[1], zv, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        draw.line([x_base, x_grid], fill=params.grid_color_rgb, width=max(1, int(params.grid_line_width_px)))
        draw.line([y_base, y_grid], fill=params.grid_color_rgb, width=max(1, int(params.grid_line_width_px)))
        draw.line([z_base, z_grid_x], fill=params.grid_color_rgb, width=max(1, int(params.grid_line_width_px)))
        draw.line([z_base, z_grid_y], fill=params.grid_color_rgb, width=max(1, int(params.grid_line_width_px)))
        if label_xy_ticks:
            x_label = str(int(round(xv)))
            if dataset.x_labels:
                xi = int(round(float(tick) * (len(dataset.x_labels) - 1)))
                x_label = str(dataset.x_labels[max(0, min(len(dataset.x_labels) - 1, xi))])
            _draw_text(draw, (x_base[0], x_base[1] + 20), x_label, font=axis_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="mt")
        if label_y_ticks:
            y_label = str(int(round(yv)))
            if dataset.y_labels:
                yi = int(round(float(tick) * (len(dataset.y_labels) - 1)))
                y_label = str(dataset.y_labels[max(0, min(len(dataset.y_labels) - 1, yi))])
            _draw_text(draw, (y_base[0] - 15, y_base[1] + 6), y_label, font=axis_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="rt")
        if label_z_ticks:
            _draw_text(draw, (z_base[0] - 13, z_base[1]), str(int(round(zv))), font=axis_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="rm")

    if dataset.x_axis_label:
        if compact_axis_labels:
            x_label_xy = (x_tip[0] + 34.0, x_tip[1] + 30.0)
            x_label_anchor = "lm"
        else:
            x_label_xy = (x_tip[0] + 52.0, x_tip[1] + 22.0)
            x_label_anchor = "lm"
        _draw_text(draw, x_label_xy, dataset.x_axis_label, font=label_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor=x_label_anchor)
    if dataset.y_axis_label:
        y_label_xy = _axis_line_position((x0, y0), y_tip, fraction=0.76 if compact_axis_labels else 0.82)
        _draw_text(draw, y_label_xy, dataset.y_axis_label, font=label_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="mm")
    if dataset.z_axis_label:
        z_label_xy = (z_tip[0] - (64.0 if compact_axis_labels else 86.0), z_tip[1] - 10.0)
        _draw_text(draw, z_label_xy, dataset.z_axis_label, font=label_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="rm")


def _draw_point_label(
    draw: ImageDraw.ImageDraw,
    *,
    point_xy: tuple[float, float],
    label: str,
    font: ImageFont.ImageFont,
    params: Surface3DRenderParams,
    radius: float,
) -> None:
    px, py = float(point_xy[0]), float(point_xy[1])
    candidates = (
        ((px + radius + 8.0, py - radius - 8.0), "lt"),
        ((px - radius - 8.0, py - radius - 8.0), "rt"),
        ((px + radius + 8.0, py + radius + 8.0), "la"),
        ((px - radius - 8.0, py + radius + 8.0), "ra"),
    )
    best_xy, best_anchor = candidates[0]
    for xy, anchor in candidates:
        box = _text_bbox(draw, xy, str(label), font, stroke_width=2, anchor=anchor)
        if 10 <= box[0] and box[2] <= float(params.canvas_width) - 10 and 10 <= box[1] and box[3] <= float(params.canvas_height) - 10:
            best_xy, best_anchor = xy, anchor
            break
    _draw_text(draw, best_xy, str(label), font=font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor=best_anchor)


def _draw_point(
    draw: ImageDraw.ImageDraw,
    *,
    point: Point3D,
    plot_bbox: Sequence[float],
    dataset: Surface3DDataset,
    params: Surface3DRenderParams,
    point_font: ImageFont.ImageFont,
    label_points: bool = True,
) -> BBox:
    px, py = _project_3d(point.x_value, point.y_value, point.z_value, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
    radius = float(params.point_radius_px)
    bbox = _point_bbox(px, py, radius)
    if point.shape == "square":
        draw.rounded_rectangle(bbox, radius=2, fill=point.color_rgb, outline=params.marker_outline_rgb, width=2)
    elif point.shape == "triangle":
        draw.polygon([(px, py - radius), (px - radius, py + radius), (px + radius, py + radius)], fill=point.color_rgb, outline=params.marker_outline_rgb)
    else:
        draw.ellipse(bbox, fill=point.color_rgb, outline=params.marker_outline_rgb, width=2)
    if label_points and point.label:
        _draw_point_label(draw, point_xy=(px, py), label=point.label, font=point_font, params=params, radius=radius)
    return bbox


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: RGB,
    width: int,
    dash_px: float = 8.0,
    gap_px: float = 6.0,
) -> None:
    """Draw one dashed line segment in screen space."""

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = x1 - x0
    dy = y1 - y0
    distance = math.hypot(dx, dy)
    if distance <= 0.0:
        return
    ux = dx / distance
    uy = dy / distance
    cursor = 0.0
    dash = max(1.0, float(dash_px))
    gap = max(0.0, float(gap_px))
    while cursor < distance:
        segment_end = min(distance, cursor + dash)
        draw.line(
            [
                (x0 + ux * cursor, y0 + uy * cursor),
                (x0 + ux * segment_end, y0 + uy * segment_end),
            ],
            fill=fill,
            width=max(1, int(width)),
        )
        cursor = segment_end + gap


def _draw_scatter_floor_guides(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Surface3DDataset,
    plot_bbox: Sequence[float],
    params: Surface3DRenderParams,
) -> list[dict[str, Any]]:
    """Draw floor projections and optional y-reference line for 3D scatter readout."""

    entities: list[dict[str, Any]] = []
    z_floor = float(dataset.z_range[0])
    guide_rgb = _blend(params.grid_color_rgb, params.text_color_rgb, 0.28)
    reference_rgb = _blend(params.axis_color_rgb, params.surface_high_rgb, 0.45)
    if dataset.reference_y_value is not None:
        y_value = max(float(dataset.y_range[0]), min(float(dataset.y_range[1]), float(dataset.reference_y_value)))
        start_xy = _project_3d(dataset.x_range[0], y_value, z_floor, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        end_xy = _project_3d(dataset.x_range[1], y_value, z_floor, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        _draw_dashed_line(draw, start_xy, end_xy, fill=reference_rgb, width=max(2, int(params.axis_line_width_px)), dash_px=12.0, gap_px=7.0)
        label_font = load_font(max(12, int(params.tick_font_size_px) - 2), bold=True)
        label_bbox = _draw_text(
            draw,
            (start_xy[0] - 12.0, start_xy[1] + 3.0),
            f"y={int(round(y_value))}",
            font=label_font,
            fill=params.text_color_rgb,
            stroke_fill=params.text_stroke_rgb,
            stroke_width=1,
            anchor="rt",
        )
        entities.append(
            {
                "entity_id": "surface_3d_y_reference_line",
                "entity_type": "surface_3d_y_reference_line",
                "bbox_xyxy": _bbox(
                    bbox_union_raw(
                        [
                            list(label_bbox),
                            [
                                min(float(start_xy[0]), float(end_xy[0])),
                                min(float(start_xy[1]), float(end_xy[1])),
                                max(float(start_xy[0]), float(end_xy[0])),
                                max(float(start_xy[1]), float(end_xy[1])),
                            ],
                        ]
                    )
                ),
                "attrs": {"y_value": float(round(y_value, 3)), "excluded_from_annotation": True},
            }
        )
    for point in dataset.points:
        point_xy = _project_3d(point.x_value, point.y_value, point.z_value, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        floor_xy = _project_3d(point.x_value, point.y_value, z_floor, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        _draw_dashed_line(draw, floor_xy, point_xy, fill=guide_rgb, width=max(1, int(params.grid_line_width_px)), dash_px=7.0, gap_px=5.0)
        radius = max(3.0, float(params.point_radius_px) * 0.45)
        floor_bbox = _point_bbox(floor_xy[0], floor_xy[1], radius)
        draw.ellipse(floor_bbox, outline=guide_rgb, width=max(1, int(params.grid_line_width_px)))
        entities.append(
            {
                "entity_id": f"{point.point_id}_floor_projection",
                "entity_type": "surface_3d_floor_projection",
                "bbox_xyxy": list(floor_bbox),
                "attrs": {"point_id": str(point.point_id), "label": str(point.label), "excluded_from_annotation": True},
            }
        )
    return entities


def _draw_series_legend(
    draw: ImageDraw.ImageDraw,
    *,
    legend_bbox: Sequence[float],
    series_items: Sequence[tuple[str, RGB]],
    params: Surface3DRenderParams,
) -> tuple[dict[str, BBox], list[dict[str, Any]]]:
    """Draw a side legend for connected 3D series labels."""

    x0, y0, x1, y1 = (float(value) for value in legend_bbox)
    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=8,
        fill=params.panel_fill_rgb,
        outline=params.panel_border_rgb,
        width=1,
    )
    title_font = load_font(max(12, int(params.label_font_size_px) - 5), bold=False)
    label_font = load_font(max(12, int(params.label_font_size_px) - 6), bold=False)
    _draw_text(
        draw,
        (x0 + 14.0, y0 + 12.0),
        "Series",
        font=title_font,
        fill=params.text_color_rgb,
        stroke_fill=params.text_stroke_rgb,
        stroke_width=0,
        anchor="la",
    )
    row_top = y0 + 50.0
    row_step = max(28.0, min(42.0, (y1 - row_top - 12.0) / max(1.0, float(len(series_items)))))
    bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []
    for index, (label, color_rgb) in enumerate(series_items):
        cy = float(row_top + (float(index) * row_step) + (0.5 * row_step))
        swatch_x0 = x0 + 14.0
        swatch_x1 = swatch_x0 + 34.0
        draw.line(
            [(swatch_x0, cy), (swatch_x1, cy)],
            fill=tuple(int(channel) for channel in color_rgb),
            width=max(2, int(params.line_width_px)),
        )
        marker_radius = max(4.0, float(params.point_radius_px) - 2.0)
        marker_box = _point_bbox((swatch_x0 + swatch_x1) / 2.0, cy, marker_radius)
        draw.ellipse(
            marker_box,
            fill=tuple(int(channel) for channel in color_rgb),
            outline=params.marker_outline_rgb,
            width=1,
        )
        text_xy = (swatch_x1 + 12.0, cy - 0.5 * float(label_font.size))
        text_box = _draw_text(
            draw,
            text_xy,
            str(label),
            font=label_font,
            fill=params.text_color_rgb,
            stroke_fill=params.text_stroke_rgb,
            stroke_width=0,
        )
        row_bbox = _bbox(bbox_union_raw([marker_box, text_box]))
        bboxes[str(label)] = row_bbox
        entities.append(
            {
                "entity_id": f"legend_{label}",
                "entity_type": "series_legend_entry",
                "bbox_xyxy": list(row_bbox),
                "attrs": {"label": str(label), "color_rgb": [int(channel) for channel in color_rgb]},
            }
        )
    return bboxes, entities


def _render_scatter_or_lines(image: Image.Image, *, dataset: Surface3DDataset, params: Surface3DRenderParams) -> RenderedSurface3D:
    """Render point clouds or connected 3D series while preserving per-marker projected boxes."""

    draw = ImageDraw.Draw(image)
    legend_width = 188.0 if bool(dataset.connect_points_by_label) else 0.0
    legend_gap = 22.0 if bool(dataset.connect_points_by_label) else 0.0
    plot_bbox = [
        float(params.plot_margin_left_px),
        float(params.plot_margin_top_px),
        float(params.canvas_width - params.plot_margin_right_px - legend_width - legend_gap),
        float(params.canvas_height - params.plot_margin_bottom_px),
    ]
    title_font = load_font(int(params.title_font_size_px), bold=True)
    point_font = load_font(max(15, int(params.label_font_size_px) - 2), bold=True)
    draw.rounded_rectangle([plot_bbox[0] - 42, plot_bbox[1] - 50, plot_bbox[2] + 42, plot_bbox[3] + 68], radius=8, fill=params.panel_fill_rgb, outline=params.panel_border_rgb, width=2)
    draw.rectangle(plot_bbox, fill=params.plot_fill_rgb)
    _draw_text(draw, (plot_bbox[0], plot_bbox[1] - 38), dataset.title, font=title_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=1)
    _draw_3d_axes(
        draw,
        plot_bbox=plot_bbox,
        dataset=dataset,
        params=params,
        tick_values=(0.0, 0.25, 0.5, 0.75, 1.0),
        label_y_ticks=not bool(dataset.connect_points_by_label),
    )

    point_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []
    if not dataset.connect_points_by_label:
        entities.extend(_draw_scatter_floor_guides(draw, dataset=dataset, plot_bbox=plot_bbox, params=params))
    if dataset.connect_points_by_label:
        by_label: dict[str, list[Point3D]] = {}
        for point in dataset.points:
            by_label.setdefault(str(point.label), []).append(point)
        for label, points in by_label.items():
            ordered = sorted(points, key=lambda item: float(item.x_value))
            projected = [
                _project_3d(point.x_value, point.y_value, point.z_value, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
                for point in ordered
            ]
            if len(projected) >= 2:
                draw.line(projected, fill=ordered[0].color_rgb, width=int(params.line_width_px))
            entities.append({"entity_id": f"series_{label}", "entity_type": "series_3d", "bbox_xyxy": _bbox(bbox_union_raw([_point_bbox(px, py, params.point_radius_px) for px, py in projected])), "attrs": {"label": str(label)}})
        legend_items = [
            (str(label), tuple(int(channel) for channel in points[0].color_rgb))
            for label, points in by_label.items()
            if points
        ]
        _legend_bboxes, legend_entities = _draw_series_legend(
            draw,
            legend_bbox=[
                float(plot_bbox[2] + legend_gap),
                float(plot_bbox[1] + 28.0),
                float(params.canvas_width - params.plot_margin_right_px),
                float(plot_bbox[1] + 28.0 + min(260.0, max(130.0, 48.0 + 36.0 * len(legend_items)))),
            ],
            series_items=legend_items,
            params=params,
        )
        entities.extend(legend_entities)

    for point in sorted(dataset.points, key=lambda item: (float(item.y_value), float(item.x_value), float(item.z_value))):
        label_points = not dataset.connect_points_by_label
        bbox = _draw_point(draw, point=point, plot_bbox=plot_bbox, dataset=dataset, params=params, point_font=point_font, label_points=label_points)
        point_bboxes[str(point.point_id)] = bbox
        entities.append(
            {
                "entity_id": str(point.point_id),
                "entity_type": "point_3d",
                "bbox_xyxy": list(bbox),
                "attrs": {"label": str(point.label), "x_value": round(float(point.x_value), 3), "y_value": round(float(point.y_value), 3), "z_value": round(float(point.z_value), 3)},
            }
        )
    return RenderedSurface3D(image=image, entities=tuple(entities), plot_bbox_px=_bbox(plot_bbox), point_bboxes_px=dict(point_bboxes), surface_cell_bboxes_px={}, panel_bboxes_px={})


def _draw_categorical_axis_labels(draw: ImageDraw.ImageDraw, *, plot_bbox: Sequence[float], dataset: Surface3DDataset, params: Surface3DRenderParams) -> None:
    tick_font = load_font(max(12, int(params.tick_font_size_px)), bold=True)
    for x_index, x_label in enumerate(dataset.x_labels):
        px, py = _project_3d(float(x_index), dataset.y_range[0], dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        _draw_text(draw, (px, py + 24), str(x_label), font=tick_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="mt")
    for y_index, y_label in enumerate(dataset.y_labels):
        px, py = _project_3d(dataset.x_range[0], float(y_index), dataset.z_range[0], plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        _draw_text(draw, (px - 20, py + 4), str(y_label), font=tick_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=2, anchor="rt")


def _render_surface(image: Image.Image, *, dataset: Surface3DDataset, params: Surface3DRenderParams) -> RenderedSurface3D:
    draw = ImageDraw.Draw(image)
    plot_bbox = [
        float(params.plot_margin_left_px),
        float(params.plot_margin_top_px),
        float(params.canvas_width - params.plot_margin_right_px),
        float(params.canvas_height - params.plot_margin_bottom_px),
    ]
    title_font = load_font(int(params.title_font_size_px), bold=True)
    draw.rounded_rectangle([plot_bbox[0] - 42, plot_bbox[1] - 50, plot_bbox[2] + 42, plot_bbox[3] + 68], radius=8, fill=params.panel_fill_rgb, outline=params.panel_border_rgb, width=2)
    draw.rectangle(plot_bbox, fill=params.plot_fill_rgb)
    _draw_text(draw, (plot_bbox[0], plot_bbox[1] - 38), dataset.title, font=title_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=1)
    _draw_3d_axes(draw, plot_bbox=plot_bbox, dataset=dataset, params=params, label_xy_ticks=False)
    _draw_categorical_axis_labels(draw, plot_bbox=plot_bbox, dataset=dataset, params=params)

    x_count = len(dataset.x_labels)
    y_count = len(dataset.y_labels)
    values_by_xy = {(cell.x_index, cell.y_index): int(cell.value) for cell in dataset.surface_cells}
    cell_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []
    low, high = dataset.z_range
    for y_index in reversed(range(max(0, y_count - 1))):
        for x_index in range(max(0, x_count - 1)):
            corners: list[tuple[float, float]] = []
            corner_values: list[float] = []
            for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
                value = float(values_by_xy[(x_index + dx, y_index + dy)])
                corner_values.append(value)
                corners.append(_project_3d(float(x_index + dx), float(y_index + dy), value, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range))
            color = _blend(params.surface_low_rgb, params.surface_high_rgb, (sum(corner_values) / len(corner_values) - low) / max(1.0, high - low))
            draw.polygon(corners, fill=color)
            draw.line([*corners, corners[0]], fill=params.surface_edge_rgb, width=max(1, int(params.grid_line_width_px)))

    for cell in dataset.surface_cells:
        px, py = _project_3d(float(cell.x_index), float(cell.y_index), float(cell.value), plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        radius = max(4.0, float(params.point_radius_px) - 2.0)
        bbox = _point_bbox(px, py, radius)
        draw.ellipse(bbox, fill=(255, 255, 255), outline=params.marker_outline_rgb, width=2)
        cell_bboxes[str(cell.cell_id)] = bbox
        entities.append({"entity_id": str(cell.cell_id), "entity_type": "surface_cell", "bbox_xyxy": list(bbox), "attrs": {"x_label": str(cell.x_label), "y_label": str(cell.y_label), "x_index": int(cell.x_index), "y_index": int(cell.y_index), "value": int(cell.value)}})
    return RenderedSurface3D(image=image, entities=tuple(entities), plot_bbox_px=_bbox(plot_bbox), point_bboxes_px={}, surface_cell_bboxes_px=dict(cell_bboxes), panel_bboxes_px={})


def _render_small_multiples(image: Image.Image, *, dataset: Surface3DDataset, params: Surface3DRenderParams) -> RenderedSurface3D:
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(params.title_font_size_px), bold=True)
    panel_font = load_font(int(params.panel_title_font_size_px), bold=True)
    _draw_text(draw, (46, 28), dataset.title, font=title_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=1)
    cols = 3 if len(dataset.panels) > 4 else 2
    rows = int(math.ceil(len(dataset.panels) / cols))
    outer_left = float(params.plot_margin_left_px)
    outer_top = float(params.plot_margin_top_px)
    outer_right = float(params.canvas_width - params.plot_margin_right_px)
    outer_bottom = float(params.canvas_height - params.plot_margin_bottom_px)
    gap = float(params.panel_gap_px)
    panel_w = (outer_right - outer_left - gap * float(cols - 1)) / float(cols)
    panel_h = (outer_bottom - outer_top - gap * float(rows - 1)) / float(rows)
    panel_bboxes: dict[str, BBox] = {}
    point_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []
    for index, panel in enumerate(dataset.panels):
        _draw_one_panel(
            draw,
            dataset=dataset,
            panel=panel,
            index=int(index),
            cols=int(cols),
            panel_w=float(panel_w),
            panel_h=float(panel_h),
            gap=float(gap),
            outer_left=float(outer_left),
            outer_top=float(outer_top),
            params=params,
            panel_font=panel_font,
            panel_bboxes=panel_bboxes,
            point_bboxes=point_bboxes,
            entities=entities,
        )
    return RenderedSurface3D(image=image, entities=tuple(entities), plot_bbox_px=_bbox([outer_left, outer_top, outer_right, outer_bottom]), point_bboxes_px=dict(point_bboxes), surface_cell_bboxes_px={}, panel_bboxes_px=dict(panel_bboxes))


def _draw_one_panel(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Surface3DDataset,
    panel: Panel3D,
    index: int,
    cols: int,
    panel_w: float,
    panel_h: float,
    gap: float,
    outer_left: float,
    outer_top: float,
    params: Surface3DRenderParams,
    panel_font: ImageFont.ImageFont,
    panel_bboxes: dict[str, BBox],
    point_bboxes: dict[str, BBox],
    entities: list[dict[str, Any]],
) -> None:
    """Draw one mini 3D chart panel and record its projected witnesses."""

    col = int(index) % int(cols)
    row = int(index) // int(cols)
    box = [outer_left + float(col) * (panel_w + gap), outer_top + float(row) * (panel_h + gap), outer_left + float(col) * (panel_w + gap) + panel_w, outer_top + float(row) * (panel_h + gap) + panel_h]
    panel_bboxes[str(panel.panel_label)] = _bbox(box)
    draw.rounded_rectangle(box, radius=7, fill=params.panel_fill_rgb, outline=params.panel_border_rgb, width=2)
    _draw_text(draw, (box[0] + 12, box[1] + 10), str(panel.panel_label), font=panel_font, fill=params.text_color_rgb, stroke_fill=params.text_stroke_rgb, stroke_width=1)
    plot_bbox = [box[0] + 34, box[1] + 46, box[2] - 30, box[3] - 30]
    _draw_3d_axes(draw, plot_bbox=plot_bbox, dataset=dataset, params=params, tick_values=(0.0, 1.0), label_xy_ticks=False, label_z_ticks=False)
    projected: list[tuple[float, float]] = []
    for step, value in enumerate(panel.values):
        point = Point3D(
            point_id=f"panel_{panel.panel_label}_point_{step}",
            label=str(panel.panel_label),
            x_value=float(step),
            y_value=float(step % 2),
            z_value=float(value),
            color_rgb=panel.color_rgb,
        )
        px, py = _project_3d(point.x_value, point.y_value, point.z_value, plot_bbox=plot_bbox, x_range=dataset.x_range, y_range=dataset.y_range, z_range=dataset.z_range)
        projected.append((px, py))
        point_bboxes[str(point.point_id)] = _point_bbox(px, py, max(4.0, float(params.point_radius_px) - 2.0))
    if len(projected) >= 2:
        draw.line(projected, fill=panel.color_rgb, width=max(2, int(params.line_width_px)))
    for step, (px, py) in enumerate(projected):
        bbox = point_bboxes[f"panel_{panel.panel_label}_point_{step}"]
        draw.ellipse(bbox, fill=panel.color_rgb, outline=params.marker_outline_rgb, width=1)
    entities.append({"entity_id": f"panel_{panel.panel_label}", "entity_type": "three_d_panel", "bbox_xyxy": list(panel_bboxes[str(panel.panel_label)]), "attrs": {"panel_label": str(panel.panel_label), "values": [int(value) for value in panel.values], "value_range": int(max(panel.values) - min(panel.values))}})


def _render_dataset(image: Image.Image, *, dataset: Surface3DDataset, params: Surface3DRenderParams) -> RenderedSurface3D:
    if str(dataset.scene_variant) == SURFACE_VARIANT:
        return _render_surface(image, dataset=dataset, params=params)
    if str(dataset.scene_variant) == PANEL_VARIANT:
        return _render_small_multiples(image, dataset=dataset, params=params)
    if str(dataset.scene_variant) == SCATTER_VARIANT:
        return _render_scatter_or_lines(image, dataset=dataset, params=params)
    raise ValueError(f"unsupported 3D chart scene variant: {dataset.scene_variant}")


def render_surface_3d_dataset(
    *,
    dataset: Surface3DDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    font_namespace: str,
) -> Surface3DRenderArtifacts:
    """Render a dataset with sampled background, font, and post-image noise."""

    resolved_params = resolve_render_params(params)
    protected_colors = [
        *(tuple(int(channel) for channel in point.color_rgb) for point in dataset.points),
        *(tuple(int(channel) for channel in panel.color_rgb) for panel in dataset.panels),
        resolved_params.surface_low_rgb,
        resolved_params.surface_high_rgb,
        resolved_params.surface_edge_rgb,
    ]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="surface_3d",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=str(font_namespace),
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = _render_dataset(background.convert("RGB"), dataset=dataset, params=render_params)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return Surface3DRenderArtifacts(
        image=image,
        rendered_scene=rendered,
        render_params=render_params,
        background_style={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_image_noise=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


__all__ = ["render_surface_3d_dataset", "resolve_render_params"]
