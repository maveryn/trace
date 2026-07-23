"""Rendering primitives for density-curve chart scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import (
    clip_bbox_to_container as cartesian_clip_bbox_to_container,
    project_xy,
    round_bbox,
    union_bboxes as cartesian_union_bboxes,
)
from trace_tasks.tasks.charts.density_curve.shared.state import (
    BBox,
    DensityCurve,
    DensityCurveDataset,
    DensityCurveRendered,
    RGB,
)
from trace_tasks.tasks.shared.text_legibility import draw_readable_text, draw_text_traced, resolve_readable_text_style
from trace_tasks.tasks.shared.text_rendering import load_font


DASH_ON_PX = 14.0
DASH_OFF_PX = 5.0
DOT_SPACING_PX = 10.0


def bbox(values: Sequence[float]) -> BBox:
    """Return a rounded bbox list."""

    return round_bbox(values)


def bbox_union(boxes: Sequence[Sequence[float]]) -> BBox:
    """Return the union of non-empty bbox lists."""

    return cartesian_union_bboxes(boxes)


def value_to_px(
    plot_bbox: Sequence[float],
    *,
    x_value: float,
    y_value: float,
    x_min: float,
    x_max: float,
    y_max: float,
) -> Tuple[float, float]:
    """Project one chart value into pixel coordinates."""

    return project_xy(
        x_value=float(x_value),
        y_value=float(y_value),
        plot_bbox=plot_bbox,
        x_min=float(x_min),
        x_max=float(x_max),
        y_min=0.0,
        y_max=float(y_max),
        min_span=1e-9,
        clamp=True,
    )


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    fill: RGB,
    width: int,
    style: str,
) -> None:
    """Draw one styled density curve."""

    if len(points) < 2:
        return
    if str(style) == "solid":
        draw.line(tuple(points), fill=tuple(fill) + (255,), width=int(width), joint="curve")
        return
    if str(style) == "dot":
        radius = max(1.4, float(width) * 0.50)
        for point in _points_at_spacing(points, spacing_px=DOT_SPACING_PX):
            x, y = float(point[0]), float(point[1])
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tuple(fill) + (255,))
        return
    _draw_dashed_polyline(
        draw,
        points,
        fill=fill,
        width=int(width),
        dash_on_px=DASH_ON_PX,
        dash_off_px=DASH_OFF_PX,
    )


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Return Euclidean distance between two pixel points."""

    return float(math.hypot(float(point_b[0]) - float(point_a[0]), float(point_b[1]) - float(point_a[1])))


def _interpolate(point_a: Tuple[float, float], point_b: Tuple[float, float], fraction: float) -> Tuple[float, float]:
    """Interpolate between two pixel points."""

    t = max(0.0, min(1.0, float(fraction)))
    return (
        float(point_a[0]) + (float(point_b[0]) - float(point_a[0])) * t,
        float(point_a[1]) + (float(point_b[1]) - float(point_a[1])) * t,
    )


def _points_at_spacing(
    points: Sequence[Tuple[float, float]],
    *,
    spacing_px: float,
) -> Tuple[Tuple[float, float], ...]:
    """Return points sampled at approximately even pixel spacing."""

    if not points:
        return tuple()
    spacing = max(1.0, float(spacing_px))
    sampled = [tuple(float(value) for value in points[0])]
    carried = 0.0
    previous = tuple(float(value) for value in points[0])
    for target_raw in points[1:]:
        target = tuple(float(value) for value in target_raw)
        segment_length = _distance(previous, target)
        while segment_length > 1e-6 and carried + segment_length >= spacing:
            needed = spacing - carried
            point = _interpolate(previous, target, needed / segment_length)
            sampled.append(point)
            previous = point
            segment_length = _distance(previous, target)
            carried = 0.0
        carried += segment_length
        previous = target
    return tuple(sampled)


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    fill: RGB,
    width: int,
    dash_on_px: float,
    dash_off_px: float,
) -> None:
    """Draw one polyline with short, readable dash gaps."""

    if len(points) < 2:
        return
    drawing = True
    remaining = max(1.0, float(dash_on_px))
    on_length = max(1.0, float(dash_on_px))
    off_length = max(1.0, float(dash_off_px))
    previous = tuple(float(value) for value in points[0])
    for target_raw in points[1:]:
        target = tuple(float(value) for value in target_raw)
        segment_start = previous
        segment_length = _distance(segment_start, target)
        while segment_length > 1e-6:
            step = min(float(remaining), float(segment_length))
            segment_end = _interpolate(segment_start, target, step / segment_length)
            if drawing:
                draw.line((segment_start, segment_end), fill=tuple(fill) + (255,), width=int(width), joint="curve")
            segment_start = segment_end
            segment_length = _distance(segment_start, target)
            remaining -= step
            if remaining <= 1e-6:
                drawing = not drawing
                remaining = on_length if drawing else off_length
        previous = target


def curve_points_px(
    curve: DensityCurve,
    plot_bbox: Sequence[float],
    *,
    x_min: float,
    x_max: float,
    y_max: float,
) -> Tuple[Tuple[float, float], ...]:
    """Project all points for one curve."""

    return tuple(
        value_to_px(
            plot_bbox,
            x_value=point.x_value,
            y_value=point.y_value,
            x_min=x_min,
            x_max=x_max,
            y_max=y_max,
        )
        for point in curve.points
    )


def curve_point_at_x_px(
    curve: DensityCurve,
    plot_bbox: Sequence[float],
    *,
    x_value: float,
    x_min: float,
    x_max: float,
    y_max: float,
) -> list[float]:
    """Project one interpolated point on a curve at a target x-value."""

    if not curve.points:
        return []
    target_x = max(float(x_min), min(float(x_max), float(x_value)))
    previous = curve.points[0]
    for current in curve.points[1:]:
        prev_x = float(previous.x_value)
        curr_x = float(current.x_value)
        if prev_x <= target_x <= curr_x or curr_x <= target_x <= prev_x:
            fraction = 0.0 if abs(curr_x - prev_x) < 1e-9 else (target_x - prev_x) / (curr_x - prev_x)
            y_value = float(previous.y_value) + max(0.0, min(1.0, fraction)) * (
                float(current.y_value) - float(previous.y_value)
            )
            x_px, y_px = value_to_px(
                plot_bbox,
                x_value=target_x,
                y_value=y_value,
                x_min=x_min,
                x_max=x_max,
                y_max=y_max,
            )
            return [round(float(x_px), 3), round(float(y_px), 3)]
        previous = current
    nearest = min(curve.points, key=lambda point: abs(float(point.x_value) - target_x))
    x_px, y_px = value_to_px(
        plot_bbox,
        x_value=float(nearest.x_value),
        y_value=float(nearest.y_value),
        x_min=x_min,
        x_max=x_max,
        y_max=y_max,
    )
    return [round(float(x_px), 3), round(float(y_px), 3)]


def curve_bbox(points: Sequence[Tuple[float, float]], *, pad: float = 3.0) -> BBox:
    """Return a bbox covering projected curve points."""

    if not points:
        return []
    return bbox(
        (
            min(x for x, _y in points) - pad,
            min(y for _x, y in points) - pad,
            max(x for x, _y in points) + pad,
            max(y for _x, y in points) + pad,
        )
    )


def clip_bbox_to_container(bbox_values: Sequence[float], container: Sequence[float]) -> BBox:
    """Clip a bbox to a container bbox."""

    if len(bbox_values) < 4 or len(container) < 4:
        return []
    return cartesian_clip_bbox_to_container(bbox_values, container)


def render_density_curve_scene(
    image: Image.Image,
    *,
    dataset: DensityCurveDataset,
    render_params: Any,
) -> DensityCurveRendered:
    """Render one density-curve scene and record projection maps.

    This is scene-owned rendering: it may branch on semantic visible roles such
    as mean, mode, interval, or density-at-x, but it must not know public task
    ids or query ids.
    """

    draw = ImageDraw.Draw(image, "RGBA")
    label_font = load_font(int(render_params.label_font_size_px))
    tick_font = load_font(int(render_params.tick_font_size_px))
    legend_font = load_font(max(10, int(render_params.tick_font_size_px) + 1))
    plot_bbox = plot_bbox_from_margins(
        canvas_width=float(render_params.canvas_width),
        canvas_height=float(render_params.canvas_height),
        margin_left_px=float(render_params.plot_margin_left_px),
        margin_right_px=float(render_params.plot_margin_right_px),
        margin_top_px=float(render_params.plot_margin_top_px),
        margin_bottom_px=float(render_params.plot_margin_bottom_px),
    )
    px0, py0, px1, py1 = [float(value) for value in plot_bbox]
    draw.rectangle(
        plot_bbox,
        fill=tuple(render_params.plot_fill_rgb) + (255,),
        outline=tuple(render_params.axis_color_rgb) + (255,),
        width=1,
    )

    y_scale_max = float(dataset.y_max) * 1.16
    for tick in (0, 25, 50, 75, 100):
        x_tick, _ = value_to_px(
            plot_bbox,
            x_value=float(tick),
            y_value=0.0,
            x_min=dataset.x_min,
            x_max=dataset.x_max,
            y_max=y_scale_max,
        )
        draw.line(
            (x_tick, py0, x_tick, py1),
            fill=tuple(render_params.grid_color_rgb) + (255,),
            width=int(render_params.grid_line_width_px),
        )
        draw.line(
            (x_tick, py1, x_tick, py1 + int(render_params.tick_length_px)),
            fill=tuple(render_params.axis_color_rgb) + (255,),
            width=int(render_params.axis_line_width_px),
        )
        draw_text_traced(
            draw,
            (x_tick, py1 + 8.0),
            str(tick),
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
            anchor="ma",
            role="axis_tick",
            required=False,
        )
    for frac, label in ((0.25, "low"), (0.5, "mid"), (0.75, "high")):
        y_tick = py1 - float(frac) * (py1 - py0)
        draw.line(
            (px0, y_tick, px1, y_tick),
            fill=tuple(render_params.grid_color_rgb) + (255,),
            width=int(render_params.grid_line_width_px),
        )
        draw_text_traced(
            draw,
            (px0 - 8.0, y_tick),
            str(label),
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
            anchor="rm",
            role="axis_tick",
            required=False,
        )

    interval_start_px, _ = value_to_px(
        plot_bbox,
        x_value=dataset.query.interval_start,
        y_value=0.0,
        x_min=dataset.x_min,
        x_max=dataset.x_max,
        y_max=y_scale_max,
    )
    interval_end_px, _ = value_to_px(
        plot_bbox,
        x_value=dataset.query.interval_end,
        y_value=0.0,
        x_min=dataset.x_min,
        x_max=dataset.x_max,
        y_max=y_scale_max,
    )
    if dataset.query.visible_role == "interval_mass":
        draw.rectangle(
            (interval_start_px, py0, interval_end_px, py1),
            fill=(90, 110, 135, 34),
            outline=(90, 110, 135, 100),
            width=1,
        )
        draw_text_traced(
            draw,
            ((interval_start_px + interval_end_px) / 2.0, py0 + 6.0),
            f"range {dataset.query.trace['interval_label']}",
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
            anchor="ma",
            role="readout",
            required=False,
        )

    curve_bboxes: Dict[str, BBox] = {}
    mean_marker_bboxes: Dict[str, BBox] = {}
    mode_marker_bboxes: Dict[str, BBox] = {}
    interval_mass_bboxes: Dict[str, BBox] = {}
    interval_mass_points: Dict[str, list[float]] = {}
    density_at_x_points: Dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []

    curve_points_by_label = {
        str(curve.label): curve_points_px(
            curve,
            plot_bbox,
            x_min=dataset.x_min,
            x_max=dataset.x_max,
            y_max=y_scale_max,
        )
        for curve in dataset.curves
    }

    if dataset.query.visible_role == "interval_mass":
        interval_mid = (float(dataset.query.interval_start) + float(dataset.query.interval_end)) / 2.0
        for curve in dataset.curves:
            points = curve_points_by_label[str(curve.label)]
            interval_mass_points[str(curve.label)] = curve_point_at_x_px(
                curve,
                plot_bbox,
                x_value=float(interval_mid),
                x_min=dataset.x_min,
                x_max=dataset.x_max,
                y_max=y_scale_max,
            )
            area_points = [(interval_start_px, py1)]
            area_points.extend(
                point
                for point, source in zip(points, curve.points)
                if float(dataset.query.interval_start) <= float(source.x_value) <= float(dataset.query.interval_end)
            )
            area_points.append((interval_end_px, py1))
            if len(area_points) >= 3:
                draw.polygon(tuple(area_points), fill=tuple(curve.color_rgb) + (34,))
                interval_mass_bboxes[str(curve.label)] = clip_bbox_to_container(
                    curve_bbox(area_points, pad=2.0),
                    plot_bbox,
                )

    reference_x_px, _ = value_to_px(
        plot_bbox,
        x_value=dataset.query.reference_x,
        y_value=0.0,
        x_min=dataset.x_min,
        x_max=dataset.x_max,
        y_max=y_scale_max,
    )
    if dataset.query.visible_role == "density_at_x":
        draw.line(
            (reference_x_px, py0, reference_x_px, py1),
            fill=tuple(render_params.axis_color_rgb) + (130,),
            width=max(1, int(render_params.guide_line_width_px)),
        )
        draw_text_traced(
            draw,
            (reference_x_px, py0 + 6.0),
            f"x = {dataset.query.trace['reference_x_label']}",
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
            anchor="ma",
            role="readout",
            required=False,
        )

    for curve in dataset.curves:
        points = curve_points_by_label[str(curve.label)]
        draw_polyline(
            draw,
            points,
            fill=curve.color_rgb,
            width=int(render_params.line_width_px),
            style=str(curve.line_style),
        )
        curve_bboxes[str(curve.label)] = curve_bbox(points, pad=3.0)

    if dataset.query.visible_role == "mean":
        for curve in dataset.curves:
            x_px, _ = value_to_px(
                plot_bbox,
                x_value=curve.mean_x,
                y_value=0.0,
                x_min=dataset.x_min,
                x_max=dataset.x_max,
                y_max=y_scale_max,
            )
            marker = (x_px - 6.0, py1 + 14.0, x_px, py1 + 2.0, x_px + 6.0, py1 + 14.0)
            draw.polygon(marker, fill=tuple(curve.color_rgb) + (255,), outline=tuple(render_params.text_stroke_rgb) + (255,))
            draw.line(
                (x_px, py1, x_px, py0),
                fill=tuple(curve.color_rgb) + (84,),
                width=max(1, int(render_params.guide_line_width_px)),
            )
            mean_marker_bboxes[str(curve.label)] = bbox((x_px - 7.0, py1 + 1.0, x_px + 7.0, py1 + 15.0))

    if dataset.query.visible_role == "mode":
        for curve in dataset.curves:
            x_px, y_px = value_to_px(
                plot_bbox,
                x_value=curve.mode_x,
                y_value=curve.mode_y,
                x_min=dataset.x_min,
                x_max=dataset.x_max,
                y_max=y_scale_max,
            )
            radius = max(5.0, float(render_params.point_radius_px))
            draw.ellipse(
                (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                fill=tuple(curve.color_rgb) + (255,),
                outline=tuple(render_params.text_stroke_rgb) + (255,),
                width=2,
            )
            mode_marker_bboxes[str(curve.label)] = bbox((x_px - radius, y_px - radius, x_px + radius, y_px + radius))

    if dataset.query.visible_role == "density_at_x":
        radius = max(4.5, float(render_params.point_radius_px) * 0.85)
        for curve in dataset.curves:
            x_px, y_px = value_to_px(
                plot_bbox,
                x_value=dataset.query.reference_x,
                y_value=curve.density_at_x,
                x_min=dataset.x_min,
                x_max=dataset.x_max,
                y_max=y_scale_max,
            )
            draw.ellipse(
                (x_px - radius, y_px - radius, x_px + radius, y_px + radius),
                fill=tuple(curve.color_rgb) + (255,),
                outline=tuple(render_params.text_stroke_rgb) + (255,),
                width=2,
            )
            density_at_x_points[str(curve.label)] = [round(float(x_px), 3), round(float(y_px), 3)]

    draw_text_traced(
        draw,
        ((px0 + px1) / 2.0, py1 + 44.0),
        "x value",
        font=label_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=1,
        anchor="mm",
        role="chart_label",
        required=False,
    )
    draw_text_traced(
        draw,
        (px0, py0 - 10.0),
        "density",
        font=label_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=1,
        anchor="ls",
        role="chart_label",
        required=False,
    )

    legend_x0 = px1 + 18.0
    legend_y0 = py0 + 8.0
    legend_x1 = float(render_params.canvas_width) - 18.0
    row_h = max(24.0, float(render_params.tick_font_size_px) + 8.0)
    legend_y1 = min(py1, legend_y0 + (row_h * float(len(dataset.curves))) + 16.0)
    legend_bbox = bbox((legend_x0, legend_y0, legend_x1, legend_y1))
    legend_fill_rgb = tuple(int(value) for value in render_params.plot_fill_rgb)
    draw.rectangle(
        legend_bbox,
        fill=legend_fill_rgb + (235,),
        outline=tuple(render_params.grid_color_rgb) + (255,),
        width=1,
    )
    legend_text_style = resolve_readable_text_style(
        instance_seed=0,
        namespace="charts.density_curve.legend_label",
        role="legend_label",
        surface_rgbs=(legend_fill_rgb,),
        preferred_rgbs=(tuple(int(value) for value in render_params.text_color_rgb),),
    )
    legend_items: Dict[str, BBox] = {}
    for index, curve in enumerate(dataset.curves):
        row_y = legend_y0 + 12.0 + (float(index) * row_h)
        draw_polyline(
            draw,
            (
                (legend_x0 + 10.0, row_y + 7.0),
                (legend_x0 + 58.0, row_y + 7.0),
            ),
            fill=tuple(curve.color_rgb),
            width=max(3, int(render_params.line_width_px)),
            style=str(curve.line_style),
        )
        label_record = draw_readable_text(
            draw,
            xy=(legend_x0 + 66.0, row_y),
            text=str(curve.label),
            font=legend_font,
            style=legend_text_style,
            stroke_width=1,
        )
        legend_items[str(curve.label)] = list(label_record["bbox_px"])

    for curve in dataset.curves:
        entities.append(
            {
                "entity_id": f"curve:{curve.label}",
                "entity_type": "density_curve",
                "label": str(curve.label),
                "family": str(curve.family),
                "component_count": int(curve.component_count),
                "color_rgb": [int(channel) for channel in curve.color_rgb],
                "line_style": str(curve.line_style),
                "bbox_px": list(curve_bboxes[str(curve.label)]),
                "mean_marker_bbox_px": list(mean_marker_bboxes.get(str(curve.label), [])),
                "mode_marker_bbox_px": list(mode_marker_bboxes.get(str(curve.label), [])),
                "interval_mass_bbox_px": list(interval_mass_bboxes.get(str(curve.label), [])),
                "interval_mass_point_px": list(interval_mass_points.get(str(curve.label), [])),
                "density_at_x_point_px": list(density_at_x_points.get(str(curve.label), [])),
            }
        )

    return DensityCurveRendered(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        legend_bbox_px=list(legend_bbox),
        legend_item_bboxes_px=dict(legend_items),
        curve_bboxes_px=dict(curve_bboxes),
        mean_marker_bboxes_px=dict(mean_marker_bboxes),
        mode_marker_bboxes_px=dict(mode_marker_bboxes),
        interval_mass_bboxes_px=dict(interval_mass_bboxes),
        interval_mass_points_px=dict(interval_mass_points),
        density_at_x_points_px=dict(density_at_x_points),
        title_bbox_px=[],
        render_meta={
            "y_scale_max": round(float(y_scale_max), 8),
            "line_style_rendering": {
                "dash_on_px": float(DASH_ON_PX),
                "dash_off_px": float(DASH_OFF_PX),
                "dot_spacing_px": float(DOT_SPACING_PX),
            },
            "interval_visible": bool(dataset.query.visible_role == "interval_mass"),
            "mean_markers_visible": bool(dataset.query.visible_role == "mean"),
            "mode_markers_visible": bool(dataset.query.visible_role == "mode"),
            "density_at_x_markers_visible": bool(dataset.query.visible_role == "density_at_x"),
        },
    )
