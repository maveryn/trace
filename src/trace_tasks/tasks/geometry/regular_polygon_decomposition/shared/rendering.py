"""Rendering primitives for regular-polygon decomposition diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, fmt_measure, pad_bbox
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import add_scaled, mid, point_to_list, sub, unit
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import BBox, Color, Point, RegularPolygonProblem, RenderContext, RenderedRegularPolygonScene, SceneGeometry


def _blend(color_a: Color, color_b: Color, alpha: float) -> Color:
    amount = max(0.0, min(1.0, float(alpha)))
    return tuple(
        int(round((float(color_a[index]) * amount) + (float(color_b[index]) * (1.0 - amount))))
        for index in range(3)
    )  # type: ignore[return-value]


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> RenderContext:
    """Create the styled image, text resources, and transform context."""

    width = int(params.get("canvas_width", rendering_defaults.get("canvas_width", 820)))
    height = int(params.get("canvas_height", rendering_defaults.get("canvas_height", 580)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=width,
        canvas_height=height,
        allow_dark=False,
        require_grid=False,
    )
    readout_font_family = str(params.get("readout_font_family", rendering_defaults.get("readout_font_family", "roboto")))
    diagram_style_trace = dict(diagram_style_meta)
    diagram_style_trace["readout_font_family"] = readout_font_family
    panel_fill = tuple(int(value) for value in diagram_style.panel_fill_rgb)
    accent = tuple(int(value) for value in diagram_style.accent_rgb)
    image = background.convert("RGB")
    return RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=width,
        height=height,
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        accent_color=accent,
        fill_color=tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
        shaded_fill_color=_blend(accent, panel_fill, 0.34),
        panel_fill_color=panel_fill,
        panel_border_color=tuple(int(value) for value in diagram_style.panel_border_rgb),
        line_width=max(2, int(params.get("line_width", rendering_defaults.get("line_width", 3)))),
        label_stroke_width=max(0, min(1, int(diagram_style.label_stroke_width_px))),
        font=load_font(
            int(params.get("label_font_size", rendering_defaults.get("label_font_size", 22))),
            bold=False,
            font_family=readout_font_family,
        ),
        small_font=load_font(
            int(params.get("small_label_font_size", rendering_defaults.get("small_label_font_size", 18))),
            bold=False,
            font_family=readout_font_family,
        ),
        diagram_style_meta=diagram_style_trace,
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), f"{SCENE_ID}.scene_transform"),
            params=params,
            render_defaults=rendering_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _draw_text_centered(
    ctx: RenderContext,
    text: str,
    center: Point,
    *,
    small: bool = True,
    role: str = "readout",
    stroke_width_override: int | None = None,
) -> BBox:
    font = ctx.small_font if bool(small) else ctx.font
    stroke_width = max(0, int(ctx.label_stroke_width if stroke_width_override is None else stroke_width_override))
    draw_text_traced(
        ctx.draw,
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        fill=ctx.label_color,
        stroke_width=stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role=str(role),
        required=False,
    )
    bbox = ctx.draw.textbbox((float(center[0]), float(center[1])), str(text), anchor="mm", font=font, stroke_width=stroke_width)
    return pad_bbox(bbox, 3.0, width=ctx.width, height=ctx.height)


def _draw_readout_panel(ctx: RenderContext, lines: Sequence[str]) -> dict[str, BBox]:
    if not lines:
        return {}
    padding_x = 14
    padding_y = 10
    line_gap = 6
    text_boxes = [ctx.draw.textbbox((0, 0), line, font=ctx.small_font, stroke_width=ctx.label_stroke_width) for line in lines]
    text_width = max(box[2] - box[0] for box in text_boxes)
    text_height = sum(box[3] - box[1] for box in text_boxes) + (len(lines) - 1) * line_gap
    left = 28
    top = 26
    panel = (
        float(left),
        float(top),
        float(left + text_width + (2 * padding_x)),
        float(top + text_height + (2 * padding_y)),
    )
    ctx.draw.rounded_rectangle(panel, radius=8, fill=ctx.panel_fill_color, outline=ctx.panel_border_color, width=max(1, ctx.line_width - 2))
    bboxes: dict[str, BBox] = {"readout_panel": pad_bbox(panel, 0, width=ctx.width, height=ctx.height)}
    cursor_y = top + padding_y
    for index, line in enumerate(lines):
        box = text_boxes[index]
        x = left + padding_x
        y = cursor_y - box[1]
        draw_text_traced(
            ctx.draw,
            (float(x), float(y)),
            str(line),
            font=ctx.small_font,
            fill=ctx.label_color,
            stroke_width=ctx.label_stroke_width,
            stroke_fill=ctx.label_stroke_color,
            role="readout",
            required=False,
        )
        bboxes[f"readout_{index}"] = pad_bbox((x, y + box[1], x + (box[2] - box[0]), y + box[3]), 3.0, width=ctx.width, height=ctx.height)
        cursor_y += (box[3] - box[1]) + line_gap
    return bboxes


def _polygon_vertices(center: Point, radius: float, n_sides: int, rotation_degrees: float) -> tuple[Point, ...]:
    points: list[Point] = []
    for index in range(int(n_sides)):
        theta = math.radians(float(rotation_degrees) + (360.0 * float(index) / float(n_sides)))
        points.append((float(center[0]) + float(radius) * math.cos(theta), float(center[1]) + float(radius) * math.sin(theta)))
    return tuple(points)


def _point_on_ray(center: Point, radius: float, degrees: float) -> Point:
    theta = math.radians(float(degrees))
    return (float(center[0]) + float(radius) * math.cos(theta), float(center[1]) + float(radius) * math.sin(theta))


def _draw_arc_polyline(ctx: RenderContext, center: Point, radius: float, start_degrees: float, span_degrees: float) -> BBox:
    sample_count = max(8, int(abs(float(span_degrees)) // 5) + 2)
    points = [
        _point_on_ray(center, radius, float(start_degrees) + (float(span_degrees) * float(i) / float(sample_count - 1)))
        for i in range(sample_count)
    ]
    ctx.draw.line(points, fill=ctx.accent_color, width=max(3, ctx.line_width + 1), joint="curve")
    return bbox_from_points(points, width=ctx.width, height=ctx.height, pad=ctx.line_width + 4)


def _draw_dimension_label(ctx: RenderContext, start: Point, end: Point, label: str, *, center_reference: Point, distance: float) -> BBox:
    direction = unit(sub(end, start))
    normal = (-direction[1], direction[0])
    midpoint = mid(start, end)
    candidate = add_scaled(midpoint, normal, distance)
    if math.hypot(candidate[0] - center_reference[0], candidate[1] - center_reference[1]) < math.hypot(midpoint[0] - center_reference[0], midpoint[1] - center_reference[1]):
        normal = (-normal[0], -normal[1])
    return _draw_text_centered(ctx, label, add_scaled(midpoint, normal, distance), small=False, stroke_width_override=0)


def _draw_midpoint_label(ctx: RenderContext, side_start: Point, side_end: Point, center_reference: Point) -> BBox:
    side_mid = mid(side_start, side_end)
    tangent = unit(sub(side_end, side_start))
    outward = unit(sub(side_mid, center_reference))
    # Keep M close to the side midpoint, but away from the side-length label that sits on the outward normal.
    label_center = add_scaled(add_scaled(side_mid, tangent, 24.0), outward, 9.0)
    return _draw_text_centered(ctx, "M", label_center, small=True, role="label")


def _draw_apothem(ctx: RenderContext, center: Point, side_start: Point, side_end: Point, label: str) -> BBox:
    side_mid = mid(side_start, side_end)
    direction = unit(sub(side_mid, center))
    label_center = add_scaled(mid(center, side_mid), direction, 14.0)
    stroke_width = 0
    label_raw_bbox = ctx.draw.textbbox(label_center, str(label), anchor="mm", font=ctx.font, stroke_width=stroke_width)
    label_bbox = pad_bbox(label_raw_bbox, 4.0, width=ctx.width, height=ctx.height)
    total_length = math.hypot(float(side_mid[0]) - float(center[0]), float(side_mid[1]) - float(center[1]))
    label_distance = (
        (float(label_center[0]) - float(center[0])) * float(direction[0])
        + (float(label_center[1]) - float(center[1])) * float(direction[1])
    )
    gap_radius = max(float(label_bbox[2]) - float(label_bbox[0]), float(label_bbox[3]) - float(label_bbox[1])) * 0.55
    gap_start = max(0.0, float(label_distance) - float(gap_radius) - 5.0)
    gap_end = min(float(total_length), float(label_distance) + float(gap_radius) + 5.0)
    line_segments: list[tuple[Point, Point]] = []
    if gap_start > 4.0:
        line_segments.append((center, add_scaled(center, direction, gap_start)))
    if gap_end < total_length - 4.0:
        line_segments.append((add_scaled(center, direction, gap_end), side_mid))
    for segment_start, segment_end in line_segments:
        ctx.draw.line((segment_start, segment_end), fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
    line_points = [point for segment in line_segments for point in segment] or [center, side_mid]
    bbox_line = bbox_from_points(line_points, width=ctx.width, height=ctx.height, pad=ctx.line_width + 2)
    label_bbox = _draw_text_centered(ctx, label, label_center, small=False, stroke_width_override=0)
    return bbox_from_points(
        ((bbox_line[0], bbox_line[1]), (bbox_line[2], bbox_line[3]), (label_bbox[0], label_bbox[1]), (label_bbox[2], label_bbox[3])),
        width=ctx.width,
        height=ctx.height,
        pad=2.0,
    )


def _readout_lines(problem: RegularPolygonProblem) -> tuple[str, ...]:
    lines: list[str] = []
    if problem.show_total_area_readout and problem.total_area is not None:
        lines.append(f"Total polygon area = {fmt_measure(float(problem.total_area))}")
    if problem.show_perimeter_readout and problem.perimeter is not None:
        lines.append(f"Perimeter = {fmt_measure(float(problem.perimeter))}")
    if problem.show_wedge_area_readout and problem.wedge_area is not None:
        lines.append(f"Wedge area = {fmt_measure(float(problem.wedge_area))}")
    return tuple(lines)


def render_regular_polygon_scene(
    ctx: RenderContext,
    problem: RegularPolygonProblem,
) -> RenderedRegularPolygonScene:
    """Render the selected polygon decomposition after final transform placement."""

    rng = spawn_rng(int(problem.layout_seed), f"{SCENE_ID}.layout")
    radius = min(float(ctx.width) * 0.285, float(ctx.height) * 0.345) * rng.uniform(0.92, 1.05)
    center = (
        float(ctx.width) * 0.53 + rng.uniform(-24.0, 28.0),
        float(ctx.height) * 0.57 + rng.uniform(-18.0, 18.0),
    )
    rotation = -90.0 + (180.0 / float(problem.n_sides)) + rng.uniform(-8.0, 8.0)
    vertices = _polygon_vertices(center, radius, problem.n_sides, rotation)
    selected_indices = tuple(range(int(problem.start_index), int(problem.start_index + problem.wedge_count)))
    end_index = (int(problem.start_index) + int(problem.wedge_count)) % int(problem.n_sides)
    start_degrees = rotation + (360.0 * float(problem.start_index) / float(problem.n_sides))
    angle_span = 360.0 * float(problem.wedge_count) / float(problem.n_sides)
    target_midpoint = _point_on_ray(center, radius * 0.54, start_degrees + (angle_span / 2.0))
    transform = ctx.scene_transform.resolve((center, *vertices, target_midpoint))
    center = transform.point(center)
    vertices = transform.points(vertices)
    target_midpoint = transform.point(target_midpoint)
    radius = float(radius) * float(transform.scale)
    start_degrees += float(transform.angle_degrees)

    construction_bboxes: dict[str, BBox] = {}
    readout_bboxes: dict[str, BBox] = {}
    polygon_points = tuple(vertices)
    ctx.draw.polygon(polygon_points, fill=ctx.fill_color)
    if problem.show_shaded_region:
        for wedge_index in selected_indices:
            ctx.draw.polygon((center, vertices[wedge_index], vertices[(wedge_index + 1) % problem.n_sides]), fill=ctx.shaded_fill_color)
    ctx.draw.line((*polygon_points, polygon_points[0]), fill=ctx.line_color, width=ctx.line_width, joint="curve")
    for vertex in vertices:
        ctx.draw.line((center, vertex), fill=ctx.secondary_color, width=max(1, ctx.line_width - 1))
    selected_boundary_vertices = (vertices[problem.start_index], vertices[end_index])
    for boundary_vertex in selected_boundary_vertices:
        ctx.draw.line((center, boundary_vertex), fill=ctx.accent_color, width=max(3, ctx.line_width + 1))
    center_dot = max(4, ctx.line_width + 2)
    ctx.draw.ellipse((center[0] - center_dot, center[1] - center_dot, center[0] + center_dot, center[1] + center_dot), fill=ctx.line_color)

    construction_bboxes["regular_polygon"] = bbox_from_points(vertices, width=ctx.width, height=ctx.height, pad=ctx.line_width + 4)
    side_start = vertices[problem.start_index]
    side_end = vertices[(problem.start_index + 1) % problem.n_sides]
    side_mid = mid(side_start, side_end)
    construction_bboxes["selected_region"] = bbox_from_points(
        (center, vertices[problem.start_index], vertices[end_index]),
        width=ctx.width,
        height=ctx.height,
        pad=ctx.line_width + 4,
    )

    if problem.show_angle_unknown:
        construction_bboxes["marked_angle_arc"] = _draw_arc_polyline(ctx, center, radius * 0.31, start_degrees, angle_span)
        readout_bboxes["unknown_angle_label"] = _draw_text_centered(ctx, "?", _point_on_ray(center, radius * 0.43, start_degrees + (angle_span / 2.0)), small=False)
    side_label_distance = 66.0 if problem.show_midpoint_label else 36.0
    if problem.show_known_side_length and problem.side_length is not None:
        readout_bboxes["side_length_label"] = _draw_dimension_label(
            ctx,
            side_start,
            side_end,
            f"s = {fmt_measure(float(problem.side_length))}",
            center_reference=center,
            distance=side_label_distance,
        )
    if problem.show_unknown_side_length:
        readout_bboxes["target_side_label"] = _draw_dimension_label(
            ctx,
            side_start,
            side_end,
            "s = ?",
            center_reference=center,
            distance=side_label_distance,
        )
    if problem.show_apothem and problem.apothem is not None:
        construction_bboxes["apothem"] = _draw_apothem(ctx, center, side_start, side_end, f"a = {fmt_measure(float(problem.apothem))}")
    readout_bboxes.update(_draw_readout_panel(ctx, _readout_lines(problem)))

    start_vertex = vertices[problem.start_index]
    label_end_vertex = side_end if problem.show_apothem else vertices[end_index]
    readout_bboxes["label_O"] = _draw_text_centered(ctx, "O", add_scaled(center, (0.0, -22.0)), small=True, role="label")
    if problem.show_side_endpoint_labels:
        readout_bboxes["label_A"] = _draw_text_centered(ctx, "A", add_scaled(start_vertex, unit(sub(start_vertex, center)), 25.0), small=True, role="label")
        readout_bboxes["label_B"] = _draw_text_centered(ctx, "B", add_scaled(label_end_vertex, unit(sub(label_end_vertex, center)), 25.0), small=True, role="label")
    if problem.show_region_label:
        readout_bboxes["label_W"] = _draw_text_centered(ctx, "W", target_midpoint, small=True, role="label")
    if problem.show_midpoint_label:
        readout_bboxes["label_M"] = _draw_midpoint_label(ctx, side_start, side_end, center)

    annotation_points = {
        "O": center,
        "A": start_vertex,
        "B": label_end_vertex,
        "M": side_mid,
        "W": target_midpoint,
    }
    geometry = SceneGeometry(
        center=center,
        vertices=vertices,
        selected_wedge_indices=selected_indices,
        selected_region_midpoint=target_midpoint,
        angle_span_degrees=float(angle_span),
    )
    render_map = {
        "center": point_to_list(center),
        "vertices": [point_to_list(point) for point in vertices],
        "selected_wedge_indices": [int(index) for index in selected_indices],
        "readout_bboxes": geometry_json_ready(readout_bboxes),
        "construction_bboxes": geometry_json_ready(construction_bboxes),
        "single_object_scene_rotation": ctx.scene_transform.metadata(),
    }
    return RenderedRegularPolygonScene(
        image=ctx.image,
        geometry=geometry,
        annotation_points=annotation_points,
        readout_bboxes=readout_bboxes,
        construction_bboxes=construction_bboxes,
        render_map=render_map,
    )


__all__ = ["create_render_context", "render_regular_polygon_scene"]
