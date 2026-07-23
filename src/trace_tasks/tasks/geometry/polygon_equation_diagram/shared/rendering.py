"""Rendering primitives for polygon equation diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, pad_bbox
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import add_scaled, mid, sub, unit
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .algebra import side_name
from .defaults import SCENE_ID
from .state import BBox, Point, PolygonEquationCase, RenderContext, RenderedPolygonEquationScene


def make_render_context(
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> RenderContext:
    """Create one styled canvas for a polygon equation diagram."""

    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 840)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 620)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
        require_grid=False,
        style_profile="analytical_diagram",
    )
    line_width = int(params.get("line_width", group_default(rendering_defaults, "line_width", 3)))
    label_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_label_size = int(
        params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18))
    )
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", int(diagram_style.label_stroke_width_px)),
        )
    )
    return RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        fill_color=tuple(int(value) for value in diagram_style.muted_fill_rgb),
        accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(int(label_size), bold=False),
        small_font=load_font(int(small_label_size), bold=False),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), f"{SCENE_ID}.scene_transform"),
            params=params,
            render_defaults=rendering_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _center(points: Sequence[Point]) -> Point:
    return (
        sum(float(point[0]) for point in points) / float(len(points)),
        sum(float(point[1]) for point in points) / float(len(points)),
    )


def _draw_text_centered(ctx: RenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    font = ctx.small_font if bool(small) else ctx.font
    draw_text_traced(
        ctx.draw,
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        fill=ctx.label_color,
        stroke_width=max(0, int(ctx.label_stroke_width)),
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    bbox = ctx.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=max(0, int(ctx.label_stroke_width)),
    )
    return pad_bbox(bbox, 3.0, width=ctx.width, height=ctx.height)


def _polygon_points(ctx: RenderContext, side_count: int, *, instance_seed: int) -> tuple[Point, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.polygon_points")
    center = (
        (ctx.width / 2.0) + float(rng.uniform(-42.0, 42.0)),
        (ctx.height / 2.0) + float(rng.uniform(-34.0, 36.0)),
    )
    radius_x = float(rng.uniform(210.0, 270.0)) * (0.94 if int(side_count) >= 5 else 1.0)
    radius_y = float(rng.uniform(160.0, 215.0))
    phase = float(rng.uniform(-math.pi, math.pi))
    raw: list[Point] = []
    for index in range(int(side_count)):
        theta = phase + (2.0 * math.pi * float(index) / float(side_count))
        theta += float(rng.uniform(-0.08, 0.08))
        raw.append(
            (
                center[0] + (radius_x * float(rng.uniform(0.9, 1.08)) * math.cos(theta)),
                center[1] + (radius_y * float(rng.uniform(0.9, 1.08)) * math.sin(theta)),
            )
        )
    return ctx.scene_transform.points(raw)


def _draw_polygon(ctx: RenderContext, points: Sequence[Point]) -> BBox:
    polygon = [(float(x), float(y)) for x, y in points]
    ctx.draw.polygon(polygon, fill=ctx.fill_color)
    ctx.draw.line(polygon + [polygon[0]], fill=ctx.line_color, width=int(ctx.line_width), joint="curve")
    return bbox_from_points(polygon, width=ctx.width, height=ctx.height, pad=ctx.line_width + 2)


def _draw_vertex_labels(
    ctx: RenderContext,
    points: Sequence[Point],
    labels: Sequence[str],
) -> dict[str, BBox]:
    center = _center(points)
    bboxes: dict[str, BBox] = {}
    for label, point in zip(labels, points):
        direction = unit(sub(point, center))
        bboxes[str(label)] = _draw_text_centered(ctx, str(label), add_scaled(point, direction, 24.0), small=True)
    return bboxes


def _side_indices(labels: Sequence[str], side: str) -> tuple[int, int]:
    for index in range(len(labels)):
        if side_name(tuple(labels), index) == str(side):
            return index, (index + 1) % len(labels)
    raise ValueError(f"unknown side label: {side}")


def _side_normal(points: Sequence[Point], side: tuple[int, int]) -> Point:
    a = points[int(side[0])]
    b = points[int(side[1])]
    tangent = unit(sub(b, a))
    normal = (-tangent[1], tangent[0])
    midpoint = mid(a, b)
    to_center = sub(_center(points), midpoint)
    if normal[0] * to_center[0] + normal[1] * to_center[1] > 0:
        normal = (-normal[0], -normal[1])
    return normal


def _draw_side_label(ctx: RenderContext, points: Sequence[Point], side: tuple[int, int], text: str) -> BBox:
    a = points[int(side[0])]
    b = points[int(side[1])]
    return _draw_text_centered(ctx, str(text), add_scaled(mid(a, b), _side_normal(points, side), 54.0), small=True)


def _draw_side_tick(ctx: RenderContext, points: Sequence[Point], side: tuple[int, int], *, count: int) -> BBox:
    a = points[int(side[0])]
    b = points[int(side[1])]
    tangent = unit(sub(b, a))
    normal = _side_normal(points, side)
    side_mid = mid(a, b)
    drawn: list[Point] = []
    for tick_index in range(int(count)):
        shift = (float(tick_index) - (float(count) - 1.0) / 2.0) * 9.0
        tick_center = add_scaled(add_scaled(side_mid, tangent, shift), normal, 4.0)
        start = add_scaled(tick_center, normal, -10.0)
        end = add_scaled(tick_center, normal, 10.0)
        ctx.draw.line((start, end), fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
        drawn.extend([start, end])
    return bbox_from_points(drawn, width=ctx.width, height=ctx.height, pad=5.0)


def _angle_arc_points(vertex: Point, previous: Point, next_point: Point, radius: float) -> list[Point]:
    prev_unit = unit(sub(previous, vertex))
    next_unit = unit(sub(next_point, vertex))
    angle_prev = math.atan2(prev_unit[1], prev_unit[0])
    angle_next = math.atan2(next_unit[1], next_unit[0])
    delta = (angle_next - angle_prev) % (2.0 * math.pi)
    if delta > math.pi:
        angle_prev, angle_next = angle_next, angle_prev
        delta = (angle_next - angle_prev) % (2.0 * math.pi)
    return [
        (
            float(vertex[0]) + (float(radius) * math.cos(angle_prev + delta * (step / 18.0))),
            float(vertex[1]) + (float(radius) * math.sin(angle_prev + delta * (step / 18.0))),
        )
        for step in range(19)
    ]


def _draw_angle_marker(
    ctx: RenderContext,
    points: Sequence[Point],
    vertex_index: int,
    *,
    text: str,
    count: int,
) -> tuple[BBox, BBox]:
    vertex = points[int(vertex_index)]
    previous = points[(int(vertex_index) - 1) % len(points)]
    next_point = points[(int(vertex_index) + 1) % len(points)]
    drawn: list[Point] = []
    for arc_index in range(max(1, int(count))):
        arc = _angle_arc_points(vertex, previous, next_point, 22.0 + (arc_index * 6.0))
        ctx.draw.line(arc, fill=ctx.accent_color, width=max(2, ctx.line_width - 1), joint="curve")
        drawn.extend(arc)
    center_direction = unit(sub(_center(points), vertex))
    label_bbox = _draw_text_centered(ctx, str(text), add_scaled(vertex, center_direction, 74.0), small=True)
    arc_bbox = bbox_from_points(drawn, width=ctx.width, height=ctx.height, pad=5.0)
    return arc_bbox, label_bbox


def render_polygon_equation_case(
    *,
    case: PolygonEquationCase,
    ctx: RenderContext,
    instance_seed: int,
) -> RenderedPolygonEquationScene:
    """Render one resolved polygon equation case."""

    labels = case.vertex_labels()
    points = _polygon_points(ctx, int(case.side_count), instance_seed=int(instance_seed))
    _draw_polygon(ctx, points)
    point_label_bboxes = _draw_vertex_labels(ctx, points, labels)

    marker_bboxes: dict[str, BBox] = {}
    side_mark_counts = {
        str(side_label): int(count)
        for side_label, count in (
            case.side_mark_counts.items()
            if case.side_mark_counts
            else {str(side_label): 1 for side_label in case.equal_sides}.items()
        )
    }
    for side_label, mark_count in sorted(side_mark_counts.items()):
        side = _side_indices(labels, str(side_label))
        marker_bboxes[f"equal_side_{side_label}"] = _draw_side_tick(ctx, points, side, count=int(mark_count))
    side_label_bboxes: dict[str, BBox] = {}
    for side_label, text in sorted(case.side_labels.items()):
        side = _side_indices(labels, str(side_label))
        side_label_bboxes[str(side_label)] = _draw_side_label(ctx, points, side, str(text))

    angle_label_bboxes: dict[str, BBox] = {}
    angle_mark_counts = {
        str(vertex_label): int(count)
        for vertex_label, count in (
            case.angle_mark_counts.items()
            if case.angle_mark_counts
            else {
                str(vertex_label): 2 if str(vertex_label) in set(case.equal_angles) else 1
                for vertex_label in case.angle_labels
            }.items()
        )
    }
    for vertex_label, text in sorted(case.angle_labels.items()):
        vertex_index = labels.index(str(vertex_label))
        marker_count = int(angle_mark_counts.get(str(vertex_label), 1))
        arc_bbox, label_bbox = _draw_angle_marker(
            ctx,
            points,
            vertex_index,
            text=str(text),
            count=marker_count,
        )
        marker_bboxes[f"angle_arc_{vertex_label}"] = arc_bbox
        angle_label_bboxes[str(vertex_label)] = label_bbox

    keyed_points = {str(label): point for label, point in zip(labels, points)}
    return RenderedPolygonEquationScene(
        image=ctx.image,
        annotation_keyed_points=dict(keyed_points),
        annotation_roles=tuple(labels),
        vertex_points=dict(keyed_points),
        point_label_bboxes=dict(point_label_bboxes),
        side_label_bboxes=dict(side_label_bboxes),
        angle_label_bboxes=dict(angle_label_bboxes),
        marker_bboxes=dict(marker_bboxes),
        vertices=tuple(points),
    )


__all__ = ["make_render_context", "render_polygon_equation_case"]
