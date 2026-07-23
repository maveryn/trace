"""Rendering primitives for composite-shape geometry scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_right_angle_marker,
    draw_label,
    fmt_measure,
    pad_bbox,
    round1,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import BBox, Color, CompositeRenderContext, CompositeShapeProblem, Point, RenderedCompositeShape
from .styles import resolve_composite_shape_style

_DIAGRAM_LEFT = 84.0
_READOUT_STRIP_WIDTH = 250.0
_MIN_DIAGRAM_WIDTH = 320.0


def _diagram_width(ctx: CompositeRenderContext) -> float:
    """Width available for the geometric drawing before the right readout strip."""

    diagram_right = max(_DIAGRAM_LEFT + _MIN_DIAGRAM_WIDTH, float(ctx.width) - _READOUT_STRIP_WIDTH)
    return max(_MIN_DIAGRAM_WIDTH, diagram_right - _DIAGRAM_LEFT)


def create_composite_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    render_namespace: str,
) -> tuple[CompositeRenderContext, Dict[str, Any]]:
    """Resolve canvas, background, palette, and readout font choices."""

    rng = spawn_rng(int(instance_seed), f"{render_namespace}.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        require_grid=False,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    )
    composite_style = resolve_composite_shape_style(
        instance_seed=int(instance_seed),
        params=params,
        render_namespace=str(render_namespace),
        diagram_style=diagram_style,
        background_meta=background_meta,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 24)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 22)))
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    label_stroke_width = int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 0)))
    ctx = CompositeRenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        background_color=tuple(int(value) for value in diagram_style.canvas_rgb),
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        label_color=tuple(int(value) for value in composite_style.label_color),
        label_stroke_color=tuple(int(value) for value in composite_style.label_stroke_color),
        accent_color=tuple(int(value) for value in composite_style.accent_color),
        fill_color=tuple(int(value) for value in composite_style.fill_color),
        secondary_fill_color=tuple(int(value) for value in composite_style.secondary_fill_color),
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        scene_transform=LazySceneTransform(
            rng,
            params=params,
            render_defaults=render_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "diagram_style": dict(diagram_style_meta),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "label_stroke_width": int(ctx.label_stroke_width),
        "label_font_bold": False,
        "fill_color": list(ctx.fill_color),
        "secondary_fill_color": list(ctx.secondary_fill_color),
        "accent_color": list(ctx.accent_color),
        "composite_fill_style": dict(composite_style.metadata),
    }
    return ctx, render_meta


def _place_points(ctx: CompositeRenderContext, points: Sequence[Point]) -> tuple[Point, ...]:
    """Apply the optional scene-level rigid transform before drawing/annotation."""

    resolved = tuple((float(point[0]), float(point[1])) for point in points)
    if ctx.scene_transform is None:
        return resolved
    return ctx.scene_transform.points(resolved)


def _place_point(ctx: CompositeRenderContext, point: Point) -> Point:
    if ctx.scene_transform is None:
        return (float(point[0]), float(point[1]))
    return ctx.scene_transform.point((float(point[0]), float(point[1])))


def _draw_text(
    ctx: CompositeRenderContext,
    text: str,
    center: Point,
    *,
    font: Any | None = None,
    fill: Color | None = None,
    stroke_width: int | None = None,
) -> BBox:
    active_font = font if font is not None else ctx.font
    active_fill = fill if fill is not None else ctx.label_color
    active_stroke_width = int(ctx.label_stroke_width if stroke_width is None else stroke_width)
    x, y = float(center[0]), float(center[1])
    draw_text_traced(
        ctx.draw,
        (x, y),
        str(text),
        anchor="mm",
        font=active_font,
        fill=active_fill,
        stroke_width=max(0, int(active_stroke_width)),
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    bbox = ctx.draw.textbbox(
        (x, y),
        str(text),
        anchor="mm",
        font=active_font,
        stroke_width=max(0, int(active_stroke_width)),
    )
    return pad_bbox(bbox, 2.0, width=ctx.width, height=ctx.height)


def _draw_segment_label(ctx: CompositeRenderContext, text: str, a: Point, b: Point, *, offset: float = 30.0) -> BBox:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    length = max(1.0, math.hypot(dx, dy))
    nx = -dy / length
    ny = dx / length
    center = (
        (float(a[0]) + float(b[0])) / 2.0 + nx * float(offset),
        (float(a[1]) + float(b[1])) / 2.0 + ny * float(offset),
    )
    return _draw_text(ctx, str(text), center)


def _draw_point_label_outward(
    ctx: CompositeRenderContext,
    label: str,
    point: Point,
    *,
    center: Point,
    distance: float = 24.0,
) -> BBox:
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    length = max(1.0, math.hypot(dx, dy))
    label_center = (
        float(point[0]) + (dx / length) * float(distance),
        float(point[1]) + (dy / length) * float(distance),
    )
    return _draw_text(ctx, str(label), label_center, font=ctx.small_font)


def _draw_labeled_points(
    ctx: CompositeRenderContext,
    points: Mapping[str, Point],
    *,
    center: Point | None = None,
    distance: float = 24.0,
) -> tuple[dict[str, Point], dict[str, BBox]]:
    """Draw visible point labels and return the same final points for annotation."""

    point_items = [(str(label), (float(point[0]), float(point[1]))) for label, point in points.items()]
    if center is None:
        center = (
            sum(point[0] for _label, point in point_items) / max(1, len(point_items)),
            sum(point[1] for _label, point in point_items) / max(1, len(point_items)),
        )
    label_bboxes: dict[str, BBox] = {}
    keyed_points: dict[str, Point] = {}
    for label, point in point_items:
        keyed_points[label] = point
        label_bboxes[label] = _draw_point_label_outward(ctx, label, point, center=center, distance=distance)
    return keyed_points, label_bboxes


def _draw_point_marker(ctx: CompositeRenderContext, point: Point, *, radius: float = 4.0) -> BBox:
    x, y = float(point[0]), float(point[1])
    bbox = (x - float(radius), y - float(radius), x + float(radius), y + float(radius))
    ctx.draw.ellipse(bbox, fill=ctx.line_color)
    return pad_bbox(bbox, 2.0, width=ctx.width, height=ctx.height)


def _draw_polygon(
    ctx: CompositeRenderContext,
    points: Sequence[Point],
    *,
    fill: Color | None = None,
    outline: Color | None = None,
    width: int | None = None,
) -> BBox:
    if fill is not None:
        ctx.draw.polygon([(float(x), float(y)) for x, y in points], fill=fill)
    line_fill = outline if outline is not None else ctx.line_color
    line_width = int(width if width is not None else ctx.line_width)
    closed = list(points) + [points[0]]
    ctx.draw.line([(float(x), float(y)) for x, y in closed], fill=line_fill, width=line_width, joint="curve")
    return bbox_from_points(points, width=ctx.width, height=ctx.height, pad=line_width + 2)


def _arc_points(
    center: Point,
    radius: float,
    *,
    start_degrees: float,
    end_degrees: float,
    steps: int | None = None,
) -> tuple[Point, ...]:
    """Sample an arc in PIL/screen coordinates so it can be transformed as geometry."""

    span = float(end_degrees) - float(start_degrees)
    sample_count = int(steps) if steps is not None else max(12, int(abs(span) / 4.0) + 1)
    if sample_count <= 1:
        sample_count = 2
    points: list[Point] = []
    for index in range(sample_count):
        t = float(index) / float(sample_count - 1)
        angle = math.radians(float(start_degrees) + (span * t))
        points.append(
            (
                float(center[0]) + float(radius) * math.cos(angle),
                float(center[1]) + float(radius) * math.sin(angle),
            )
        )
    return tuple(points)


def _draw_polyline(
    ctx: CompositeRenderContext,
    points: Sequence[Point],
    *,
    fill: Color | None = None,
    width: int | None = None,
) -> BBox:
    line_width = int(width if width is not None else ctx.line_width)
    line_fill = fill if fill is not None else ctx.line_color
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 2:
        ctx.draw.line(pts, fill=line_fill, width=line_width, joint="curve")
    return bbox_from_points(pts, width=ctx.width, height=ctx.height, pad=line_width + 2)


def _draw_dashed_line(
    ctx: CompositeRenderContext,
    start: Point,
    end: Point,
    *,
    fill: Color | None = None,
    width: int | None = None,
    dash: float = 11.0,
    gap: float = 8.0,
) -> BBox:
    """Draw a dashed construction/reference segment after final placement."""

    line_width = int(width if width is not None else max(2, int(ctx.line_width) - 2))
    line_fill = fill if fill is not None else ctx.line_color
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return bbox_from_points((start, end), width=ctx.width, height=ctx.height, pad=line_width + 2)
    step = float(dash) + float(gap)
    distance = 0.0
    while distance < length:
        seg_start = distance
        seg_end = min(length, distance + float(dash))
        a = seg_start / length
        b = seg_end / length
        ctx.draw.line(
            [
                (x0 + dx * a, y0 + dy * a),
                (x0 + dx * b, y0 + dy * b),
            ],
            fill=line_fill,
            width=line_width,
        )
        distance += step
    return bbox_from_points((start, end), width=ctx.width, height=ctx.height, pad=line_width + 2)


def _draw_radius_witness(
    ctx: CompositeRenderContext,
    center: Point,
    endpoint: Point,
    *,
    label: str = "r",
) -> dict[str, BBox]:
    """Draw a visual radius witness without making it part of annotation."""

    line_width = max(1, int(ctx.line_width) - 2)
    line_bbox = _draw_polyline(ctx, [center, endpoint], width=line_width)
    dx = float(endpoint[0]) - float(center[0])
    dy = float(endpoint[1]) - float(center[1])
    length = max(1.0, math.hypot(dx, dy))
    nx = -dy / length
    ny = dx / length
    label_center = (
        float(center[0]) + dx * 0.58 + nx * 16.0,
        float(center[1]) + dy * 0.58 + ny * 16.0,
    )
    label_bbox = _draw_text(ctx, label, label_center, font=ctx.small_font)
    return {
        "radius_witness_segment": line_bbox,
        "radius_witness_label": label_bbox,
    }


def _fill_polygon(ctx: CompositeRenderContext, points: Sequence[Point], *, fill: Color) -> BBox:
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 3:
        ctx.draw.polygon(pts, fill=fill)
    return bbox_from_points(pts, width=ctx.width, height=ctx.height, pad=2.0)


def _draw_dimension(
    ctx: CompositeRenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point = (0.0, 0.0),
) -> BBox:
    ctx.draw.line([start, end], fill=ctx.label_color, width=max(2, ctx.line_width - 1))
    tick = 7.0
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length > 1e-9:
        nx = -dy / length
        ny = dx / length
        for point in (start, end):
            ctx.draw.line(
                [
                    (float(point[0]) - tick * nx, float(point[1]) - tick * ny),
                    (float(point[0]) + tick * nx, float(point[1]) + tick * ny),
                ],
                fill=ctx.label_color,
                width=max(2, ctx.line_width - 1),
            )
    center = (
        (float(start[0]) + float(end[0])) / 2.0 + float(label_offset[0]),
        (float(start[1]) + float(end[1])) / 2.0 + float(label_offset[1]),
    )
    return draw_label(ctx, label, center, small=True)


def _draw_measurement_list(
    ctx: CompositeRenderContext,
    labels: Sequence[str],
    *,
    anchor: Point | None = None,
    line_gap: float = 30.0,
    keys: Sequence[str] | None = None,
) -> dict[str, BBox]:
    """Draw a compact visible measurement list away from the shape."""

    if anchor is None:
        anchor = (float(ctx.width) - 96.0, 150.0)
    if keys is not None and len(keys) != len(labels):
        raise ValueError("measurement list keys must match labels")
    bboxes: dict[str, BBox] = {}
    for index, label in enumerate(labels):
        text = str(label)
        key = str(keys[index]) if keys is not None else f"measure_{index}"
        bboxes[key] = _draw_text(
            ctx,
            text,
            (float(anchor[0]), float(anchor[1]) + (float(index) * float(line_gap))),
            font=ctx.small_font,
        )
    return bboxes


def _vector_from(vertex: Point, endpoint: Point) -> Point:
    return (float(endpoint[0]) - float(vertex[0]), float(endpoint[1]) - float(vertex[1]))


def _draw_right_angle_notation(
    ctx: CompositeRenderContext,
    vertex: Point,
    arm_a: Point,
    arm_b: Point,
    *,
    side_px: float | None = None,
) -> BBox:
    return draw_right_angle_marker(
        ctx,
        vertex,
        arm_a=_vector_from(vertex, arm_a),
        arm_b=_vector_from(vertex, arm_b),
        side_px=float(side_px) if side_px is not None else max(12.0, float(ctx.line_width) * 3.5),
        color=ctx.line_color,
        width=max(1, int(ctx.line_width) - 2),
    )


def _draw_polygon_right_angle_notation(
    ctx: CompositeRenderContext,
    points: Sequence[Point],
    *,
    key_prefix: str,
) -> dict[str, BBox]:
    bboxes: dict[str, BBox] = {}
    point_list = list(points)
    for index, vertex in enumerate(point_list):
        bboxes[f"{key_prefix}_{index}_right_angle"] = _draw_right_angle_notation(
            ctx,
            vertex,
            point_list[index - 1],
            point_list[(index + 1) % len(point_list)],
        )
    return bboxes


def _draw_equal_side_ticks(
    ctx: CompositeRenderContext,
    a: Point,
    b: Point,
    *,
    count: int,
) -> BBox:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    nx = -uy
    ny = ux
    center_x = (float(a[0]) + float(b[0])) / 2.0
    center_y = (float(a[1]) + float(b[1])) / 2.0
    tick_len = max(12.0, float(ctx.line_width) * 3.5)
    spacing = max(5.0, float(ctx.line_width) * 1.6)
    drawn_points: list[Point] = []
    for index in range(max(1, int(count))):
        along = (float(index) - (float(count) - 1.0) / 2.0) * spacing
        cx = center_x + ux * along
        cy = center_y + uy * along
        start = (cx - nx * tick_len / 2.0, cy - ny * tick_len / 2.0)
        end = (cx + nx * tick_len / 2.0, cy + ny * tick_len / 2.0)
        ctx.draw.line([start, end], fill=ctx.line_color, width=max(2, int(ctx.line_width) - 1))
        drawn_points.extend([start, end])
    return bbox_from_points(drawn_points, width=ctx.width, height=ctx.height, pad=4.0)


def _render_rect_cut(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Draw a rectangle with a visible triangular cutout for area subtraction."""

    values = dict(problem.dimensions)
    width_value = int(values["width"])
    height_value = int(values["height"])
    cut_base = int(values["cut_base"])
    cut_height = int(values["cut_height"])
    left, top = 150.0, 135.0
    w_px = 390.0
    h_px = 270.0
    raw_rect = [(left, top), (left + w_px, top), (left + w_px, top + h_px), (left, top + h_px)]
    raw_tri = [
        (left + w_px, top + h_px),
        (left + w_px - (w_px * cut_base / width_value), top + h_px),
        (left + w_px, top + h_px - (h_px * cut_height / height_value)),
    ]
    placed = _place_points(ctx, [*raw_rect, *raw_tri])
    rect = list(placed[:4])
    tri = list(placed[4:7])
    _draw_polygon(ctx, rect, fill=ctx.fill_color)
    ctx.draw.polygon([(float(x), float(y)) for x, y in tri], fill=ctx.background_color)
    _draw_polygon(ctx, rect)
    _draw_polygon(ctx, tri, outline=ctx.accent_color, width=max(2, ctx.line_width - 1))
    notation_bboxes = {
        **_draw_polygon_right_angle_notation(ctx, rect, key_prefix="outer_corner"),
        "cutout_right_angle": _draw_right_angle_notation(ctx, tri[0], tri[1], tri[2]),
    }
    region_bbox = bbox_from_points(rect, width=ctx.width, height=ctx.height, pad=2.0)
    cutout_bbox = bbox_from_points(tri, width=ctx.width, height=ctx.height, pad=4.0)
    label_bboxes = _draw_measurement_list(
        ctx,
        (
            f"DC={width_value}",
            f"AD={height_value}",
            f"EC={cut_base}",
            f"CF={cut_height}",
        ),
        keys=("outer_width", "outer_height", "cutout_base", "cutout_height"),
    )
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {
            "A": rect[0],
            "B": rect[1],
            "C": rect[2],
            "D": rect[3],
            "E": tri[1],
            "F": tri[2],
        },
        center=((rect[0][0] + rect[2][0]) / 2.0, (rect[0][1] + rect[2][1]) / 2.0),
    )
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_points),
        annotation_keyed_points=annotation_points,
        scene_entities=(
            {
                "type": "rectangle_with_triangular_cutout",
                "outer": rect,
                "cutout": tri,
                "points": dict(annotation_points),
            },
        ),
        render_map={
            "outer_region_bbox": bbox_to_list(region_bbox),
            "cutout_region_bbox": bbox_to_list(cutout_bbox),
            "measurement_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in label_bboxes.items()},
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "visual_notation_bboxes": {key: bbox_to_list(bbox) for key, bbox in notation_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={
            "outer_width": width_value,
            "outer_height": height_value,
            "cutout_base": cut_base,
            "cutout_height": cut_height,
            "answer_area": int(problem.answer_value),
        },
    )


def _render_l_profile(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Draw an L-shaped region with a missing-corner witness bbox."""

    values = dict(problem.dimensions)
    width_value = int(values["width"])
    height_value = int(values["height"])
    cut_width = int(values["cut_width"])
    cut_height = int(values["cut_height"])
    left, top = 145.0, 125.0
    w_px, h_px = 420.0, 300.0
    cut_w_px = w_px * cut_width / width_value
    cut_h_px = h_px * cut_height / height_value
    raw_pts = [
        (left, top),
        (left + w_px, top),
        (left + w_px, top + h_px - cut_h_px),
        (left + w_px - cut_w_px, top + h_px - cut_h_px),
        (left + w_px - cut_w_px, top + h_px),
        (left, top + h_px),
    ]
    raw_cutout_corner = (raw_pts[2][0], raw_pts[4][1])
    placed = _place_points(ctx, [*raw_pts, raw_cutout_corner])
    pts = list(placed[:6])
    cutout_corner = placed[6]
    _draw_polygon(ctx, pts, fill=ctx.fill_color)
    _draw_polygon(ctx, pts)
    notation_bboxes = _draw_polygon_right_angle_notation(ctx, pts, key_prefix="corner")
    cutout_rect = [pts[3], pts[2], cutout_corner, pts[4]]
    region_bbox = bbox_from_points(pts, width=ctx.width, height=ctx.height, pad=2.0)
    cutout_bbox = bbox_from_points(cutout_rect, width=ctx.width, height=ctx.height, pad=4.0)
    label_bboxes = _draw_measurement_list(
        ctx,
        (
            f"AB={width_value}",
            f"AF={height_value}",
            f"DC={cut_width}",
            f"DE={cut_height}",
        ),
        keys=("outer_width", "outer_height", "missing_width", "missing_height"),
    )
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {label: point for label, point in zip(("A", "B", "C", "D", "E", "F"), pts)},
    )
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_points),
        annotation_keyed_points=annotation_points,
        scene_entities=({"type": "rectilinear_corner_cutout", "outline": pts, "points": dict(annotation_points)},),
        render_map={
            "outer_region_bbox": bbox_to_list(region_bbox),
            "missing_corner_bbox": bbox_to_list(cutout_bbox),
            "measurement_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in label_bboxes.items()},
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "visual_notation_bboxes": {key: bbox_to_list(bbox) for key, bbox in notation_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={
            "outer_width": width_value,
            "outer_height": height_value,
            "missing_width": cut_width,
            "missing_height": cut_height,
            "answer_area": int(problem.answer_value),
        },
    )


def _render_house(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Draw a pentagonal house outline for boundary-perimeter reasoning."""

    values = dict(problem.dimensions)
    width_value = int(values["width"])
    wall_height = int(values["wall_height"])
    roof_side = int(values["roof_side"])
    roof_height_units = math.sqrt(max(1.0, float(roof_side) ** 2 - (float(width_value) / 2.0) ** 2))
    scale = min(
        26.0,
        (float(ctx.width) - 220.0) / max(1.0, float(width_value)),
        (float(ctx.height) - 170.0) / max(1.0, float(wall_height) + roof_height_units),
    )
    scale = max(8.0, float(scale))
    w_px = float(width_value) * scale
    half_w = w_px / 2.0
    roof_h = math.sqrt(max(1.0, (float(roof_side) * scale) ** 2 - (half_w**2)))
    left = (float(ctx.width) - w_px) / 2.0
    base_y = 78.0 + (float(wall_height) * scale) + roof_h
    raw_a = (left, base_y)
    raw_b = (left + w_px, base_y)
    raw_c = (left + w_px, base_y - (float(wall_height) * scale))
    raw_d = (left + w_px / 2.0, raw_c[1] - roof_h)
    raw_e = (left, raw_c[1])
    a, b, c, d, e = _place_points(ctx, (raw_a, raw_b, raw_c, raw_d, raw_e))
    pts = [a, b, c, d, e]
    _draw_polygon(ctx, pts, fill=ctx.fill_color)
    _draw_polygon(ctx, pts)
    notation_bboxes = {
        "left_base_wall_right_angle": _draw_right_angle_notation(ctx, a, b, e),
        "right_base_wall_right_angle": _draw_right_angle_notation(ctx, b, a, c),
        "left_wall_equal_tick": _draw_equal_side_ticks(ctx, a, e, count=1),
        "right_wall_equal_tick": _draw_equal_side_ticks(ctx, b, c, count=1),
        "left_roof_equal_tick": _draw_equal_side_ticks(ctx, e, d, count=2),
        "right_roof_equal_tick": _draw_equal_side_ticks(ctx, c, d, count=2),
    }
    target_bbox = bbox_from_points(pts, width=ctx.width, height=ctx.height, pad=4.0)
    label_bboxes = _draw_measurement_list(
        ctx,
        (
            f"AB={width_value}",
            f"BC={wall_height}",
            f"CD={roof_side}",
            f"DE={roof_side}",
            f"EA={wall_height}",
        ),
        keys=(
            "base_length_AB",
            "wall_height_BC",
            "roof_side_CD",
            "roof_side_DE",
            "wall_height_EA",
        ),
    )
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {"A": a, "B": b, "C": c, "D": d, "E": e},
    )
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_points),
        annotation_keyed_points=annotation_points,
        scene_entities=({"type": "pentagonal_house_outline", "points": dict(annotation_points)},),
        render_map={
            "target_boundary_bbox": bbox_to_list(target_bbox),
            "measurement_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in label_bboxes.items()},
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "visual_notation_bboxes": {key: bbox_to_list(bbox) for key, bbox in notation_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={
            "AB": width_value,
            "AE": wall_height,
            "CD_equals_DE": roof_side,
            "BC_equals_AE": True,
            "answer_perimeter": int(problem.answer_value),
        },
    )


def _render_tabbed(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Draw a rectilinear tabbed outline for perimeter reasoning."""

    values = dict(problem.dimensions)
    width_value = int(values["width"])
    height_value = int(values["height"])
    tab_height = int(values["tab_height"])
    tab_width = int(values.get("tab_width", max(4, round(float(width_value) * 0.42))))
    shoulder_width = int(values.get("shoulder_width", max(1, (int(width_value) - int(tab_width)) // 2)))
    scale = min(
        25.0,
        (float(ctx.width) - 220.0) / max(1.0, float(width_value)),
        (float(ctx.height) - 170.0) / max(1.0, float(height_value + tab_height)),
    )
    scale = max(8.0, float(scale))
    w_px, h_px, tab_h_px = float(width_value) * scale, float(height_value) * scale, float(tab_height) * scale
    tab_w_px = float(tab_width) * scale
    left = (float(ctx.width) - w_px) / 2.0
    bottom = 78.0 + h_px + tab_h_px
    x0 = left + (float(shoulder_width) * scale)
    x1 = x0 + tab_w_px
    raw_pts = [
        (left, bottom),
        (left + w_px, bottom),
        (left + w_px, bottom - h_px),
        (x1, bottom - h_px),
        (x1, bottom - h_px - tab_h_px),
        (x0, bottom - h_px - tab_h_px),
        (x0, bottom - h_px),
        (left, bottom - h_px),
    ]
    pts = list(_place_points(ctx, raw_pts))
    _draw_polygon(ctx, pts, fill=ctx.fill_color)
    _draw_polygon(ctx, pts)
    notation_bboxes = _draw_polygon_right_angle_notation(ctx, pts, key_prefix="corner")
    target_bbox = bbox_from_points(pts, width=ctx.width, height=ctx.height, pad=4.0)
    label_bboxes = _draw_measurement_list(
        ctx,
        (
            f"AB={width_value}",
            f"BC={height_value}",
            f"CD={shoulder_width}",
            f"DE={tab_height}",
            f"EF={tab_width}",
            f"FG={tab_height}",
            f"GH={shoulder_width}",
            f"HA={height_value}",
        ),
        anchor=(float(ctx.width) - 70.0, 150.0),
    )
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {label: point for label, point in zip(("A", "B", "C", "D", "E", "F", "G", "H"), pts)},
        distance=22.0,
    )
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_points),
        annotation_keyed_points=annotation_points,
        scene_entities=({"type": "tabbed_rectilinear_polygon", "outline": pts, "points": dict(annotation_points)},),
        render_map={
            "target_boundary_bbox": bbox_to_list(target_bbox),
            "measurement_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in label_bboxes.items()},
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "visual_notation_bboxes": {key: bbox_to_list(bbox) for key, bbox in notation_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={
            "overall_width": width_value,
            "main_height": height_value,
            "tab_height": tab_height,
            "tab_width": tab_width,
            "shoulder_width": shoulder_width,
            "answer_perimeter": int(problem.answer_value),
        },
    )


def _boundary_width(ctx: CompositeRenderContext) -> int:
    return max(int(ctx.line_width) + 2, 6)


def _render_semicircle(ctx: CompositeRenderContext, problem: CompositeShapeProblem, *, cutout: bool) -> RenderedCompositeShape:
    """Draw a rectangle combined with one semicircular add/remove component."""

    values = dict(problem.dimensions)
    width_units = int(values["width_units"])
    height_units = int(values["height_units"])
    radius_units = int(values["radius_units"])
    total_width_units = float(width_units + radius_units)
    total_height_units = max(float(height_units), float(2 * radius_units))
    diagram_width = _diagram_width(ctx)
    scale = min(
        22.0,
        diagram_width / max(1.0, total_width_units),
        (float(ctx.height) - 200.0) / max(1.0, total_height_units),
    )
    scale = max(7.0, float(scale))
    rect_w = float(width_units) * scale
    rect_h = float(height_units) * scale
    radius_px = float(radius_units) * scale
    left = _DIAGRAM_LEFT + ((diagram_width - (total_width_units * scale)) / 2.0)
    top = max(112.0, (float(ctx.height) - rect_h) / 2.0)
    right = left + rect_w
    bottom = top + rect_h
    mid_y = (top + bottom) / 2.0
    cap_start_y = mid_y - radius_px
    cap_end_y = mid_y + radius_px
    arc_start, arc_end = (90.0, 270.0) if cutout else (-90.0, 90.0)
    raw_center = (right, mid_y)
    raw_arc = _arc_points(raw_center, radius_px, start_degrees=arc_start, end_degrees=arc_end)
    raw_rect = ((left, top), (right, top), (right, bottom), (left, bottom))
    raw_cap_start = (right, cap_start_y)
    raw_cap_end = (right, cap_end_y)
    fit_points = (*raw_rect, *raw_arc)
    if ctx.scene_transform is not None:
        ctx.scene_transform.resolve(fit_points)
    rect = list(_place_points(ctx, raw_rect))
    p_left_top, p_right_top, p_right_bottom, p_left_bottom = rect
    center = _place_point(ctx, raw_center)
    arc = list(_place_points(ctx, raw_arc))
    cap_start = _place_point(ctx, raw_cap_start)
    cap_end = _place_point(ctx, raw_cap_end)
    _fill_polygon(ctx, rect, fill=ctx.fill_color)
    if cutout:
        _fill_polygon(ctx, [center, *arc], fill=ctx.background_color)
    else:
        _fill_polygon(ctx, [center, *arc], fill=ctx.fill_color)
    _draw_polyline(ctx, [p_left_top, p_right_top])
    _draw_polyline(ctx, [p_left_bottom, p_right_bottom])
    _draw_polyline(ctx, [p_left_top, p_left_bottom])
    if cap_start_y > top + 1.0:
        _draw_polyline(ctx, [p_right_top, cap_start])
    if cap_end_y < bottom - 1.0:
        _draw_polyline(ctx, [cap_end, p_right_bottom])
    _draw_polyline(ctx, arc)
    radius_midpoint = _place_point(
        ctx,
        (
            right + (radius_px * (-1.0 if cutout else 1.0)),
            mid_y,
        ),
    )
    top_right_reference = cap_start if cap_start_y > top + 1.0 else p_right_bottom
    bottom_right_reference = cap_end if cap_end_y < bottom - 1.0 else p_right_top
    notation_bboxes = {
        "top_left_right_angle": _draw_right_angle_notation(ctx, p_left_top, p_right_top, p_left_bottom),
        "bottom_left_right_angle": _draw_right_angle_notation(ctx, p_left_bottom, p_left_top, p_right_bottom),
        "top_right_right_angle": _draw_right_angle_notation(ctx, p_right_top, p_left_top, top_right_reference),
        "bottom_right_right_angle": _draw_right_angle_notation(ctx, p_right_bottom, bottom_right_reference, p_left_bottom),
    }
    if problem.metric_kind != "perimeter":
        notation_bboxes.update(_draw_radius_witness(ctx, center, radius_midpoint))
    if problem.metric_kind == "perimeter":
        highlight_width = _boundary_width(ctx)
        _draw_polyline(ctx, [p_left_top, p_right_top], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, [p_left_bottom, p_right_bottom], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, [p_left_top, p_left_bottom], fill=ctx.accent_color, width=highlight_width)
        if cap_start_y > top + 1.0:
            _draw_polyline(ctx, [p_right_top, cap_start], fill=ctx.accent_color, width=highlight_width)
        if cap_end_y < bottom - 1.0:
            _draw_polyline(ctx, [cap_end, p_right_bottom], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, arc, fill=ctx.accent_color, width=highlight_width)
    support_roles: list[str] = []
    support_bboxes: list[BBox] = []
    if problem.metric_kind == "perimeter":
        remainder_units = round1((float(height_units) - (2.0 * float(radius_units))) / 2.0)
        list_bboxes = _draw_measurement_list(
            ctx,
            (
                f"AB={fmt_measure(width_units)}",
                f"BE={fmt_measure(remainder_units)}",
                f"arc EF={fmt_measure(float(values['arc_length']))}",
                f"FC={fmt_measure(remainder_units)}",
                f"CD={fmt_measure(width_units)}",
                f"DA={fmt_measure(height_units)}",
            ),
            keys=(
                "top_width_label",
                "upper_remainder_label",
                "arc_length_label",
                "lower_remainder_label",
                "bottom_width_label",
                "height_label",
            ),
        )
        support_roles.extend(list(list_bboxes))
        support_bboxes.extend(list_bboxes.values())
    else:
        list_labels: list[str] = []
        list_keys: list[str] = []
        if problem.metric_kind != "missing_width":
            list_labels.append(f"AB={fmt_measure(width_units)}")
            list_keys.append("width_label")
        list_labels.extend((f"AD={fmt_measure(height_units)}", f"r={fmt_measure(radius_units)}"))
        list_keys.extend(("height_label", "radius_label"))
        if problem.metric_kind == "missing_width":
            list_labels.append(f"Area={float(values['total_area']):.1f}")
            list_keys.append("total_area_label")
        list_bboxes = _draw_measurement_list(ctx, tuple(list_labels), keys=tuple(list_keys))
        support_roles.extend(list(list_bboxes))
        support_bboxes.extend(list_bboxes.values())
        if problem.metric_kind == "missing_width":
            width_dim_y = min(bottom + 44.0, float(ctx.height) - 54.0)
            width_label_offset_y = -30.0 if width_dim_y >= float(ctx.height) - 62.0 else 30.0
            width_bbox = _draw_dimension(
                ctx,
                _place_point(ctx, (left, width_dim_y)),
                _place_point(ctx, (right, width_dim_y)),
                "?",
                label_offset=(0.0, width_label_offset_y),
            )
            support_roles.append("width_label")
            support_bboxes.append(width_bbox)
    center_marker_bbox = _draw_point_marker(ctx, center)
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {
            "A": p_left_top,
            "B": p_right_top,
            "C": p_right_bottom,
            "D": p_left_bottom,
            **({"E": cap_start, "F": cap_end} if problem.metric_kind == "perimeter" else {}),
            "O": center,
        },
        center=_place_point(ctx, ((left + right) / 2.0, (top + bottom) / 2.0)),
        distance=24.0,
    )
    target_points = [*rect, *arc] if not cutout else rect
    target_bbox = bbox_from_points(target_points, width=ctx.width, height=ctx.height, pad=8.0)
    curved_component_bbox = bbox_from_points([center, *arc], width=ctx.width, height=ctx.height, pad=6.0)
    annotation_roles = tuple(annotation_points)
    if problem.metric_kind == "perimeter":
        annotation_roles = tuple(annotation_points)
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_roles),
        annotation_keyed_points=annotation_points,
        scene_entities=(
            {
                "entity_id": "target_shape",
                "entity_type": "curvilinear_composite",
                "bbox": bbox_to_list(target_bbox),
                "points": dict(annotation_points),
                "components": [
                    {"kind": "rectangle", "width": width_units, "height": height_units},
                    {"kind": "semicircle", "radius": radius_units, "operation": "subtract" if cutout else "add"},
                ],
            },
        ),
        render_map={
            "target_bbox": bbox_to_list(target_bbox),
            "curved_component_bbox": bbox_to_list(curved_component_bbox),
            "support_bboxes": [bbox_to_list(bbox) for bbox in support_bboxes],
            "support_roles": list(support_roles),
            "center_marker_bbox": bbox_to_list(center_marker_bbox),
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "visual_notation_bboxes": {key: bbox_to_list(bbox) for key, bbox in notation_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={
            "formula_family": problem.formula_family,
            "operation": "subtract" if cutout else "add",
            **dict(values),
        },
    )


def _render_quarter_sector(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Draw a rectangle whose top-right corner has a quarter-sector cutout."""

    values = dict(problem.dimensions)
    width_units = int(values["width_units"])
    height_units = int(values["height_units"])
    radius_units = int(values["radius_units"])
    diagram_width = _diagram_width(ctx)
    scale = min(
        22.0,
        diagram_width / max(1.0, float(width_units)),
        (float(ctx.height) - 200.0) / max(1.0, float(height_units)),
    )
    scale = max(8.0, float(scale))
    rect_w = float(width_units) * scale
    rect_h = float(height_units) * scale
    radius_px = float(radius_units) * scale
    left = _DIAGRAM_LEFT + ((diagram_width - rect_w) / 2.0)
    top = max(112.0, (float(ctx.height) - rect_h) / 2.0)
    right = left + rect_w
    bottom = top + rect_h
    raw_center = (right, top)
    raw_arc = _arc_points(raw_center, radius_px, start_degrees=90.0, end_degrees=180.0)
    raw_a = (left, top)
    raw_b = raw_center
    raw_c = (right, bottom)
    raw_d = (left, bottom)
    raw_e = (right - radius_px, top)
    raw_f = (right, top + radius_px)
    fit_points = (raw_a, raw_b, raw_c, raw_d, *raw_arc)
    if ctx.scene_transform is not None:
        ctx.scene_transform.resolve(fit_points)
    a, b, c, d, e, f = _place_points(ctx, (raw_a, raw_b, raw_c, raw_d, raw_e, raw_f))
    arc = list(_place_points(ctx, raw_arc))
    _fill_polygon(ctx, [a, b, c, d], fill=ctx.secondary_fill_color)
    _fill_polygon(ctx, [b, *arc], fill=ctx.background_color)
    _draw_polyline(ctx, [a, d])
    _draw_polyline(ctx, [d, c])
    _draw_polyline(ctx, [f, c])
    _draw_polyline(ctx, [e, a])
    _draw_polyline(ctx, arc)
    guide_bboxes = {
        "original_top_extension_guide": _draw_dashed_line(ctx, e, b),
        "original_right_extension_guide": _draw_dashed_line(ctx, b, f),
    }
    if problem.metric_kind == "perimeter":
        highlight_width = _boundary_width(ctx)
        _draw_polyline(ctx, [a, e], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, [a, d], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, [d, c], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, [f, c], fill=ctx.accent_color, width=highlight_width)
        _draw_polyline(ctx, arc, fill=ctx.accent_color, width=highlight_width)
    notation_bboxes = {
        "top_left_right_angle": _draw_right_angle_notation(ctx, a, e, d),
        "bottom_left_right_angle": _draw_right_angle_notation(ctx, d, a, c),
        "bottom_right_right_angle": _draw_right_angle_notation(ctx, c, f, d),
        "quarter_sector_right_angle": _draw_right_angle_notation(
            ctx,
            b,
            e,
            f,
        ),
        **guide_bboxes,
    }
    if problem.metric_kind == "perimeter":
        list_bboxes = _draw_measurement_list(
            ctx,
            (
                f"AE={fmt_measure(width_units - radius_units)}",
                f"FC={fmt_measure(height_units - radius_units)}",
                f"CD={fmt_measure(width_units)}",
                f"AD={fmt_measure(height_units)}",
                f"arc={fmt_measure(float(values['arc_length']))}",
            ),
            keys=(
                "top_remainder_label",
                "right_remainder_label",
                "bottom_width_label",
                "left_height_label",
                "arc_length_label",
            ),
            anchor=(float(ctx.width) - 98.0, 150.0),
            line_gap=30.0,
        )
        support_bboxes = list(list_bboxes.values())
        support_roles = list(list_bboxes)
    else:
        list_bboxes = _draw_measurement_list(
            ctx,
            (
                f"AB={fmt_measure(width_units)}",
                f"AD={fmt_measure(height_units)}",
                f"BE={fmt_measure(radius_units)}",
                f"BF={fmt_measure(radius_units)}",
            ),
            keys=("width_label", "height_label", "radius_horizontal_label", "radius_vertical_label"),
            anchor=(float(ctx.width) - 94.0, 198.0),
            line_gap=30.0,
        )
        support_bboxes = list(list_bboxes.values())
        support_roles = list(list_bboxes)
    center_marker_bbox = _draw_point_marker(ctx, b)
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "E": e,
            "F": f,
        },
        center=_place_point(ctx, ((left + right) / 2.0, (top + bottom) / 2.0)),
        distance=24.0,
    )
    target_bbox = bbox_from_points((a, b, c, d), width=ctx.width, height=ctx.height, pad=8.0)
    curved_component_bbox = bbox_from_points((b, *arc), width=ctx.width, height=ctx.height, pad=6.0)
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_points),
        annotation_keyed_points=annotation_points,
        scene_entities=(
            {
                "entity_id": "target_shape",
                "entity_type": "curvilinear_composite",
                "bbox": bbox_to_list(target_bbox),
                "points": dict(annotation_points),
                "components": [
                    {"kind": "rectangle", "width": width_units, "height": height_units},
                    {"kind": "sector", "radius": radius_units, "theta_degrees": 90, "operation": "subtract"},
                ],
            },
        ),
        render_map={
            "target_bbox": bbox_to_list(target_bbox),
            "curved_component_bbox": bbox_to_list(curved_component_bbox),
            "support_bboxes": [bbox_to_list(bbox) for bbox in support_bboxes],
            "support_roles": list(support_roles),
            "center_marker_bbox": bbox_to_list(center_marker_bbox),
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "visual_notation_bboxes": {key: bbox_to_list(bbox) for key, bbox in notation_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={"formula_family": problem.formula_family, **dict(values)},
    )


def _render_sector(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Draw a circular sector with a missing central angle value."""

    values = dict(problem.dimensions)
    theta = int(values["theta_degrees"])
    radius_units = int(values["radius_units"])
    radius_px = 180.0
    raw_center = (310.0, 310.0)
    start_deg = -135.0
    end_deg = start_deg + float(theta)
    start_rad = math.radians(start_deg)
    end_rad = math.radians(end_deg)
    raw_p0 = (raw_center[0] + radius_px * math.cos(start_rad), raw_center[1] + radius_px * math.sin(start_rad))
    raw_p1 = (raw_center[0] + radius_px * math.cos(end_rad), raw_center[1] + radius_px * math.sin(end_rad))
    raw_arc = _arc_points(raw_center, radius_px, start_degrees=start_deg, end_degrees=end_deg)
    if ctx.scene_transform is not None:
        ctx.scene_transform.resolve((raw_center, raw_p0, raw_p1, *raw_arc))
    center, p0, p1 = _place_points(ctx, (raw_center, raw_p0, raw_p1))
    arc = list(_place_points(ctx, raw_arc))
    _fill_polygon(ctx, [center, *arc], fill=ctx.fill_color)
    _draw_polyline(ctx, [center, p0])
    _draw_polyline(ctx, arc)
    _draw_polyline(ctx, [center, p1])
    center_marker_bbox = _draw_point_marker(ctx, center)
    mid_rad = math.radians((start_deg + end_deg) / 2.0)
    target_bbox = draw_label(
        ctx,
        "?",
        _place_point(ctx, (raw_center[0] + 54.0 * math.cos(mid_rad), raw_center[1] + 54.0 * math.sin(mid_rad))),
        small=False,
    )
    if problem.metric_kind == "sector_from_arc":
        measure_text = f"arc={float(values['arc_length']):.1f}"
        measure_role = "arc_length_label"
    else:
        measure_text = f"Area={float(values['sector_area']):.1f}"
        measure_role = "sector_area_label"
    list_bboxes = _draw_measurement_list(
        ctx,
        (f"OA={fmt_measure(radius_units)}", f"OB={fmt_measure(radius_units)}", measure_text),
        keys=("radius_OA_label", "radius_OB_label", measure_role),
    )
    sector_bbox = bbox_from_points((center, *arc), width=ctx.width, height=ctx.height, pad=8.0)
    annotation_points, point_label_bboxes = _draw_labeled_points(
        ctx,
        {"O": center, "A": p0, "B": p1},
        center=((center[0] + p0[0] + p1[0]) / 3.0, (center[1] + p0[1] + p1[1]) / 3.0),
        distance=24.0,
    )
    return RenderedCompositeShape(
        image=ctx.image,
        answer_value=problem.answer_value,
        annotation_roles=tuple(annotation_points),
        annotation_keyed_points=annotation_points,
        scene_entities=(
            {
                "entity_id": "target_sector",
                "entity_type": "sector",
                "bbox": bbox_to_list(sector_bbox),
                "points": dict(annotation_points),
                "radius": radius_units,
                "theta_degrees": theta,
                "arc_length": float(values["arc_length"]),
                "sector_area": float(values["sector_area"]),
            },
        ),
        render_map={
            "target_bbox": bbox_to_list(target_bbox),
            "sector_bbox": bbox_to_list(sector_bbox),
            "support_bboxes": [bbox_to_list(bbox) for bbox in list_bboxes.values()],
            "support_roles": list(list_bboxes),
            "center_marker_bbox": bbox_to_list(center_marker_bbox),
            "point_label_bboxes": {key: bbox_to_list(bbox) for key, bbox in point_label_bboxes.items()},
            "coord_space": "pixel",
        },
        witness={"formula_family": problem.formula_family, **dict(values)},
    )


def render_composite_shape(ctx: CompositeRenderContext, problem: CompositeShapeProblem) -> RenderedCompositeShape:
    """Dispatch from semantic shape family to the corresponding renderer primitive."""

    family = str(problem.shape_family)
    if family == "rect_cut":
        return _render_rect_cut(ctx, problem)
    if family == "l_profile":
        return _render_l_profile(ctx, problem)
    if family == "house":
        return _render_house(ctx, problem)
    if family == "tabbed":
        return _render_tabbed(ctx, problem)
    if family == "semi_cap":
        return _render_semicircle(ctx, problem, cutout=False)
    if family == "semi_cut":
        return _render_semicircle(ctx, problem, cutout=True)
    if family == "quarter_cut":
        return _render_quarter_sector(ctx, problem)
    if family == "sector":
        return _render_sector(ctx, problem)
    raise ValueError(f"unsupported composite shape family: {family}")
