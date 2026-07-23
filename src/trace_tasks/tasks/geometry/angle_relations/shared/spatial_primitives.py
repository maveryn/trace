"""Scene-local drawing primitives for angle-relations diagrams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points as _bbox_from_points,
    pad_bbox as _pad_bbox,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import (
    add_scaled as _add,
    mid as _mid,
    perp as _perp,
    sub as _sub,
    unit as _unit,
)
from trace_tasks.tasks.shared.text_legibility import draw_text_traced


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

ANGLE_RELATIONS_NOISE_DEFAULTS: Dict[str, Any] = {
    "apply_prob": 0.5,
    "edit_types": ["blur", "downsample", "jpeg", "noise"],
    "edit_count_range": [1, 1],
    "value_ranges": {
        "blur": {"radius": [0.12, 0.32]},
        "downsample": {"scale": [0.93, 0.98]},
        "jpeg": {"quality": [84, 94]},
        "noise": {"alpha": [0.01, 0.03]},
    },
}


@dataclass
class RenderContext:
    """Resolved rendering inputs shared by angle-relations case builders."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    fill_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    layout_offset: Point = (0.0, 0.0)
    font_family: str = ""
    scene_transform: LazySceneTransform | None = None


def _combine_bboxes(bboxes: Sequence[Sequence[float]], *, width: int, height: int, pad: float = 0.0) -> BBox:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    return _bbox_from_points(
        [(bbox[0], bbox[1]) for bbox in bboxes] + [(bbox[2], bbox[3]) for bbox in bboxes],
        width=width,
        height=height,
        pad=pad,
    )


def _offset_points(ctx: RenderContext, points: Sequence[Point]) -> tuple[Point, ...]:
    """Apply the resolved non-semantic scene placement jitter to several points."""

    offset_points = tuple(
        (
            float(point[0]) + float(ctx.layout_offset[0]),
            float(point[1]) + float(ctx.layout_offset[1]),
        )
        for point in points
    )
    if ctx.scene_transform is not None:
        return ctx.scene_transform.points(offset_points)
    return offset_points


def _draw_text(
    ctx: RenderContext,
    text: str,
    center: Point,
    *,
    font: Any | None = None,
    fill: Color | None = None,
    stroke_width: int | None = None,
) -> BBox:
    """Draw one readout label and return its padded bbox for trace metadata."""

    active_font = font if font is not None else ctx.font
    active_fill = fill if fill is not None else ctx.label_color
    active_stroke_width = int(ctx.label_stroke_width if stroke_width is None else stroke_width)
    x, y = float(center[0]), float(center[1])
    try:
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
    except Exception:
        draw_text_traced(
            ctx.draw,
            (x, y),
            str(text),
            font=active_font,
            fill=active_fill,
            role="readout",
            required=False,
        )
        bbox = ctx.draw.textbbox((x, y), str(text), font=active_font)
    return _pad_bbox(bbox, 2.0, width=ctx.width, height=ctx.height)


def _draw_point_labels(
    ctx: RenderContext,
    points: Mapping[str, Point],
    *,
    offsets: Mapping[str, Point] | None = None,
) -> Dict[str, BBox]:
    bboxes: Dict[str, BBox] = {}
    for label, point in points.items():
        x, y = float(point[0]), float(point[1])
        if offsets is not None and str(label) in offsets:
            offset = offsets[str(label)]
        else:
            if y > ctx.height * 0.68:
                offset = (0.0, 22.0)
            elif y < ctx.height * 0.25:
                offset = (0.0, -22.0)
            elif x < ctx.width * 0.28:
                offset = (-22.0, 0.0)
            else:
                offset = (22.0, 0.0)
        bboxes[str(label)] = _draw_text(
            ctx,
            str(label),
            (x + offset[0], y + offset[1]),
            font=ctx.small_font,
        )
    return bboxes


def _draw_polyline(
    ctx: RenderContext,
    points: Sequence[Point],
    *,
    fill: Color | None = None,
    width: int | None = None,
) -> BBox:
    line_fill = fill if fill is not None else ctx.line_color
    line_width = int(width if width is not None else ctx.line_width)
    ctx.draw.line([(float(x), float(y)) for x, y in points], fill=line_fill, width=line_width, joint="curve")
    return _bbox_from_points(points, width=ctx.width, height=ctx.height, pad=line_width + 2)


def _draw_polygon(
    ctx: RenderContext,
    points: Sequence[Point],
    *,
    outline: Color | None = None,
    fill: Color | None = None,
    width: int | None = None,
) -> BBox:
    if fill is not None:
        ctx.draw.polygon([(float(x), float(y)) for x, y in points], fill=fill)
    closed = list(points) + [points[0]]
    return _draw_polyline(ctx, closed, fill=outline, width=width)


def _draw_angle_arc(ctx: RenderContext, vertex: Point, arm_a: Point, arm_b: Point, *, radius: float = 38.0) -> BBox:
    va = _unit(_sub(arm_a, vertex))
    vb = _unit(_sub(arm_b, vertex))
    angle_a = math.atan2(va[1], va[0])
    angle_b = math.atan2(vb[1], vb[0])
    delta = (angle_b - angle_a + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(8, int(abs(delta) / math.radians(7.5)))
    pts: list[Point] = []
    for idx in range(steps + 1):
        angle = angle_a + (delta * (float(idx) / float(steps)))
        pts.append(
            (
                float(vertex[0]) + (math.cos(angle) * float(radius)),
                float(vertex[1]) + (math.sin(angle) * float(radius)),
            )
        )
    _draw_polyline(ctx, pts, fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
    return _bbox_from_points(pts, width=ctx.width, height=ctx.height, pad=5.0)


def _angle_label_center(vertex: Point, arm_a: Point, arm_b: Point, *, radius: float = 62.0) -> Point:
    va = _unit(_sub(arm_a, vertex))
    vb = _unit(_sub(arm_b, vertex))
    bisector = _unit((va[0] + vb[0], va[1] + vb[1]))
    if math.hypot(bisector[0], bisector[1]) <= 1e-6:
        bisector = _unit(_perp(va))
    return _add(vertex, bisector, radius)


def _draw_angle_label(
    ctx: RenderContext,
    text: str,
    vertex: Point,
    arm_a: Point,
    arm_b: Point,
    *,
    radius: float = 62.0,
) -> tuple[BBox, BBox]:
    arc_bbox = _draw_angle_arc(ctx, vertex, arm_a, arm_b, radius=max(28.0, radius - 24.0))
    text_bbox = _draw_text(ctx, str(text), _angle_label_center(vertex, arm_a, arm_b, radius=radius))
    return arc_bbox, text_bbox


def _angle_annotation_point(vertex: Point, _arm_a: Point, _arm_b: Point, *, label_radius: float = 62.0) -> Point:
    """Public angle annotation points to the angle vertex, not the angle mark."""

    _ = label_radius
    return (float(vertex[0]), float(vertex[1]))


def _intersect_rays(origin_a: Point, angle_a_degrees: float, origin_b: Point, angle_b_degrees: float) -> Point:
    ax, ay = float(origin_a[0]), float(origin_a[1])
    bx, by = float(origin_b[0]), float(origin_b[1])
    da = (math.cos(math.radians(angle_a_degrees)), -math.sin(math.radians(angle_a_degrees)))
    db = (math.cos(math.radians(angle_b_degrees)), -math.sin(math.radians(angle_b_degrees)))
    det = (da[0] * (-db[1])) - (da[1] * (-db[0]))
    if abs(det) <= 1e-9:
        return ((ax + bx) / 2.0, min(ay, by) - 180.0)
    rhs = (bx - ax, by - ay)
    t = ((rhs[0] * (-db[1])) - (rhs[1] * (-db[0]))) / det
    return (ax + (t * da[0]), ay + (t * da[1]))


def _triangle_from_base_angles(
    *,
    left_angle: int,
    right_angle: int,
    width: float = 390.0,
    base_y: float = 410.0,
    canvas_width: float = 720.0,
    canvas_height: float = 560.0,
) -> tuple[Point, Point, Point]:
    top_margin = 94.0
    bottom_margin = 54.0
    side_margin = 58.0
    requested_width = float(width)
    max_base_y = max(top_margin + 120.0, float(canvas_height) - bottom_margin)
    max_height = max(120.0, max_base_y - top_margin)

    local_left = (0.0, 0.0)
    local_right = (requested_width, 0.0)
    local_apex = _intersect_rays(local_left, float(left_angle), local_right, 180.0 - float(right_angle))
    local_height = max(1.0, -float(local_apex[1]))
    scale = min(1.0, max_height / local_height)
    fitted_width = requested_width * scale

    local_right = (fitted_width, 0.0)
    local_apex = _intersect_rays(local_left, float(left_angle), local_right, 180.0 - float(right_angle))
    fitted_height = max(1.0, -float(local_apex[1]))
    fitted_base_y = min(max_base_y, max(float(base_y), top_margin + fitted_height))

    local_points = (
        (0.0, fitted_base_y),
        (float(local_apex[0]), fitted_base_y + float(local_apex[1])),
        (fitted_width, fitted_base_y),
    )
    min_x = min(point[0] for point in local_points)
    max_x = max(point[0] for point in local_points)
    desired_center_x = 160.0 + (requested_width / 2.0)
    x_offset = desired_center_x - ((min_x + max_x) / 2.0)
    x_offset = max(side_margin - min_x, min(float(canvas_width) - side_margin - max_x, x_offset))
    return tuple((float(x) + x_offset, float(y)) for x, y in local_points)  # type: ignore[return-value]


__all__ = [
    "ANGLE_RELATIONS_NOISE_DEFAULTS",
    "BBox",
    "Color",
    "Point",
    "RenderContext",
    "_angle_annotation_point",
    "_bbox_from_points",
    "_draw_angle_label",
    "_draw_point_labels",
    "_draw_polygon",
    "_draw_polyline",
    "_offset_points",
    "_triangle_from_base_angles",
]
