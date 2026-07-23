"""Rendering primitives for triangle-relations analytical diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.text_legibility import (
    READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
    READ_REQUIRED_TEXT_MIN_LAB_DISTANCE,
    contrast_ratio,
)
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_readout_centered,
    pad_bbox,
)
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import add_scaled, mid, perp, point_to_list, sub, unit
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import (
    DOMAIN,
    SCENE_ID,
    AngleLabel,
    Point,
    RenderContext,
    RenderedTriangleRelationsScene,
    RightAngleMark,
    SegmentLabel,
    TickGroup,
    TriangleRelationsProblem,
)

BBox = tuple[float, float, float, float]
_DARK_READOUT: tuple[int, int, int] = (10, 14, 22)
_LIGHT_READOUT: tuple[int, int, int] = (250, 252, 255)


def _high_contrast_readout_color(*surfaces: Sequence[int]) -> tuple[int, int, int]:
    """Use a conservative ink for dense triangle measurement readouts."""

    normalized = [tuple(int(channel) for channel in surface[:3]) for surface in surfaces if len(surface) >= 3]
    if not normalized:
        return _DARK_READOUT
    return max(
        (_DARK_READOUT, _LIGHT_READOUT),
        key=lambda candidate: min(float(contrast_ratio(candidate, surface)) for surface in normalized),
    )


def _opposite_readout_color(color: Sequence[int]) -> tuple[int, int, int]:
    return _LIGHT_READOUT if tuple(int(channel) for channel in color[:3]) == _DARK_READOUT else _DARK_READOUT


def _dual_ink_readout_metadata(
    *,
    fill: Sequence[int],
    stroke: Sequence[int],
    surfaces: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Record contrast for readouts whose fill and outline work as a pair."""

    fill_rgb = tuple(int(channel) for channel in fill[:3])
    stroke_rgb = tuple(int(channel) for channel in stroke[:3])
    surface_rgbs = tuple(tuple(int(channel) for channel in surface[:3]) for surface in surfaces if len(surface) >= 3)
    min_contrast = min(
        (max(float(contrast_ratio(fill_rgb, surface)), float(contrast_ratio(stroke_rgb, surface))) for surface in surface_rgbs),
        default=float("inf"),
    )
    min_lab = min(
        (
            max(
                float(color_distance(fill_rgb, surface, distance_space="lab")),
                float(color_distance(stroke_rgb, surface, distance_space="lab")),
            )
            for surface in surface_rgbs
        ),
        default=float("inf"),
    )
    return {
        "surface_rgbs": [list(surface) for surface in surface_rgbs],
        "surface_sample_method": "triangle_relations_dual_ink_surface_anchors",
        "min_contrast_ratio": round(float(min_contrast), 3),
        "min_lab_distance": round(float(min_lab), 3),
        "min_contrast_required": round(float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO), 3),
        "min_lab_distance_required": round(float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE), 3),
        "passes": bool(
            float(min_contrast) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
            and float(min_lab) >= float(READ_REQUIRED_TEXT_MIN_LAB_DISTANCE)
        ),
    }


def _bbox_has_visible_area(bbox: Sequence[float]) -> bool:
    """Return whether a bbox survived clamping with a nonzero visible area."""

    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (x1 - x0) > 1.0 and (y1 - y0) > 1.0


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RenderContext:
    """Create a styled analytical-diagram render context for this scene."""

    rng = spawn_rng(int(instance_seed), f"{DOMAIN}.{SCENE_ID}.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 580)))
    image, background_meta, diagram_style, diagram_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
        require_grid=False,
        style_profile="analytical_diagram",
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"geometry.{SCENE_ID}.font_family",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 3)))
    label_stroke_width = int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 1)))
    readout_color = _high_contrast_readout_color(
        diagram_style.canvas_rgb,
        diagram_style.paper_rgb,
        diagram_style.panel_fill_rgb,
        diagram_style.panel_alt_fill_rgb,
        diagram_style.option_fill_rgb,
    )
    readout_stroke_color = _opposite_readout_color(readout_color)
    readout_metadata = _dual_ink_readout_metadata(
        fill=readout_color,
        stroke=readout_stroke_color,
        surfaces=(
            diagram_style.canvas_rgb,
            diagram_style.paper_rgb,
            diagram_style.panel_fill_rgb,
            diagram_style.panel_alt_fill_rgb,
            diagram_style.option_fill_rgb,
            readout_color,
            readout_stroke_color,
        ),
    )
    diagram_meta = {
        **geometry_diagram_style_metadata(diagram_style),
        **dict(diagram_meta),
        "font_family": font_record.to_trace(),
        "font_asset_version": font_asset_version(),
        "triangle_relations_readout_ink": list(readout_color),
        "triangle_relations_readout_stroke": list(readout_stroke_color),
        "triangle_relations_readout_contrast": dict(readout_metadata),
    }
    rgb_image = image.convert("RGB")
    return RenderContext(
        rng=rng,
        image=rgb_image,
        draw=ImageDraw.Draw(rgb_image),
        width=int(width),
        height=int(height),
        line_color=readout_color,
        label_color=readout_color,
        label_stroke_color=readout_stroke_color,
        accent_color=readout_color,
        fill_color=tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
        alt_fill_color=tuple(int(value) for value in diagram_style.option_fill_rgb),
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        readout_text_metadata=dict(readout_metadata),
        font=load_font(max(12, int(font_size)), bold=False, font_family=str(font_family)),
        small_font=load_font(max(10, int(small_font_size)), bold=False, font_family=str(font_family)),
        diagram_style_meta=diagram_meta,
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            rng,
            params=params,
            render_defaults=render_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _draw_line(ctx: RenderContext, points: Sequence[Point], *, color: tuple[int, int, int] | None = None, width: int | None = None) -> tuple[float, float, float, float]:
    ctx.draw.line(list(points), fill=color or ctx.line_color, width=int(width or ctx.line_width), joint="curve")
    return bbox_from_points(tuple(points), width=ctx.width, height=ctx.height, pad=float(width or ctx.line_width) + 3.0)


def _draw_polygon(ctx: RenderContext, points: Sequence[Point], *, fill: tuple[int, int, int] | None = None) -> tuple[float, float, float, float]:
    if fill is not None:
        ctx.draw.polygon(list(points), fill=fill)
    return _draw_line(ctx, tuple(points) + (tuple(points)[0],))


def _coerce_bbox(bbox: Sequence[float]) -> BBox:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _bbox_overlap_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = _coerce_bbox(a)
    bx0, by0, bx1, by1 = _coerce_bbox(b)
    overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    return overlap_w * overlap_h


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / length_sq))
    closest = (ax + vx * t, ay + vy * t)
    return math.hypot(px - closest[0], py - closest[1])


def _point_to_bbox_distance(point: Point, bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = _coerce_bbox(bbox)
    px, py = float(point[0]), float(point[1])
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 <= 0.0 <= o2 or o2 <= 0.0 <= o1) and (o3 <= 0.0 <= o4 or o4 <= 0.0 <= o3)


def _segment_intersects_bbox(start: Point, end: Point, bbox: Sequence[float]) -> bool:
    x0, y0, x1, y1 = _coerce_bbox(bbox)
    if x0 <= float(start[0]) <= x1 and y0 <= float(start[1]) <= y1:
        return True
    if x0 <= float(end[0]) <= x1 and y0 <= float(end[1]) <= y1:
        return True
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    edges = tuple(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges)


def _bbox_segment_clearance(bbox: Sequence[float], start: Point, end: Point) -> float:
    if _segment_intersects_bbox(start, end, bbox):
        return 0.0
    x0, y0, x1, y1 = _coerce_bbox(bbox)
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    distances = [_point_to_segment_distance(corner, start, end) for corner in corners]
    distances.append(_point_to_bbox_distance(start, bbox))
    distances.append(_point_to_bbox_distance(end, bbox))
    return min(distances)


def _expand_bbox(bbox: Sequence[float], pad: float) -> BBox:
    x0, y0, x1, y1 = _coerce_bbox(bbox)
    return (x0 - float(pad), y0 - float(pad), x1 + float(pad), y1 + float(pad))


def _readout_bbox(ctx: RenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    font = ctx.small_font if bool(small) else ctx.font
    bbox = ctx.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=max(0, int(ctx.label_stroke_width)),
    )
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (x0 - 4.0, y0 - 4.0, x1 + 4.0, y1 + 4.0)


def _placement_penalty(
    ctx: RenderContext,
    bbox: Sequence[float],
    *,
    avoid_bboxes: Sequence[Sequence[float]],
    avoid_segments: Sequence[tuple[Point, Point]] = (),
    anchor: Point,
    center: Point,
) -> float:
    x0, y0, x1, y1 = _coerce_bbox(bbox)
    margin = 16.0
    edge_penalty = 0.0
    if x0 < margin:
        edge_penalty += 1_000_000.0 + (margin - x0) * 120.0
    if y0 < margin:
        edge_penalty += 1_000_000.0 + (margin - y0) * 120.0
    if x1 > float(ctx.width) - margin:
        edge_penalty += 1_000_000.0 + (x1 - (float(ctx.width) - margin)) * 120.0
    if y1 > float(ctx.height) - margin:
        edge_penalty += 1_000_000.0 + (y1 - (float(ctx.height) - margin)) * 120.0
    overlap_penalty = 0.0
    for avoid_bbox in avoid_bboxes:
        overlap_area = _bbox_overlap_area(bbox, avoid_bbox)
        if overlap_area > 0.0:
            overlap_penalty += 1_000_000.0 + overlap_area * 100.0
    for segment_start, segment_end in avoid_segments:
        clearance = _bbox_segment_clearance(bbox, segment_start, segment_end)
        if clearance < 10.0:
            overlap_penalty += 1_000_000.0 + (10.0 - clearance) * 80_000.0
        elif clearance < 24.0:
            overlap_penalty += (24.0 - clearance) * 4_000.0
    distance_penalty = math.hypot(float(center[0]) - float(anchor[0]), float(center[1]) - float(anchor[1])) * 0.08
    return edge_penalty + overlap_penalty + distance_penalty


def _choose_segment_label_center(
    ctx: RenderContext,
    a: Point,
    b: Point,
    text: str,
    offset: float,
    *,
    avoid_bboxes: Sequence[Sequence[float]],
) -> Point:
    tangent = unit(sub(b, a))
    normal = perp(tangent)
    midpoint = mid(a, b)
    base_offset = max(30.0, abs(float(offset)))
    preferred_sign = 1.0 if float(offset) >= 0.0 else -1.0
    segment_length = math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    is_target_label = "?" in str(text)
    offset_scales = (1.0, 1.4, 1.9, 2.5, 3.2)
    shift_scales = (0.0, -0.18, 0.18, -0.34, 0.34, -0.52, 0.52)
    if is_target_label:
        shift_scales = (0.0, -0.18, 0.18, -0.34, 0.34, -0.52, 0.52, -0.68, 0.68)
    candidates: list[Point] = []
    for sign in (preferred_sign, -preferred_sign):
        for offset_scale in offset_scales:
            for shift_scale in shift_scales:
                center = add_scaled(add_scaled(midpoint, normal, sign * base_offset * offset_scale), tangent, segment_length * shift_scale)
                candidates.append(center)
    return min(
        candidates,
        key=lambda center: _placement_penalty(
            ctx,
            _readout_bbox(ctx, text, center, small=True),
            avoid_bboxes=avoid_bboxes,
            anchor=midpoint,
            center=center,
        ),
    )


def _draw_segment_label(
    ctx: RenderContext,
    points: Mapping[str, Point],
    label: SegmentLabel,
    *,
    avoid_bboxes: Sequence[Sequence[float]] = (),
) -> BBox:
    a = points[str(label.segment[0])]
    b = points[str(label.segment[1])]
    center = _choose_segment_label_center(ctx, a, b, str(label.text), float(label.offset), avoid_bboxes=avoid_bboxes)
    return draw_readout_centered(
        ctx,
        str(label.text),
        center,
        small=True,
        backed=False,
        extra_metadata=dict(ctx.readout_text_metadata),
    )


def _segment_name(label: SegmentLabel) -> str:
    return "".join(str(part) for part in label.segment)


def _is_target_segment_label(label: SegmentLabel) -> bool:
    return "?" in str(label.text)


def _use_side_readout(label: SegmentLabel) -> bool:
    placement = str(label.placement)
    if placement == "side_readout":
        return True
    if placement == "segment":
        return False
    return not _is_target_segment_label(label)


def _side_readout_text(label: SegmentLabel) -> str:
    name = _segment_name(label)
    text = str(label.text)
    if "=" in text:
        left, right = text.split("=", 1)
        if left.strip() == name:
            return text
        return f"{name}={right.strip()}"
    return f"{name}={text}"


def _readout_list_bboxes(ctx: RenderContext, texts: Sequence[str], origin: Point) -> tuple[tuple[BBox, Point], ...]:
    x0, y0 = float(origin[0]), float(origin[1])
    line_gap = 8.0
    rows: list[tuple[BBox, Point]] = []
    cursor_y = y0
    for text in texts:
        text_bbox = ctx.draw.textbbox(
            (0.0, 0.0),
            str(text),
            anchor="lt",
            font=ctx.small_font,
            stroke_width=max(0, int(ctx.label_stroke_width)),
        )
        width = float(text_bbox[2] - text_bbox[0]) + 8.0
        height = float(text_bbox[3] - text_bbox[1]) + 8.0
        bbox = (x0, cursor_y, x0 + width, cursor_y + height)
        rows.append((bbox, (x0 + width / 2.0, cursor_y + height / 2.0)))
        cursor_y += height + line_gap
    return tuple(rows)


def _readout_list_size(ctx: RenderContext, texts: Sequence[str]) -> tuple[float, float]:
    rows = _readout_list_bboxes(ctx, texts, (0.0, 0.0))
    if not rows:
        return (0.0, 0.0)
    width = max(bbox[2] - bbox[0] for bbox, _center in rows)
    height = max(bbox[3] for bbox, _center in rows)
    return (float(width), float(height))


def _choose_readout_list_origin(
    ctx: RenderContext,
    texts: Sequence[str],
    *,
    diagram_bbox: Sequence[float],
    avoid_bboxes: Sequence[Sequence[float]],
) -> Point:
    """Place operand readouts near the diagram without covering geometry."""

    width, height = _readout_list_size(ctx, texts)
    margin = 22.0
    diagram_x0, diagram_y0, diagram_x1, diagram_y1 = _coerce_bbox(diagram_bbox)
    diagram_cx = (diagram_x0 + diagram_x1) / 2.0
    diagram_cy = (diagram_y0 + diagram_y1) / 2.0
    candidates: list[tuple[str, Point]] = [
        ("right", (diagram_x1 + margin, diagram_cy - height / 2.0)),
        ("left", (diagram_x0 - width - margin, diagram_cy - height / 2.0)),
        ("top", (diagram_cx - width / 2.0, diagram_y0 - height - margin)),
        ("bottom", (diagram_cx - width / 2.0, diagram_y1 + margin)),
        ("right_top", (diagram_x1 + margin, diagram_y0)),
        ("right_bottom", (diagram_x1 + margin, diagram_y1 - height)),
        ("left_top", (diagram_x0 - width - margin, diagram_y0)),
        ("left_bottom", (diagram_x0 - width - margin, diagram_y1 - height)),
        ("top_left", (diagram_x0, diagram_y0 - height - margin)),
        ("top_right", (diagram_x1 - width, diagram_y0 - height - margin)),
        ("bottom_left", (diagram_x0, diagram_y1 + margin)),
        ("bottom_right", (diagram_x1 - width, diagram_y1 + margin)),
    ]
    side_bias = {
        "right": 0.0,
        "left": 25.0,
        "top": 50.0,
        "bottom": 75.0,
        "right_top": 100.0,
        "right_bottom": 105.0,
        "left_top": 125.0,
        "left_bottom": 130.0,
        "top_left": 150.0,
        "top_right": 155.0,
        "bottom_left": 175.0,
        "bottom_right": 180.0,
    }

    def candidate_penalty(item: tuple[str, Point]) -> float:
        name, origin = item
        penalty = side_bias.get(name, 0.0)
        for bbox, center in _readout_list_bboxes(ctx, texts, origin):
            penalty += _placement_penalty(ctx, bbox, avoid_bboxes=avoid_bboxes, anchor=center, center=center)
            list_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            penalty += math.hypot(list_center[0] - diagram_cx, list_center[1] - diagram_cy) * 0.02
        return penalty

    return min(candidates, key=candidate_penalty)[1]


def _draw_side_readout_labels(
    ctx: RenderContext,
    labels: Sequence[SegmentLabel],
    *,
    diagram_bbox: Sequence[float],
    avoid_bboxes: Sequence[Sequence[float]] = (),
) -> dict[str, BBox]:
    if not labels:
        return {}
    texts = tuple(_side_readout_text(label) for label in labels)
    origin = _choose_readout_list_origin(ctx, texts, diagram_bbox=diagram_bbox, avoid_bboxes=avoid_bboxes)
    rows = _readout_list_bboxes(ctx, texts, origin)
    bboxes: dict[str, BBox] = {}
    for label, text, (_expected_bbox, center) in zip(labels, texts, rows, strict=True):
        bboxes[label.role or "".join(label.segment)] = _coerce_bbox(
            draw_readout_centered(
                ctx,
                text,
                center,
                small=True,
                backed=False,
                extra_metadata=dict(ctx.readout_text_metadata),
            )
        )
    return bboxes


def _angle_points(vertex: Point, arm_a: Point, arm_b: Point, radius: float) -> tuple[Point, ...]:
    start = math.atan2(float(arm_a[1]) - float(vertex[1]), float(arm_a[0]) - float(vertex[0]))
    end = math.atan2(float(arm_b[1]) - float(vertex[1]), float(arm_b[0]) - float(vertex[0]))
    delta = (end - start + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(10, int(abs(delta) / math.radians(8.0)))
    return tuple(
        (
            float(vertex[0]) + math.cos(start + delta * (idx / steps)) * float(radius),
            float(vertex[1]) + math.sin(start + delta * (idx / steps)) * float(radius),
        )
        for idx in range(steps + 1)
    )


def _angle_marker_radius(ctx: RenderContext, vertex: Point, arm_a: Point, arm_b: Point, angle: AngleLabel) -> float:
    """Keep angle arcs legible without letting them spill outside tight triangles."""

    requested = float(angle.radius)
    arm_a_length = math.hypot(float(arm_a[0]) - float(vertex[0]), float(arm_a[1]) - float(vertex[1]))
    arm_b_length = math.hypot(float(arm_b[0]) - float(vertex[0]), float(arm_b[1]) - float(vertex[1]))
    short_arm = max(1.0, min(arm_a_length, arm_b_length))
    has_text = bool(str(angle.text))
    max_radius = min(42.0 if has_text else 30.0, short_arm * (0.28 if has_text else 0.22))
    min_radius = 16.0 if has_text else 12.0
    return max(min_radius, min(requested, max_radius))


def _draw_angle(
    ctx: RenderContext,
    points: Mapping[str, Point],
    angle: AngleLabel,
    *,
    avoid_bboxes: Sequence[Sequence[float]] = (),
) -> tuple[float, float, float, float]:
    vertex = points[str(angle.vertex)]
    arm_a = points[str(angle.arm_a)]
    arm_b = points[str(angle.arm_b)]
    radius = _angle_marker_radius(ctx, vertex, arm_a, arm_b, angle)
    arc = _angle_points(vertex, arm_a, arm_b, radius)
    arc_bbox = _draw_line(ctx, arc, color=ctx.accent_color, width=max(2, ctx.line_width - 1))
    if not str(angle.text):
        return arc_bbox
    text_bbox = draw_readout_centered(
        ctx,
        str(angle.text),
        _choose_angle_text_center(ctx, vertex, arm_a, arm_b, str(angle.text), radius, avoid_bboxes=avoid_bboxes),
        small=True,
        backed=False,
        extra_metadata=dict(ctx.readout_text_metadata),
    )
    return bbox_from_points(((arc_bbox[0], arc_bbox[1]), (arc_bbox[2], arc_bbox[3]), (text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[3])), width=ctx.width, height=ctx.height, pad=2.0)


def _choose_angle_text_center(
    ctx: RenderContext,
    vertex: Point,
    arm_a: Point,
    arm_b: Point,
    text: str,
    radius: float,
    *,
    avoid_bboxes: Sequence[Sequence[float]],
) -> Point:
    """Place an angle readout near the mark without sitting on its arms."""

    u = unit(sub(arm_a, vertex))
    v = unit(sub(arm_b, vertex))
    direction = unit(add_scaled(u, v, 1.0))
    if math.hypot(direction[0], direction[1]) <= 1e-6:
        direction = perp(unit(sub(arm_a, vertex)))
    candidate_directions = (
        direction,
        perp(u),
        perp(v),
        (-direction[0], -direction[1]),
        (-perp(u)[0], -perp(u)[1]),
        (-perp(v)[0], -perp(v)[1]),
    )
    candidates: list[Point] = []
    for candidate_direction in candidate_directions:
        normalized = unit(candidate_direction)
        for distance in (float(radius) + 28.0, float(radius) + 42.0, float(radius) + 58.0, float(radius) + 76.0):
            candidates.append(add_scaled(vertex, normalized, distance))
    return min(
        candidates,
        key=lambda center: _placement_penalty(
            ctx,
            _readout_bbox(ctx, text, center, small=True),
            avoid_bboxes=avoid_bboxes,
            anchor=vertex,
            center=center,
        ),
    )


def _draw_right_angle(ctx: RenderContext, points: Mapping[str, Point], mark: RightAngleMark) -> tuple[float, float, float, float]:
    vertex = points[str(mark.vertex)]
    u = unit(sub(points[str(mark.arm_a)], vertex))
    v = unit(sub(points[str(mark.arm_b)], vertex))
    size = 22.0
    p1 = add_scaled(vertex, u, size)
    p2 = add_scaled(p1, v, size)
    p3 = add_scaled(vertex, v, size)
    return _draw_line(ctx, (p1, p2, p3), color=ctx.accent_color, width=max(2, ctx.line_width - 1))


def _draw_tick_group(ctx: RenderContext, points: Mapping[str, Point], group: TickGroup) -> tuple[float, float, float, float]:
    """Draw equal/parallel marks; angle-bisector groups are already shown by arcs."""

    if str(group.kind) == "angle_bisector" or int(group.count) <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    tick_points: list[Point] = []
    for a_label, b_label in group.segments:
        a = points[str(a_label)]
        b = points[str(b_label)]
        tangent = unit(sub(b, a))
        normal = perp(tangent)
        center = mid(a, b)
        for idx in range(int(group.count)):
            shift = (idx - (int(group.count) - 1) / 2.0) * 11.0
            tick_center = add_scaled(center, tangent, shift)
            p0 = add_scaled(tick_center, normal, -9.0)
            p1 = add_scaled(tick_center, normal, 9.0)
            _draw_line(ctx, (p0, p1), color=ctx.accent_color, width=max(2, ctx.line_width - 1))
            tick_points.extend((p0, p1))
    return bbox_from_points(tuple(tick_points), width=ctx.width, height=ctx.height, pad=5.0) if tick_points else (0.0, 0.0, 0.0, 0.0)


def _choose_point_label_center(
    ctx: RenderContext,
    label: str,
    point: Point,
    *,
    avoid_bboxes: Sequence[Sequence[float]],
    avoid_segments: Sequence[tuple[Point, Point]] = (),
) -> Point:
    """Choose a readable point-label position without changing projected geometry."""

    x, y = float(point[0]), float(point[1])
    outward_x = -1.0 if x < float(ctx.width) / 2.0 else 1.0
    outward_y = -1.0 if y < float(ctx.height) / 2.0 else 1.0
    offsets = (
        (0.0, -28.0),
        (28.0, 0.0),
        (0.0, 28.0),
        (-28.0, 0.0),
        (24.0 * outward_x, 24.0 * outward_y),
        (24.0 * outward_x, -24.0 * outward_y),
        (-24.0 * outward_x, 24.0 * outward_y),
        (-24.0 * outward_x, -24.0 * outward_y),
        (0.0, -38.0),
        (38.0, 0.0),
        (0.0, 38.0),
        (-38.0, 0.0),
        (0.0, -52.0),
        (52.0, 0.0),
        (0.0, 52.0),
        (-52.0, 0.0),
        (38.0 * outward_x, 38.0 * outward_y),
        (38.0 * outward_x, -38.0 * outward_y),
        (-38.0 * outward_x, 38.0 * outward_y),
        (-38.0 * outward_x, -38.0 * outward_y),
        (0.0, -66.0),
        (66.0, 0.0),
        (0.0, 66.0),
        (-66.0, 0.0),
        (0.0, -84.0),
        (84.0, 0.0),
        (0.0, 84.0),
        (-84.0, 0.0),
        (60.0 * outward_x, 60.0 * outward_y),
        (60.0 * outward_x, -60.0 * outward_y),
        (-60.0 * outward_x, 60.0 * outward_y),
        (-60.0 * outward_x, -60.0 * outward_y),
        (0.0, -104.0),
        (104.0, 0.0),
        (0.0, 104.0),
        (-104.0, 0.0),
    )
    candidates = [(x + dx, y + dy) for dx, dy in offsets]
    for radius in (76.0, 96.0, 118.0, 140.0):
        for degrees in range(0, 360, 30):
            radians = math.radians(float(degrees))
            candidates.append((x + math.cos(radians) * radius, y + math.sin(radians) * radius))
    return min(
        candidates,
        key=lambda center: _placement_penalty(
            ctx,
            _readout_bbox(ctx, label, center, small=True),
            avoid_bboxes=avoid_bboxes,
            avoid_segments=avoid_segments,
            anchor=point,
            center=center,
        ),
    )


def _draw_point_labels(
    ctx: RenderContext,
    points: Mapping[str, Point],
    *,
    avoid_bboxes: Sequence[Sequence[float]] = (),
    point_avoid_bboxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    point_avoid_segments: Mapping[str, Sequence[tuple[Point, Point]]] | None = None,
) -> dict[str, list[float]]:
    bboxes: dict[str, list[float]] = {}
    placed_avoid_bboxes: list[BBox] = []
    for label, point in points.items():
        point_specific = tuple((point_avoid_bboxes or {}).get(str(label), ()))
        center = _choose_point_label_center(
            ctx,
            str(label),
            point,
            avoid_bboxes=tuple(avoid_bboxes) + point_specific + tuple(placed_avoid_bboxes),
            avoid_segments=tuple((point_avoid_segments or {}).get(str(label), ())),
        )
        bbox = draw_readout_centered(ctx, str(label), center, small=True, backed=False, required=False)
        bboxes[str(label)] = bbox_to_list(bbox)
        placed_avoid_bboxes.append(_expand_bbox(bbox, 10.0))
    return bboxes


def _point_line_guard_bboxes(
    ctx: RenderContext,
    points: Mapping[str, Point],
    edges: Sequence[tuple[str, str]],
) -> dict[str, tuple[BBox, ...]]:
    """Return local line guards so point labels do not sit on incident segments."""

    result: dict[str, list[BBox]] = {str(label): [] for label in points}
    half_span = 98.0
    max_distance = max(10.0, float(ctx.line_width) + 7.0)
    for point_label, point in points.items():
        px, py = float(point[0]), float(point[1])
        for edge_start, edge_end in edges:
            a = points[str(edge_start)]
            b = points[str(edge_end)]
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            vx, vy = bx - ax, by - ay
            length = math.hypot(vx, vy)
            if length <= 1e-6:
                continue
            t = ((px - ax) * vx + (py - ay) * vy) / length
            if t < -max_distance or t > length + max_distance:
                continue
            clamped_t = max(0.0, min(length, t))
            closest = (ax + vx * clamped_t / length, ay + vy * clamped_t / length)
            distance = math.hypot(px - closest[0], py - closest[1])
            if distance > max_distance:
                continue
            start_t = max(0.0, clamped_t - half_span)
            end_t = min(length, clamped_t + half_span)
            guard_start = (ax + vx * start_t / length, ay + vy * start_t / length)
            guard_end = (ax + vx * end_t / length, ay + vy * end_t / length)
            result[str(point_label)].append(
                bbox_from_points(
                    (guard_start, guard_end),
                    width=ctx.width,
                    height=ctx.height,
                    pad=max(22.0, float(ctx.line_width) + 18.0),
                )
            )
    return {label: tuple(bboxes) for label, bboxes in result.items()}


def _point_line_guard_segments(
    points: Mapping[str, Point],
    edges: Sequence[tuple[str, str]],
) -> dict[str, tuple[tuple[Point, Point], ...]]:
    segments = tuple((points[str(edge_start)], points[str(edge_end)]) for edge_start, edge_end in edges)
    return {str(label): segments for label in points}


def _vertex_guard_bboxes(ctx: RenderContext, points: Mapping[str, Point], *, radius: float = 18.0) -> dict[str, BBox]:
    return {
        key: pad_bbox((float(point[0]), float(point[1]), float(point[0]), float(point[1])), float(radius), width=int(ctx.width), height=int(ctx.height))
        for key, point in points.items()
    }


def _draw_point_markers(
    ctx: RenderContext,
    points: Mapping[str, Point],
    labels: Sequence[str],
) -> dict[str, list[float]]:
    """Draw small visual markers for special interior construction points."""

    bboxes: dict[str, list[float]] = {}
    radius = max(4.0, float(ctx.line_width) + 2.5)
    outline_width = max(2, int(ctx.line_width) - 1)
    for raw_label in labels:
        label = str(raw_label)
        if label not in points:
            continue
        x, y = points[label]
        bbox = (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius)
        ctx.draw.ellipse(
            bbox,
            fill=ctx.accent_color,
            outline=ctx.label_stroke_color,
            width=outline_width,
        )
        bboxes[label] = bbox_to_list(bbox)
    return bboxes


def render_triangle_relations_scene(
    ctx: RenderContext,
    problem: TriangleRelationsProblem,
) -> RenderedTriangleRelationsScene:
    """Render one resolved construction and project witnesses after transform."""

    case = problem.case
    source_points = tuple(case.vertices.values())
    ctx.scene_transform.resolve(source_points)
    points = ctx.scene_transform.keyed_points(case.vertices)
    diagram_bbox = bbox_from_points(tuple(points.values()), width=ctx.width, height=ctx.height, pad=12.0)
    polygon_bboxes: dict[str, Any] = {}
    for polygon in case.filled_polygons:
        polygon_bboxes["filled_" + "_".join(polygon)] = bbox_to_list(_draw_polygon(ctx, [points[label] for label in polygon], fill=ctx.fill_color))
    for polygon in case.polygons:
        polygon_bboxes["_".join(polygon)] = bbox_to_list(_draw_polygon(ctx, [points[label] for label in polygon]))
    edge_bboxes: dict[str, list[float]] = {}
    edge_bbox_by_segment: dict[frozenset[str], BBox] = {}
    for a, b in case.edges:
        bbox = _draw_line(ctx, (points[a], points[b]), color=ctx.line_color)
        edge_bboxes[f"{a}{b}"] = bbox_to_list(bbox)
        edge_bbox_by_segment[frozenset((a, b))] = _coerce_bbox(bbox)
    vertex_guards = _vertex_guard_bboxes(ctx, points)
    angle_bboxes: dict[str, list[float]] = {}
    angle_bbox_values: list[BBox] = []
    for angle in case.angle_labels:
        angle_edges = (
            edge_bbox_by_segment.get(frozenset((str(angle.vertex), str(angle.arm_a)))),
            edge_bbox_by_segment.get(frozenset((str(angle.vertex), str(angle.arm_b)))),
            vertex_guards.get(str(angle.vertex)),
        )
        bbox = _draw_angle(ctx, points, angle, avoid_bboxes=tuple(bbox for bbox in angle_edges if bbox is not None))
        angle_bboxes[angle.role or f"{angle.arm_a}{angle.vertex}{angle.arm_b}"] = bbox_to_list(bbox)
        angle_bbox_values.append(_coerce_bbox(bbox))
    right_angle_bboxes: dict[str, list[float]] = {}
    right_angle_bbox_values: list[BBox] = []
    for mark in case.right_angles:
        bbox = _draw_right_angle(ctx, points, mark)
        right_angle_bboxes[mark.vertex] = bbox_to_list(bbox)
        right_angle_bbox_values.append(_coerce_bbox(bbox))
    tick_bboxes: dict[str, list[float]] = {}
    tick_bbox_values: list[BBox] = []
    for idx, group in enumerate(case.tick_groups):
        bbox = _draw_tick_group(ctx, points, group)
        tick_bboxes[f"{group.kind}_{idx}"] = bbox_to_list(bbox)
        tick_bbox_values.append(_coerce_bbox(bbox))
    point_marker_bboxes = _draw_point_markers(ctx, points, case.point_mark_labels)
    construction_bboxes = tuple(angle_bbox_values + right_angle_bbox_values + tick_bbox_values)
    label_bboxes: dict[str, list[float]] = {}
    label_bbox_values: list[BBox] = []
    label_avoid_bboxes: list[BBox] = []
    side_readout_labels = tuple(label for label in case.segment_labels if _use_side_readout(label))
    segment_labels = tuple(label for label in case.segment_labels if not _use_side_readout(label))
    for label in segment_labels:
        owned_segment = frozenset(str(value) for value in label.segment)
        avoid_bboxes = (
            tuple(vertex_guards.values())
            + tuple(bbox for segment, bbox in edge_bbox_by_segment.items() if segment != owned_segment)
            + construction_bboxes
            + tuple(label_avoid_bboxes)
        )
        bbox = _draw_segment_label(ctx, points, label, avoid_bboxes=avoid_bboxes)
        raw_bbox = _coerce_bbox(bbox)
        if _bbox_has_visible_area(raw_bbox):
            label_bboxes[label.role or "".join(label.segment)] = bbox_to_list(raw_bbox)
            label_bbox_values.append(raw_bbox)
            label_avoid_bboxes.append(_expand_bbox(raw_bbox, 14.0))
    side_label_bboxes = _draw_side_readout_labels(
        ctx,
        side_readout_labels,
        diagram_bbox=diagram_bbox,
        avoid_bboxes=(
            tuple(vertex_guards.values())
            + tuple(edge_bbox_by_segment.values())
            + construction_bboxes
            + tuple(label_avoid_bboxes)
            + (_expand_bbox(diagram_bbox, 8.0),)
        ),
    )
    for key, raw_bbox in side_label_bboxes.items():
        if _bbox_has_visible_area(raw_bbox):
            label_bboxes[key] = bbox_to_list(raw_bbox)
            label_bbox_values.append(raw_bbox)
            label_avoid_bboxes.append(_expand_bbox(raw_bbox, 14.0))
    point_label_bboxes = _draw_point_labels(
        ctx,
        points,
        avoid_bboxes=construction_bboxes + tuple(label_avoid_bboxes),
        point_avoid_bboxes=_point_line_guard_bboxes(ctx, points, case.edges),
        point_avoid_segments=_point_line_guard_segments(points, case.edges),
    )
    annotation_segment = None
    annotation_point = None
    annotation_points: dict[str, Point] = {}
    roles: tuple[str, ...] = ()
    if problem.annotation_mode == "segment" and case.target_segment is not None:
        annotation_segment = (points[case.target_segment[0]], points[case.target_segment[1]])
        roles = ("target_segment",)
    elif problem.annotation_mode == "point" and case.target_point is not None:
        annotation_point = points[case.target_point]
        roles = ("target_point",)
    elif problem.annotation_mode == "point_map":
        annotation_points = {label: points[label] for label in case.point_annotation_labels}
        roles = tuple(annotation_points)
    return RenderedTriangleRelationsScene(
        image=ctx.image,
        answer=case.answer,
        annotation_mode=str(problem.annotation_mode),
        annotation_segment=annotation_segment,
        annotation_point=annotation_point,
        annotation_points=annotation_points,
        annotation_roles=roles,
        scene_entities=(
            {
                "type": "triangle_relations_diagram",
                "points": {key: point_to_list(value) for key, value in points.items()},
                "case_kind": str(case.case_kind),
            },
        ),
        render_map={
            "coord_space": "pixel",
            "vertices": {key: point_to_list(value) for key, value in points.items()},
            "point_label_bboxes": point_label_bboxes,
            "edge_bboxes": edge_bboxes,
            "polygon_bboxes": polygon_bboxes,
            "measurement_label_bboxes": label_bboxes,
            "angle_bboxes": angle_bboxes,
            "right_angle_bboxes": right_angle_bboxes,
            "tick_bboxes": tick_bboxes,
            "point_marker_bboxes": point_marker_bboxes,
            "single_object_scene_rotation": ctx.scene_transform.metadata(),
        },
        witness=geometry_json_ready(case.trace_values),
        reasoning_steps=int(case.reasoning_steps),
    )


__all__ = ["create_render_context", "render_triangle_relations_scene"]
