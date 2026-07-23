"""Rendering helpers for circle-theorem value tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.visual.noise import apply_post_image_noise
from ....shared.config_defaults import group_default
from ....shared.text_rendering import (
    draw_text_centered,
    load_font,
    resolve_text_label_center,
)
from ...shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    geometry_diagram_style_metadata,
    geometry_shape_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from ...shared.render_variation import sample_int_render_param
from ...shared.scene_transform import LazySceneTransform

from .state import (
    Point,
    BBox,
    DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    RenderedCircleTheoremScene,
    SCENE_ID,
    _text_bbox_for_center,
    _bbox_to_list,
)


def _model_transform(
    *,
    canvas_size: int,
    margin_px: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> Callable[[Point], Point]:
    width = max(1e-6, float(max_x) - float(min_x))
    height = max(1e-6, float(max_y) - float(min_y))
    usable = max(10.0, float(canvas_size) - (2.0 * float(margin_px)))
    scale = min(float(usable) / float(width), float(usable) / float(height))
    center_x = 0.5 * (float(min_x) + float(max_x))
    center_y = 0.5 * (float(min_y) + float(max_y))
    canvas_center = 0.5 * float(canvas_size)

    def transform(point: Point) -> Point:
        return (
            float(canvas_center + ((float(point[0]) - center_x) * scale)),
            float(canvas_center - ((float(point[1]) - center_y) * scale)),
        )

    transform.scale = float(scale)  # type: ignore[attr-defined]
    return transform

def _draw_centered_text_with_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Point,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int,
) -> BBox:
    draw_text_centered(
        draw,
        text=str(text),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=int(stroke_width),
    )
    return _text_bbox_for_center(
        draw,
        text=str(text),
        center=(float(center[0]), float(center[1])),
        font=font,
        stroke_width=int(stroke_width),
    )

def _segment_label_center(
    p0: Point, p1: Point, *, side: float, offset_px: float
) -> Point:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    norm = max(1e-6, math.hypot(dx, dy))
    nx, ny = -dy / norm, dx / norm
    return (
        float(0.5 * (x0 + x1) + (float(side) * float(offset_px) * nx)),
        float(0.5 * (y0 + y1) + (float(side) * float(offset_px) * ny)),
    )

def _segment_label_direction(p0: Point, p1: Point, *, side: float) -> Point:
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    norm = max(1e-6, math.hypot(dx, dy))
    return (
        float(float(side) * (-dy / norm)),
        float(float(side) * (dx / norm)),
    )

def _circle_boundary_segments(
    center_px: Point, radius_px: float, *, steps: int = 72
) -> List[Tuple[Point, Point]]:
    cx, cy = float(center_px[0]), float(center_px[1])
    radius = float(max(1.0, radius_px))
    count = max(12, int(steps))
    points = [
        (
            float(cx + (radius * math.cos((2.0 * math.pi * index) / float(count)))),
            float(cy + (radius * math.sin((2.0 * math.pi * index) / float(count)))),
        )
        for index in range(count)
    ]
    return [(points[index], points[(index + 1) % count]) for index in range(count)]

def _draw_measurement_label(
    draw: ImageDraw.ImageDraw,
    *,
    token: str,
    p0: Point,
    p1: Point,
    side: float,
    offset_px: float,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int,
    blocked_segments: Sequence[Tuple[Point, Point]],
    blocked_points: Sequence[Point],
    occupied_boxes: List[BBox],
    canvas_size: int,
) -> BBox:
    """Place one measurement label while avoiding existing geometry and label boxes."""
    anchor = (

        float(0.5 * (float(p0[0]) + float(p1[0]))),
        float(0.5 * (float(p0[1]) + float(p1[1]))),
    )
    center, bbox = resolve_text_label_center(
        draw,
        text=str(token),
        anchor=anchor,
        base_direction=_segment_label_direction(p0, p1, side=float(side)),
        offset_px=max(36.0, float(offset_px)),
        font=font,
        blocked_segments=blocked_segments,
        blocked_points=blocked_points,
        occupied_boxes=occupied_boxes,
        stroke_width=int(stroke_width),
        line_clearance_px=10.0,
        point_clearance_px=8.0,
        canvas_size=int(canvas_size),
    )
    draw_text_centered(
        draw,
        text=str(token),
        center=center,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=int(stroke_width),
    )
    return tuple(float(value) for value in bbox)

def _screen_angle_path(
    start_angle: float, end_angle: float, *, steps: int
) -> List[Point]:
    start = float(start_angle)
    delta = ((float(end_angle) - start + math.pi) % (2.0 * math.pi)) - math.pi
    count = max(2, int(steps))
    return [
        (
            math.cos(start + (delta * (index / float(count - 1)))),
            math.sin(start + (delta * (index / float(count - 1)))),
        )
        for index in range(count)
    ]

def _draw_angle_annotation(
    draw: ImageDraw.ImageDraw,
    *,
    token: str | None,
    vertex_px: Point,
    arm0_px: Point,
    arm1_px: Point,
    radius_px: float,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int,
    line_fill: Tuple[int, int, int],
    line_width: int,
    blocked_segments: Sequence[Tuple[Point, Point]],
    blocked_points: Sequence[Point],
    occupied_boxes: List[BBox],
    canvas_size: int,
) -> BBox | None:
    """Draw an angle marker and attach its token without obscuring construction geometry."""
    vx, vy = float(vertex_px[0]), float(vertex_px[1])

    vectors: List[Point] = []
    for point in (arm0_px, arm1_px):
        dx, dy = float(point[0]) - vx, float(point[1]) - vy
        norm = max(1e-6, math.hypot(dx, dy))
        vectors.append((dx / norm, dy / norm))
    angle0 = math.atan2(vectors[0][1], vectors[0][0])
    angle1 = math.atan2(vectors[1][1], vectors[1][0])
    unit_path = _screen_angle_path(angle0, angle1, steps=16)
    path = [
        (vx + (float(radius_px) * ux), vy + (float(radius_px) * uy))
        for ux, uy in unit_path
    ]
    if len(path) >= 2:
        draw.line(path, fill=line_fill, width=max(1, int(line_width)))
    if token is None:
        return None
    midpoint = unit_path[len(unit_path) // 2]
    center, bbox = resolve_text_label_center(
        draw,
        text=str(token),
        anchor=(vx, vy),
        base_direction=(float(midpoint[0]), float(midpoint[1])),
        offset_px=float(radius_px) + 42.0,
        font=font,
        blocked_segments=blocked_segments,
        blocked_points=blocked_points,
        occupied_boxes=occupied_boxes,
        stroke_width=int(stroke_width),
        line_clearance_px=10.0,
        point_clearance_px=8.0,
        canvas_size=int(canvas_size),
    )
    draw_text_centered(
        draw,
        text=str(token),
        center=center,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=int(stroke_width),
    )
    return tuple(float(value) for value in bbox)

def _draw_right_angle_marker(
    draw: ImageDraw.ImageDraw,
    *,
    vertex_px: Point,
    arm0_px: Point,
    arm1_px: Point,
    size_px: float,
    line_fill: Tuple[int, int, int],
    line_width: int,
) -> None:
    vx, vy = float(vertex_px[0]), float(vertex_px[1])
    unit_vectors: List[Point] = []
    for point in (arm0_px, arm1_px):
        dx, dy = float(point[0]) - vx, float(point[1]) - vy
        norm = max(1e-6, math.hypot(dx, dy))
        unit_vectors.append((dx / norm, dy / norm))
    size = max(8.0, float(size_px))
    p0 = (vx + (size * unit_vectors[0][0]), vy + (size * unit_vectors[0][1]))
    p1 = (
        vx + (size * (unit_vectors[0][0] + unit_vectors[1][0])),
        vy + (size * (unit_vectors[0][1] + unit_vectors[1][1])),
    )
    p2 = (vx + (size * unit_vectors[1][0]), vy + (size * unit_vectors[1][1]))
    draw.line((p0, p1, p2), fill=line_fill, width=max(1, int(line_width)))

def _draw_circle_arc_annotation(
    draw: ImageDraw.ImageDraw,
    *,
    token: str | None,
    center_px: Point,
    radius_px: float,
    start_px: Point,
    end_px: Point,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int,
    line_fill: Tuple[int, int, int],
    line_width: int,
    blocked_segments: Sequence[Tuple[Point, Point]],
    blocked_points: Sequence[Point],
    occupied_boxes: List[BBox],
    canvas_size: int,
) -> BBox | None:
    """Draw an arc marker and place its token outside the circle when possible."""
    cx, cy = float(center_px[0]), float(center_px[1])

    angle0 = math.atan2(float(start_px[1]) - cy, float(start_px[0]) - cx)
    angle1 = math.atan2(float(end_px[1]) - cy, float(end_px[0]) - cx)
    unit_path = _screen_angle_path(angle0, angle1, steps=28)
    arc_radius = float(radius_px) * 1.012
    path = [(cx + (arc_radius * ux), cy + (arc_radius * uy)) for ux, uy in unit_path]
    if len(path) >= 2:
        draw.line(path, fill=line_fill, width=max(1, int(line_width)))
    if token is None:
        return None
    midpoint = unit_path[len(unit_path) // 2]
    anchor = (
        float(cx + (float(radius_px) * midpoint[0])),
        float(cy + (float(radius_px) * midpoint[1])),
    )
    center, bbox = resolve_text_label_center(
        draw,
        text=str(token),
        anchor=anchor,
        base_direction=(float(midpoint[0]), float(midpoint[1])),
        offset_px=52.0,
        font=font,
        blocked_segments=blocked_segments,
        blocked_points=blocked_points,
        occupied_boxes=occupied_boxes,
        stroke_width=int(stroke_width),
        line_clearance_px=12.0,
        point_clearance_px=8.0,
        canvas_size=int(canvas_size),
    )
    draw_text_centered(
        draw,
        text=str(token),
        center=center,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=int(stroke_width),
    )
    return tuple(float(value) for value in bbox)

def _draw_point_label(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    point_px: Point,
    base_direction: Point,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int,
    point_label_offset_px: float,
    blocked_segments: Sequence[Tuple[Point, Point]],
    blocked_points: Sequence[Point],
    occupied_boxes: List[BBox],
    canvas_size: int,
) -> BBox:
    """Place one point label near its construction point without crossing geometry."""
    center, bbox = resolve_text_label_center(

        draw,
        text=str(label),
        anchor=(float(point_px[0]), float(point_px[1])),
        base_direction=(float(base_direction[0]), float(base_direction[1])),
        offset_px=float(point_label_offset_px),
        font=font,
        blocked_segments=blocked_segments,
        blocked_points=blocked_points,
        occupied_boxes=occupied_boxes,
        stroke_width=int(stroke_width),
        line_clearance_px=8.0,
        point_clearance_px=8.0,
        canvas_size=int(canvas_size),
    )
    draw_text_centered(
        draw,
        text=str(label),
        center=center,
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=int(stroke_width),
    )
    occupied_boxes.append(tuple(float(value) for value in bbox))
    return tuple(float(value) for value in bbox)

def _render_base_scene(
    *,
    rng,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    point_model: Mapping[str, Point],
    circle_center: Point,
    circle_radius: float,
    segments: Mapping[str, Tuple[str, str]],
    measurement_specs: Sequence[Tuple[str, str, float]],
    support_measurement_tokens: Sequence[str],
    annotation_point_labels: Sequence[str],
    annotation_values: Mapping[str, int],
    theorem_trace: Mapping[str, Any],
    angle_marker_specs: Sequence[Mapping[str, Any]] | None = None,
    right_angle_marker_specs: Sequence[Mapping[str, Any]] | None = None,
    circle_arc_specs: Sequence[Mapping[str, Any]] | None = None,
) -> RenderedCircleTheoremScene:
    """Render one already-selected circle-theorem construction.

    The caller owns the task/query semantics; this function only projects
    model-space points, draws the diagram, and returns pixel-space metadata.
    """
    canvas_min = int(
        group_default(render_defaults, "canvas_size_min", DEFAULTS.canvas_size_min)
    )
    canvas_max = int(
        group_default(render_defaults, "canvas_size_max", DEFAULTS.canvas_size_max)
    )
    explicit_canvas_size = params.get("canvas_size")
    if explicit_canvas_size is not None:
        canvas_size = max(256, int(explicit_canvas_size))
    else:
        canvas_size = int(
            rng.randint(min(canvas_min, canvas_max), max(canvas_min, canvas_max))
        )
    margin_px = float(
        params.get(
            "outer_margin_px",
            group_default(
                render_defaults, "outer_margin_px", DEFAULTS.outer_margin_px
            ),
        )
    )
    line_width = int(
        sample_int_render_param(
            rng,
            params=params,
            render_defaults=render_defaults,
            key="line_width",
            fallback=DEFAULTS.line_width,
            minimum_value=1,
        )
    )
    circle_line_width = int(
        params.get(
            "circle_line_width",
            group_default(
                render_defaults,
                "circle_line_width",
                max(line_width, DEFAULTS.circle_line_width),
            ),
        )
    )
    point_radius_px = int(
        params.get(
            "point_radius_px",
            group_default(
                render_defaults, "point_radius_px", DEFAULTS.point_radius_px
            ),
        )
    )
    measurement_offset_px = float(
        params.get(
            "measurement_label_offset_px",
            group_default(
                render_defaults,
                "measurement_label_offset_px",
                DEFAULTS.measurement_label_offset_px,
            ),
        )
    )
    point_label_offset_px = float(
        params.get(
            "point_label_offset_px",
            group_default(
                render_defaults,
                "point_label_offset_px",
                DEFAULTS.point_label_offset_px,
            ),
        )
    )
    label_font_size_px = int(
        rng.randint(
            int(
                group_default(
                    render_defaults,
                    "label_font_size_min",
                    DEFAULTS.label_font_size_min,
                )
            ),
            int(
                group_default(
                    render_defaults,
                    "label_font_size_max",
                    DEFAULTS.label_font_size_max,
                )
            ),
        )
    )
    measurement_font_size_px = int(
        rng.randint(
            int(
                group_default(
                    render_defaults,
                    "measurement_font_size_min",
                    DEFAULTS.measurement_font_size_min,
                )
            ),
            int(
                group_default(
                    render_defaults,
                    "measurement_font_size_max",
                    DEFAULTS.measurement_font_size_max,
                )
            ),
        )
    )

    min_x = float(circle_center[0] - circle_radius)
    max_x = float(circle_center[0] + circle_radius)
    min_y = float(circle_center[1] - circle_radius)
    max_y = float(circle_center[1] + circle_radius)
    for point in point_model.values():
        min_x = min(min_x, float(point[0]))
        max_x = max(max_x, float(point[0]))
        min_y = min(min_y, float(point[1]))
        max_y = max(max_y, float(point[1]))
    pad = max(2.0, 0.12 * max(float(max_x - min_x), float(max_y - min_y)))
    transform = _model_transform(
        canvas_size=int(canvas_size),
        margin_px=float(margin_px),
        min_x=float(min_x - pad),
        min_y=float(min_y - pad),
        max_x=float(max_x + pad),
        max_y=float(max_y + pad),
    )
    point_pixels = {
        label: list(transform(point)) for label, point in point_model.items()
    }
    center_px = transform(circle_center)
    radius_px = float(circle_radius) * float(getattr(transform, "scale"))
    scene_transform = LazySceneTransform(
        rng,
        params=params,
        render_defaults=render_defaults,
        canvas_width=int(canvas_size),
        canvas_height=int(canvas_size),
    )
    scene_transform.resolve(
        (
            center_px,
            (center_px[0] - radius_px, center_px[1]),
            (center_px[0] + radius_px, center_px[1]),
            (center_px[0], center_px[1] - radius_px),
            (center_px[0], center_px[1] + radius_px),
            *(tuple(float(value) for value in point) for point in point_pixels.values()),
        )
    )
    point_pixels = {
        label: list(scene_transform.point(tuple(float(value) for value in point)))
        for label, point in point_pixels.items()
    }
    center_px = scene_transform.point(center_px)
    radius_px *= float(scene_transform.transform.scale)

    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(canvas_size),
        canvas_height=int(canvas_size),
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id=SCENE_ID,
        allow_dark=False,
        require_grid=False,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    )
    draw = ImageDraw.Draw(image)
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    line_color = tuple(int(value) for value in shape_style.line_color)
    label_color = tuple(int(value) for value in shape_style.label_color)
    stroke_color = tuple(int(value) for value in shape_style.label_stroke_color)
    label_font = load_font(label_font_size_px, bold=False)
    measurement_font = load_font(measurement_font_size_px, bold=False)
    label_stroke_width = 0
    measurement_stroke_width = 0

    circle_bbox = (
        float(center_px[0] - radius_px),
        float(center_px[1] - radius_px),
        float(center_px[0] + radius_px),
        float(center_px[1] + radius_px),
    )
    draw.ellipse(circle_bbox, outline=line_color, width=max(1, int(circle_line_width)))

    segment_pixels: Dict[str, List[List[float]]] = {}
    blocked_segments: List[Tuple[Point, Point]] = []
    for segment_id, (label0, label1) in segments.items():
        p0 = tuple(float(value) for value in point_pixels[str(label0)])
        p1 = tuple(float(value) for value in point_pixels[str(label1)])
        segment_pixels[str(segment_id)] = [list(p0), list(p1)]
        blocked_segments.append((p0, p1))
        draw.line((p0, p1), fill=line_color, width=max(1, int(line_width)))
    circle_blocked_segments = _circle_boundary_segments(center_px, float(radius_px))
    label_blocked_segments: List[Tuple[Point, Point]] = list(blocked_segments) + list(
        circle_blocked_segments
    )

    for point in point_pixels.values():
        px, py = float(point[0]), float(point[1])
        r = max(2, int(point_radius_px))
        draw.ellipse((px - r, py - r, px + r, py + r), fill=line_color)
    blocked_points = [
        tuple(float(value) for value in point) for point in point_pixels.values()
    ]

    occupied_boxes: List[BBox] = []
    token_bboxes: Dict[str, List[float]] = {}
    for token, segment_id, side in measurement_specs:
        point_pair = segment_pixels[str(segment_id)]
        bbox = _draw_measurement_label(
            draw,
            token=str(token),
            p0=tuple(point_pair[0]),
            p1=tuple(point_pair[1]),
            side=float(side),
            offset_px=float(measurement_offset_px),
            font=measurement_font,
            fill=label_color,
            stroke_fill=stroke_color,
            stroke_width=int(measurement_stroke_width),
            blocked_segments=label_blocked_segments,
            blocked_points=blocked_points,
            occupied_boxes=occupied_boxes,
            canvas_size=int(canvas_size),
        )
        bbox_list = _bbox_to_list(bbox)
        token_bboxes[str(token)] = list(bbox_list)
        occupied_boxes.append(tuple(float(value) for value in bbox))

    for spec in right_angle_marker_specs or ():
        vertex = str(spec["vertex"])
        arm0 = str(spec["arm0"])
        arm1 = str(spec["arm1"])
        _draw_right_angle_marker(
            draw,
            vertex_px=tuple(float(value) for value in point_pixels[vertex]),
            arm0_px=tuple(float(value) for value in point_pixels[arm0]),
            arm1_px=tuple(float(value) for value in point_pixels[arm1]),
            size_px=float(spec.get("size_px", 24.0)),
            line_fill=line_color,
            line_width=max(1, int(line_width) - 1),
        )

    for spec in angle_marker_specs or ():
        vertex = str(spec["vertex"])
        arm0 = str(spec["arm0"])
        arm1 = str(spec["arm1"])
        token_value = spec.get("token")
        bbox = _draw_angle_annotation(
            draw,
            token=None if token_value is None else str(token_value),
            vertex_px=tuple(float(value) for value in point_pixels[vertex]),
            arm0_px=tuple(float(value) for value in point_pixels[arm0]),
            arm1_px=tuple(float(value) for value in point_pixels[arm1]),
            radius_px=float(spec.get("radius_px", 42.0)),
            font=measurement_font,
            fill=label_color,
            stroke_fill=stroke_color,
            stroke_width=int(measurement_stroke_width),
            line_fill=line_color,
            line_width=max(1, int(line_width) - 1),
            blocked_segments=label_blocked_segments,
            blocked_points=blocked_points,
            occupied_boxes=occupied_boxes,
            canvas_size=int(canvas_size),
        )
        if bbox is not None and token_value is not None:
            bbox_list = _bbox_to_list(bbox)
            token_bboxes[str(token_value)] = list(bbox_list)
            occupied_boxes.append(tuple(float(value) for value in bbox))

    for spec in circle_arc_specs or ():
        start_label = str(spec["start"])
        end_label = str(spec["end"])
        token_value = spec.get("token")
        bbox = _draw_circle_arc_annotation(
            draw,
            token=None if token_value is None else str(token_value),
            center_px=center_px,
            radius_px=float(radius_px),
            start_px=tuple(float(value) for value in point_pixels[start_label]),
            end_px=tuple(float(value) for value in point_pixels[end_label]),
            font=measurement_font,
            fill=label_color,
            stroke_fill=stroke_color,
            stroke_width=int(measurement_stroke_width),
            line_fill=line_color,
            line_width=max(2, int(line_width)),
            blocked_segments=label_blocked_segments,
            blocked_points=blocked_points,
            occupied_boxes=occupied_boxes,
            canvas_size=int(canvas_size),
        )
        if bbox is not None and token_value is not None:
            bbox_list = _bbox_to_list(bbox)
            token_bboxes[str(token_value)] = list(bbox_list)
            occupied_boxes.append(tuple(float(value) for value in bbox))

    point_label_bboxes: Dict[str, List[float]] = {}
    for label, point_px_list in point_pixels.items():
        px, py = float(point_px_list[0]), float(point_px_list[1])
        base_direction = (float(px - center_px[0]), float(py - center_px[1]))
        point_label_bbox = _draw_point_label(
            draw,
            label=str(label),
            point_px=(float(px), float(py)),
            base_direction=base_direction,
            font=label_font,
            fill=label_color,
            stroke_fill=stroke_color,
            stroke_width=int(label_stroke_width),
            point_label_offset_px=float(point_label_offset_px),
            blocked_segments=label_blocked_segments,
            blocked_points=blocked_points,
            occupied_boxes=occupied_boxes,
            canvas_size=int(canvas_size),
        )
        point_label_bboxes[str(label)] = _bbox_to_list(point_label_bbox)

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    scene_entities = [
        {
            "entity_id": "circle_O",
            "type": "circle",
            "center_model": [float(circle_center[0]), float(circle_center[1])],
            "radius_model": float(circle_radius),
            "center_pixel": [
                float(round(center_px[0], 2)),
                float(round(center_px[1], 2)),
            ],
            "radius_px": float(round(radius_px, 2)),
        }
    ]
    scene_entities.extend(
        {
            "entity_id": f"point_{label}",
            "type": "point",
            "label": str(label),
            "model_point": [
                float(point_model[str(label)][0]),
                float(point_model[str(label)][1]),
            ],
            "pixel_point": [
                float(round(point_pixels[str(label)][0], 2)),
                float(round(point_pixels[str(label)][1], 2)),
            ],
        }
        for label in sorted(point_model)
    )
    scene_entities.extend(
        {
            "entity_id": f"segment_{segment_id}",
            "type": "segment",
            "labels": [str(labels[0]), str(labels[1])],
            "pixel_endpoints": [
                list(point) for point in segment_pixels[str(segment_id)]
            ],
        }
        for segment_id, labels in sorted(segments.items())
    )
    scene_entities.extend(
        {
            "entity_id": f"annotation_{index}",
            "type": "measurement_annotation",
            "token": str(token),
            "bbox": list(token_bboxes[str(token)]),
        }
        for index, token in enumerate(token_bboxes)
    )

    return RenderedCircleTheoremScene(
        image=image,
        answer_value=float(theorem_trace["answer_value"]),
        support_measurement_tokens=[
            str(token) for token in support_measurement_tokens
        ],
        annotation_point_labels=[
            str(label)
            for label in dict.fromkeys(str(item) for item in annotation_point_labels)
            if str(label) in point_pixels
        ],
        token_bboxes={str(key): list(value) for key, value in token_bboxes.items()},
        point_pixels={
            str(key): [float(round(value[0], 2)), float(round(value[1], 2))]
            for key, value in point_pixels.items()
        },
        point_label_bboxes={
            str(key): list(value) for key, value in point_label_bboxes.items()
        },
        point_model={
            str(key): [float(value[0]), float(value[1])]
            for key, value in point_model.items()
        },
        segment_pixels={
            str(key): [
                [float(round(point[0], 2)), float(round(point[1], 2))]
                for point in value
            ]
            for key, value in segment_pixels.items()
        },
        circle_center_pixel=[
            float(round(center_px[0], 2)),
            float(round(center_px[1], 2)),
        ],
        circle_center_model=[float(circle_center[0]), float(circle_center[1])],
        circle_radius_model=float(circle_radius),
        circle_radius_px=float(round(radius_px, 2)),
        annotation_values={
            str(key): int(value) for key, value in annotation_values.items()
        },
        theorem_trace=dict(theorem_trace),
        scene_entities=list(scene_entities),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        shape_style=dict(shape_style.to_trace_dict()),
        render_params={
            "canvas_size": int(canvas_size),
            "line_width": int(line_width),
            "circle_line_width": int(circle_line_width),
            "point_radius_px": int(point_radius_px),
            "label_font_size_px": int(label_font_size_px),
            "measurement_font_size_px": int(measurement_font_size_px),
            "font_bold": False,
            "label_stroke_width": int(label_stroke_width),
            "measurement_stroke_width": int(measurement_stroke_width),
            "single_object_scene_rotation": scene_transform.metadata(),
            "measurement_label_offset_px": float(measurement_offset_px),
            "point_label_offset_px": float(point_label_offset_px),
            "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
            "technical_diagram_style_resolution": dict(diagram_style_meta),
        },
    )

__all__ = [
    '_model_transform',
    '_draw_centered_text_with_bbox',
    '_segment_label_center',
    '_segment_label_direction',
    '_circle_boundary_segments',
    '_draw_measurement_label',
    '_screen_angle_path',
    '_draw_angle_annotation',
    '_draw_right_angle_marker',
    '_draw_circle_arc_annotation',
    '_draw_point_label',
    '_render_base_scene',
]
