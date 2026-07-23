"""Rendering helpers for circle-polygon-composite diagrams."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced, text_legibility_metadata_for_surfaces
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    pad_bbox,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import (
    add as _add,
    mul as _mul,
    point_to_list as _point_to_list,
    sub as _sub,
    unit as _unit,
)

from .construction import tangential_local_geometry
from .state import (
    AngleDiagramSpec,
    BBox,
    CirclePolygonRenderContext,
    Color,
    Point,
    RenderedAngleScene,
    RenderedTangentialScene,
    SCENE_ID,
    TangentialDiagramSpec,
)


def render_with_layout_retry(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    max_attempts: int,
    build_context: Callable[[int, Mapping[str, Any]], Any],
    draw_scene: Callable[[Any, int], Any],
) -> tuple[Any, Any]:
    """Retry stochastic layout without choosing task/query behavior."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_params = dict(task_params)
        attempt_params["_render_attempt"] = int(attempt)
        attempt_seed = int(instance_seed) + int(attempt)
        try:
            render_context = build_context(attempt_seed, attempt_params)
            rendered = draw_scene(render_context, attempt_seed)
            return render_context, rendered
        except Exception as exc:
            last_error = exc
    raise RuntimeError("failed to render circle-polygon-composite scene") from last_error


def create_circle_polygon_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fill_namespace: str,
    transform_namespace: str = f"{SCENE_ID}.scene_transform",
) -> CirclePolygonRenderContext:
    """Create a styled drawing context shared by this scene's renderers."""

    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 600)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=False,
        require_grid=None,
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    )
    fill_palettes: Tuple[Tuple[Color, Color], ...] = (
        ((238, 246, 255), (255, 252, 232)),
        ((241, 248, 239), (247, 243, 255)),
        ((255, 243, 234), (235, 246, 250)),
        ((248, 247, 240), (232, 242, 255)),
    )
    fill_rng = spawn_rng(int(instance_seed), str(fill_namespace))
    polygon_fill, circle_fill = uniform_choice(fill_rng, fill_palettes)
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18))
    )
    line_width = int(params.get("line_width", group_default(rendering_defaults, "line_width", 3)))
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", int(diagram_style.label_stroke_width_px)),
        )
    )
    return CirclePolygonRenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        label_backing_color=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        polygon_fill=tuple(int(value) for value in polygon_fill),
        circle_fill=tuple(int(value) for value in circle_fill),
        accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(max(12, int(font_size))),
        small_font=load_font(max(10, int(small_font_size))),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), str(transform_namespace)),
            params=params,
            render_defaults=rendering_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _rotate(point: Point, angle_radians: float) -> Point:
    x, y = float(point[0]), float(point[1])
    cos_a = math.cos(float(angle_radians))
    sin_a = math.sin(float(angle_radians))
    return ((x * cos_a) - (y * sin_a), (x * sin_a) + (y * cos_a))


def _transform(point: Point, *, angle_radians: float, scale: float, offset: Point) -> Point:
    rotated = _rotate(point, angle_radians)
    return ((rotated[0] * float(scale)) + float(offset[0]), (rotated[1] * float(scale)) + float(offset[1]))


def _transform_local(point: Point, *, scale: float, offset: Point) -> Point:
    return ((float(point[0]) * float(scale)) + float(offset[0]), float(offset[1]) - (float(point[1]) * float(scale)))


def _base_projector(*, scale: float, offset: Point) -> Callable[[Point], Point]:
    """Return the unrotated local-to-canvas projection for one fitted layout."""

    return lambda point: _transform_local(point, scale=scale, offset=offset)


def _centroid(points: Sequence[Point]) -> Point:
    return (
        sum(float(point[0]) for point in points) / float(len(points)),
        sum(float(point[1]) for point in points) / float(len(points)),
    )


def _draw_text_centered(ctx: CirclePolygonRenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    font = ctx.small_font if bool(small) else ctx.font
    bbox = ctx.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=0,
    )
    backing_bbox = (
        max(0.0, float(bbox[0]) - 5.0),
        max(0.0, float(bbox[1]) - 4.0),
        min(float(ctx.width), float(bbox[2]) + 5.0),
        min(float(ctx.height), float(bbox[3]) + 4.0),
    )
    ctx.draw.rounded_rectangle(
        backing_bbox,
        radius=4,
        fill=ctx.label_backing_color,
        outline=ctx.label_stroke_color,
        width=1,
    )
    draw_text_traced(
        ctx.draw,
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        fill=ctx.label_color,
        stroke_width=0,
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=True,
        extra_metadata=text_legibility_metadata_for_surfaces(
            fill_rgb=ctx.label_color,
            surface_rgbs=(ctx.label_backing_color,),
        ),
    )
    return pad_bbox(bbox, 4.0, width=ctx.width, height=ctx.height)


def _outside_label_point(start: Point, end: Point, centroid: Point, *, offset: float) -> Point:
    midpoint = ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
    vector = _sub(end, start)
    normal = _unit((-vector[1], vector[0]))
    if ((midpoint[0] - centroid[0]) * normal[0]) + ((midpoint[1] - centroid[1]) * normal[1]) < 0.0:
        normal = (-normal[0], -normal[1])
    return _add(midpoint, _mul(normal, float(offset)))


def _assert_bboxes_inside(bboxes: Sequence[BBox], *, width: int, height: int) -> None:
    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        if x0 <= 4.0 or y0 <= 4.0 or x1 >= float(width) - 4.0 or y1 >= float(height) - 4.0:
            raise ValueError("circle-polygon label too close to canvas edge")


def _line_rectangle_intersections(point: Point, direction: Point, bounds: tuple[float, float, float, float]) -> Tuple[Point, ...]:
    """Find the two clipped endpoints where a local tangent line crosses a box."""

    xmin, ymin, xmax, ymax = [float(value) for value in bounds]
    px, py = float(point[0]), float(point[1])
    dx, dy = float(direction[0]), float(direction[1])
    candidates: list[tuple[float, Point]] = []
    if abs(dx) > 1e-9:
        for x in (xmin, xmax):
            t = (x - px) / dx
            y = py + (t * dy)
            if ymin - 1e-7 <= y <= ymax + 1e-7:
                candidates.append((float(t), (float(x), float(y))))
    if abs(dy) > 1e-9:
        for y in (ymin, ymax):
            t = (y - py) / dy
            x = px + (t * dx)
            if xmin - 1e-7 <= x <= xmax + 1e-7:
                candidates.append((float(t), (float(x), float(y))))
    unique: list[tuple[float, Point]] = []
    for t, candidate in candidates:
        if not any(math.hypot(candidate[0] - other[1][0], candidate[1] - other[1][1]) <= 1e-6 for other in unique):
            unique.append((t, candidate))
    unique.sort(key=lambda item: item[0])
    if len(unique) < 2:
        raise ValueError("tangent line did not cross shape bounds clearly")
    return (unique[0][1], unique[-1][1])


def _draw_local_polyline(
    ctx: CirclePolygonRenderContext,
    points: Sequence[Point],
    *,
    project: Callable[[Point], Point],
    fill: Color,
    width: int,
) -> Tuple[Point, ...]:
    transformed = tuple(project(point) for point in points)
    ctx.draw.line(transformed, fill=fill, width=max(1, int(width)), joint="curve")
    return transformed


def _arc_points(*, center: Point, radius: float, start_degrees: float, end_degrees: float, steps: int = 18) -> Tuple[Point, ...]:
    return tuple(
        (
            float(center[0])
            + (
                float(radius)
                * math.cos(math.radians(start_degrees + ((end_degrees - start_degrees) * i / max(1, steps))))
            ),
            float(center[1])
            + (
                float(radius)
                * math.sin(math.radians(start_degrees + ((end_degrees - start_degrees) * i / max(1, steps))))
            ),
        )
        for i in range(int(steps) + 1)
    )


def _angle_between_degrees(vector: Point) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _draw_angle_arc_at_vertex(
    ctx: CirclePolygonRenderContext,
    *,
    vertex: Point,
    arm_a: Point,
    arm_b: Point,
    radius: float,
    project: Callable[[Point], Point],
) -> Tuple[Point, ...]:
    angle_a = _angle_between_degrees(_sub(arm_a, vertex))
    angle_b = _angle_between_degrees(_sub(arm_b, vertex))
    while angle_b - angle_a > 180.0:
        angle_b -= 360.0
    while angle_b - angle_a < -180.0:
        angle_b += 360.0
    if abs(angle_b - angle_a) > 120.0:
        angle_b = angle_a + (360.0 - abs(angle_b - angle_a)) * (1.0 if angle_b < angle_a else -1.0)
    points = _arc_points(center=vertex, radius=float(radius), start_degrees=angle_a, end_degrees=angle_b, steps=16)
    return _draw_local_polyline(
        ctx,
        points,
        project=project,
        fill=ctx.accent_color,
        width=max(2, ctx.line_width - 1),
    )


def _draw_semicircle(
    ctx: CirclePolygonRenderContext,
    *,
    center: Point,
    radius: float,
    project: Callable[[Point], Point],
) -> Tuple[Point, ...]:
    points = _arc_points(center=center, radius=float(radius), start_degrees=0.0, end_degrees=180.0, steps=48)
    return _draw_local_polyline(
        ctx,
        points,
        project=project,
        fill=ctx.secondary_color,
        width=max(2, ctx.line_width - 1),
    )


def render_angle_scene(
    ctx: CirclePolygonRenderContext,
    spec: AngleDiagramSpec,
    *,
    instance_seed: int,
    render_namespace: str,
) -> RenderedAngleScene:
    """Render the tangent-angle diagram after task code has bound the target angle."""

    rng = spawn_rng(int(instance_seed), str(render_namespace))
    theta = math.radians(float(spec.angle_degrees))
    sign = int(spec.side_sign)
    is_semicircle = str(spec.construction_kind) == "semicircle"
    if is_semicircle:
        bounds = (-1.25, 0.0, 1.25, 1.25)
        corners = {
            "A": (-1.25, 1.25),
            "B": (1.25, 1.25),
            "C": (1.25, 0.0),
            "D": (-1.25, 0.0),
        }
        scale = min((ctx.width - 180.0) / 2.5, (ctx.height - 150.0) / 1.35) * float(rng.uniform(0.78, 0.84))
        offset = (
            (ctx.width / 2.0) + float(rng.uniform(-30.0, 30.0)),
            (ctx.height * 0.70) + float(rng.uniform(-18.0, 18.0)),
        )
    else:
        bounds = (-1.0, -1.0, 1.0, 1.0)
        corners = {
            "A": (-1.0, 1.0),
            "B": (1.0, 1.0),
            "C": (1.0, -1.0),
            "D": (-1.0, -1.0),
        }
        scale = min((ctx.width - 180.0) / 2.05, (ctx.height - 150.0) / 2.05) * float(rng.uniform(0.86, 0.94))
        offset = (
            (ctx.width / 2.0) + float(rng.uniform(-28.0, 28.0)),
            (ctx.height / 2.0) + float(rng.uniform(-20.0, 20.0)),
        )

    tangent_point = (float(sign) * math.sin(theta), math.cos(theta))
    tangent_direction = (-float(sign) * math.cos(theta), math.sin(theta))
    tangent_endpoints = _line_rectangle_intersections(tangent_point, tangent_direction, bounds)
    known_angle_vertex = max(tangent_endpoints, key=lambda point: point[1])
    known_angle_reference_point = (known_angle_vertex[0] + (float(sign) * 0.36), known_angle_vertex[1])
    target_angle_vertex = (0.0, 0.0)
    target_reference_point = (0.0, 1.0)
    circle_center = (0.0, 0.0)

    corner_label_locals: Dict[str, Point] = {}
    for key, local_point in corners.items():
        direction = _unit(local_point)
        corner_label_locals[str(key)] = _add(local_point, _mul(direction, 0.13))
    center_label_local = (0.12, -0.12)
    tangent_label_local = _add(tangent_point, _mul(_unit(tangent_point), 0.15))
    known_label_direction = _unit(
        _add(
            _unit(_sub(known_angle_reference_point, known_angle_vertex)),
            _unit(_sub(tangent_point, known_angle_vertex)),
        )
    )
    known_label_local = _add(known_angle_vertex, _mul(known_label_direction, 0.30))
    target_label_local = _add(
        target_angle_vertex,
        _mul(_unit(_add(_sub(target_reference_point, target_angle_vertex), _sub(tangent_point, target_angle_vertex))), 0.43),
    )

    base_project = _base_projector(scale=scale, offset=offset)
    circle_fit_points = (
        _arc_points(center=(0.0, 0.0), radius=1.0, start_degrees=0.0, end_degrees=180.0, steps=24)
        if is_semicircle
        else _arc_points(center=(0.0, 0.0), radius=1.0, start_degrees=0.0, end_degrees=360.0, steps=48)
    )
    fit_local_points: Tuple[Point, ...] = (
        *corners.values(),
        *tangent_endpoints,
        tangent_point,
        circle_center,
        known_angle_vertex,
        known_angle_reference_point,
        target_reference_point,
        *circle_fit_points,
    )
    ctx.scene_transform.resolve(tuple(base_project(point) for point in fit_local_points))

    def project(point: Point) -> Point:
        return ctx.scene_transform.point(base_project(point))

    corner_px = {key: project(value) for key, value in corners.items()}
    tangent_point_px = project(tangent_point)
    circle_center_px = project(circle_center)
    known_angle_vertex_px = project(known_angle_vertex)
    known_angle_reference_px = project(known_angle_reference_point)
    target_reference_px = project(target_reference_point)

    polygon = (corner_px["A"], corner_px["B"], corner_px["C"], corner_px["D"])
    _assert_bboxes_inside(
        (bbox_from_points(polygon, width=ctx.width, height=ctx.height, pad=4.0),),
        width=ctx.width,
        height=ctx.height,
    )
    ctx.draw.polygon(polygon, fill=ctx.polygon_fill)
    ctx.draw.line([*polygon, polygon[0]], fill=ctx.line_color, width=ctx.line_width, joint="curve")
    if is_semicircle:
        _draw_semicircle(ctx, center=(0.0, 0.0), radius=1.0, project=project)
        diameter = (
            project((-1.0, 0.0)),
            project((1.0, 0.0)),
        )
        ctx.draw.line(diameter, fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
    else:
        radius_px = float(scale) * float(ctx.scene_transform.transform.scale)
        circle_bbox = (
            circle_center_px[0] - radius_px,
            circle_center_px[1] - radius_px,
            circle_center_px[0] + radius_px,
            circle_center_px[1] + radius_px,
        )
        ctx.draw.ellipse(circle_bbox, fill=ctx.circle_fill, outline=ctx.secondary_color, width=max(2, ctx.line_width - 1))

    _draw_local_polyline(ctx, tangent_endpoints, project=project, fill=ctx.line_color, width=ctx.line_width)
    _draw_local_polyline(ctx, (circle_center, tangent_point), project=project, fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
    _draw_local_polyline(ctx, (circle_center, target_reference_point), project=project, fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
    target_arc = _draw_angle_arc_at_vertex(
        ctx,
        vertex=target_angle_vertex,
        arm_a=target_reference_point,
        arm_b=tangent_point,
        radius=0.28,
        project=project,
    )
    known_arc = _draw_angle_arc_at_vertex(
        ctx,
        vertex=known_angle_vertex,
        arm_a=known_angle_reference_point,
        arm_b=tangent_point,
        radius=0.20,
        project=project,
    )

    dot_radius = max(3, int(ctx.line_width + 1))
    for point in (circle_center_px, tangent_point_px):
        ctx.draw.ellipse(
            (point[0] - dot_radius, point[1] - dot_radius, point[0] + dot_radius, point[1] + dot_radius),
            fill=ctx.accent_color,
            outline=ctx.line_color,
            width=1,
        )

    label_bboxes: Dict[str, BBox] = {}
    for key in corner_px:
        label_bboxes[f"corner_{key}_label"] = _draw_text_centered(
            ctx,
            str(key),
            project(corner_label_locals[str(key)]),
            small=True,
        )
    label_bboxes["center_label"] = _draw_text_centered(ctx, "O", project(center_label_local), small=True)
    label_bboxes["tangent_label"] = _draw_text_centered(
        ctx,
        "T",
        project(tangent_label_local),
        small=True,
    )
    degree = "\N{DEGREE SIGN}"
    label_bboxes["known_angle_label"] = _draw_text_centered(
        ctx,
        f"{int(spec.angle_degrees)}{degree}",
        project(known_label_local),
        small=True,
    )
    label_bboxes["target_angle_label"] = _draw_text_centered(
        ctx,
        "?",
        project(target_label_local),
        small=True,
    )
    _assert_bboxes_inside(label_bboxes.values(), width=ctx.width, height=ctx.height)

    annotation = {
        "A": corner_px["A"],
        "B": corner_px["B"],
        "C": corner_px["C"],
        "D": corner_px["D"],
        "O": circle_center_px,
        "T": tangent_point_px,
    }
    render_map = {
        "coord_space": "pixel",
        "construction_kind": str(spec.construction_kind),
        "angle_value_degrees": int(spec.angle_degrees),
        "side_sign": int(spec.side_sign),
        "shape_corners": {key: _point_to_list(value) for key, value in corner_px.items()},
        "circle_center": _point_to_list(circle_center_px),
        "tangent_point": _point_to_list(tangent_point_px),
        "known_angle_vertex": _point_to_list(known_angle_vertex_px),
        "known_angle_reference_point": _point_to_list(known_angle_reference_px),
        "target_angle_vertex": _point_to_list(circle_center_px),
        "target_reference_point": _point_to_list(target_reference_px),
        "tangent_segment": [_point_to_list(project(point)) for point in tangent_endpoints],
        "known_angle_arc_bbox": bbox_to_list(bbox_from_points(known_arc, width=ctx.width, height=ctx.height, pad=2.0)),
        "target_angle_arc_bbox": bbox_to_list(bbox_from_points(target_arc, width=ctx.width, height=ctx.height, pad=2.0)),
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "scale_px_per_unit": round(float(scale), 3),
        "offset": [round(float(offset[0]), 3), round(float(offset[1]), 3)],
    }
    return RenderedAngleScene(
        image=ctx.image,
        annotation_keyed_points={key: tuple(value) for key, value in annotation.items()},
        annotation_roles=tuple(spec.annotation_roles),
        label_bboxes=dict(label_bboxes),
        render_map=dict(render_map),
    )


def render_tangential_scene(
    ctx: CirclePolygonRenderContext,
    spec: TangentialDiagramSpec,
    *,
    instance_seed: int,
    render_namespace: str,
) -> RenderedTangentialScene:
    """Render the tangential quadrilateral after task code has hidden target sides."""

    rng = spawn_rng(int(instance_seed), str(render_namespace))
    local_vertices, local_tangencies, local_radius = tangential_local_geometry(spec.vertex_tangents)
    local_all_points = [
        *local_vertices.values(),
        *local_tangencies.values(),
        (local_radius, local_radius),
        (-local_radius, local_radius),
        (local_radius, -local_radius),
        (-local_radius, -local_radius),
    ]
    base_rotation = float(rng.choice((0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)))
    angle = base_rotation + math.radians(float(rng.uniform(-10.0, 10.0)))
    rotated = [_rotate(point, angle) for point in local_all_points]
    min_x = min(point[0] for point in rotated)
    max_x = max(point[0] for point in rotated)
    min_y = min(point[1] for point in rotated)
    max_y = max(point[1] for point in rotated)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((ctx.width - 180.0) / span_x, (ctx.height - 150.0) / span_y)
    scale *= float(rng.uniform(0.88, 0.96))
    center_after_scale = ((min_x + max_x) * scale / 2.0, (min_y + max_y) * scale / 2.0)
    target_center = (
        (ctx.width / 2.0) + float(rng.uniform(-28.0, 28.0)),
        (ctx.height / 2.0) + float(rng.uniform(-20.0, 20.0)),
    )
    offset = (target_center[0] - center_after_scale[0], target_center[1] - center_after_scale[1])

    vertices = {
        key: _transform(point, angle_radians=angle, scale=scale, offset=offset)
        for key, point in local_vertices.items()
    }
    tangencies = {
        key: _transform(point, angle_radians=angle, scale=scale, offset=offset)
        for key, point in local_tangencies.items()
    }
    center = _transform((0.0, 0.0), angle_radians=angle, scale=scale, offset=offset)
    radius_px = float(local_radius) * float(scale)
    ctx.scene_transform.resolve((*vertices.values(), *tangencies.values(), center))
    vertices = ctx.scene_transform.keyed_points(vertices)
    tangencies = ctx.scene_transform.keyed_points(tangencies)
    center = ctx.scene_transform.point(center)
    radius_px *= float(ctx.scene_transform.transform.scale)

    polygon = (vertices["A"], vertices["B"], vertices["C"], vertices["D"])
    polygon_bbox = bbox_from_points(polygon, width=ctx.width, height=ctx.height, pad=2.0)
    _assert_bboxes_inside((polygon_bbox,), width=ctx.width, height=ctx.height)
    circle_bbox = (
        center[0] - radius_px,
        center[1] - radius_px,
        center[0] + radius_px,
        center[1] + radius_px,
    )
    _assert_bboxes_inside((circle_bbox,), width=ctx.width, height=ctx.height)

    ctx.draw.polygon(polygon, fill=ctx.polygon_fill)
    ctx.draw.line([*polygon, polygon[0]], fill=ctx.line_color, width=ctx.line_width, joint="curve")
    ctx.draw.ellipse(circle_bbox, fill=ctx.circle_fill, outline=ctx.secondary_color, width=max(2, ctx.line_width - 1))
    for point in tangencies.values():
        x, y = float(point[0]), float(point[1])
        dot_radius = max(3, int(ctx.line_width + 1))
        ctx.draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=ctx.accent_color,
            outline=ctx.line_color,
            width=1,
        )

    label_bboxes: Dict[str, BBox] = {}
    label_offset = float(max(28, ctx.line_width * 8))
    poly_center = _centroid(polygon)
    unknown_sides = set(str(side) for side in spec.unknown_sides)
    for side, start_key, end_key in (
        ("AB", "A", "B"),
        ("BC", "B", "C"),
        ("CD", "C", "D"),
        ("DA", "D", "A"),
    ):
        text = f"{side} ?" if side in unknown_sides else f"{side}={int(spec.side_lengths[side])}"
        label_center = _outside_label_point(
            vertices[start_key],
            vertices[end_key],
            poly_center,
            offset=label_offset,
        )
        label_bboxes[f"{side}_label"] = _draw_text_centered(ctx, text, label_center, small=True)

    vertex_label_offset = float(max(18, ctx.line_width * 5))
    for vertex_key, point in vertices.items():
        direction = _unit(_sub(point, poly_center))
        label_center = _add(point, _mul(direction, vertex_label_offset))
        label_bboxes[f"{vertex_key}_label"] = _draw_text_centered(ctx, str(vertex_key), label_center, small=True)
        x, y = float(point[0]), float(point[1])
        dot_radius = max(3, int(ctx.line_width + 1))
        ctx.draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=ctx.line_color)
    _assert_bboxes_inside(label_bboxes.values(), width=ctx.width, height=ctx.height)

    annotation = {
        "A": vertices["A"],
        "B": vertices["B"],
        "C": vertices["C"],
        "D": vertices["D"],
    }
    render_map = {
        "coord_space": "pixel",
        "vertices": {key: _point_to_list(value) for key, value in vertices.items()},
        "tangency_points": {key: _point_to_list(value) for key, value in tangencies.items()},
        "incircle_center": _point_to_list(center),
        "incircle_radius_px": round(float(radius_px), 3),
        "polygon_bbox": bbox_to_list(polygon_bbox),
        "circle_bbox": bbox_to_list(circle_bbox),
        "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
        "rotation_degrees": round(math.degrees(float(angle)), 3),
        "scale_px_per_unit": round(float(scale), 3),
    }
    return RenderedTangentialScene(
        image=ctx.image,
        annotation_keyed_points={key: tuple(value) for key, value in annotation.items()},
        annotation_roles=tuple(spec.annotation_roles),
        vertices=dict(vertices),
        tangency_points=dict(tangencies),
        label_bboxes=dict(label_bboxes),
        render_map=dict(render_map),
    )


__all__ = [
    "create_circle_polygon_render_context",
    "render_angle_scene",
    "render_tangential_scene",
]
