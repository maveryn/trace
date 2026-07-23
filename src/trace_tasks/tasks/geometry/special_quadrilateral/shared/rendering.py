"""Rendering primitives for special-quadrilateral theorem diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, pad_bbox
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import add_scaled, mid, point_to_list, sub, unit
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import BBox, Point, QuadrilateralCase, RenderContext, RenderedSpecialQuadrilateralScene, SCENE_ID, SpecialQuadrilateralProblem

RENDER_PARALLELOGRAM_OPPOSITE_ANGLES = "parallelogram_opposite_angles"
RENDER_PARALLELOGRAM_CONSECUTIVE_ANGLES = "parallelogram_consecutive_angles"
RENDER_RHOMBUS_HALF_ANGLE_EXPRESSION = "rhombus_half_angle_expression"
RENDER_KITE_OPPOSITE_ANGLES = "kite_opposite_angles"
RENDER_PARALLELOGRAM_OPPOSITE_SIDES = "parallelogram_opposite_sides"
RENDER_RHOMBUS_ALL_SIDES = "rhombus_all_sides"
RENDER_KITE_ADJACENT_SIDES = "kite_adjacent_sides"
RENDER_PARALLELOGRAM_BISECTED_DIAGONAL = "parallelogram_bisected_diagonal"


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RenderContext:
    """Create one styled drawing context with scene-level rotation support."""

    width = int(params.get("canvas_width", render_defaults.get("canvas_width", 760)))
    height = int(params.get("canvas_height", render_defaults.get("canvas_height", 560)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=width,
        canvas_height=height,
        allow_dark=False,
        require_grid=False,
    )
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = int(params.get("line_width", render_defaults.get("line_width", 3)))
    font_size = int(params.get("label_font_size", render_defaults.get("label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", render_defaults.get("small_label_font_size", 18)))
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            render_defaults.get("label_stroke_width", int(diagram_style.label_stroke_width_px)),
        )
    )
    return RenderContext(
        image=image,
        draw=draw,
        width=width,
        height=height,
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        secondary_color=tuple(int(value) for value in diagram_style.secondary_stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
        muted_color=tuple(int(value) for value in diagram_style.grid_major_rgb),
        fill_color=tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
        line_width=max(2, line_width),
        label_stroke_width=max(0, int(label_stroke_width)),
        font=load_font(font_size),
        small_font=load_font(small_font_size),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), f"{SCENE_ID}.scene_transform"),
            params=params,
            render_defaults=render_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _offset_from_segment(a: Point, b: Point, distance: float) -> Point:
    tangent = unit(sub(b, a))
    return (-tangent[1] * float(distance), tangent[0] * float(distance))


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


def _draw_vertex_labels(ctx: RenderContext, vertices: Mapping[str, Point]) -> dict[str, BBox]:
    center = (
        sum(point[0] for point in vertices.values()) / float(len(vertices)),
        sum(point[1] for point in vertices.values()) / float(len(vertices)),
    )
    label_bboxes: dict[str, BBox] = {}
    for label, point in vertices.items():
        direction = unit(sub(point, center))
        label_center = add_scaled(point, direction, 26.0)
        label_bboxes[str(label)] = _draw_text_centered(ctx, str(label), label_center, small=True)
    return label_bboxes


def _draw_angle_marker(
    ctx: RenderContext,
    *,
    vertex: Point,
    ray_a: Point,
    ray_b: Point,
    label: str,
    radius: float = 34.0,
) -> tuple[BBox, BBox, Point]:
    vector_a = unit(sub(ray_a, vertex))
    vector_b = unit(sub(ray_b, vertex))
    angle_a = math.atan2(vector_a[1], vector_a[0])
    angle_b = math.atan2(vector_b[1], vector_b[0])
    delta = ((angle_b - angle_a + math.pi) % (2.0 * math.pi)) - math.pi
    if abs(delta) > math.pi * 0.92:
        delta = -math.copysign((2.0 * math.pi) - abs(delta), delta)
    steps = max(8, int(abs(delta) / (math.pi / 20.0)))
    arc_points: list[Point] = []
    for step in range(steps + 1):
        theta = angle_a + delta * (float(step) / float(steps))
        arc_points.append((float(vertex[0]) + radius * math.cos(theta), float(vertex[1]) + radius * math.sin(theta)))
    ctx.draw.line(arc_points, fill=ctx.accent_color, width=max(2, int(ctx.line_width) - 1), joint="curve")
    bisector = unit((vector_a[0] + vector_b[0], vector_a[1] + vector_b[1]))
    if math.hypot(bisector[0], bisector[1]) <= 1e-6:
        bisector = unit(sub((ctx.width / 2.0, ctx.height / 2.0), vertex))
    label_center = add_scaled(vertex, bisector, radius + 24.0)
    label_bbox = _draw_text_centered(ctx, str(label), label_center, small=True)
    arc_bbox = bbox_from_points(arc_points, width=ctx.width, height=ctx.height, pad=4.0)
    annotation_point = add_scaled(vertex, bisector, radius * 0.45)
    return arc_bbox, label_bbox, annotation_point


def _draw_segment_label(ctx: RenderContext, *, a: Point, b: Point, text: str, offset: float) -> BBox:
    label_center = add_scaled(mid(a, b), _offset_from_segment(a, b, offset), 1.0)
    return _draw_text_centered(ctx, str(text), label_center, small=True)


def _draw_tick(ctx: RenderContext, a: Point, b: Point, *, offset: float = 0.0, count: int = 1) -> BBox:
    center = mid(a, b)
    tangent = unit(sub(b, a))
    normal = (-tangent[1], tangent[0])
    tick_points: list[Point] = []
    spacing = 10.0
    for index in range(int(count)):
        shift = (float(index) - (float(count) - 1.0) / 2.0) * spacing
        tick_center = add_scaled(add_scaled(center, tangent, shift), normal, offset)
        p0 = add_scaled(tick_center, normal, -9.0)
        p1 = add_scaled(tick_center, normal, 9.0)
        ctx.draw.line((p0, p1), fill=ctx.accent_color, width=max(2, int(ctx.line_width) - 1))
        tick_points.extend([p0, p1])
    return bbox_from_points(tick_points, width=ctx.width, height=ctx.height, pad=3.0)


def _base_vertices(shape_kind: str, *, width: int, height: int, instance_seed: int) -> dict[str, Point]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{shape_kind}.layout")
    jitter_x = rng.uniform(-24.0, 24.0)
    jitter_y = rng.uniform(-18.0, 18.0)
    scale = rng.uniform(0.92, 1.06)
    center = (float(width) / 2.0 + jitter_x, float(height) / 2.0 + jitter_y)

    if shape_kind == "rhombus":
        offsets = {
            "A": (0.0, -150.0),
            "B": (190.0, 0.0),
            "C": (0.0, 150.0),
            "D": (-190.0, 0.0),
        }
    elif shape_kind == "kite":
        offsets = {
            "A": (0.0, -170.0),
            "B": (180.0, 0.0),
            "C": (0.0, 160.0),
            "D": (-145.0, 0.0),
        }
    elif shape_kind == "parallelogram":
        offsets = {
            "A": (-190.0, 105.0),
            "B": (170.0, 105.0),
            "C": (245.0, -105.0),
            "D": (-115.0, -105.0),
        }
    else:
        raise ValueError(f"unknown special quadrilateral shape: {shape_kind}")

    return {label: (center[0] + dx * scale, center[1] + dy * scale) for label, (dx, dy) in offsets.items()}


def _transformed_base_vertices(ctx: RenderContext, shape_kind: str, *, instance_seed: int) -> dict[str, Point]:
    vertices = _base_vertices(shape_kind, width=ctx.width, height=ctx.height, instance_seed=instance_seed)
    ctx.scene_transform.resolve(tuple(vertices.values()))
    return ctx.scene_transform.keyed_points(vertices)


def _draw_base_shape(ctx: RenderContext, vertices: Mapping[str, Point], *, case: QuadrilateralCase) -> dict[str, BBox]:
    polygon = [vertices[label] for label in ("A", "B", "C", "D")]
    ctx.draw.polygon(polygon, fill=ctx.fill_color)
    ctx.draw.line(polygon + [polygon[0]], fill=ctx.line_color, width=ctx.line_width, joint="curve")
    bboxes: dict[str, BBox] = {
        "outline": bbox_from_points(polygon, width=ctx.width, height=ctx.height, pad=ctx.line_width + 2)
    }
    if case.shape_kind == "rhombus":
        for side in ("AB", "BC", "CD", "DA"):
            bboxes[f"tick_{side}"] = _draw_tick(ctx, vertices[side[0]], vertices[side[1]], count=1)
    elif case.shape_kind == "kite":
        bboxes["tick_AB"] = _draw_tick(ctx, vertices["A"], vertices["B"], count=1)
        bboxes["tick_AD"] = _draw_tick(ctx, vertices["A"], vertices["D"], count=1)
        bboxes["tick_BC"] = _draw_tick(ctx, vertices["B"], vertices["C"], count=2)
        bboxes["tick_CD"] = _draw_tick(ctx, vertices["C"], vertices["D"], count=2)
    else:
        bboxes["tick_AB"] = _draw_tick(ctx, vertices["A"], vertices["B"], count=1)
        bboxes["tick_CD"] = _draw_tick(ctx, vertices["C"], vertices["D"], count=1)
        bboxes["tick_BC"] = _draw_tick(ctx, vertices["B"], vertices["C"], count=2)
        bboxes["tick_DA"] = _draw_tick(ctx, vertices["D"], vertices["A"], count=2)
    return bboxes


def _draw_algebraic_angle_relation(ctx: RenderContext, case: QuadrilateralCase, vertices: Mapping[str, Point]) -> tuple[dict[str, BBox], dict[str, BBox], dict[str, Point]]:
    """Draw one algebraic angle-expression construction."""

    construction_bboxes: dict[str, BBox] = {}
    readout_bboxes: dict[str, BBox] = {}
    extra_points: dict[str, Point] = {}
    if case.render_kind == RENDER_PARALLELOGRAM_OPPOSITE_ANGLES:
        _, readout_bboxes["support_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["A"], ray_a=vertices["B"], ray_b=vertices["D"], label=case.support_label)
        _, readout_bboxes["target_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["C"], ray_a=vertices["D"], ray_b=vertices["B"], label=case.target_label)
    elif case.render_kind == RENDER_PARALLELOGRAM_CONSECUTIVE_ANGLES:
        _, readout_bboxes["support_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["A"], ray_a=vertices["B"], ray_b=vertices["D"], label=case.support_label)
        _, readout_bboxes["target_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["B"], ray_a=vertices["C"], ray_b=vertices["A"], label=case.target_label)
    elif case.render_kind == RENDER_RHOMBUS_HALF_ANGLE_EXPRESSION:
        ctx.draw.line((vertices["B"], vertices["D"]), fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
        o = mid(vertices["B"], vertices["D"])
        construction_bboxes["diagonal_BD"] = bbox_from_points((vertices["B"], vertices["D"]), width=ctx.width, height=ctx.height, pad=4.0)
        readout_bboxes["point_label_O"] = _draw_text_centered(ctx, "O", add_scaled(o, (0.0, 1.0), 22.0), small=True)
        _, readout_bboxes["support_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["B"], ray_a=vertices["A"], ray_b=o, label=case.support_label)
        _, readout_bboxes["target_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["B"], ray_a=o, ray_b=vertices["C"], label=case.target_label)
        extra_points["O"] = o
    elif case.render_kind == RENDER_KITE_OPPOSITE_ANGLES:
        _, readout_bboxes["target_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["B"], ray_a=vertices["A"], ray_b=vertices["C"], label=case.target_label)
        _, readout_bboxes["support_angle_label"], _ = _draw_angle_marker(ctx, vertex=vertices["D"], ray_a=vertices["C"], ray_b=vertices["A"], label=case.support_label)
    else:
        raise ValueError(f"unsupported special quadrilateral algebraic construction: {case.render_kind}")
    return construction_bboxes, readout_bboxes, extra_points


def _draw_segment_relation(ctx: RenderContext, case: QuadrilateralCase, vertices: Mapping[str, Point]) -> tuple[dict[str, BBox], dict[str, BBox], dict[str, Point]]:
    """Draw one algebraic side or diagonal segment construction."""

    construction_bboxes: dict[str, BBox] = {}
    readout_bboxes: dict[str, BBox] = {}
    extra_points: dict[str, Point] = {}
    if case.render_kind == RENDER_PARALLELOGRAM_OPPOSITE_SIDES:
        readout_bboxes["support_segment_label"] = _draw_segment_label(ctx, a=vertices["D"], b=vertices["A"], text=case.support_label, offset=-34.0)
        readout_bboxes["target_segment_label"] = _draw_segment_label(ctx, a=vertices["B"], b=vertices["C"], text=case.target_label, offset=34.0)
    elif case.render_kind == RENDER_RHOMBUS_ALL_SIDES:
        readout_bboxes["support_segment_label"] = _draw_segment_label(ctx, a=vertices["A"], b=vertices["B"], text=case.support_label, offset=-34.0)
        readout_bboxes["target_segment_label"] = _draw_segment_label(ctx, a=vertices["C"], b=vertices["D"], text=case.target_label, offset=34.0)
    elif case.render_kind == RENDER_KITE_ADJACENT_SIDES:
        readout_bboxes["support_segment_label"] = _draw_segment_label(ctx, a=vertices["A"], b=vertices["B"], text=case.support_label, offset=-34.0)
        readout_bboxes["target_segment_label"] = _draw_segment_label(ctx, a=vertices["A"], b=vertices["D"], text=case.target_label, offset=34.0)
    elif case.render_kind == RENDER_PARALLELOGRAM_BISECTED_DIAGONAL:
        ctx.draw.line((vertices["A"], vertices["C"]), fill=ctx.secondary_color, width=max(2, ctx.line_width - 1))
        o = mid(vertices["A"], vertices["C"])
        construction_bboxes["diagonal_AC"] = bbox_from_points((vertices["A"], vertices["C"]), width=ctx.width, height=ctx.height, pad=4.0)
        point_radius = max(3, int(ctx.line_width))
        ctx.draw.ellipse((o[0] - point_radius, o[1] - point_radius, o[0] + point_radius, o[1] + point_radius), fill=ctx.accent_color)
        readout_bboxes["support_segment_label"] = _draw_segment_label(ctx, a=vertices["A"], b=o, text=case.support_label, offset=-30.0)
        readout_bboxes["target_segment_label"] = _draw_segment_label(ctx, a=o, b=vertices["C"], text=case.target_label, offset=30.0)
        readout_bboxes["intersection_label"] = _draw_text_centered(ctx, "O", add_scaled(o, (0.0, 1.0), 24.0), small=True)
        extra_points["O"] = o
    else:
        raise ValueError(f"unsupported special quadrilateral segment construction: {case.render_kind}")
    return construction_bboxes, readout_bboxes, extra_points


def draw_special_quadrilateral_scene(
    *,
    problem: SpecialQuadrilateralProblem,
    ctx: RenderContext,
) -> RenderedSpecialQuadrilateralScene:
    """Render the task-selected construction after layout transform is resolved."""

    case = problem.case
    vertices = _transformed_base_vertices(ctx, case.shape_kind, instance_seed=problem.layout_seed)
    construction_bboxes = _draw_base_shape(ctx, vertices, case=case)
    if case.render_kind in {
        RENDER_PARALLELOGRAM_OPPOSITE_ANGLES,
        RENDER_PARALLELOGRAM_CONSECUTIVE_ANGLES,
        RENDER_RHOMBUS_HALF_ANGLE_EXPRESSION,
        RENDER_KITE_OPPOSITE_ANGLES,
    }:
        extra_construction, readout_bboxes, extra_points = _draw_algebraic_angle_relation(ctx, case, vertices)
    else:
        extra_construction, readout_bboxes, extra_points = _draw_segment_relation(ctx, case, vertices)
    construction_bboxes.update(extra_construction)
    point_label_bboxes = _draw_vertex_labels(ctx, vertices)
    annotation = {**dict(vertices), **dict(extra_points)}
    render_map = {
        "vertices": {key: point_to_list(point) for key, point in vertices.items()},
        "readout_bboxes": geometry_json_ready(readout_bboxes),
        "construction_bboxes": geometry_json_ready(construction_bboxes),
        "point_label_bboxes": geometry_json_ready(point_label_bboxes),
    }
    return RenderedSpecialQuadrilateralScene(
        image=ctx.image,
        vertices=dict(vertices),
        annotation_points=dict(annotation),
        point_label_bboxes=dict(point_label_bboxes),
        readout_bboxes=dict(readout_bboxes),
        construction_bboxes=dict(construction_bboxes),
        render_map=dict(render_map),
    )


__all__ = [
    "RENDER_KITE_ADJACENT_SIDES",
    "RENDER_KITE_OPPOSITE_ANGLES",
    "RENDER_PARALLELOGRAM_BISECTED_DIAGONAL",
    "RENDER_PARALLELOGRAM_CONSECUTIVE_ANGLES",
    "RENDER_PARALLELOGRAM_OPPOSITE_ANGLES",
    "RENDER_PARALLELOGRAM_OPPOSITE_SIDES",
    "RENDER_RHOMBUS_ALL_SIDES",
    "RENDER_RHOMBUS_HALF_ANGLE_EXPRESSION",
    "create_render_context",
    "draw_special_quadrilateral_scene",
]
