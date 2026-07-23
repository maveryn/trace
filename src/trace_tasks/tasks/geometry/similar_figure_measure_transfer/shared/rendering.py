"""Rendering primitives for the similar-figure measurement scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, pad_bbox
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import add_scaled, mid, point_to_list, sub, unit
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import (
    BBox,
    Color,
    FigureGeometry,
    Point,
    RenderedSimilarScene,
    SCENE_ID,
    Side,
    SimilarEquationCase,
    SimilarMeasureCase,
)


@dataclass
class RenderContext:
    """Canvas, style, and transform state shared by one scene render."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    source_fill: Color
    target_fill: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    scene_transform: LazySceneTransform


def build_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> RenderContext:
    """Build one styled canvas for a similar-figure diagram."""

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
        accent_color=tuple(int(value) for value in diagram_style.accent_rgb),
        source_fill=tuple(int(value) for value in diagram_style.panel_alt_fill_rgb),
        target_fill=tuple(int(value) for value in diagram_style.option_fill_rgb),
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


def render_measure_scene(case: SimilarMeasureCase, *, context: RenderContext, instance_seed: int) -> RenderedSimilarScene:
    """Render a numeric similar-figure measurement case after final transform."""

    geometry = _figure_geometry(
        str(case.shape_kind),
        str(case.layout_kind),
        scale_factor=float(case.scale_factor),
        context=context,
        instance_seed=int(instance_seed),
    )
    boxes = _draw_figures(context, geometry, layout_kind=str(case.layout_kind))
    point_label_bboxes = _draw_vertex_labels(context, geometry)
    readouts: dict[str, BBox] = {}
    constructions: dict[str, BBox] = dict(boxes)

    _draw_side_measurements(
        context,
        geometry,
        readouts=readouts,
        constructions=constructions,
        layout_kind=str(case.layout_kind),
        target_side=case.target_side,
        support_side=case.support_side,
        source_target_label=(None if case.source_target_side_value is None else str(case.source_target_side_value)),
        target_target_label=(
            None
            if case.target_target_side_value is None
            else ("?" if case.relation != "scale_factor_from_corresponding_side_lengths" else str(case.target_target_side_value))
        ),
        support_source_label=(None if case.support_source_side_value is None else str(case.support_source_side_value)),
        support_target_label=(None if case.support_target_side_value is None else str(case.support_target_side_value)),
    )
    _draw_metric_captions(context, geometry, case, readouts, point_label_bboxes)
    return _rendered_scene(context, geometry, point_label_bboxes, readouts, constructions)


def render_equation_scene(case: SimilarEquationCase, *, context: RenderContext, instance_seed: int) -> RenderedSimilarScene:
    """Render an expression-labeled similar-figure case after final transform."""

    geometry = _figure_geometry(
        str(case.shape_kind),
        "rotated_pair",
        scale_factor=float(case.scale_factor),
        context=context,
        instance_seed=int(instance_seed),
    )
    boxes = _draw_figures(context, geometry, layout_kind="rotated_pair")
    point_label_bboxes = _draw_vertex_labels(context, geometry)
    readouts: dict[str, BBox] = {}
    constructions: dict[str, BBox] = dict(boxes)
    support_side = (1, 2) if str(case.shape_kind) == "triangle" else (2, 3)
    _draw_side_measurements(
        context,
        geometry,
        readouts=readouts,
        constructions=constructions,
        layout_kind="rotated_pair",
        target_side=(0, 1),
        support_side=support_side,
        source_target_label=str(case.source_target_label),
        target_target_label=str(case.target_target_label),
        support_source_label=str(case.support_source_label),
        support_target_label=str(case.support_target_label),
    )
    return _rendered_scene(context, geometry, point_label_bboxes, readouts, constructions)


def _shape_template(shape_kind: str) -> tuple[Point, ...]:
    if shape_kind == "triangle":
        return ((0.0, -1.05), (1.16, 0.82), (-1.08, 0.82))
    if shape_kind == "quadrilateral":
        return ((-1.1, -0.82), (0.86, -0.98), (1.18, 0.66), (-0.86, 0.96))
    if shape_kind == "pentagon":
        return ((0.0, -1.12), (1.08, -0.34), (0.76, 0.96), (-0.46, 1.08), (-1.14, 0.12))
    raise ValueError(f"unknown shape_kind={shape_kind!r}")


def _labels_for_shape(shape_kind: str, *, target: bool = False) -> tuple[str, ...]:
    base = tuple(chr(ord("A") + index) for index in range(len(_shape_template(str(shape_kind)))))
    if target:
        return tuple(f"{label}'" for label in base)
    return base


def _transform_template(points: Sequence[Point], *, center: Point, scale: float, rotation_degrees: float) -> tuple[Point, ...]:
    theta = math.radians(float(rotation_degrees))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    transformed: list[Point] = []
    for x, y in points:
        sx = float(x) * float(scale)
        sy = float(y) * float(scale)
        transformed.append((center[0] + sx * cos_t - sy * sin_t, center[1] + sx * sin_t + sy * cos_t))
    return tuple(transformed)


def _figure_geometry(
    shape_kind: str,
    layout_kind: str,
    *,
    scale_factor: float,
    context: RenderContext,
    instance_seed: int,
) -> FigureGeometry:
    """Place both figures and apply optional whole-diagram rotation once."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.{shape_kind}.{layout_kind}.layout")
    template = _shape_template(str(shape_kind))
    visual_scale = min(1.92, 1.12 + 0.14 * float(scale_factor))
    source_scale = rng.uniform(56.0, 70.0) if str(layout_kind) != "nested" else rng.uniform(47.0, 58.0)
    target_scale = source_scale * visual_scale
    if str(layout_kind) == "nested":
        center = (context.width * 0.50 + rng.uniform(-12.0, 12.0), context.height * 0.51 + rng.uniform(-10.0, 12.0))
        source_center = target_center = center
        source_rotation = target_rotation = rng.uniform(-7.0, 7.0)
    else:
        source_center = (context.width * 0.32 + rng.uniform(-16.0, 12.0), context.height * 0.53 + rng.uniform(-18.0, 18.0))
        target_center = (context.width * 0.69 + rng.uniform(-12.0, 16.0), context.height * 0.52 + rng.uniform(-18.0, 18.0))
        source_rotation = rng.uniform(-8.0, 8.0)
        target_rotation = rng.uniform(13.0, 24.0) if str(layout_kind) == "rotated_pair" else source_rotation + rng.uniform(-2.0, 2.0)
    source_points = _transform_template(template, center=source_center, scale=source_scale, rotation_degrees=source_rotation)
    target_points = _transform_template(template, center=target_center, scale=target_scale, rotation_degrees=target_rotation)
    all_points = tuple((*source_points, *target_points))
    context.scene_transform.resolve(all_points)
    return FigureGeometry(
        source_vertices=context.scene_transform.transform.points(source_points),
        target_vertices=context.scene_transform.transform.points(target_points),
        source_labels=_labels_for_shape(str(shape_kind)),
        target_labels=_labels_for_shape(str(shape_kind), target=True),
        transform_metadata=context.scene_transform.metadata(),
    )


def _draw_figures(context: RenderContext, geometry: FigureGeometry, *, layout_kind: str) -> dict[str, BBox]:
    if str(layout_kind) == "nested":
        target = _draw_polygon(context, geometry.target_vertices, fill=context.target_fill, outline=context.secondary_color)
        source = _draw_polygon(context, geometry.source_vertices, fill=context.source_fill, outline=context.line_color)
    else:
        source = _draw_polygon(context, geometry.source_vertices, fill=context.source_fill, outline=context.line_color)
        target = _draw_polygon(context, geometry.target_vertices, fill=context.target_fill, outline=context.secondary_color)
    return {"source_figure": source, "target_figure": target}


def _draw_polygon(context: RenderContext, points: Sequence[Point], *, fill: Color, outline: Color) -> BBox:
    polygon = [(float(x), float(y)) for x, y in points]
    context.draw.polygon(polygon, fill=fill)
    context.draw.line(polygon + [polygon[0]], fill=outline, width=context.line_width, joint="curve")
    return bbox_from_points(polygon, width=context.width, height=context.height, pad=context.line_width + 2)


def _draw_vertex_labels(context: RenderContext, geometry: FigureGeometry) -> dict[str, BBox]:
    boxes: dict[str, BBox] = {}
    for prefix, points, labels, offset in (
        ("source", geometry.source_vertices, geometry.source_labels, 24.0),
        ("target", geometry.target_vertices, geometry.target_labels, 28.0),
    ):
        center = (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))
        for label, point in zip(labels, points):
            label_center = add_scaled(point, unit(sub(point, center)), offset)
            boxes[f"{prefix}_{label}"] = _draw_text_centered(context, str(label), label_center, small=True)
    return boxes


def _draw_side_measurements(
    context: RenderContext,
    geometry: FigureGeometry,
    *,
    readouts: dict[str, BBox],
    constructions: dict[str, BBox],
    layout_kind: str,
    target_side: Side,
    support_side: Side,
    source_target_label: str | None,
    target_target_label: str | None,
    support_source_label: str | None,
    support_target_label: str | None,
) -> None:
    """Draw the two marked side pairs when their labels are present."""

    if source_target_label is not None and target_target_label is not None:
        if str(layout_kind) == "nested":
            readouts["source_target_side_label"] = _draw_side_label_relative(
                context,
                geometry.source_vertices,
                target_side,
                source_target_label,
                distance=24.0,
                outward=False,
            )
            readouts["target_target_side_label"] = _draw_side_label_relative(
                context,
                geometry.target_vertices,
                target_side,
                target_target_label,
                distance=30.0,
                outward=True,
            )
        else:
            readouts["source_target_side_label"] = _draw_side_label(context, geometry.source_vertices, target_side, source_target_label, offset=-30.0)
            readouts["target_target_side_label"] = _draw_side_label(context, geometry.target_vertices, target_side, target_target_label, offset=34.0)
        constructions["source_target_side_tick"] = _draw_tick(context, geometry.source_vertices[target_side[0]], geometry.source_vertices[target_side[1]], count=1)
        constructions["target_target_side_tick"] = _draw_tick(context, geometry.target_vertices[target_side[0]], geometry.target_vertices[target_side[1]], count=1)
    if support_source_label is not None and support_target_label is not None:
        if str(layout_kind) == "nested":
            readouts["support_source_side_label"] = _draw_side_label_relative(
                context,
                geometry.source_vertices,
                support_side,
                support_source_label,
                distance=24.0,
                outward=False,
            )
            readouts["support_target_side_label"] = _draw_side_label_relative(
                context,
                geometry.target_vertices,
                support_side,
                support_target_label,
                distance=30.0,
                outward=True,
            )
        else:
            readouts["support_source_side_label"] = _draw_side_label(context, geometry.source_vertices, support_side, support_source_label, offset=32.0)
            readouts["support_target_side_label"] = _draw_side_label(context, geometry.target_vertices, support_side, support_target_label, offset=-36.0)
        constructions["support_source_side_tick"] = _draw_tick(context, geometry.source_vertices[support_side[0]], geometry.source_vertices[support_side[1]], count=2)
        constructions["support_target_side_tick"] = _draw_tick(context, geometry.target_vertices[support_side[0]], geometry.target_vertices[support_side[1]], count=2)


def _draw_metric_captions(
    context: RenderContext,
    geometry: FigureGeometry,
    case: SimilarMeasureCase,
    readouts: dict[str, BBox],
    point_label_bboxes: Mapping[str, BBox],
) -> None:
    """Place area/perimeter readouts outside figures without changing task semantics."""

    occupied_labels = tuple(point_label_bboxes.values())
    if case.source_perimeter is not None and case.target_perimeter is not None:
        readouts["source_perimeter_label"] = _draw_caption_near_polygon(
            context,
            f"perimeter = {case.source_perimeter}",
            geometry.source_vertices,
            occupied=(*occupied_labels, *readouts.values()),
            preferred=("below", "above", "left", "right"),
        )
        readouts["target_perimeter_label"] = _draw_caption_near_polygon(
            context,
            f"perimeter = {case.target_perimeter}",
            geometry.target_vertices,
            occupied=(*occupied_labels, *readouts.values()),
            preferred=("below", "above", "right", "left"),
        )
    if case.source_area is not None and case.target_area is not None:
        if str(case.construction_family) == "area_ratio_label":
            readouts["source_area_label"] = _draw_caption_near_polygon(
                context,
                f"area = {case.source_area}",
                geometry.source_vertices,
                occupied=(*occupied_labels, *readouts.values()),
                preferred=("below", "above", "left", "right"),
            )
            readouts["area_ratio_label"] = _draw_caption_near_polygon(
                context,
                f"area ratio = {case.area_ratio_label}",
                (*geometry.source_vertices, *geometry.target_vertices),
                occupied=(*occupied_labels, *readouts.values()),
                preferred=("above", "below", "right", "left"),
            )
        elif str(case.layout_kind) == "nested":
            readouts["source_area_label"] = _draw_caption_near_polygon(
                context,
                f"inner area = {case.source_area}",
                geometry.target_vertices,
                occupied=(*occupied_labels, *readouts.values()),
                preferred=("below", "above", "right", "left"),
            )
            readouts["target_area_label"] = _draw_caption_near_polygon(
                context,
                f"outer area = {case.target_area}",
                geometry.target_vertices,
                occupied=(*occupied_labels, *readouts.values()),
                preferred=("above", "below", "right", "left"),
            )
        else:
            readouts["source_area_label"] = _draw_caption_near_polygon(
                context,
                f"area = {case.source_area}",
                geometry.source_vertices,
                occupied=(*occupied_labels, *readouts.values()),
                preferred=("below", "above", "left", "right"),
            )
            readouts["target_area_label"] = _draw_caption_near_polygon(
                context,
                f"area = {case.target_area}",
                geometry.target_vertices,
                occupied=(*occupied_labels, *readouts.values()),
                preferred=("below", "above", "right", "left"),
            )


def _draw_text_centered(context: RenderContext, text: str, center: Point, *, small: bool = True) -> BBox:
    font = context.small_font if small else context.font
    draw_text_traced(
        context.draw,
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        fill=context.label_color,
        stroke_width=max(0, int(context.label_stroke_width)),
        stroke_fill=context.label_stroke_color,
        role="readout",
        required=False,
    )
    bbox = context.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=font,
        stroke_width=max(0, int(context.label_stroke_width)),
    )
    return pad_bbox(bbox, 3.0, width=context.width, height=context.height)


def _draw_side_label(context: RenderContext, points: Sequence[Point], side: Side, text: str, *, offset: float) -> BBox:
    a = points[int(side[0])]
    b = points[int(side[1])]
    label_center = add_scaled(mid(a, b), _offset_from_segment(a, b, offset), 1.0)
    return _draw_text_centered(context, str(text), label_center, small=True)


def _draw_side_label_relative(
    context: RenderContext,
    points: Sequence[Point],
    side: Side,
    text: str,
    *,
    distance: float,
    outward: bool,
) -> BBox:
    a = points[int(side[0])]
    b = points[int(side[1])]
    side_mid = mid(a, b)
    center = _polygon_center(points)
    tangent = unit(sub(b, a))
    normal = (-tangent[1], tangent[0])
    option_a = add_scaled(side_mid, normal, float(distance))
    option_b = add_scaled(side_mid, normal, -float(distance))
    distance_a = math.hypot(option_a[0] - center[0], option_a[1] - center[1])
    distance_b = math.hypot(option_b[0] - center[0], option_b[1] - center[1])
    if bool(outward):
        label_center = option_a if distance_a >= distance_b else option_b
    else:
        label_center = option_a if distance_a < distance_b else option_b
    return _draw_text_centered(context, str(text), label_center, small=True)


def _draw_tick(context: RenderContext, a: Point, b: Point, *, count: int = 1) -> BBox:
    center = mid(a, b)
    tangent = unit(sub(b, a))
    normal = (-tangent[1], tangent[0])
    tick_points: list[Point] = []
    for index in range(int(count)):
        shift = (float(index) - (float(count) - 1.0) / 2.0) * 9.0
        tick_center = add_scaled(center, tangent, shift)
        p0 = add_scaled(tick_center, normal, -9.0)
        p1 = add_scaled(tick_center, normal, 9.0)
        context.draw.line((p0, p1), fill=context.accent_color, width=max(2, context.line_width - 1))
        tick_points.extend([p0, p1])
    return bbox_from_points(tick_points, width=context.width, height=context.height, pad=4.0)


def _rendered_scene(
    context: RenderContext,
    geometry: FigureGeometry,
    point_label_bboxes: Mapping[str, BBox],
    readouts: Mapping[str, BBox],
    constructions: Mapping[str, BBox],
) -> RenderedSimilarScene:
    render_map = {
        "source_vertices": {label: point_to_list(point) for label, point in zip(geometry.source_labels, geometry.source_vertices)},
        "target_vertices": {label: point_to_list(point) for label, point in zip(geometry.target_labels, geometry.target_vertices)},
        "point_label_bboxes": geometry_json_ready(point_label_bboxes),
        "readout_bboxes": geometry_json_ready(readouts),
        "construction_bboxes": geometry_json_ready(constructions),
        "image_id": "img0",
    }
    return RenderedSimilarScene(
        image=context.image,
        figure_geometry=geometry,
        point_label_bboxes=dict(point_label_bboxes),
        readout_bboxes=dict(readouts),
        construction_bboxes=dict(constructions),
        style_metadata=dict(context.diagram_style_meta),
        background_metadata=dict(context.background_meta),
        render_map=render_map,
    )


def _offset_from_segment(a: Point, b: Point, distance: float) -> Point:
    tangent = unit(sub(b, a))
    return (-tangent[1] * float(distance), tangent[0] * float(distance))


def _polygon_center(points: Sequence[Point]) -> Point:
    return (sum(point[0] for point in points) / float(len(points)), sum(point[1] for point in points) / float(len(points)))


def _draw_caption_near_polygon(
    context: RenderContext,
    text: str,
    points: Sequence[Point],
    *,
    occupied: Sequence[BBox],
    preferred: Sequence[str],
) -> BBox:
    """Draw a readout outside the visible polygon bbox with stable fallbacks."""

    polygon_bbox = bbox_from_points(points, width=context.width, height=context.height, pad=8.0)
    blocked = tuple(occupied)
    first_inside_candidate: Point | None = None
    for direction in preferred:
        candidate = _caption_candidate_center(context, str(text), polygon_bbox, direction=str(direction))
        bbox = _text_bbox_at(context, str(text), candidate)
        if not _bbox_inside_canvas(context, bbox):
            continue
        if first_inside_candidate is None:
            first_inside_candidate = candidate
        if not _bbox_overlaps_any(bbox, (*blocked, polygon_bbox)):
            return _draw_text_centered(context, str(text), candidate, small=True)
    if first_inside_candidate is not None:
        return _draw_text_centered(context, str(text), first_inside_candidate, small=True)
    center = _clamp_caption_center(context, str(text), _caption_candidate_center(context, str(text), polygon_bbox, direction=str(preferred[0])))
    return _draw_text_centered(context, str(text), center, small=True)


def _caption_candidate_center(context: RenderContext, text: str, polygon_bbox: BBox, *, direction: str) -> Point:
    font = context.small_font
    text_bbox = context.draw.textbbox((0.0, 0.0), str(text), anchor="mm", font=font, stroke_width=max(0, int(context.label_stroke_width)))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    x0, y0, x1, y1 = (float(value) for value in polygon_bbox)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    gap = 14.0
    if str(direction) == "above":
        return (cx, y0 - gap - text_height / 2.0)
    if str(direction) == "below":
        return (cx, y1 + gap + text_height / 2.0)
    if str(direction) == "left":
        return (x0 - gap - text_width / 2.0, cy)
    if str(direction) == "right":
        return (x1 + gap + text_width / 2.0, cy)
    raise ValueError(f"unknown caption direction: {direction!r}")


def _text_bbox_at(context: RenderContext, text: str, center: Point) -> BBox:
    bbox = context.draw.textbbox(
        (float(center[0]), float(center[1])),
        str(text),
        anchor="mm",
        font=context.small_font,
        stroke_width=max(0, int(context.label_stroke_width)),
    )
    return (
        float(bbox[0]) - 3.0,
        float(bbox[1]) - 3.0,
        float(bbox[2]) + 3.0,
        float(bbox[3]) + 3.0,
    )


def _bbox_inside_canvas(context: RenderContext, bbox: BBox, *, margin: float = 32.0) -> bool:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return x0 >= margin and y0 >= margin and x1 <= float(context.width) - margin and y1 <= float(context.height) - margin


def _bbox_overlaps_any(bbox: BBox, others: Sequence[BBox]) -> bool:
    return any(_bbox_overlaps(bbox, other) for other in others)


def _bbox_overlaps(a: BBox, b: BBox) -> bool:
    ax0, ay0, ax1, ay1 = (float(value) for value in a)
    bx0, by0, bx1, by1 = (float(value) for value in b)
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def _clamp_caption_center(context: RenderContext, text: str, center: Point) -> Point:
    bbox = _text_bbox_at(context, str(text), center)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    cx, cy = float(center[0]), float(center[1])
    margin = 10.0
    if x0 < margin:
        cx += margin - x0
    if x1 > float(context.width) - margin:
        cx -= x1 - (float(context.width) - margin)
    if y0 < margin:
        cy += margin - y0
    if y1 > float(context.height) - margin:
        cy -= y1 - (float(context.height) - margin)
    return (cx, cy)
