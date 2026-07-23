"""Rendering primitives for concentric-chord diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    geometry_shape_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_label,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.vector2d import unit
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import ANNOTATION_KEYS, BBox, Color, ConcentricChordDiagramSpec, Point, RenderContext, RenderedConcentricChordScene


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    random_namespace: str,
) -> tuple[RenderContext, Dict[str, Any]]:
    """Resolve deterministic style, font, canvas, and transform state."""

    rng = spawn_rng(int(instance_seed), str(random_namespace))
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_resolution = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
    )
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    accents: Tuple[Color, ...] = (
        tuple(int(value) for value in diagram_style.accent_rgb),
        tuple(int(value) for value in diagram_style.secondary_accent_rgb),
        tuple(int(value) for value in diagram_style.highlight_rgb),
        tuple(int(value) for value in diagram_style.guide_rgb),
    )
    accent_rng = spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.accent")
    accent_color = uniform_choice(accent_rng, accents)
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18)))
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    label_stroke_width = int(params.get("label_stroke_width", group_default(render_defaults, "label_stroke_width", 0)))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"geometry.{SCENE_ID}.font_family",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    ctx = RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        accent_color=accent_color,
        line_width=max(2, int(line_width)),
        font=load_font(max(12, int(font_size)), bold=False, font_family=str(font_family)),
        small_font=load_font(max(10, int(small_font_size)), bold=False, font_family=str(font_family)),
        label_stroke_width=max(0, int(label_stroke_width)),
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
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "label_stroke_width": int(ctx.label_stroke_width),
        "accent_color": list(accent_color),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_resolution),
        "font_family": font_record.to_trace(),
        "font_asset_version": font_asset_version(),
    }
    return ctx, render_meta


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point = (0.0, 0.0),
    label_center: Point | None = None,
    gap_for_label: bool = False,
    color: Color | None = None,
) -> BBox:
    draw_color = color if color is not None else ctx.label_color
    tick = 7.0
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if label_center is not None:
        center = (float(label_center[0]), float(label_center[1]))
    else:
        center = (
            (float(start[0]) + float(end[0])) / 2.0 + float(label_offset[0]),
            (float(start[1]) + float(end[1])) / 2.0 + float(label_offset[1]),
        )
    line_width = max(2, ctx.line_width - 1)
    if bool(gap_for_label) and length > 1e-9:
        ux = dx / length
        uy = dy / length
        projection = ((center[0] - float(start[0])) * ux) + ((center[1] - float(start[1])) * uy)
        label_bbox = _prospective_label_bbox(ctx, label, center)
        label_w = float(label_bbox[2]) - float(label_bbox[0])
        label_h = float(label_bbox[3]) - float(label_bbox[1])
        half_gap = max(18.0, (math.hypot(label_w, label_h) / 2.0) + 8.0)
        before = max(0.0, min(float(length), projection - half_gap))
        after = max(0.0, min(float(length), projection + half_gap))
        if before > 2.0:
            ctx.draw.line(
                [start, (float(start[0]) + ux * before, float(start[1]) + uy * before)],
                fill=draw_color,
                width=line_width,
            )
        if after < float(length) - 2.0:
            ctx.draw.line(
                [(float(start[0]) + ux * after, float(start[1]) + uy * after), end],
                fill=draw_color,
                width=line_width,
            )
    else:
        ctx.draw.line([start, end], fill=draw_color, width=line_width)
    if length > 1e-9:
        nx = -dy / length
        ny = dx / length
        for point in (start, end):
            ctx.draw.line(
                [
                    (float(point[0]) - tick * nx, float(point[1]) - tick * ny),
                    (float(point[0]) + tick * nx, float(point[1]) + tick * ny),
                ],
                fill=draw_color,
                width=line_width,
            )
    return draw_label(ctx, label, center, small=True)


def _prospective_label_bbox(ctx: RenderContext, label: str, center: Point) -> BBox:
    bbox = ctx.draw.textbbox((0, 0), str(label), font=ctx.small_font, stroke_width=ctx.label_stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    x0 = float(center[0]) - (text_w / 2.0) - 4.0
    y0 = float(center[1]) - (text_h / 2.0) - 4.0
    x1 = float(center[0]) + (text_w / 2.0) + 4.0
    y1 = float(center[1]) + (text_h / 2.0) + 4.0
    return (
        max(0.0, min(float(ctx.width), x0)),
        max(0.0, min(float(ctx.height), y0)),
        max(0.0, min(float(ctx.width), x1)),
        max(0.0, min(float(ctx.height), y1)),
    )


def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _outer_radius_label_center(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    avoid_bboxes: Tuple[BBox, ...],
) -> Point:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
    ux = dx / length
    uy = dy / length
    nx = -uy
    ny = ux
    candidates: list[tuple[float, Point]] = []
    for along in (0.48, 0.56, 0.64):
        base = (float(start[0]) + dx * along, float(start[1]) + dy * along)
        for distance in (44.0, 56.0, 68.0):
            for side in (1.0, -1.0):
                center = (base[0] + nx * distance * side, base[1] + ny * distance * side)
                bbox = _prospective_label_bbox(ctx, label, center)
                overflow = (
                    max(0.0, -float(center[0]))
                    + max(0.0, float(center[0]) - float(ctx.width))
                    + max(0.0, -float(center[1]))
                    + max(0.0, float(center[1]) - float(ctx.height))
                )
                edge_penalty = 0.0
                if bbox[0] <= 2.0 or bbox[1] <= 2.0 or bbox[2] >= float(ctx.width) - 2.0 or bbox[3] >= float(ctx.height) - 2.0:
                    edge_penalty = 5000.0
                overlap = sum(_bbox_intersection_area(bbox, other) for other in avoid_bboxes)
                preference = abs(along - 0.58) * 50.0 + abs(distance - 56.0) * 1.5
                candidates.append((overflow * 10000.0 + edge_penalty + overlap * 25.0 + preference, center))
    return min(candidates, key=lambda item: item[0])[1]


def render_concentric_chord_scene(
    ctx: RenderContext,
    spec: ConcentricChordDiagramSpec,
) -> RenderedConcentricChordScene:
    """Draw one tangent-chord construction and project its labeled points."""

    raw_center = (326.0, 292.0)
    raw_outer_px = 178.0
    raw_inner_px = raw_outer_px * float(spec.inner_radius) / float(spec.outer_radius)
    raw_half_chord_px = raw_outer_px * float(spec.half_chord) / float(spec.outer_radius)
    raw_chord_y = raw_center[1] - raw_inner_px
    raw_left = (raw_center[0] - raw_half_chord_px, raw_chord_y)
    raw_right = (raw_center[0] + raw_half_chord_px, raw_chord_y)
    raw_tangent = (raw_center[0], raw_chord_y)
    raw_dim_left = (raw_left[0], raw_chord_y - 46.0)
    raw_dim_right = (raw_right[0], raw_chord_y - 46.0)
    raw_outer_bounds = (
        (raw_center[0] - raw_outer_px, raw_center[1]),
        (raw_center[0] + raw_outer_px, raw_center[1]),
        (raw_center[0], raw_center[1] - raw_outer_px),
        (raw_center[0], raw_center[1] + raw_outer_px),
    )
    raw_points = (raw_center, raw_left, raw_right, raw_tangent, raw_dim_left, raw_dim_right, *raw_outer_bounds)
    ctx.scene_transform.resolve(raw_points)
    center, left, right, tangent, dim_left, dim_right, *_outer_bounds = ctx.scene_transform.points(raw_points)
    outer_px = raw_outer_px * float(ctx.scene_transform.transform.scale)
    inner_px = raw_inner_px * float(ctx.scene_transform.transform.scale)

    outer_box = (center[0] - outer_px, center[1] - outer_px, center[0] + outer_px, center[1] + outer_px)
    inner_box = (center[0] - inner_px, center[1] - inner_px, center[0] + inner_px, center[1] + inner_px)
    ctx.draw.ellipse(outer_box, outline=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(inner_box, outline=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([left, right], fill=ctx.accent_color, width=ctx.line_width + 1)
    ctx.draw.line([center, tangent], fill=ctx.line_color, width=max(2, ctx.line_width - 1))
    ctx.draw.line([center, right], fill=ctx.line_color, width=max(2, ctx.line_width - 1))

    marker = 16.0 * float(ctx.scene_transform.transform.scale)
    chord_unit = unit((float(right[0]) - float(tangent[0]), float(right[1]) - float(tangent[1])))
    radius_unit = unit((float(center[0]) - float(tangent[0]), float(center[1]) - float(tangent[1])))
    m0 = tangent
    m1 = (m0[0] + chord_unit[0] * marker, m0[1] + chord_unit[1] * marker)
    m2 = (m1[0] + radius_unit[0] * marker, m1[1] + radius_unit[1] * marker)
    m3 = (m0[0] + radius_unit[0] * marker, m0[1] + radius_unit[1] * marker)
    ctx.draw.line([m0, m1, m2, m3], fill=ctx.line_color, width=2)

    point_label_bboxes: list[BBox] = []
    for label, point in (("O", center), ("A", left), ("B", right), ("T", tangent)):
        px, py = float(point[0]), float(point[1])
        ctx.draw.ellipse((px - 4.0, py - 4.0, px + 4.0, py + 4.0), fill=ctx.line_color)
        offset = {"O": (0.0, 24.0), "A": (-18.0, -18.0), "B": (18.0, -18.0), "T": (20.0, 20.0)}[label]
        point_label_bboxes.append(draw_label(ctx, label, (px + offset[0], py + offset[1]), small=True))

    label_bboxes: Dict[str, BBox] = {}
    label_bboxes["inner_radius"] = _draw_dimension(
        ctx,
        center,
        tangent,
        spec.inner_radius_label,
        label_offset=(-38.0, -4.0),
    )
    right_angle_bbox = bbox_from_points(
        (tangent, (tangent[0] + marker, tangent[1]), (tangent[0] + marker, tangent[1] + marker), (tangent[0], tangent[1] + marker)),
        width=ctx.width,
        height=ctx.height,
        pad=5.0,
    )
    label_bboxes["outer_radius"] = _draw_dimension(
        ctx,
        center,
        right,
        spec.outer_radius_label,
        label_center=_outer_radius_label_center(
            ctx,
            center,
            right,
            spec.outer_radius_label,
            tuple(point_label_bboxes + [label_bboxes["inner_radius"], right_angle_bbox]),
        ),
        gap_for_label=True,
    )
    label_bboxes["chord"] = _draw_dimension(
        ctx,
        dim_left,
        dim_right,
        spec.chord_label,
        label_offset=(0.0, -18.0),
        color=ctx.accent_color,
    )
    label_bboxes["right_angle"] = right_angle_bbox

    annotation_keyed_points = {"O": center, "A": left, "B": right, "T": tangent}
    chord_bbox = bbox_from_points((left, right), width=ctx.width, height=ctx.height, pad=18.0)
    scene_entities = (
        {
            "entity_id": "outer_circle",
            "entity_type": "circle",
            "center": [round(center[0], 3), round(center[1], 3)],
            "radius_px": round(outer_px, 3),
            "radius_units": int(spec.outer_radius),
            "bbox": bbox_to_list(outer_box),
        },
        {
            "entity_id": "inner_circle",
            "entity_type": "circle",
            "center": [round(center[0], 3), round(center[1], 3)],
            "radius_px": round(inner_px, 3),
            "radius_units": int(spec.inner_radius),
            "bbox": bbox_to_list(inner_box),
        },
        {
            "entity_id": "outer_chord",
            "entity_type": "segment",
            "endpoints": [[round(left[0], 3), round(left[1], 3)], [round(right[0], 3), round(right[1], 3)]],
            "length_units": int(spec.chord_length),
            "bbox": bbox_to_list(chord_bbox),
        },
    )
    return RenderedConcentricChordScene(
        image=ctx.image,
        annotation_roles=ANNOTATION_KEYS,
        annotation_keyed_points=dict(annotation_keyed_points),
        label_bboxes=dict(label_bboxes),
        scene_entities=scene_entities,
        render_map={
            "center": [round(center[0], 3), round(center[1], 3)],
            "outer_radius_px": round(outer_px, 3),
            "inner_radius_px": round(inner_px, 3),
            "chord_endpoints": [[round(left[0], 3), round(left[1], 3)], [round(right[0], 3), round(right[1], 3)]],
            "tangent_point": [round(tangent[0], 3), round(tangent[1], 3)],
            "construction_points": {key: [round(point[0], 3), round(point[1], 3)] for key, point in annotation_keyed_points.items()},
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "coord_space": "pixel",
        },
        measurements={
            "outer_radius": int(spec.outer_radius),
            "inner_radius": int(spec.inner_radius),
            "half_chord": int(spec.half_chord),
            "chord_length": int(spec.chord_length),
            "formula_family": str(spec.formula_family),
            "unknown_measure": str(spec.unknown_measure),
            "answer_value": int(spec.answer),
        },
    )


def render_concentric_chord_with_retries(
    *,
    spec: ConcentricChordDiagramSpec,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> tuple[RenderedConcentricChordScene, Dict[str, Any]]:
    """Render a fixed diagram spec, retrying only visual layout/style failures."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            attempt_params = dict(params)
            attempt_params["_render_attempt"] = int(attempt_index)
            ctx, render_meta = create_render_context(
                instance_seed=int(instance_seed) + int(attempt_index),
                params=attempt_params,
                render_defaults=render_defaults,
                random_namespace=str(random_namespace),
            )
            rendered = render_concentric_chord_scene(ctx, spec)
            render_meta = dict(render_meta)
            render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            return rendered, render_meta
        except Exception as exc:
            last_error = exc
    raise RuntimeError("failed to render concentric chord diagram") from last_error


__all__ = [
    "create_render_context",
    "render_concentric_chord_scene",
    "render_concentric_chord_with_retries",
]
