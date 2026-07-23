"""Rendering primitives for Pythagorean square-dissection diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

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
    draw_readout_centered,
    draw_right_angle_marker,
)
from trace_tasks.tasks.geometry.shared.readout_fill_style import resolve_readout_fill_style
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import BBox, Color, Point, PythagoreanDissectionPlan, RenderContext, RenderedPythagoreanDissectionScene

ORIENTATIONS: tuple[tuple[str, int, int], ...] = (
    ("up_right", 1, -1),
    ("up_left", -1, -1),
    ("down_right", 1, 1),
    ("down_left", -1, 1),
)


def _polygon_center(points: Sequence[Point]) -> Point:
    return (
        sum(float(point[0]) for point in points) / float(len(points)),
        sum(float(point[1]) for point in points) / float(len(points)),
    )


def _select_orientation(*, params: Mapping[str, Any], instance_seed: int) -> tuple[str, int, int]:
    forced = params.get("orientation")
    if forced is not None:
        orientation = str(forced)
        for item in ORIENTATIONS:
            if item[0] == orientation:
                return item
        raise ValueError(f"unsupported pythagorean dissection orientation: {orientation}")
    orientation_rng = spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.orientation")
    return uniform_choice(orientation_rng, ORIENTATIONS)


def make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> tuple[RenderContext, dict[str, Any]]:
    """Resolve deterministic theme, font, canvas, and orientation state."""

    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 560)))
    background, background_meta, diagram_style, diagram_style_resolution = (
        prepare_geometry_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=int(width),
            canvas_height=int(height),
            require_grid=False,
            allow_dark=True,
            style_profile="analytical_diagram",
        )
    )
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    palettes: tuple[tuple[Color, Color, Color], ...] = (
        (
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.option_fill_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.muted_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
        ),
    )
    readout_style = resolve_readout_fill_style(
        instance_seed=int(instance_seed),
        namespace=f"geometry.{SCENE_ID}.readout_fill",
        diagram_style=diagram_style,
        background_meta=background_meta,
        candidate_palettes=palettes,
        params=params,
    )
    leg_fill, other_leg_fill, central_fill = readout_style.fill_colors
    orientation_key, orientation_sign_x, orientation_sign_y = _select_orientation(
        params=params,
        instance_seed=int(instance_seed),
    )
    font_family = str(
        params.get(
            "readout_font_family",
            sample_font_family(
                role="readout",
                instance_seed=int(instance_seed),
                namespace=f"geometry.{SCENE_ID}.font_family",
                params=params,
            ),
        )
    )
    font_record = get_font_family_record(str(font_family))
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get(
            "small_label_font_size",
            group_default(rendering_defaults, "small_label_font_size", 18),
        )
    )
    line_width = int(params.get("line_width", group_default(rendering_defaults, "line_width", 3)))
    label_stroke_width = int(
        params.get(
            "label_stroke_width",
            group_default(rendering_defaults, "label_stroke_width", diagram_style.label_stroke_width_px),
        )
    )
    image = background.convert("RGB")
    ctx = RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        secondary_color=tuple(int(v) for v in diagram_style.secondary_stroke_rgb),
        label_color=readout_style.label_color,
        label_stroke_color=readout_style.label_stroke_color,
        leg_fill_color=leg_fill,
        other_leg_fill_color=other_leg_fill,
        central_fill_color=central_fill,
        line_width=max(2, int(line_width)),
        label_stroke_width=max(0, min(1, int(label_stroke_width))),
        font=load_font(max(12, int(font_size)), bold=False, font_family=str(font_family)),
        small_font=load_font(max(10, int(small_font_size)), bold=False, font_family=str(font_family)),
        readout_text_metadata=dict(readout_style.metadata["readout_text_style"]),
        orientation_key=str(orientation_key),
        orientation_sign_x=int(orientation_sign_x),
        orientation_sign_y=int(orientation_sign_y),
        diagram_style_meta=geometry_diagram_style_metadata(diagram_style),
        background_meta=dict(background_meta),
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.scene_transform"),
            params=params,
            render_defaults=rendering_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )
    style_meta = dict(ctx.diagram_style_meta)
    style_meta["technical_diagram_style_resolution"] = dict(diagram_style_resolution)
    style_meta["readout_font_family"] = font_record.to_trace()
    return ctx, {
        "canvas": {"width": int(width), "height": int(height)},
        "style": {
            "technical_diagram": dict(style_meta),
            "background": dict(background_meta),
        },
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "label_stroke_width": int(ctx.label_stroke_width),
        "fill_colors": {
            "leg_a": list(leg_fill),
            "leg_b": list(other_leg_fill),
            "central_square": list(central_fill),
        },
        "readout_fill_style": dict(readout_style.metadata),
        "font_asset_version": font_asset_version(),
        "orientation": str(orientation_key),
        "orientation_sign_x": int(orientation_sign_x),
        "orientation_sign_y": int(orientation_sign_y),
    }


def _draw_label(ctx: RenderContext, text: str, center: Point) -> BBox:
    return draw_readout_centered(
        ctx,
        str(text),
        center,
        small=True,
        backed=False,
        extra_metadata=ctx.readout_text_metadata,
    )


def _offset_from_center(point: Point, center: Point, distance: float) -> Point:
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    length = max(1.0, math.hypot(dx, dy))
    return (
        float(point[0]) + (dx / length) * float(distance),
        float(point[1]) + (dy / length) * float(distance),
    )


def _draw_vertex_labels(
    ctx: RenderContext,
    *,
    prefix: str,
    vertices: Mapping[str, Point],
    center: Point,
    offset: float,
) -> dict[str, BBox]:
    return {
        f"{prefix}_{label}": _draw_label(
            ctx,
            label,
            _offset_from_center(point, center, float(offset)),
        )
        for label, point in vertices.items()
    }


def _display_segment_labels(ctx: RenderContext, plan: PythagoreanDissectionPlan) -> dict[str, str]:
    leg_a = int(plan.leg_a)
    leg_b = int(plan.leg_b)
    outer_side = int(plan.outer_square_side)
    if ctx.orientation_key == "up_right":
        return {"outer": f"AB={outer_side}", "leg_a": f"AE={leg_a}", "leg_b": f"EB={leg_b}"}
    if ctx.orientation_key == "up_left":
        return {"outer": f"BC={outer_side}", "leg_a": f"BF={leg_a}", "leg_b": f"FC={leg_b}"}
    if ctx.orientation_key == "down_right":
        return {"outer": f"CD={outer_side}", "leg_a": f"CG={leg_a}", "leg_b": f"GD={leg_b}"}
    return {"outer": f"DA={outer_side}", "leg_a": f"DH={leg_a}", "leg_b": f"HA={leg_b}"}


def render_pythagorean_dissection_scene(
    ctx: RenderContext,
    plan: PythagoreanDissectionPlan,
) -> RenderedPythagoreanDissectionScene:
    """Draw the attached-square dissection after resolving final scene transform."""

    outer_side_units = int(plan.outer_square_side)
    square_size = min(float(ctx.width) * 0.62, float(ctx.height) * 0.72)
    left = (float(ctx.width) - square_size) / 2.0
    top = (float(ctx.height) - square_size) / 2.0
    right = left + square_size
    bottom = top + square_size
    scale = square_size / float(outer_side_units)
    short_px = float(plan.leg_a) * scale
    long_px = float(plan.leg_b) * scale

    outer_top_left = (left, top)
    outer_top_right = (right, top)
    outer_bottom_right = (right, bottom)
    outer_bottom_left = (left, bottom)
    center_top = (left + short_px, top)
    center_right = (right, top + short_px)
    center_bottom = (left + long_px, bottom)
    center_left = (left, top + long_px)
    outer_square = (outer_top_left, outer_top_right, outer_bottom_right, outer_bottom_left)
    central_square = (center_top, center_right, center_bottom, center_left)
    corner_triangles = (
        (outer_top_left, center_top, center_left),
        (outer_top_right, center_right, center_top),
        (outer_bottom_right, center_bottom, center_right),
        (outer_bottom_left, center_left, center_bottom),
    )
    marker = 16.0
    angle_marks_raw = (
        ((left + marker, top), (left + marker, top + marker), (left, top + marker)),
        ((right - marker, top), (right - marker, top + marker), (right, top + marker)),
        ((right - marker, bottom), (right - marker, bottom - marker), (right, bottom - marker)),
        ((left + marker, bottom), (left + marker, bottom - marker), (left, bottom - marker)),
    )
    if ctx.orientation_key == "up_right":
        outer_label_center_raw = ((left + right) / 2.0, top - 32.0)
        leg_label_center_raw = ((left + center_top[0]) / 2.0, top + 28.0)
        other_leg_label_center_raw = ((center_top[0] + right) / 2.0, top + 28.0)
    elif ctx.orientation_key == "up_left":
        outer_label_center_raw = (left - 56.0, (top + bottom) / 2.0)
        leg_label_center_raw = (right - 34.0, (top + center_right[1]) / 2.0)
        other_leg_label_center_raw = (right - 42.0, (center_right[1] + bottom) / 2.0)
    elif ctx.orientation_key == "down_right":
        outer_label_center_raw = ((left + right) / 2.0, bottom + 30.0)
        leg_label_center_raw = ((center_bottom[0] + right) / 2.0, bottom - 28.0)
        other_leg_label_center_raw = ((left + center_bottom[0]) / 2.0, bottom - 28.0)
    else:
        outer_label_center_raw = (right + 56.0, (top + bottom) / 2.0)
        leg_label_center_raw = (left + 34.0, (center_left[1] + bottom) / 2.0)
        other_leg_label_center_raw = (left + 42.0, (top + center_left[1]) / 2.0)

    ctx.scene_transform.resolve(
        (
            *outer_square,
            *central_square,
            *(point for triangle in corner_triangles for point in triangle),
            outer_label_center_raw,
            leg_label_center_raw,
            other_leg_label_center_raw,
            *(point for mark in angle_marks_raw for point in mark),
        )
    )
    (
        outer_top_left,
        outer_top_right,
        outer_bottom_right,
        outer_bottom_left,
        center_top,
        center_right,
        center_bottom,
        center_left,
    ) = ctx.scene_transform.points(
        (
            outer_top_left,
            outer_top_right,
            outer_bottom_right,
            outer_bottom_left,
            center_top,
            center_right,
            center_bottom,
            center_left,
        )
    )
    outer_square = (outer_top_left, outer_top_right, outer_bottom_right, outer_bottom_left)
    central_square = (center_top, center_right, center_bottom, center_left)
    corner_triangles = tuple(ctx.scene_transform.points(triangle) for triangle in corner_triangles)
    angle_marks = tuple(ctx.scene_transform.points(mark) for mark in angle_marks_raw)
    outer_label_center = ctx.scene_transform.point(outer_label_center_raw)
    leg_label_center = ctx.scene_transform.point(leg_label_center_raw)
    other_leg_label_center = ctx.scene_transform.point(other_leg_label_center_raw)

    triangle_fills = (
        ctx.leg_fill_color,
        ctx.other_leg_fill_color,
        ctx.leg_fill_color,
        ctx.other_leg_fill_color,
    )
    for triangle, fill in zip(corner_triangles, triangle_fills):
        ctx.draw.polygon(triangle, fill=fill)
    ctx.draw.polygon(central_square, fill=ctx.central_fill_color)
    ctx.draw.line(list(outer_square) + [outer_square[0]], fill=ctx.line_color, width=ctx.line_width, joint="curve")
    for triangle in corner_triangles:
        ctx.draw.line(
            list(triangle) + [triangle[0]],
            fill=ctx.line_color,
            width=max(2, ctx.line_width - 1),
            joint="curve",
        )
    ctx.draw.line(list(central_square) + [central_square[0]], fill=ctx.line_color, width=ctx.line_width, joint="curve")

    for mark in angle_marks:
        draw_right_angle_marker(
            ctx,
            mark[0],
            arm_a=(mark[1][0] - mark[0][0], mark[1][1] - mark[0][1]),
            arm_b=(mark[2][0] - mark[0][0], mark[2][1] - mark[0][1]),
            side_px=10.0,
        )

    segment_labels = _display_segment_labels(ctx, plan)
    label_bboxes: dict[str, BBox] = {
        "outer_square_side": _draw_label(ctx, segment_labels["outer"], outer_label_center),
        "leg_a_label": _draw_label(ctx, segment_labels["leg_a"], leg_label_center),
        "leg_b_label": _draw_label(ctx, segment_labels["leg_b"], other_leg_label_center),
        "central_square_target": _draw_label(ctx, "Area=?", _polygon_center(central_square)),
    }
    outer_vertices = {
        "A": outer_top_left,
        "B": outer_top_right,
        "C": outer_bottom_right,
        "D": outer_bottom_left,
    }
    central_vertices = {
        "E": center_top,
        "F": center_right,
        "G": center_bottom,
        "H": center_left,
    }
    label_bboxes.update(
        _draw_vertex_labels(
            ctx,
            prefix="outer_vertex",
            vertices=outer_vertices,
            center=_polygon_center(outer_square),
            offset=20.0,
        )
    )
    label_bboxes.update(
        _draw_vertex_labels(
            ctx,
            prefix="central_vertex",
            vertices=central_vertices,
            center=_polygon_center(central_square),
            offset=18.0,
        )
    )
    central_square_bbox = bbox_from_points(central_square, width=ctx.width, height=ctx.height, pad=ctx.line_width + 2)
    annotation_roles = ("E", "F", "G", "H")
    annotation_keyed_points = {label: central_vertices[label] for label in annotation_roles}
    square_entities = (
        {
            "entity_id": "outer_square",
            "entity_type": "square",
            "side_units": int(outer_side_units),
            "area_units": int(outer_side_units * outer_side_units),
            "vertices": {label: [round(x, 3), round(y, 3)] for label, (x, y) in outer_vertices.items()},
            "bbox": bbox_to_list(bbox_from_points(outer_square, width=ctx.width, height=ctx.height)),
        },
        {
            "entity_id": "repeated_corner_triangle",
            "entity_type": "right_triangle",
            "multiplicity": 4,
            "leg_a_units": int(plan.leg_a),
            "leg_b_units": int(plan.leg_b),
            "area_units_each": float(plan.leg_a * plan.leg_b / 2.0),
            "vertices_by_instance": [
                [[round(x, 3), round(y, 3)] for x, y in triangle]
                for triangle in corner_triangles
            ],
            "bbox": bbox_to_list(
                bbox_from_points(
                    [point for triangle in corner_triangles for point in triangle],
                    width=ctx.width,
                    height=ctx.height,
                )
            ),
        },
        {
            "entity_id": "central_square",
            "entity_type": "square",
            "side_units": round(float(plan.central_square_side), 3),
            "area_units": int(plan.answer),
            "vertices": {label: [round(x, 3), round(y, 3)] for label, (x, y) in central_vertices.items()},
            "bbox": bbox_to_list(central_square_bbox),
        },
    )
    witness = {
        **dict(plan.witness),
        "orientation": str(ctx.orientation_key),
        "annotation_roles": list(annotation_roles),
        "displayed_segment_labels": dict(segment_labels),
    }
    return RenderedPythagoreanDissectionScene(
        image=ctx.image,
        annotation_roles=tuple(annotation_roles),
        annotation_keyed_points=dict(annotation_keyed_points),
        label_bboxes=dict(label_bboxes),
        scene_entities=square_entities,
        render_map={
            "coord_space": "pixel",
            "square_vertices": {
                "outer_square": {label: [round(x, 3), round(y, 3)] for label, (x, y) in outer_vertices.items()},
                "central_square": {
                    label: [round(x, 3), round(y, 3)]
                    for label, (x, y) in central_vertices.items()
                },
            },
            "corner_triangle_vertices": [
                [[round(x, 3), round(y, 3)] for x, y in triangle]
                for triangle in corner_triangles
            ],
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "annotation_points": {
                key: [round(float(point[0]), 3), round(float(point[1]), 3)]
                for key, point in annotation_keyed_points.items()
            },
            "orientation": str(ctx.orientation_key),
            "displayed_segment_labels": dict(segment_labels),
        },
        witness=witness,
    )


__all__ = ["ORIENTATIONS", "make_render_context", "render_pythagorean_dissection_scene"]
