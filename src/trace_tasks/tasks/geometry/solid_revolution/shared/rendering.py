"""Rendering primitives for solid-revolution diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_shape_style_from_diagram_style,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_dimension_line,
    draw_readout_centered,
    draw_right_angle_marker,
    pad_bbox,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_role_trace, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import SCENE_ID
from .measurements import format_measure
from .state import BBox, Color, Point, RenderContext, RenderedSolidRevolutionScene, SolidRevolutionProblem


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    random_namespace: str,
) -> RenderContext:
    """Resolve deterministic style, font, canvas, and color state."""

    rng = spawn_rng(int(instance_seed), str(random_namespace))
    width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 580)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(width),
        canvas_height=int(height),
        allow_dark=True,
    )
    shape_style = geometry_shape_style_from_diagram_style(diagram_style)
    palettes: tuple[tuple[Color, Color, Color, Color], ...] = (
        (
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.accent_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.muted_fill_rgb),
            tuple(int(v) for v in diagram_style.secondary_accent_rgb),
            tuple(int(v) for v in diagram_style.secondary_stroke_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.highlight_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
    )
    palette_rng = spawn_rng(int(instance_seed), f"{random_namespace}.palette")
    palette_index = int(palette_rng.randrange(len(palettes)))
    fill_color, solid_fill_color, accent_color, muted_color = palettes[palette_index]
    font_size = int(params.get("label_font_size", group_default(rendering_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(rendering_defaults, "small_label_font_size", 18))
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{random_namespace}.font",
        params=params,
    )
    with temporary_default_font_family(str(font_family)):
        font = load_font(max(12, int(font_size)), bold=False)
        small_font = load_font(max(10, int(small_font_size)), bold=False)
    return RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        fill_color=fill_color,
        solid_fill_color=solid_fill_color,
        accent_color=accent_color,
        muted_color=muted_color,
        line_width=max(2, int(params.get("line_width", group_default(rendering_defaults, "line_width", 4)))),
        font=font,
        small_font=small_font,
        label_stroke_width=0,
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        font_meta=font_role_trace(str(font_family), role="readout"),
        palette_meta={
            "fill_color": list(fill_color),
            "solid_fill_color": list(solid_fill_color),
            "accent_color": list(accent_color),
            "muted_color": list(muted_color),
            "palette_index": int(palette_index),
        },
    )


def _draw_dashed_line(
    ctx: RenderContext,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    dash: float = 10.0,
    gap: float = 7.0,
) -> None:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return
    ux = dx / length
    uy = dy / length
    distance = 0.0
    while distance < length:
        segment_end = min(length, distance + dash)
        ctx.draw.line(
            [
                (float(start[0]) + ux * distance, float(start[1]) + uy * distance),
                (float(start[0]) + ux * segment_end, float(start[1]) + uy * segment_end),
            ],
            fill=fill,
            width=width,
        )
        distance += dash + gap


def _draw_arrowhead(ctx: RenderContext, end: Point, angle: float) -> None:
    head = 14.0
    spread = math.radians(28.0)
    left = (
        float(end[0]) - head * math.cos(angle - spread),
        float(end[1]) - head * math.sin(angle - spread),
    )
    right = (
        float(end[0]) - head * math.cos(angle + spread),
        float(end[1]) - head * math.sin(angle + spread),
    )
    ctx.draw.polygon((end, left, right), fill=ctx.accent_color)


def _draw_arrow(ctx: RenderContext, start: Point, end: Point) -> None:
    ctx.draw.line([start, end], fill=ctx.accent_color, width=max(3, ctx.line_width - 1))
    angle = math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
    _draw_arrowhead(ctx, end, angle)


def _draw_rotation_cue(ctx: RenderContext, center: Point) -> BBox:
    box = (
        float(center[0]) - 60.0,
        float(center[1]) - 28.0,
        float(center[0]) + 60.0,
        float(center[1]) + 28.0,
    )
    ctx.draw.arc(box, start=205, end=515, fill=ctx.accent_color, width=3)
    end_angle = math.radians(155.0)
    end = (
        float(center[0]) + 60.0 * math.cos(end_angle),
        float(center[1]) + 28.0 * math.sin(end_angle),
    )
    _draw_arrowhead(ctx, end, end_angle + math.pi / 2.0)
    return draw_readout_centered(
        ctx,
        "360°",
        (float(center[0]) + 6.0, float(center[1]) - 36.0),
        small=True,
        required=False,
    )


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point,
    target: bool = False,
) -> BBox:
    return draw_dimension_line(
        ctx,
        start,
        end,
        label,
        label_offset=label_offset,
        color=ctx.accent_color if bool(target) else ctx.label_color,
        backed=True,
    )


def _draw_cylinder_preview(ctx: RenderContext, center: Point) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    top_y = cy - 120.0
    bottom_y = cy + 120.0
    left_x = cx - 86.0
    right_x = cx + 86.0
    ellipse_h = 42.0
    ctx.draw.rectangle((left_x, top_y, right_x, bottom_y), fill=ctx.solid_fill_color)
    ctx.draw.line([(left_x, top_y), (left_x, bottom_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([(right_x, top_y), (right_x, bottom_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(
        (left_x, top_y - ellipse_h / 2.0, right_x, top_y + ellipse_h / 2.0),
        fill=ctx.solid_fill_color,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (left_x, bottom_y - ellipse_h / 2.0, right_x, bottom_y + ellipse_h / 2.0),
        start=0,
        end=180,
        fill=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (left_x, bottom_y - ellipse_h / 2.0, right_x, bottom_y + ellipse_h / 2.0),
        start=180,
        end=360,
        fill=ctx.muted_color,
        width=max(2, ctx.line_width - 1),
    )
    return pad_bbox((left_x, top_y - ellipse_h / 2.0, right_x, bottom_y + ellipse_h / 2.0), 12.0, width=ctx.width, height=ctx.height)


def _draw_cone_preview(ctx: RenderContext, center: Point) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    apex = (cx, cy - 135.0)
    left = (cx - 94.0, cy + 118.0)
    right = (cx + 94.0, cy + 118.0)
    ctx.draw.polygon((apex, left, right), fill=ctx.solid_fill_color)
    ctx.draw.line([apex, left], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([apex, right], fill=ctx.line_color, width=ctx.line_width)
    base_box = (left[0], left[1] - 22.0, right[0], right[1] + 22.0)
    ctx.draw.arc(base_box, start=0, end=180, fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.arc(base_box, start=180, end=360, fill=ctx.muted_color, width=max(2, ctx.line_width - 1))
    return bbox_from_points((apex, left, right), width=ctx.width, height=ctx.height, pad=24.0)


def _draw_double_cone_preview(ctx: RenderContext, center: Point) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    top = (cx, cy - 142.0)
    mid_left = (cx - 92.0, cy)
    mid_right = (cx + 92.0, cy)
    bottom = (cx, cy + 142.0)
    ctx.draw.polygon((top, mid_left, mid_right), fill=ctx.solid_fill_color)
    ctx.draw.polygon((bottom, mid_left, mid_right), fill=ctx.solid_fill_color)
    for start, end in ((top, mid_left), (top, mid_right), (bottom, mid_left), (bottom, mid_right)):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    mid_box = (mid_left[0], cy - 22.0, mid_right[0], cy + 22.0)
    ctx.draw.arc(mid_box, start=0, end=180, fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.arc(mid_box, start=180, end=360, fill=ctx.muted_color, width=max(2, ctx.line_width - 1))
    return bbox_from_points((top, mid_left, mid_right, bottom), width=ctx.width, height=ctx.height, pad=22.0)


def _draw_frustum_preview(ctx: RenderContext, center: Point) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    top_y = cy - 120.0
    bottom_y = cy + 120.0
    top_left = cx - 54.0
    top_right = cx + 54.0
    bottom_left = cx - 104.0
    bottom_right = cx + 104.0
    ellipse_h = 36.0
    ctx.draw.polygon(
        ((top_left, top_y), (top_right, top_y), (bottom_right, bottom_y), (bottom_left, bottom_y)),
        fill=ctx.solid_fill_color,
    )
    ctx.draw.line([(top_left, top_y), (bottom_left, bottom_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([(top_right, top_y), (bottom_right, bottom_y)], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(
        (top_left, top_y - ellipse_h / 2.0, top_right, top_y + ellipse_h / 2.0),
        fill=ctx.solid_fill_color,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (bottom_left, bottom_y - ellipse_h / 2.0, bottom_right, bottom_y + ellipse_h / 2.0),
        start=0,
        end=180,
        fill=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.arc(
        (bottom_left, bottom_y - ellipse_h / 2.0, bottom_right, bottom_y + ellipse_h / 2.0),
        start=180,
        end=360,
        fill=ctx.muted_color,
        width=max(2, ctx.line_width - 1),
    )
    return pad_bbox((bottom_left, top_y - ellipse_h / 2.0, bottom_right, bottom_y + ellipse_h / 2.0), 12.0, width=ctx.width, height=ctx.height)


def _render_map(
    *,
    generating_shape: str,
    solid_kind: str,
    figure_points: tuple[Point, ...],
    figure_bbox: BBox,
    axis_bbox: BBox,
    solid_bbox: BBox,
    label_bboxes: Mapping[str, BBox],
    annotation_bboxes: Mapping[str, BBox],
) -> dict[str, Any]:
    return {
        "coord_space": "pixel",
        "generating_shape": {
            "kind": str(generating_shape),
            "points": [[round(float(x), 3), round(float(y), 3)] for x, y in figure_points],
            "bbox": bbox_to_list(figure_bbox),
        },
        "rotation_axis": {"bbox": bbox_to_list(axis_bbox)},
        "solid": {"kind": str(solid_kind), "bbox": bbox_to_list(solid_bbox)},
        "label_bboxes": {str(key): bbox_to_list(value) for key, value in label_bboxes.items()},
        "annotation_bboxes": {str(key): bbox_to_list(value) for key, value in annotation_bboxes.items()},
    }


def _pack_scene(
    *,
    ctx: RenderContext,
    problem: SolidRevolutionProblem,
    figure_points: tuple[Point, ...],
    axis_bbox: BBox,
    solid_bbox: BBox,
    label_bboxes: Mapping[str, BBox],
    annotation_roles: tuple[str, ...],
) -> RenderedSolidRevolutionScene:
    """Pack rendered geometry and labels into scene-neutral output state."""

    figure_bbox = bbox_from_points(figure_points, width=ctx.width, height=ctx.height, pad=36.0)
    source_diagram_bbox = bbox_from_points(
        (
            (figure_bbox[0], figure_bbox[1]),
            (figure_bbox[2], figure_bbox[3]),
            (axis_bbox[0], axis_bbox[1]),
            (axis_bbox[2], axis_bbox[3]),
            *((bbox[0], bbox[1]) for bbox in label_bboxes.values()),
            *((bbox[2], bbox[3]) for bbox in label_bboxes.values()),
        ),
        width=ctx.width,
        height=ctx.height,
        pad=8.0,
    )
    public_annotation_bboxes = {
        "source_diagram_bbox": source_diagram_bbox,
        "resulting_solid_bbox": solid_bbox,
    }
    measurements = {
        "solid_kind": str(problem.solid_kind),
        "generating_shape": str(problem.generating_shape),
        "formula_family": str(problem.formula_family),
        "formula": str(problem.formula),
        "radius": None if problem.radius is None else float(problem.radius),
        "diameter": None if problem.diameter is None else float(problem.diameter),
        "radial_input_kind": problem.radial_input_kind,
        "height": None if problem.height is None else float(problem.height),
        "slant_height": None if problem.slant_height is None else float(problem.slant_height),
        "diagonal": None if problem.diagonal is None else float(problem.diagonal),
        "half_height": None if problem.half_height is None else float(problem.half_height),
        "top_radius": None if problem.top_radius is None else float(problem.top_radius),
        "bottom_radius": None if problem.bottom_radius is None else float(problem.bottom_radius),
        "total_height": None if problem.total_height is None else float(problem.total_height),
        "rotation_degrees": 360,
        "answer_value": float(problem.answer),
    }
    return RenderedSolidRevolutionScene(
        image=ctx.image,
        annotation_bboxes=dict(public_annotation_bboxes),
        annotation_roles=tuple(annotation_roles),
        label_bboxes=dict(label_bboxes),
        scene_entities=(
            {
                "entity_id": "generating_shape",
                "entity_type": str(problem.generating_shape),
                "bbox": bbox_to_list(figure_bbox),
                "rotation_axis": "marked_axis",
            },
            {
                "entity_id": "revolved_solid",
                "entity_type": str(problem.solid_kind),
                "bbox": bbox_to_list(solid_bbox),
            },
        ),
        render_map=_render_map(
            generating_shape=problem.generating_shape,
            solid_kind=problem.solid_kind,
            figure_points=figure_points,
            figure_bbox=figure_bbox,
            axis_bbox=axis_bbox,
            solid_bbox=solid_bbox,
            label_bboxes=label_bboxes,
            annotation_bboxes=public_annotation_bboxes,
        ),
        measurements=measurements,
    )


def _draw_common_cues(ctx: RenderContext, *, axis_start: Point, axis_end: Point, preview_center: Point) -> BBox:
    _draw_dashed_line(ctx, axis_start, axis_end, fill=ctx.accent_color, width=3)
    axis_bbox = pad_bbox(
        (float(axis_start[0]), float(axis_start[1]), float(axis_end[0]), float(axis_end[1])),
        12.0,
        width=ctx.width,
        height=ctx.height,
    )
    _draw_rotation_cue(ctx, (float(axis_start[0]) + 78.0, float(axis_start[1]) - 28.0))
    _draw_arrow(ctx, (395.0, 288.0), (494.0, 288.0))
    draw_readout_centered(
        ctx,
        "V=?",
        (float(preview_center[0]), float(preview_center[1]) - 168.0),
        small=True,
        required=True,
        backed=True,
    )
    return axis_bbox


def render_cylinder_revolution(
    ctx: RenderContext,
    problem: SolidRevolutionProblem,
) -> RenderedSolidRevolutionScene:
    """Render a rectangle rotating into a cylinder."""

    top_y, bottom_y = 142.0, 424.0
    axis_x = 236.0
    rect_left_x, rect_right_x = 132.0, 340.0
    figure_points = (
        (rect_left_x, top_y),
        (rect_right_x, top_y),
        (rect_right_x, bottom_y),
        (rect_left_x, bottom_y),
    )
    ctx.draw.polygon(figure_points, fill=ctx.fill_color)
    for start, end in zip(figure_points, figure_points[1:] + figure_points[:1]):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    preview_center = (638.0, 288.0)
    axis_bbox = _draw_common_cues(
        ctx,
        axis_start=(axis_x, top_y - 34.0),
        axis_end=(axis_x, bottom_y + 34.0),
        preview_center=preview_center,
    )
    label_bboxes: dict[str, BBox] = {
        "height_label": _draw_dimension(
            ctx,
            (rect_left_x - 36.0, top_y),
            (rect_left_x - 36.0, bottom_y),
            f"h={format_measure(problem.height or 0)}",
            label_offset=(-28.0, 0.0),
        ),
    }
    if str(problem.radial_input_kind) == "diagonal":
        label_bboxes["radial_input_label"] = _draw_dimension(
            ctx,
            (rect_left_x, top_y),
            (rect_right_x, bottom_y),
            f"diag={format_measure(problem.diagonal or 0)}",
            label_offset=(34.0, -16.0),
        )
    else:
        label_bboxes["radial_input_label"] = _draw_dimension(
            ctx,
            (rect_left_x, bottom_y + 34.0),
            (rect_right_x, bottom_y + 34.0),
            f"d={format_measure(problem.diameter or 0)}",
            label_offset=(0.0, 26.0),
        )
    solid_bbox = _draw_cylinder_preview(ctx, preview_center)
    return _pack_scene(
        ctx=ctx,
        problem=problem,
        figure_points=figure_points,
        axis_bbox=axis_bbox,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_roles=(
            "source_diagram_bbox",
            "resulting_solid_bbox",
        ),
    )


def render_cone_revolution(
    ctx: RenderContext,
    problem: SolidRevolutionProblem,
) -> RenderedSolidRevolutionScene:
    """Render a right triangle rotating into a cone."""

    left_x, right_x = 200.0, 340.0
    top_y, bottom_y = 142.0, 424.0
    figure_points = ((left_x, top_y), (left_x, bottom_y), (right_x, bottom_y))
    ctx.draw.polygon(figure_points, fill=ctx.fill_color)
    for start, end in zip(figure_points, figure_points[1:] + figure_points[:1]):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    draw_right_angle_marker(ctx, (left_x, bottom_y), arm_a=(left_x, top_y), arm_b=(right_x, bottom_y), color=ctx.line_color)
    preview_center = (638.0, 288.0)
    axis_bbox = _draw_common_cues(
        ctx,
        axis_start=(left_x, top_y - 34.0),
        axis_end=(left_x, bottom_y + 34.0),
        preview_center=preview_center,
    )
    label_bboxes = {
        "height_label": _draw_dimension(
            ctx,
            (left_x - 36.0, top_y),
            (left_x - 36.0, bottom_y),
            f"h={format_measure(problem.height or 0)}",
            label_offset=(-28.0, 0.0),
        ),
        "slant_height_label": _draw_dimension(
            ctx,
            (left_x, top_y),
            (right_x, bottom_y),
            f"slant={format_measure(problem.slant_height or 0)}",
            label_offset=(30.0, -16.0),
        ),
    }
    solid_bbox = _draw_cone_preview(ctx, preview_center)
    return _pack_scene(
        ctx=ctx,
        problem=problem,
        figure_points=figure_points,
        axis_bbox=axis_bbox,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_roles=(
            "source_diagram_bbox",
            "resulting_solid_bbox",
        ),
    )


def render_double_cone_revolution(
    ctx: RenderContext,
    problem: SolidRevolutionProblem,
) -> RenderedSolidRevolutionScene:
    """Render an isosceles triangle rotating into a double cone."""

    left_x, right_x = 200.0, 340.0
    top_y, bottom_y = 142.0, 424.0
    mid_y = (top_y + bottom_y) / 2.0
    figure_points = ((left_x, top_y), (left_x, bottom_y), (right_x, mid_y))
    ctx.draw.polygon(figure_points, fill=ctx.fill_color)
    for start, end in zip(figure_points, figure_points[1:] + figure_points[:1]):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    preview_center = (638.0, 288.0)
    axis_bbox = _draw_common_cues(
        ctx,
        axis_start=(left_x, top_y - 34.0),
        axis_end=(left_x, bottom_y + 34.0),
        preview_center=preview_center,
    )
    label_bboxes = {
        "half_height_label": _draw_dimension(
            ctx,
            (left_x - 36.0, top_y),
            (left_x - 36.0, mid_y),
            f"h={format_measure(problem.half_height or 0)}",
            label_offset=(-28.0, 0.0),
        ),
        "radius_label": _draw_dimension(
            ctx,
            (left_x, mid_y + 34.0),
            (right_x, mid_y + 34.0),
            f"r={format_measure(problem.radius or 0)}",
            label_offset=(0.0, 26.0),
        ),
    }
    solid_bbox = _draw_double_cone_preview(ctx, preview_center)
    return _pack_scene(
        ctx=ctx,
        problem=problem,
        figure_points=figure_points,
        axis_bbox=axis_bbox,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_roles=(
            "source_diagram_bbox",
            "resulting_solid_bbox",
        ),
    )


def render_frustum_revolution(
    ctx: RenderContext,
    problem: SolidRevolutionProblem,
) -> RenderedSolidRevolutionScene:
    """Render a right trapezoid rotating into a frustum."""

    left_x, top_y, bottom_y = 200.0, 142.0, 424.0
    top_right, bottom_right = left_x + 86.0, 340.0
    figure_points = ((left_x, top_y), (top_right, top_y), (bottom_right, bottom_y), (left_x, bottom_y))
    ctx.draw.polygon(figure_points, fill=ctx.fill_color)
    for start, end in zip(figure_points, figure_points[1:] + figure_points[:1]):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    draw_right_angle_marker(ctx, (left_x, bottom_y), arm_a=(left_x, top_y), arm_b=(bottom_right, bottom_y), color=ctx.line_color)
    preview_center = (638.0, 288.0)
    axis_bbox = _draw_common_cues(
        ctx,
        axis_start=(left_x, top_y - 34.0),
        axis_end=(left_x, bottom_y + 34.0),
        preview_center=preview_center,
    )
    label_bboxes = {
        "height_label": _draw_dimension(
            ctx,
            (left_x - 36.0, top_y),
            (left_x - 36.0, bottom_y),
            f"h={format_measure(problem.height or 0)}",
            label_offset=(-28.0, 0.0),
        ),
        "top_radius_label": _draw_dimension(
            ctx,
            (left_x, top_y - 34.0),
            (top_right, top_y - 34.0),
            f"small r={format_measure(problem.top_radius or 0)}",
            label_offset=(0.0, -24.0),
        ),
        "bottom_radius_label": _draw_dimension(
            ctx,
            (left_x, bottom_y + 34.0),
            (bottom_right, bottom_y + 34.0),
            f"large R={format_measure(problem.bottom_radius or 0)}",
            label_offset=(0.0, 26.0),
        ),
    }
    solid_bbox = _draw_frustum_preview(ctx, preview_center)
    return _pack_scene(
        ctx=ctx,
        problem=problem,
        figure_points=figure_points,
        axis_bbox=axis_bbox,
        solid_bbox=solid_bbox,
        label_bboxes=label_bboxes,
        annotation_roles=(
            "source_diagram_bbox",
            "resulting_solid_bbox",
        ),
    )


__all__ = [
    "create_render_context",
    "render_cone_revolution",
    "render_cylinder_revolution",
    "render_double_cone_revolution",
    "render_frustum_revolution",
]
