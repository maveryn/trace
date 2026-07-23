"""Rendering primitives for visible measuring-tool diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping

from PIL import ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    pad_bbox,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .spatial_primitives import (
    add,
    normal,
    point_to_list,
    protractor_point,
    scale,
    sub,
    unit_from_degrees,
)
from .state import (
    AngleMeasurementPlan,
    BBox,
    Color,
    LengthMeasurementPlan,
    Point,
    RenderContext,
    RenderedToolScene,
)


def make_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderContext, dict[str, Any]]:
    """Create one styled rendering context for a visible measurement scene."""

    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    protected = ((205, 70, 52), (30, 126, 185), (38, 150, 95))
    image, background_meta, diagram_style, style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=width,
        canvas_height=height,
        protected_colors=protected,
        allow_dark=False,
        require_grid=False,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 17))
    )
    tiny_font_size = int(
        params.get("tiny_label_font_size", group_default(render_defaults, "tiny_label_font_size", 13))
    )
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    ctx = RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=width,
        height=height,
        line_color=tuple(int(v) for v in diagram_style.stroke_rgb),
        secondary_color=tuple(int(v) for v in diagram_style.secondary_stroke_rgb),
        guide_color=tuple(int(v) for v in diagram_style.guide_rgb),
        label_color=tuple(int(v) for v in diagram_style.label_rgb),
        label_stroke_color=tuple(int(v) for v in diagram_style.label_stroke_rgb),
        panel_fill=tuple(int(v) for v in diagram_style.panel_fill_rgb),
        panel_alt_fill=tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
        panel_border=tuple(int(v) for v in diagram_style.panel_border_rgb),
        accent_color=tuple(int(v) for v in diagram_style.accent_rgb),
        secondary_accent_color=tuple(int(v) for v in diagram_style.secondary_accent_rgb),
        line_width=max(2, line_width),
        font=load_font(max(12, font_size), bold=False),
        small_font=load_font(max(10, small_font_size), bold=False),
        tiny_font=load_font(max(8, tiny_font_size), bold=False),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(style_meta),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "tiny_label_font_size": int(tiny_font_size),
    }
    return ctx, render_meta


def draw_text(
    ctx: RenderContext,
    text: str,
    center: Point,
    *,
    font: Any | None = None,
    fill: Color | None = None,
    stroke_width: int = 0,
) -> BBox:
    """Draw centered text and return its padded bbox."""

    active_font = font or ctx.small_font
    active_fill = fill or ctx.label_color
    bbox = ctx.draw.textbbox((0, 0), str(text), font=active_font, stroke_width=stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - (text_w / 2.0)
    top = float(center[1]) - (text_h / 2.0)
    draw_text_traced(
        ctx.draw,
        (left, top),
        str(text),
        font=active_font,
        fill=active_fill,
        stroke_width=int(stroke_width),
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    return pad_bbox((left, top, left + text_w, top + text_h), 3.0, width=ctx.width, height=ctx.height)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    dash: float = 8.0,
    gap: float = 6.0,
) -> None:
    """Draw a deterministic dashed guide segment."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        seg_end = min(length, pos + dash)
        draw.line(
            [(sx + ux * pos, sy + uy * pos), (sx + ux * seg_end, sy + uy * seg_end)],
            fill=fill,
            width=max(1, int(width)),
        )
        pos = seg_end + gap


def draw_rotated_ruler(
    ctx: RenderContext,
    *,
    zero_point: Point,
    axis: Point,
    ruler_normal: Point,
    unit_px: float,
    ruler_max_cm: int,
    highlight_start_cm: int,
    highlight_end_cm: int,
    half_width: float = 30.0,
) -> tuple[BBox, Point, Point]:
    """Draw a ruler aligned with a segment and return highlighted ticks."""

    axis_len = max(1e-9, math.hypot(float(axis[0]), float(axis[1])))
    normal_len = max(1e-9, math.hypot(float(ruler_normal[0]), float(ruler_normal[1])))
    axis = scale(axis, 1.0 / axis_len)
    ruler_normal = scale(ruler_normal, 1.0 / normal_len)
    p0 = zero_point
    p1 = add(zero_point, scale(axis, float(ruler_max_cm) * float(unit_px)))
    corners = (
        add(p0, scale(ruler_normal, -half_width)),
        add(p1, scale(ruler_normal, -half_width)),
        add(p1, scale(ruler_normal, half_width)),
        add(p0, scale(ruler_normal, half_width)),
    )
    ctx.draw.polygon(corners, fill=ctx.panel_alt_fill, outline=ctx.panel_border)
    ctx.draw.line([corners[0], corners[1], corners[2], corners[3], corners[0]], fill=ctx.panel_border, width=2)
    for half_tick in range(0, (2 * int(ruler_max_cm)) + 1):
        cm_value = half_tick / 2.0
        tick_center = add(zero_point, scale(axis, cm_value * float(unit_px)))
        major = half_tick % 2 == 0
        tick_len = half_width * (1.35 if major else 0.78)
        p_start = add(tick_center, scale(ruler_normal, -half_width))
        p_end = add(tick_center, scale(ruler_normal, -half_width + tick_len))
        ctx.draw.line([p_start, p_end], fill=ctx.secondary_color, width=2 if major else 1)
        if major:
            label_center = add(tick_center, scale(ruler_normal, half_width + 13.0))
            draw_text(ctx, str(int(cm_value)), label_center, font=ctx.tiny_font, stroke_width=0)
    unit_label = add(
        add(zero_point, scale(axis, float(ruler_max_cm) * float(unit_px) - 18.0)),
        scale(ruler_normal, 13.0),
    )
    draw_text(ctx, "cm", unit_label, font=ctx.tiny_font, stroke_width=0)
    highlight_start = add(zero_point, scale(axis, float(highlight_start_cm) * float(unit_px)))
    highlight_end = add(zero_point, scale(axis, float(highlight_end_cm) * float(unit_px)))
    ctx.draw.line(
        [add(highlight_start, scale(ruler_normal, -half_width)), add(highlight_start, scale(ruler_normal, half_width))],
        fill=ctx.secondary_accent_color,
        width=3,
    )
    ctx.draw.line(
        [add(highlight_end, scale(ruler_normal, -half_width)), add(highlight_end, scale(ruler_normal, half_width))],
        fill=ctx.secondary_accent_color,
        width=3,
    )
    return bbox_from_points(corners, width=ctx.width, height=ctx.height, pad=10.0), highlight_start, highlight_end


def draw_polygon_shape(
    ctx: RenderContext,
    *,
    shape_kind: str,
    start: Point,
    end: Point,
    axis: Point,
) -> None:
    """Draw a simple polygon whose measured side is `start -> end`."""

    shape_normal = normal(axis)
    length_px = math.hypot(float(end[0] - start[0]), float(end[1] - start[1]))
    height = max(96.0, min(145.0, length_px * 0.42))
    if shape_kind == "triangle":
        apex = add(add(start, scale(axis, length_px * 0.46)), scale(shape_normal, -height))
        vertices = (start, end, apex)
    elif shape_kind == "trapezoid":
        skew = scale(axis, length_px * 0.18)
        top_left = add(add(start, scale(shape_normal, -height)), skew)
        top_right = add(add(end, scale(shape_normal, -height)), scale(axis, -length_px * 0.20))
        vertices = (start, end, top_right, top_left)
    else:
        offset = add(scale(shape_normal, -height), scale(axis, length_px * 0.18))
        vertices = (start, end, add(end, offset), add(start, offset))
    ctx.draw.polygon(vertices, fill=ctx.panel_fill, outline=ctx.line_color)
    ctx.draw.line(list(vertices) + [vertices[0]], fill=ctx.line_color, width=max(2, ctx.line_width))
    ctx.draw.line([start, end], fill=ctx.accent_color, width=ctx.line_width + 3)
    for point in (start, end):
        r = 5.5
        ctx.draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=ctx.accent_color)


def render_length_measurement_scene(
    ctx: RenderContext,
    plan: LengthMeasurementPlan,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RenderedToolScene:
    """Render a ruler placed against a polygon side or circle radius."""

    unit_px = float(params.get("ruler_unit_px", group_default(render_defaults, "ruler_unit_px", 42)))
    length_px = float(plan.target_length_cm) * unit_px
    ruler_start_cm = int(plan.ruler_start_cm)
    ruler_end_cm = int(plan.ruler_start_cm + plan.target_length_cm)
    orientation_options = (-22.0, -12.0, 0.0, 14.0, 25.0)
    orientation_rng = spawn_rng(int(instance_seed), "geometry.measuring_tools.ruler_orientation")
    orientation = float(uniform_choice(orientation_rng, orientation_options))
    axis = unit_from_degrees(float(orientation))
    ruler_normal = normal(axis)

    if plan.shape_kind == "circle":
        center = (float(ctx.width) * 0.45, float(ctx.height) * 0.44)
        end = add(center, scale(axis, length_px))
        radius = length_px
        ctx.draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            fill=ctx.panel_fill,
            outline=ctx.line_color,
            width=max(2, ctx.line_width),
        )
        ctx.draw.line([center, end], fill=ctx.accent_color, width=ctx.line_width + 3)
        for point in (center, end):
            r = 5.5
            ctx.draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=ctx.accent_color)
        ruler_start_tick_target = add(center, scale(ruler_normal, 70.0))
        zero_point = sub(ruler_start_tick_target, scale(axis, float(ruler_start_cm) * unit_px))
        measure_start, measure_end = center, end
    else:
        center = (float(ctx.width) * 0.52, float(ctx.height) * 0.47)
        start = sub(center, scale(axis, length_px / 2.0))
        end = add(center, scale(axis, length_px / 2.0))
        draw_polygon_shape(ctx, shape_kind=str(plan.shape_kind), start=start, end=end, axis=axis)
        ruler_start_tick_target = add(start, scale(ruler_normal, 72.0))
        zero_point = sub(ruler_start_tick_target, scale(axis, float(ruler_start_cm) * unit_px))
        measure_start, measure_end = start, end

    ruler_bbox, ruler_start_tick, ruler_end_tick = draw_rotated_ruler(
        ctx,
        zero_point=zero_point,
        axis=axis,
        ruler_normal=ruler_normal,
        unit_px=unit_px,
        ruler_max_cm=int(plan.ruler_max_cm),
        highlight_start_cm=ruler_start_cm,
        highlight_end_cm=ruler_end_cm,
    )
    draw_dashed_line(ctx.draw, measure_start, ruler_start_tick, fill=ctx.guide_color, width=max(1, ctx.line_width - 2))
    draw_dashed_line(ctx.draw, measure_end, ruler_end_tick, fill=ctx.guide_color, width=max(1, ctx.line_width - 2))
    label_center = add(add(measure_start, scale(sub(measure_end, measure_start), 0.5)), scale(ruler_normal, -25.0))
    draw_text(ctx, "?", label_center, font=ctx.font, fill=ctx.secondary_accent_color)
    annotation = {
        "measure_start": point_to_list(measure_start),
        "measure_end": point_to_list(measure_end),
        "ruler_start_tick": point_to_list(ruler_start_tick),
        "ruler_end_tick": point_to_list(ruler_end_tick),
    }
    scene_entities = (
        {
            "entity_id": "ruler",
            "entity_type": "measuring_tool",
            "tool_kind": "ruler",
            "unit": "centimeter",
            "max_cm": int(plan.ruler_max_cm),
            "bbox": bbox_to_list(ruler_bbox),
        },
        {
            "entity_id": "measured_feature",
            "entity_type": "radius" if plan.shape_kind == "circle" else "side",
            "shape_kind": str(plan.shape_kind),
            "length_cm": int(plan.target_length_cm),
            "endpoints_px": [point_to_list(measure_start), point_to_list(measure_end)],
            "bbox": bbox_to_list(
                bbox_from_points((measure_start, measure_end), width=ctx.width, height=ctx.height, pad=18.0)
            ),
        },
    )
    return RenderedToolScene(
        image=ctx.image,
        answer=int(plan.target_length_cm),
        annotation_points=annotation,
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "tool_kind": "ruler",
            "measurement_kind": str(plan.measurement_kind),
            "shape_kind": str(plan.shape_kind),
            "ruler_unit_px": round(unit_px, 3),
            "ruler_start_cm": int(ruler_start_cm),
            "ruler_end_cm": int(ruler_end_cm),
            "ruler_bbox": bbox_to_list(ruler_bbox),
            "measurement_points": dict(annotation),
        },
        witness={
            "tool_kind": "ruler",
            "measurement_kind": str(plan.measurement_kind),
            "shape_kind": str(plan.shape_kind),
            "unit": "centimeter",
            "target_length_cm": int(plan.target_length_cm),
            "ruler_start_cm": int(ruler_start_cm),
            "ruler_end_cm": int(ruler_end_cm),
            "answer_value": int(plan.target_length_cm),
        },
    )


def render_length_measurement(
    ctx: RenderContext,
    plan: LengthMeasurementPlan,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RenderedToolScene:
    """Lifecycle-compatible adapter for ruler diagrams."""

    return render_length_measurement_scene(
        ctx,
        plan,
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
    )


def render_angle_measurement_scene(
    ctx: RenderContext,
    plan: AngleMeasurementPlan,
) -> RenderedToolScene:
    """Render a protractor placed on a polygon vertex."""

    cx = float(ctx.width) * 0.34
    cy = float(ctx.height) * 0.72
    center = (cx, cy)
    outer_radius = min(float(ctx.width) * 0.34, float(ctx.height) * 0.43)
    inner_radius = outer_radius - 76.0
    ray_radius = outer_radius + 28.0
    baseline_end = protractor_point(center, ray_radius, 0)
    target_end = protractor_point(center, ray_radius, float(plan.target_angle_degrees))

    if plan.shape_kind == "quadrilateral":
        top = (
            min(float(ctx.width) - 80.0, max(baseline_end[0], target_end[0]) + 35.0),
            min(baseline_end[1], target_end[1]) - 80.0,
        )
        shape_points = (center, baseline_end, top, target_end)
    else:
        shape_points = (center, baseline_end, target_end)

    protractor_box = (
        cx - outer_radius,
        cy - outer_radius,
        cx + outer_radius,
        cy + outer_radius,
    )
    inner_box = (
        cx - inner_radius,
        cy - inner_radius,
        cx + inner_radius,
        cy + inner_radius,
    )
    ctx.draw.pieslice(protractor_box, start=180, end=360, fill=ctx.panel_alt_fill, outline=ctx.panel_border, width=3)
    ctx.draw.arc(protractor_box, start=180, end=360, fill=ctx.line_color, width=3)
    ctx.draw.arc(inner_box, start=180, end=360, fill=ctx.secondary_color, width=2)
    left = protractor_point(center, outer_radius, 180)
    right = protractor_point(center, outer_radius, 0)
    ctx.draw.line([left, right], fill=ctx.line_color, width=3)
    for degree in range(0, 181, 5):
        tick_len = 22.0 if degree % 10 == 0 else 12.0
        if degree % 30 == 0:
            tick_len = 28.0
        p0 = protractor_point(center, outer_radius - 2.0, degree)
        p1 = protractor_point(center, outer_radius - tick_len, degree)
        ctx.draw.line([p0, p1], fill=ctx.secondary_color, width=2 if degree % 10 == 0 else 1)
    for degree in range(0, 181, 30):
        label_center = protractor_point(center, outer_radius - 48.0, degree)
        draw_text(ctx, str(degree), label_center, font=ctx.tiny_font, stroke_width=0)

    # Keep the measuring tool behind the measured shape; otherwise the filled
    # Keep the instrument behind the target so its body does not obscure geometry.
    ctx.draw.line(list(shape_points) + [shape_points[0]], fill=ctx.line_color, width=max(2, ctx.line_width))
    ctx.draw.line([center, baseline_end], fill=ctx.accent_color, width=ctx.line_width + 3)
    ctx.draw.line([center, target_end], fill=ctx.accent_color, width=ctx.line_width + 3)

    dot_r = 6.0
    ctx.draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=ctx.accent_color)
    arc_box = (cx - 78.0, cy - 78.0, cx + 78.0, cy + 78.0)
    ctx.draw.arc(
        arc_box,
        start=360 - int(plan.target_angle_degrees),
        end=360,
        fill=ctx.secondary_accent_color,
        width=5,
    )
    mid_degree = float(plan.target_angle_degrees) / 2.0
    question_center = protractor_point(center, 103.0, mid_degree)
    draw_text(ctx, "?", question_center, font=ctx.font, fill=ctx.secondary_accent_color)
    reading_tick = protractor_point(center, outer_radius - 10.0, float(plan.target_angle_degrees))
    annotation = {
        "angle_vertex": point_to_list(center),
        "baseline_ray_point": point_to_list(baseline_end),
        "target_ray_point": point_to_list(target_end),
        "protractor_reading_tick": point_to_list(reading_tick),
    }
    scale_bbox = bbox_to_list(
        pad_bbox((cx - outer_radius, cy - outer_radius, cx + outer_radius, cy), 10.0, width=ctx.width, height=ctx.height)
    )
    angle_bbox = bbox_to_list(bbox_from_points((center, baseline_end, target_end), width=ctx.width, height=ctx.height, pad=18.0))
    scene_entities = (
        {
            "entity_id": "protractor",
            "entity_type": "measuring_tool",
            "tool_kind": "protractor",
            "center_px": point_to_list(center),
            "outer_radius_px": round(outer_radius, 3),
            "bbox": scale_bbox,
        },
        {
            "entity_id": "target_angle",
            "entity_type": "angle",
            "shape_kind": str(plan.shape_kind),
            "degree_value": int(plan.target_angle_degrees),
            "vertex_px": point_to_list(center),
            "ray_endpoints_px": [point_to_list(baseline_end), point_to_list(target_end)],
            "bbox": angle_bbox,
        },
    )
    return RenderedToolScene(
        image=ctx.image,
        answer=int(plan.target_angle_degrees),
        annotation_points=annotation,
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "tool_kind": "protractor",
            "measurement_kind": str(plan.measurement_kind),
            "shape_kind": str(plan.shape_kind),
            "center_px": point_to_list(center),
            "outer_radius_px": round(outer_radius, 3),
            "target_angle_degrees": int(plan.target_angle_degrees),
            "measurement_points": dict(annotation),
        },
        witness={
            "tool_kind": "protractor",
            "measurement_kind": str(plan.measurement_kind),
            "shape_kind": str(plan.shape_kind),
            "target_angle_degrees": int(plan.target_angle_degrees),
            "answer_value": int(plan.target_angle_degrees),
        },
    )


def render_angle_measurement(
    ctx: RenderContext,
    plan: AngleMeasurementPlan,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> RenderedToolScene:
    """Lifecycle-compatible adapter for protractor diagrams."""

    return render_angle_measurement_scene(ctx, plan)


__all__ = [
    "draw_polygon_shape",
    "draw_rotated_ruler",
    "draw_text",
    "make_context",
    "render_angle_measurement",
    "render_angle_measurement_scene",
    "render_length_measurement",
    "render_length_measurement_scene",
]
