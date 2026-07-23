"""Rendering primitives for solid cross-section diagrams."""

from __future__ import annotations

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
    draw_label,
    fmt_measure,
    pad_bbox,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.font_assets import font_role_trace, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import SCENE_ID
from .state import BBox, Color, Point, RenderContext, RenderedSolidCrossSectionScene, SolidCrossSectionProblem


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
    palettes: tuple[tuple[Color, Color, Color, Color, Color], ...] = (
        (
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.accent_rgb),
            tuple(int(v) for v in diagram_style.highlight_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.muted_fill_rgb),
            tuple(int(v) for v in diagram_style.secondary_accent_rgb),
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.secondary_stroke_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.highlight_rgb),
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
    )
    palette_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{random_namespace}.palette",
    )
    fill_color, secondary_fill_color, accent_color, slice_fill_color, muted_color = palettes[
        int(palette_index) % len(palettes)
    ]
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
        secondary_fill_color=secondary_fill_color,
        accent_color=accent_color,
        slice_fill_color=slice_fill_color,
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
            "secondary_fill_color": list(secondary_fill_color),
            "accent_color": list(accent_color),
            "slice_fill_color": list(slice_fill_color),
            "muted_color": list(muted_color),
            "palette_index": int(palette_index) % len(palettes),
        },
    )


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point = (0.0, 0.0),
) -> BBox:
    ctx.draw.line([start, end], fill=ctx.label_color, width=max(2, ctx.line_width - 1))
    tick = 7.0
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = (dx**2 + dy**2) ** 0.5
    if length > 1e-9:
        nx = -dy / length
        ny = dx / length
        for point in (start, end):
            ctx.draw.line(
                [
                    (float(point[0]) - tick * nx, float(point[1]) - tick * ny),
                    (float(point[0]) + tick * nx, float(point[1]) + tick * ny),
                ],
                fill=ctx.label_color,
                width=max(2, ctx.line_width - 1),
            )
    center = (
        (float(start[0]) + float(end[0])) / 2.0 + float(label_offset[0]),
        (float(start[1]) + float(end[1])) / 2.0 + float(label_offset[1]),
    )
    return draw_label(ctx, label, center, small=True)


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
    length = (dx**2 + dy**2) ** 0.5
    if length <= 1e-9:
        return
    ux = dx / length
    uy = dy / length
    distance = 0.0
    while distance < length:
        seg_start = distance
        seg_end = min(length, distance + dash)
        ctx.draw.line(
            [
                (float(start[0]) + ux * seg_start, float(start[1]) + uy * seg_start),
                (float(start[0]) + ux * seg_end, float(start[1]) + uy * seg_end),
            ],
            fill=fill,
            width=width,
        )
        distance += dash + gap


def _slice_y(apex_y: float, base_y: float, solid_height: float, slice_distance: float) -> float:
    return float(apex_y) + (float(base_y) - float(apex_y)) * (float(slice_distance) / float(solid_height))


def render_cone_cross_section(ctx: RenderContext, problem: SolidCrossSectionProblem) -> RenderedSolidCrossSectionScene:
    """Render a cone with one plane parallel to the base."""

    apex = (410.0, 92.0)
    base_left = (226.0, 438.0)
    base_right = (594.0, 438.0)
    base_center = (410.0, 438.0)
    base_ellipse_h = 74.0
    slice_y = _slice_y(apex[1], base_center[1], problem.solid_height, problem.slice_distance_from_apex)
    scale = float(problem.slice_distance_from_apex) / float(problem.solid_height)
    slice_half_w = (base_right[0] - base_left[0]) * scale / 2.0
    slice_ellipse_h = base_ellipse_h * scale
    slice_bbox = (
        base_center[0] - slice_half_w,
        slice_y - slice_ellipse_h / 2.0,
        base_center[0] + slice_half_w,
        slice_y + slice_ellipse_h / 2.0,
    )

    ctx.draw.polygon((apex, base_left, base_right), fill=ctx.fill_color)
    ctx.draw.line([apex, base_left], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([apex, base_right], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(
        (base_left[0], base_left[1] - base_ellipse_h / 2.0, base_right[0], base_right[1] + base_ellipse_h / 2.0),
        fill=ctx.secondary_fill_color,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    _draw_dashed_line(ctx, apex, base_center, fill=ctx.accent_color, width=3)
    ctx.draw.ellipse(slice_bbox, fill=ctx.slice_fill_color, outline=ctx.accent_color, width=max(3, ctx.line_width))
    ctx.draw.line([(slice_bbox[0], slice_y), (slice_bbox[2], slice_y)], fill=ctx.accent_color, width=2)

    label_bboxes = {
        "target_area": draw_label(ctx, "A=?", (base_center[0], slice_y - slice_ellipse_h / 2.0 - 28.0), small=True),
        "base_radius_label": _draw_dimension(
            ctx,
            (base_center[0], base_center[1] + base_ellipse_h / 2.0 + 22.0),
            (base_right[0], base_right[1] + base_ellipse_h / 2.0 + 22.0),
            f"R={fmt_measure(problem.base_radius or 0)}",
            label_offset=(0.0, 26.0),
        ),
        "height_label": _draw_dimension(
            ctx,
            (base_right[0] + 48.0, apex[1]),
            (base_right[0] + 48.0, base_center[1]),
            f"H={fmt_measure(problem.solid_height)}",
            label_offset=(38.0, 0.0),
        ),
        "slice_distance_label": _draw_dimension(
            ctx,
            (base_center[0] - 42.0, apex[1]),
            (base_center[0] - 42.0, slice_y),
            f"d={fmt_measure(problem.slice_distance_from_apex)}",
            label_offset=(-36.0, 0.0),
        ),
    }
    solid_bbox = bbox_from_points((apex, base_left, base_right), width=ctx.width, height=ctx.height, pad=58.0)
    cross_section_bbox = pad_bbox(slice_bbox, 6.0, width=ctx.width, height=ctx.height)
    return _rendered_scene(
        ctx=ctx,
        problem=problem,
        solid_bbox=solid_bbox,
        cross_section_bbox=cross_section_bbox,
        label_bboxes=label_bboxes,
        annotation_roles=("cross_section",),
    )


def render_square_pyramid_cross_section(
    ctx: RenderContext,
    problem: SolidCrossSectionProblem,
) -> RenderedSolidCrossSectionScene:
    """Render a square pyramid with one plane parallel to the base."""

    apex = (410.0, 84.0)
    front_left = (246.0, 438.0)
    front_right = (538.0, 438.0)
    back_right = (610.0, 382.0)
    back_left = (318.0, 382.0)
    base_points = (front_left, front_right, back_right, back_left)
    base_center = (
        sum(point[0] for point in base_points) / 4.0,
        sum(point[1] for point in base_points) / 4.0,
    )
    slice_scale = float(problem.slice_distance_from_apex) / float(problem.solid_height)
    slice_points = tuple(
        (
            apex[0] + (point[0] - apex[0]) * slice_scale,
            apex[1] + (point[1] - apex[1]) * slice_scale,
        )
        for point in base_points
    )
    slice_y = _slice_y(apex[1], base_center[1], problem.solid_height, problem.slice_distance_from_apex)

    ctx.draw.polygon(base_points, fill=ctx.secondary_fill_color)
    for start, end in zip(base_points, base_points[1:] + base_points[:1]):
        ctx.draw.line([start, end], fill=ctx.line_color, width=ctx.line_width)
    for point in base_points:
        ctx.draw.line([apex, point], fill=ctx.line_color, width=ctx.line_width)
    _draw_dashed_line(ctx, apex, base_center, fill=ctx.accent_color, width=3)
    ctx.draw.polygon(slice_points, fill=ctx.slice_fill_color)
    for start, end in zip(slice_points, slice_points[1:] + slice_points[:1]):
        ctx.draw.line([start, end], fill=ctx.accent_color, width=max(3, ctx.line_width))

    cross_section_bbox = bbox_from_points(slice_points, width=ctx.width, height=ctx.height, pad=6.0)
    label_bboxes = {
        "target_area": draw_label(ctx, "A=?", (base_center[0], cross_section_bbox[1] - 22.0), small=True),
        "base_side_label": _draw_dimension(
            ctx,
            (front_left[0], front_left[1] + 34.0),
            (front_right[0], front_right[1] + 34.0),
            f"s={fmt_measure(problem.base_side or 0)}",
            label_offset=(0.0, 28.0),
        ),
        "height_label": _draw_dimension(
            ctx,
            (back_right[0] + 42.0, apex[1]),
            (back_right[0] + 42.0, base_center[1]),
            f"H={fmt_measure(problem.solid_height)}",
            label_offset=(38.0, 0.0),
        ),
        "slice_distance_label": _draw_dimension(
            ctx,
            (base_center[0] - 44.0, apex[1]),
            (base_center[0] - 44.0, slice_y),
            f"d={fmt_measure(problem.slice_distance_from_apex)}",
            label_offset=(-36.0, 0.0),
        ),
    }
    solid_bbox = bbox_from_points((apex,) + base_points, width=ctx.width, height=ctx.height, pad=58.0)
    return _rendered_scene(
        ctx=ctx,
        problem=problem,
        solid_bbox=solid_bbox,
        cross_section_bbox=cross_section_bbox,
        label_bboxes=label_bboxes,
        annotation_roles=("cross_section",),
    )


def _rendered_scene(
    *,
    ctx: RenderContext,
    problem: SolidCrossSectionProblem,
    solid_bbox: BBox,
    cross_section_bbox: BBox,
    label_bboxes: Mapping[str, BBox],
    annotation_roles: tuple[str, ...],
) -> RenderedSolidCrossSectionScene:
    """Package one rendered solid diagram with role-bound annotation bboxes."""

    annotation_bboxes = {
        "cross_section": cross_section_bbox,
        **{str(key): bbox for key, bbox in label_bboxes.items()},
    }
    measurements = {
        "solid_kind": str(problem.solid_kind),
        "base_radius": None if problem.base_radius is None else float(problem.base_radius),
        "base_side": None if problem.base_side is None else float(problem.base_side),
        "solid_height": float(problem.solid_height),
        "slice_distance_from_apex": float(problem.slice_distance_from_apex),
        "similarity_scale": float(problem.slice_distance_from_apex) / float(problem.solid_height),
        "slice_radius": None if problem.slice_radius is None else float(problem.slice_radius),
        "slice_side": None if problem.slice_side is None else float(problem.slice_side),
        "formula_family": str(problem.formula_family),
        "formula": str(problem.formula),
        "answer_value": float(problem.answer),
    }
    return RenderedSolidCrossSectionScene(
        image=ctx.image,
        annotation_bboxes=annotation_bboxes,
        annotation_roles=tuple(str(role) for role in annotation_roles),
        label_bboxes=dict(label_bboxes),
        scene_entities=(
            {
                "entity_id": "solid",
                "entity_type": str(problem.solid_kind),
                "bbox": bbox_to_list(solid_bbox),
            },
            {
                "entity_id": "cross_section",
                "entity_type": "parallel_slice",
                "bbox": bbox_to_list(cross_section_bbox),
            },
        ),
        render_map={
            "coord_space": "pixel",
            "solid": {
                "kind": str(problem.solid_kind),
                "bbox": bbox_to_list(solid_bbox),
            },
            "cross_section": {"bbox": bbox_to_list(cross_section_bbox)},
            "label_bboxes": {str(key): bbox_to_list(value) for key, value in label_bboxes.items()},
        },
        measurements=measurements,
    )


__all__ = [
    "create_render_context",
    "render_cone_cross_section",
    "render_square_pyramid_cross_section",
]
