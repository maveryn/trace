"""Rendering primitives for circular-sector formula diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import prepare_geometry_diagram_style_and_background
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_label,
    fmt_measure,
    draw_right_angle_marker,
    pad_bbox,
)
from trace_tasks.tasks.geometry.shared.metadata_serialization import geometry_json_ready
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import BBox, Color, Point, RenderContext, RenderedSectorScene, SectorProblem

DEGREE_SYMBOL = "\N{DEGREE SIGN}"


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> RenderContext:
    """Create a styled analytical geometry canvas for one sector diagram."""

    width = int(params.get("canvas_width", rendering_defaults.get("canvas_width", 760)))
    height = int(params.get("canvas_height", rendering_defaults.get("canvas_height", 560)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=width,
        canvas_height=height,
        allow_dark=True,
        require_grid=False,
    )
    fill_palette: tuple[tuple[Color, Color, Color], ...] = (
        ((92, 158, 236), (114, 204, 164), (25, 91, 168)),
        ((238, 148, 86), (117, 190, 219), (150, 72, 24)),
        ((144, 116, 220), (230, 108, 164), (96, 69, 160)),
        ((94, 188, 138), (238, 184, 82), (32, 119, 76)),
    )
    palette_rng = spawn_rng(int(instance_seed), f"{SCENE_ID}.fill_palette")
    fill_index = int(palette_rng.randrange(len(fill_palette)))
    fill_color, secondary_fill_color, accent_color = fill_palette[fill_index]
    readout_font_family = str(params.get("readout_font_family", rendering_defaults.get("readout_font_family", "roboto")))
    diagram_style_trace = dict(diagram_style_meta)
    diagram_style_trace["readout_font_family"] = readout_font_family
    diagram_style_trace["label_stroke_width_override_px"] = 0
    if isinstance(diagram_style_trace.get("stroke_widths"), dict):
        diagram_style_trace["stroke_widths"] = {
            **dict(diagram_style_trace["stroke_widths"]),
            "label_stroke_px": 0,
        }
    image = background.convert("RGB")
    return RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=width,
        height=height,
        line_color=tuple(int(value) for value in diagram_style.stroke_rgb),
        label_color=tuple(int(value) for value in diagram_style.label_rgb),
        label_stroke_color=tuple(int(value) for value in diagram_style.label_stroke_rgb),
        fill_color=fill_color,
        secondary_fill_color=secondary_fill_color,
        accent_color=accent_color,
        line_width=max(2, int(params.get("line_width", rendering_defaults.get("line_width", 4)))),
        label_stroke_width=0,
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
        fill_style_meta={
            "fill_palette_index": int(fill_index),
            "fill_color": list(fill_color),
            "secondary_fill_color": list(secondary_fill_color),
            "accent_color": list(accent_color),
        },
        scene_transform=LazySceneTransform(
            spawn_rng(int(instance_seed), f"{SCENE_ID}.scene_transform"),
            params=params,
            render_defaults=rendering_defaults,
            canvas_width=int(width),
            canvas_height=int(height),
        ),
    )


def _point_on_circle(center: Point, radius: float, degrees: float) -> Point:
    radians = math.radians(float(degrees))
    return (
        float(center[0]) + float(radius) * math.cos(radians),
        float(center[1]) + float(radius) * math.sin(radians),
    )


def _arc_points(center: Point, radius: float, start_degrees: float, end_degrees: float, *, steps: int = 24) -> tuple[Point, ...]:
    span = (float(end_degrees) - float(start_degrees)) % 360.0
    if span <= 1e-9 and abs(float(end_degrees) - float(start_degrees)) > 1e-9:
        span = 360.0
    if span <= 1e-9:
        return (_point_on_circle(center, radius, start_degrees),)
    count = max(2, int(steps))
    return tuple(
        _point_on_circle(center, radius, float(start_degrees) + span * (index / float(count - 1)))
        for index in range(count)
    )


def _draw_arc_band(ctx: RenderContext, box: BBox, *, start: float, end: float, color: Color, width_extra: int = 2) -> None:
    ctx.draw.arc(
        box,
        start=float(start),
        end=float(end),
        fill=color,
        width=max(5, int(ctx.line_width) + int(width_extra)),
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
    length = math.hypot(dx, dy)
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
    label_center = (
        (float(start[0]) + float(end[0])) / 2.0 + float(label_offset[0]),
        (float(start[1]) + float(end[1])) / 2.0 + float(label_offset[1]),
    )
    return draw_label(ctx, label, label_center, small=True)


def _unit_vector(start: Point, end: Point, *, fallback: Point = (1.0, -1.0)) -> Point:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        fallback_length = max(1e-9, math.hypot(float(fallback[0]), float(fallback[1])))
        return (float(fallback[0]) / fallback_length, float(fallback[1]) / fallback_length)
    return (dx / length, dy / length)


def _clamp_label_center(ctx: RenderContext, point: Point, *, margin: float = 20.0) -> Point:
    return (
        max(float(margin), min(float(ctx.width) - float(margin), float(point[0]))),
        max(float(margin), min(float(ctx.height) - float(margin), float(point[1]))),
    )


def _draw_point_label(
    ctx: RenderContext,
    label: str,
    point: Point,
    *,
    label_center: Point,
) -> BBox:
    """Draw one non-answer construction point label while keeping annotation roles geometric."""

    marker_radius = max(4.0, float(ctx.line_width) + 1.0)
    marker_bbox = (
        float(point[0]) - marker_radius,
        float(point[1]) - marker_radius,
        float(point[0]) + marker_radius,
        float(point[1]) + marker_radius,
    )
    ctx.draw.ellipse(
        marker_bbox,
        fill=ctx.label_color,
        outline=ctx.label_stroke_color,
        width=1,
    )
    font = ctx.small_font
    stroke_width = max(0, int(ctx.label_stroke_width))
    text_bbox = ctx.draw.textbbox((0, 0), str(label), font=font, stroke_width=stroke_width)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    left = float(label_center[0]) - (text_w / 2.0)
    top = float(label_center[1]) - (text_h / 2.0)
    draw_text_traced(
        ctx.draw,
        (left, top),
        str(label),
        font=font,
        fill=ctx.label_color,
        stroke_width=stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role="diagram_point_label",
        required=False,
    )
    return bbox_from_points(
        (
            (marker_bbox[0], marker_bbox[1]),
            (marker_bbox[2], marker_bbox[3]),
            (left, top),
            (left + text_w, top + text_h),
        ),
        width=int(ctx.width),
        height=int(ctx.height),
        pad=3.0,
    )


def _draw_construction_point_labels(
    ctx: RenderContext,
    *,
    center: Point,
    points: Mapping[str, Point],
) -> dict[str, BBox]:
    label_bboxes: dict[str, BBox] = {}
    for label, point in points.items():
        if str(label) == "O":
            label_center = _clamp_label_center(ctx, (float(point[0]) + 19.0, float(point[1]) + 20.0))
        else:
            ux, uy = _unit_vector(center, point)
            label_center = _clamp_label_center(ctx, (float(point[0]) + ux * 24.0, float(point[1]) + uy * 24.0))
        label_bboxes[str(label)] = _draw_point_label(ctx, str(label), point, label_center=label_center)
    return label_bboxes


def _draw_sector_base(ctx: RenderContext, problem: SectorProblem) -> dict[str, Any]:
    """Draw the common sector body and return projected geometry after rotation."""

    values = problem.values
    radius_px = 178.0
    center = (302.0, 306.0)
    start_deg = -142.0
    end_deg = start_deg + float(values.theta_degrees)
    ctx.scene_transform.resolve(
        (
            center,
            (center[0] - radius_px, center[1]),
            (center[0] + radius_px, center[1]),
            (center[0], center[1] - radius_px),
            (center[0], center[1] + radius_px),
        )
    )
    center = ctx.scene_transform.point(center)
    radius_px *= float(ctx.scene_transform.transform.scale)
    start_deg += float(ctx.scene_transform.transform.angle_degrees)
    end_deg += float(ctx.scene_transform.transform.angle_degrees)
    arc_box = (
        center[0] - radius_px,
        center[1] - radius_px,
        center[0] + radius_px,
        center[1] + radius_px,
    )
    ctx.draw.pieslice(
        arc_box,
        start=start_deg,
        end=end_deg,
        fill=ctx.fill_color,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    p0 = _point_on_circle(center, radius_px, start_deg)
    p1 = _point_on_circle(center, radius_px, end_deg)
    ctx.draw.line([center, p0], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([center, p1], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=ctx.line_color)
    _draw_arc_band(ctx, arc_box, start=start_deg, end=end_deg, color=ctx.accent_color, width_extra=3)
    arc_points = _arc_points(center, radius_px, start_deg, end_deg)
    sector_points = (center, *arc_points)
    return {
        "center": center,
        "radius_px": radius_px,
        "start_deg": start_deg,
        "end_deg": end_deg,
        "arc_box": arc_box,
        "p0": p0,
        "p1": p1,
        "construction_points": {
            "O": center,
            "A": p0,
            "B": p1,
        },
        "sector_bbox": bbox_from_points(sector_points, width=ctx.width, height=ctx.height, pad=22.0),
        "arc_bbox": bbox_from_points(arc_points, width=ctx.width, height=ctx.height, pad=28.0),
    }


def _fmt_given(value: float) -> str:
    return f"{float(value):.1f}"


def _draw_visible_measure(ctx: RenderContext, problem: SectorProblem) -> None:
    values = problem.values
    if problem.visible_measure_kind == "arc_length":
        draw_label(ctx, f"arc={_fmt_given(values.arc_length)}", (590.0, 214.0), small=True)
    elif problem.visible_measure_kind == "sector_area":
        draw_label(ctx, f"Area={_fmt_given(values.sector_area)}", (590.0, 214.0), small=True)


def _sector_angle_region_bbox(ctx: RenderContext, center: Point, radius: float, start_deg: float, end_deg: float) -> BBox:
    angle_radius = min(94.0, max(54.0, float(radius) * 0.46))
    angle_points = _arc_points(center, angle_radius, start_deg, end_deg, steps=18)
    return bbox_from_points((center, *angle_points), width=ctx.width, height=ctx.height, pad=14.0)


def _arc_midpoint_degrees(start_deg: float, end_deg: float) -> float:
    span = (float(end_deg) - float(start_deg)) % 360.0
    if span <= 1e-9 and abs(float(end_deg) - float(start_deg)) > 1e-9:
        span = 360.0
    return float(start_deg) + (span / 2.0)


def _angle_label_point(center: Point, radius: float, start_deg: float, end_deg: float) -> Point:
    mid_deg = _arc_midpoint_degrees(float(start_deg), float(end_deg))
    return (
        float(center[0]) + float(radius) * math.cos(math.radians(mid_deg)),
        float(center[1]) + float(radius) * math.sin(math.radians(mid_deg)),
    )


def _draw_angle_arc(
    ctx: RenderContext,
    center: Point,
    radius: float,
    start_deg: float,
    end_deg: float,
    *,
    color: Color,
    pad: float = 12.0,
) -> BBox:
    arc_box = (
        float(center[0]) - float(radius),
        float(center[1]) - float(radius),
        float(center[0]) + float(radius),
        float(center[1]) + float(radius),
    )
    _draw_arc_band(ctx, arc_box, start=float(start_deg), end=float(end_deg), color=color, width_extra=0)
    return bbox_from_points(
        _arc_points(center, float(radius), float(start_deg), float(end_deg), steps=24),
        width=ctx.width,
        height=ctx.height,
        pad=float(pad),
    )


def _draw_sector_angle_cue(
    ctx: RenderContext,
    center: Point,
    radius_px: float,
    start_deg: float,
    end_deg: float,
    *,
    label: str | None,
) -> BBox:
    angle_radius = min(94.0, max(54.0, float(radius_px) * 0.46))
    bbox = _draw_angle_arc(
        ctx,
        center,
        angle_radius,
        start_deg,
        end_deg,
        color=ctx.label_color,
        pad=14.0,
    )
    if label:
        draw_label(ctx, str(label), _angle_label_point(center, angle_radius * 0.7, start_deg, end_deg), small=False)
    return bbox_from_points(
        (center, *_arc_points(center, angle_radius, start_deg, end_deg, steps=18)),
        width=ctx.width,
        height=ctx.height,
        pad=14.0,
    )


def _draw_adjacent_angle_measure(
    ctx: RenderContext,
    center: Point,
    radius_px: float,
    start_deg: float,
    end_deg: float,
    *,
    angle_degrees: int,
) -> BBox:
    angle_radius = min(112.0, max(76.0, float(radius_px) * 0.58))
    bbox = _draw_angle_arc(
        ctx,
        center,
        angle_radius,
        start_deg,
        end_deg,
        color=ctx.secondary_fill_color,
        pad=12.0,
    )
    draw_label(
        ctx,
        f"{int(angle_degrees)}{DEGREE_SYMBOL}",
        _angle_label_point(center, angle_radius + 26.0, start_deg, end_deg),
        small=True,
    )
    return bbox


def render_sector_scene(
    ctx: RenderContext,
    problem: SectorProblem,
) -> RenderedSectorScene:
    """Render a sector diagram without public task/query routing in shared code."""

    values = problem.values
    base = _draw_sector_base(ctx, problem)
    center = base["center"]
    radius_px = float(base["radius_px"])
    start_deg = float(base["start_deg"])
    end_deg = float(base["end_deg"])
    _draw_dimension(
        ctx,
        center,
        base["p0"],
        f"r={fmt_measure(values.radius_units)}",
        label_offset=(-18.0, 24.0),
    )

    annotation_bboxes: dict[str, BBox] = {
        "target_sector_region": base["sector_bbox"],
        "target_arc": base["arc_bbox"],
    }
    construction_points: dict[str, Point] = dict(base["construction_points"])
    _draw_visible_measure(ctx, problem)

    if problem.visible_measure_kind in {"complement_relation", "supplement_relation"}:
        total = int(values.target_angle_total or 0)
        adjacent = int(values.adjacent_angle_degrees or 0)
        target_end = start_deg + float(total)
        p_target = _point_on_circle(center, radius_px, target_end)
        ctx.draw.line([center, p_target], fill=ctx.line_color, width=ctx.line_width)
        if total != 360:
            construction_points["C"] = p_target
        _draw_adjacent_angle_measure(
            ctx,
            center,
            radius_px,
            end_deg,
            target_end,
            angle_degrees=adjacent,
        )
        if total == 90:
            draw_right_angle_marker(
                ctx,
                center,
                arm_a=(math.cos(math.radians(start_deg)), math.sin(math.radians(start_deg))),
                arm_b=(math.cos(math.radians(target_end)), math.sin(math.radians(target_end))),
                side_px=22.0,
                color=ctx.secondary_fill_color,
                width=max(2, ctx.line_width - 1),
            )

    if problem.target_kind in {"sector_angle", "related_angle"} or problem.visible_measure_kind in {
        "complement_relation",
        "supplement_relation",
    }:
        angle_text = "?" if problem.target_kind == "sector_angle" else None
        sector_angle_bbox = _draw_sector_angle_cue(
            ctx,
            center,
            radius_px,
            start_deg,
            end_deg,
            label=angle_text,
        )
        if problem.target_kind == "sector_angle":
            annotation_bboxes["target_sector_angle_region"] = sector_angle_bbox

    if problem.target_kind == "related_angle":
        total = int(values.target_angle_total or 0)
        target_start = end_deg
        target_end = start_deg + float(total)
        target_radius = min(132.0, max(88.0, float(radius_px) * 0.68))
        if total != 360:
            p_target = _point_on_circle(center, radius_px, target_end)
            ctx.draw.line([center, p_target], fill=ctx.line_color, width=ctx.line_width)
            construction_points["C"] = p_target
            if total == 90:
                draw_right_angle_marker(
                    ctx,
                    center,
                    arm_a=(math.cos(math.radians(start_deg)), math.sin(math.radians(start_deg))),
                    arm_b=(math.cos(math.radians(target_end)), math.sin(math.radians(target_end))),
                    side_px=22.0,
                    color=ctx.secondary_fill_color,
                    width=max(2, ctx.line_width - 1),
                )
        annotation_bboxes["target_related_angle_arc"] = _draw_angle_arc(
            ctx,
            center,
            target_radius,
            target_start,
            target_end,
            color=ctx.secondary_fill_color,
            pad=14.0,
        )
        draw_label(
            ctx,
            "?",
            _angle_label_point(
                center,
                target_radius + (28.0 if total == 360 else 24.0),
                target_start,
                target_end,
            ),
            small=False,
        )
    construction_label_bboxes = _draw_construction_point_labels(
        ctx,
        center=center,
        points=construction_points,
    )
    scene_entities = (
        {
            "entity_id": "sector",
            "entity_type": "sector",
            "bbox": bbox_to_list(base["sector_bbox"]),
            "radius": int(values.radius_units),
            "theta_degrees": int(values.theta_degrees),
            "arc_length": float(values.arc_length),
            "sector_area": float(values.sector_area),
        },
    )
    render_map = {
        "coord_space": "pixel",
        "target_bboxes": geometry_json_ready(annotation_bboxes),
        "construction_points": geometry_json_ready(
            {label: [float(point[0]), float(point[1])] for label, point in construction_points.items()}
        ),
        "construction_label_bboxes": geometry_json_ready(construction_label_bboxes),
        "sector_bbox": bbox_to_list(base["sector_bbox"]),
        "arc_bbox": bbox_to_list(base["arc_bbox"]),
        "scene_transform": dict(ctx.scene_transform.metadata()),
    }
    witness = {
        "formula_family": str(problem.formula_family),
        "pi_value": float(math.pi),
        "radius_units": int(values.radius_units),
        "theta_degrees": int(values.theta_degrees),
        "adjacent_angle_degrees": values.adjacent_angle_degrees,
        "angle_from_arc_length": float(values.angle_from_arc_length),
        "angle_from_sector_area": float(values.angle_from_sector_area),
        "arc_length": float(values.arc_length),
        "sector_area": float(values.sector_area),
        "target_angle_total": values.target_angle_total,
        "answer_value": float(problem.answer),
    }
    return RenderedSectorScene(
        image=ctx.image,
        annotation_bboxes=dict(annotation_bboxes),
        scene_entities=scene_entities,
        render_map=render_map,
        witness=witness,
        reasoning_steps=int(problem.reasoning_steps),
    )


__all__ = ["create_render_context", "render_sector_scene"]
