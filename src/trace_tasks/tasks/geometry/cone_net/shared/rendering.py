"""Rendering primitives for cone-sector net diagrams."""

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
    pad_bbox,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .measurements import fmt_measure
from .state import BBox, Color, ConeNetDiagramSpec, Point, RenderContext, RenderedConeNetScene


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    random_namespace: str,
) -> tuple[RenderContext, Dict[str, Any]]:
    """Resolve deterministic style, font, canvas, and color state."""

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
    palettes: Tuple[Tuple[Color, Color, Color], ...] = (
        (
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.accent_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.muted_fill_rgb),
            tuple(int(v) for v in diagram_style.secondary_accent_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.muted_fill_rgb),
            tuple(int(v) for v in diagram_style.option_fill_rgb),
            tuple(int(v) for v in diagram_style.highlight_rgb),
        ),
        (
            tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
            tuple(int(v) for v in diagram_style.panel_fill_rgb),
            tuple(int(v) for v in diagram_style.guide_rgb),
        ),
    )
    palette_rng = spawn_rng(int(instance_seed), f"geometry.{SCENE_ID}.palette")
    fill_color, cone_fill_color, accent_color = uniform_choice(palette_rng, palettes)
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
        fill_color=fill_color,
        cone_fill_color=cone_fill_color,
        accent_color=accent_color,
        line_width=max(2, int(line_width)),
        font=load_font(max(12, int(font_size)), bold=False, font_family=str(font_family)),
        small_font=load_font(max(10, int(small_font_size)), bold=False, font_family=str(font_family)),
        label_stroke_width=max(0, int(label_stroke_width)),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "label_stroke_width": int(ctx.label_stroke_width),
        "fill_color": list(fill_color),
        "cone_fill_color": list(cone_fill_color),
        "accent_color": list(accent_color),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_resolution),
        "font_family": font_record.to_trace(),
        "font_asset_version": font_asset_version(),
    }
    return ctx, render_meta


def _draw_dashed_line(
    ctx: RenderContext,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    dash: float = 12.0,
    gap: float = 8.0,
) -> None:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return
    ux = dx / length
    uy = dy / length
    pos = 0.0
    while pos < length:
        segment_end = min(length, pos + dash)
        ctx.draw.line(
            [
                (float(start[0]) + ux * pos, float(start[1]) + uy * pos),
                (float(start[0]) + ux * segment_end, float(start[1]) + uy * segment_end),
            ],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def _draw_dimension(
    ctx: RenderContext,
    start: Point,
    end: Point,
    label: str,
    *,
    label_offset: Point = (0.0, 0.0),
    label_center: Point | None = None,
    color: Color | None = None,
) -> BBox:
    draw_color = color if color is not None else ctx.label_color
    ctx.draw.line([start, end], fill=draw_color, width=max(2, ctx.line_width - 1))
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
                fill=draw_color,
                width=max(2, ctx.line_width - 1),
            )
    if label_center is not None:
        center = (float(label_center[0]), float(label_center[1]))
    else:
        center = (
            (float(start[0]) + float(end[0])) / 2.0 + float(label_offset[0]),
            (float(start[1]) + float(end[1])) / 2.0 + float(label_offset[1]),
        )
    return draw_label(ctx, label, center, small=True)


def _segment_label_center(
    start: Point,
    end: Point,
    *,
    along: float,
    normal_offset: float,
    side: float = 1.0,
) -> Point:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
    nx = -dy / length
    ny = dx / length
    return (
        float(start[0]) + dx * float(along) + nx * float(normal_offset) * float(side),
        float(start[1]) + dy * float(along) + ny * float(normal_offset) * float(side),
    )


def _point_on_circle(center: Point, radius: float, degrees: float) -> Point:
    radians = math.radians(float(degrees))
    return (
        float(center[0]) + float(radius) * math.cos(radians),
        float(center[1]) + float(radius) * math.sin(radians),
    )


def _draw_arrow(ctx: RenderContext, start: Point, end: Point) -> None:
    ctx.draw.line([start, end], fill=ctx.accent_color, width=max(3, ctx.line_width - 1))
    angle = math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
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


def _draw_target_label(ctx: RenderContext, spec: ConeNetDiagramSpec, base_center: Point, base_right: Point, apex: Point) -> BBox:
    if spec.target_label_anchor == "radius_segment":
        center = ((float(base_center[0]) + float(base_right[0])) / 2.0, float(base_center[1]) + 42.0)
        return draw_label(ctx, spec.target_label, center, small=True)
    if spec.target_label_anchor == "height_segment":
        center = (float(base_center[0]) + 44.0, (float(apex[1]) + float(base_center[1])) / 2.0)
        return draw_label(ctx, spec.target_label, center, small=True)
    raise ValueError(f"unsupported cone-net target label anchor: {spec.target_label_anchor}")


def render_cone_net_scene(ctx: RenderContext, spec: ConeNetDiagramSpec) -> RenderedConeNetScene:
    """Draw one sector net, the folded cone, and the requested missing measure."""

    sector_center = (250.0, 318.0)
    sector_radius_px = 166.0
    start_deg = -136.0
    end_deg = start_deg + float(spec.theta_degrees)
    mid_deg = (start_deg + end_deg) / 2.0
    arc_box = (
        sector_center[0] - sector_radius_px,
        sector_center[1] - sector_radius_px,
        sector_center[0] + sector_radius_px,
        sector_center[1] + sector_radius_px,
    )
    p0 = _point_on_circle(sector_center, sector_radius_px, start_deg)
    p1 = _point_on_circle(sector_center, sector_radius_px, end_deg)
    ctx.draw.pieslice(
        arc_box,
        start=start_deg,
        end=end_deg,
        fill=ctx.fill_color,
        outline=ctx.line_color,
        width=ctx.line_width,
    )
    ctx.draw.line([sector_center, p0], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([sector_center, p1], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(
        (sector_center[0] - 4.0, sector_center[1] - 4.0, sector_center[0] + 4.0, sector_center[1] + 4.0),
        fill=ctx.line_color,
    )
    ctx.draw.arc(arc_box, start=start_deg, end=end_deg, fill=ctx.accent_color, width=max(5, ctx.line_width + 4))

    label_bboxes: Dict[str, BBox] = {}
    label_bboxes["slant_height"] = _draw_dimension(
        ctx,
        sector_center,
        p0,
        f"l={fmt_measure(spec.slant_height)}",
        label_center=_segment_label_center(sector_center, p0, along=0.46, normal_offset=42.0, side=-1.0),
    )
    angle_radius = 58.0
    angle_box = (
        sector_center[0] - angle_radius,
        sector_center[1] - angle_radius,
        sector_center[0] + angle_radius,
        sector_center[1] + angle_radius,
    )
    ctx.draw.arc(angle_box, start=start_deg, end=end_deg, fill=ctx.accent_color, width=max(3, ctx.line_width - 1))
    angle_center = (
        sector_center[0] + 82.0 * math.cos(math.radians(mid_deg)),
        sector_center[1] + 82.0 * math.sin(math.radians(mid_deg)),
    )
    label_bboxes["sector_angle"] = draw_label(ctx, f"θ={spec.theta_degrees}°", angle_center, small=True)

    cone_apex = (620.0, 112.0)
    cone_base_center = (620.0, 428.0)
    cone_base_left = (508.0, 428.0)
    cone_base_right = (732.0, 428.0)
    cone_base_box = (
        cone_base_left[0],
        cone_base_center[1] - 22.0,
        cone_base_right[0],
        cone_base_center[1] + 22.0,
    )
    ctx.draw.polygon((cone_apex, cone_base_left, cone_base_right), fill=ctx.cone_fill_color, outline=ctx.line_color)
    ctx.draw.line([cone_apex, cone_base_left], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([cone_apex, cone_base_right], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.ellipse(cone_base_box, outline=ctx.accent_color, width=ctx.line_width)
    ctx.draw.arc(cone_base_box, start=0, end=180, fill=ctx.accent_color, width=ctx.line_width + 1)
    _draw_dashed_line(ctx, cone_apex, cone_base_center, fill=ctx.line_color, width=max(2, ctx.line_width - 1))
    ctx.draw.line([cone_base_center, cone_base_right], fill=ctx.accent_color, width=max(2, ctx.line_width - 1))
    marker = 16.0
    ctx.draw.line(
        [
            (cone_base_center[0], cone_base_center[1] - marker),
            (cone_base_center[0] + marker, cone_base_center[1] - marker),
            (cone_base_center[0] + marker, cone_base_center[1]),
        ],
        fill=ctx.line_color,
        width=2,
    )
    label_bboxes["cone_slant_height"] = draw_label(
        ctx,
        f"l={fmt_measure(spec.slant_height)}",
        _segment_label_center(cone_apex, cone_base_right, along=0.52, normal_offset=46.0, side=-1.0),
        small=True,
    )
    _draw_arrow(ctx, (432.0, 258.0), (510.0, 352.0))
    label_bboxes["target"] = _draw_target_label(ctx, spec, cone_base_center, cone_base_right, cone_apex)

    all_points: Dict[str, Point] = {
        "S": sector_center,
        "P": p0,
        "Q": p1,
        "A": cone_apex,
        "C": cone_base_center,
        "R": cone_base_right,
    }
    point_offsets: Dict[str, Point] = {
        "S": (-20.0, 22.0),
        "P": (-18.0, -16.0),
        "Q": (18.0, 18.0),
        "A": (0.0, -20.0),
        "C": (-24.0, 20.0),
        "R": (24.0, 2.0),
    }
    annotation_points = {role: all_points[role] for role in spec.annotation_roles}
    point_label_bboxes: Dict[str, BBox] = {}
    for label, point in annotation_points.items():
        px, py = float(point[0]), float(point[1])
        ctx.draw.ellipse((px - 3.5, py - 3.5, px + 3.5, py + 3.5), fill=ctx.line_color)
        ox, oy = point_offsets[str(label)]
        point_label_bboxes[str(label)] = draw_label(ctx, str(label), (px + float(ox), py + float(oy)), small=True)

    sector_bbox = pad_bbox(arc_box, 8.0, width=ctx.width, height=ctx.height)
    cone_bbox = bbox_from_points((cone_apex, cone_base_left, cone_base_right), width=ctx.width, height=ctx.height, pad=24.0)
    scene_entities = (
        {
            "entity_id": "sector_net",
            "entity_type": "sector",
            "bbox": bbox_to_list(sector_bbox),
            "radius_units": int(spec.slant_height),
            "theta_degrees": int(spec.theta_degrees),
            "arc_length_units": float(spec.arc_length),
        },
        {
            "entity_id": "folded_cone",
            "entity_type": "cone",
            "bbox": bbox_to_list(cone_bbox),
            "slant_height_units": int(spec.slant_height),
            "base_radius_units": float(spec.base_radius),
            "height_units": float(spec.cone_height),
        },
    )
    measurements = {
        "formula_family": str(spec.formula_family),
        "target_measure": str(spec.target_measure),
        "slant_height": int(spec.slant_height),
        "theta_degrees": int(spec.theta_degrees),
        "arc_length": float(spec.arc_length),
        "base_radius": float(spec.base_radius),
        "cone_height": float(spec.cone_height),
        "net_relation": "sector arc length equals folded cone base circumference",
        "base_radius_relation": "r = theta * l / 360",
        "height_relation": "h^2 + r^2 = l^2",
        "answer_value": float(spec.answer),
        "reasoning_steps": int(spec.reasoning_steps),
    }
    return RenderedConeNetScene(
        image=ctx.image,
        annotation_roles=tuple(spec.annotation_roles),
        annotation_keyed_points=dict(annotation_points),
        label_bboxes=dict(label_bboxes),
        point_label_bboxes=dict(point_label_bboxes),
        scene_entities=scene_entities,
        render_map={
            "sector": {
                "center": [round(sector_center[0], 3), round(sector_center[1], 3)],
                "radius_px": round(sector_radius_px, 3),
                "start_degrees": round(start_deg, 3),
                "end_degrees": round(end_deg, 3),
                "endpoints": [[round(p0[0], 3), round(p0[1], 3)], [round(p1[0], 3), round(p1[1], 3)]],
            },
            "cone": {
                "apex": [round(cone_apex[0], 3), round(cone_apex[1], 3)],
                "base_center": [round(cone_base_center[0], 3), round(cone_base_center[1], 3)],
                "base_left": [round(cone_base_left[0], 3), round(cone_base_left[1], 3)],
                "base_right": [round(cone_base_right[0], 3), round(cone_base_right[1], 3)],
            },
            "construction_points": {
                key: [round(point[0], 3), round(point[1], 3)] for key, point in annotation_points.items()
            },
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "point_label_bboxes": {key: bbox_to_list(value) for key, value in point_label_bboxes.items()},
            "coord_space": "pixel",
        },
        measurements=measurements,
    )


def render_cone_net_with_retries(
    *,
    spec: ConeNetDiagramSpec,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> tuple[RenderedConeNetScene, Dict[str, Any]]:
    """Retry deterministic rendering after rare geometry/layout failures."""

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
            return render_cone_net_scene(ctx, spec), dict(render_meta)
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render cone-net scene") from last_error


__all__ = [
    "create_render_context",
    "render_cone_net_scene",
    "render_cone_net_with_retries",
]
