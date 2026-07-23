"""Rendering primitives for folded-paper diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from PIL import Image, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    bbox_from_points,
    bbox_to_list,
    draw_label,
    fmt_measure,
    pad_bbox,
    round1,
)
from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform
from trace_tasks.tasks.geometry.shared.shape_style import (
    extract_background_anchor_colors,
    sample_geometry_shape_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import SCENE_ID
from .state import (
    BBox,
    Color,
    FoldAnglePlan,
    FoldSegmentPlan,
    Point,
    RenderContext,
    RenderedPaperFoldScene,
)

DEGREE_SYMBOL = chr(176)


def _fmt_angle(value: float) -> str:
    return f"{fmt_measure(value)}{DEGREE_SYMBOL}"


def make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderContext, Dict[str, Any]]:
    """Create a styled render context for one folded-paper diagram."""

    rng = spawn_rng(int(instance_seed), "geometry.paper_fold.render")
    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 760)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 560)))
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(width),
        canvas_height=int(height),
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        namespace_suffix="paper_fold_background",
    )
    bg_color = tuple(int(value) for value in diagram_style.canvas_rgb)
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=render_defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    palette: tuple[tuple[Color, Color, Color, Color], ...] = (
        ((252, 249, 235), (96, 164, 214), (24, 107, 166), (122, 132, 142)),
        ((247, 250, 255), (226, 143, 106), (165, 74, 40), (118, 126, 136)),
        ((250, 247, 255), (126, 155, 226), (74, 86, 162), (121, 126, 140)),
        ((247, 253, 247), (96, 178, 132), (39, 124, 74), (120, 132, 122)),
    )
    palette_rng = spawn_rng(int(instance_seed), "geometry.paper_fold.paper_palette")
    paper_fill, folded_fill, crease_color, dashed_color = uniform_choice(
        palette_rng,
        palette,
    )
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(
        params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18))
    )
    point_font_size = int(
        params.get("point_label_font_size", group_default(render_defaults, "point_label_font_size", 17))
    )
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    ctx = RenderContext(
        rng=rng,
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
        background_color=(int(bg_color[0]), int(bg_color[1]), int(bg_color[2])),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        paper_fill_color=paper_fill,
        folded_fill_color=folded_fill,
        crease_color=crease_color,
        dashed_color=dashed_color,
        line_width=max(2, int(line_width)),
        label_stroke_width=0,
        font=load_font(max(12, int(font_size)), bold=False),
        small_font=load_font(max(10, int(small_font_size)), bold=False),
        point_font=load_font(max(10, int(point_font_size)), bold=False),
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
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_meta),
        "shape_style": shape_style.to_trace_dict(),
        "line_width": int(ctx.line_width),
        "label_bold": False,
        "label_stroke_width": int(ctx.label_stroke_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "point_label_font_size": int(point_font_size),
        "paper_fill_color": list(paper_fill),
        "folded_fill_color": list(folded_fill),
        "crease_color": list(crease_color),
        "dashed_color": list(dashed_color),
    }
    return ctx, render_meta


def _draw_point_label(ctx: RenderContext, label: str, point: Point, offset: Point) -> BBox:
    center = (float(point[0]) + float(offset[0]), float(point[1]) + float(offset[1]))
    stroke_width = max(0, int(ctx.label_stroke_width))
    bbox = ctx.draw.textbbox((0, 0), str(label), font=ctx.point_font, stroke_width=stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - (text_w / 2.0)
    top = float(center[1]) - (text_h / 2.0)
    draw_text_traced(
        ctx.draw,
        (left, top),
        str(label),
        font=ctx.point_font,
        fill=ctx.label_color,
        stroke_width=stroke_width,
        stroke_fill=ctx.label_stroke_color,
        role="readout",
        required=False,
    )
    return pad_bbox((left, top, left + text_w, top + text_h), 3.0, width=ctx.width, height=ctx.height)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
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
    cursor = 0.0
    while cursor < length:
        dash_end = min(length, cursor + float(dash))
        draw.line(
            [
                (float(start[0]) + ux * cursor, float(start[1]) + uy * cursor),
                (float(start[0]) + ux * dash_end, float(start[1]) + uy * dash_end),
            ],
            fill=fill,
            width=width,
        )
        cursor += float(dash) + float(gap)


def _draw_angle_arc(
    ctx: RenderContext,
    origin: Point,
    *,
    start_degrees: float,
    end_degrees: float,
    radius: float,
    label: str,
    color: Color,
    label_radius: float | None = None,
) -> BBox:
    steps = max(12, int(abs(float(end_degrees) - float(start_degrees)) // 4) + 1)
    points: list[Point] = []
    for index in range(steps + 1):
        t = float(index) / float(steps)
        degrees = float(start_degrees) + (float(end_degrees) - float(start_degrees)) * t
        radians = math.radians(degrees)
        points.append(
            (
                float(origin[0]) + float(radius) * math.cos(radians),
                float(origin[1]) + float(radius) * math.sin(radians),
            )
        )
    if len(points) > 1:
        ctx.draw.line(points, fill=color, width=max(3, ctx.line_width - 1), joint="curve")
    mid_degrees = (float(start_degrees) + float(end_degrees)) / 2.0
    label_dist = float(label_radius if label_radius is not None else radius + 22.0)
    label_center = (
        float(origin[0]) + label_dist * math.cos(math.radians(mid_degrees)),
        float(origin[1]) + label_dist * math.sin(math.radians(mid_degrees)),
    )
    return draw_label(ctx, label, label_center, small=True)


def _union_bbox(boxes: tuple[BBox, ...], *, width: int, height: int) -> BBox:
    """Return one padded bbox around multiple already-projected visual cues."""

    x0 = min(float(box[0]) for box in boxes)
    y0 = min(float(box[1]) for box in boxes)
    x1 = max(float(box[2]) for box in boxes)
    y1 = max(float(box[3]) for box in boxes)
    return pad_bbox((x0, y0, x1, y1), 3.0, width=width, height=height)


def _draw_segment_label(
    ctx: RenderContext,
    label: str,
    start: Point,
    end: Point,
    *,
    offset: float,
    small: bool = True,
) -> BBox:
    """Draw a label offset normally from a visible segment."""

    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = max(1e-6, math.hypot(dx, dy))
    normal = (-dy / length, dx / length)
    center = (
        (float(start[0]) + float(end[0])) / 2.0 + normal[0] * float(offset),
        (float(start[1]) + float(end[1])) / 2.0 + normal[1] * float(offset),
    )
    return draw_label(ctx, str(label), center, small=bool(small))


def _draw_right_angle_marker(ctx: RenderContext, vertex: Point, arm_a: Point, arm_b: Point) -> None:
    """Draw a small right-angle marker between two visible arms."""

    def unit_toward(point: Point) -> Point:
        dx = float(point[0]) - float(vertex[0])
        dy = float(point[1]) - float(vertex[1])
        length = max(1e-6, math.hypot(dx, dy))
        return (dx / length, dy / length)

    u = unit_toward(arm_a)
    v = unit_toward(arm_b)
    size = 18.0
    p1 = (float(vertex[0]) + u[0] * size, float(vertex[1]) + u[1] * size)
    p2 = (p1[0] + v[0] * size, p1[1] + v[1] * size)
    p3 = (float(vertex[0]) + v[0] * size, float(vertex[1]) + v[1] * size)
    ctx.draw.line([p1, p2, p3], fill=ctx.crease_color, width=max(2, ctx.line_width - 1))


def render_paper_fold_scene(ctx: RenderContext, plan: FoldAnglePlan) -> RenderedPaperFoldScene:
    """Render the folded-corner angle diagram after final scene transform."""

    geometry = plan.geometry
    margin_x = 94.0
    margin_y = 72.0
    scale = min(
        (float(ctx.width) - 2.0 * margin_x) / geometry.width_units,
        (float(ctx.height) - 170.0) / geometry.height_units,
    )
    paper_w = geometry.width_units * scale
    paper_h = geometry.height_units * scale
    x0 = (float(ctx.width) - paper_w) / 2.0
    y0 = margin_y

    def pt(x_units: float, y_units: float) -> Point:
        return (x0 + float(x_units) * scale, y0 + float(y_units) * scale)

    height = geometry.height_units
    offset = geometry.folded_offset_units
    upper = geometry.upper_segment_units
    crease_top_x = (height * height + offset * offset) / (2.0 * offset)

    a = pt(0.0, 0.0)
    b = pt(geometry.width_units, 0.0)
    c = pt(geometry.width_units, height)
    d = pt(0.0, height)
    e = pt(0.0, upper)
    f = pt(crease_top_x, 0.0)
    p = pt(offset, height)
    ctx.scene_transform.resolve((a, b, c, d, e, f, p))
    a, b, c, d, e, f, p = ctx.scene_transform.points((a, b, c, d, e, f, p))

    paper_bbox = bbox_from_points((a, b, c, d), width=ctx.width, height=ctx.height, pad=0.0)
    ctx.draw.polygon((a, b, c, d), fill=ctx.paper_fill_color)
    ctx.draw.line((a, b, c, d, a), fill=ctx.line_color, width=ctx.line_width)

    overlay = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    folded_fill = (*ctx.folded_fill_color[:3], 88)
    overlay_draw.polygon([e, f, p], fill=folded_fill)
    ctx.image.paste(Image.alpha_composite(ctx.image.convert("RGBA"), overlay).convert("RGB"))
    ctx.draw = ImageDraw.Draw(ctx.image)

    _draw_dashed_line(ctx.draw, a, e, fill=ctx.dashed_color, width=max(2, ctx.line_width - 1))
    _draw_dashed_line(ctx.draw, a, f, fill=ctx.dashed_color, width=max(2, ctx.line_width - 1))
    ctx.draw.line([e, p, f], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([e, f], fill=ctx.crease_color, width=ctx.line_width + 1)
    for point in (a, b, c, d, e, f, p):
        radius = 4.0
        ctx.draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=ctx.label_color,
            outline=ctx.label_stroke_color,
            width=1,
        )

    _draw_point_label(ctx, "A", a, (-14.0, -16.0))
    _draw_point_label(ctx, "B", b, (14.0, -16.0))
    _draw_point_label(ctx, "C", c, (14.0, 16.0))
    _draw_point_label(ctx, "D", d, (-15.0, 17.0))
    _draw_point_label(ctx, "E", e, (-18.0, -2.0))
    _draw_point_label(ctx, "F", f, (0.0, -18.0))
    _draw_point_label(ctx, "P", p, (0.0, 20.0))

    rotation_degrees = float(ctx.scene_transform.transform.angle_degrees)
    radius_scale = float(ctx.scene_transform.transform.scale)
    theta_crease = -math.degrees(math.atan(offset / height)) + rotation_degrees
    theta_folded = -90.0 + 2.0 * (90.0 - math.degrees(math.atan(offset / height))) + rotation_degrees
    theta_paper_edge = 90.0 + rotation_degrees
    known_angle_bbox = _draw_angle_arc(
        ctx,
        e,
        start_degrees=theta_folded,
        end_degrees=theta_paper_edge,
        radius=78.0 * radius_scale,
        label=_fmt_angle(float(plan.params["given_angle_degrees"])),
        color=ctx.crease_color,
        label_radius=106.0 * radius_scale,
    )
    target_upper_bbox = _draw_angle_arc(
        ctx,
        e,
        start_degrees=theta_crease,
        end_degrees=-90.0 + rotation_degrees,
        radius=50.0 * radius_scale,
        label="x",
        color=ctx.crease_color,
        label_radius=70.0 * radius_scale,
    )
    target_lower_bbox = _draw_angle_arc(
        ctx,
        e,
        start_degrees=theta_crease,
        end_degrees=theta_folded,
        radius=42.0 * radius_scale,
        label="x",
        color=ctx.crease_color,
        label_radius=62.0 * radius_scale,
    )
    target_bbox = _union_bbox(
        (target_upper_bbox, target_lower_bbox),
        width=ctx.width,
        height=ctx.height,
    )

    folded_bbox = bbox_from_points((e, f, p), width=ctx.width, height=ctx.height, pad=8.0)
    original_fold_bbox = bbox_from_points((a, e, f), width=ctx.width, height=ctx.height, pad=8.0)
    crease_bbox = bbox_from_points((e, f), width=ctx.width, height=ctx.height, pad=8.0)
    scene_entities = (
        {
            "entity_id": "paper_rectangle",
            "entity_type": "paper_rectangle",
            "bbox": bbox_to_list(paper_bbox),
            "height_units": float(height),
            "width_units": float(geometry.width_units),
        },
        {
            "entity_id": "original_corner",
            "entity_type": "dashed_original_fold_region",
            "bbox": bbox_to_list(original_fold_bbox),
        },
        {
            "entity_id": "folded_corner",
            "entity_type": "folded_flap",
            "bbox": bbox_to_list(folded_bbox),
        },
        {
            "entity_id": "fold_crease",
            "entity_type": "fold_crease",
            "bbox": bbox_to_list(crease_bbox),
        },
    )
    point_map = {
        "A": bbox_to_list(pad_bbox((a[0], a[1], a[0], a[1]), 3.0, width=ctx.width, height=ctx.height)),
        "D": bbox_to_list(pad_bbox((d[0], d[1], d[0], d[1]), 3.0, width=ctx.width, height=ctx.height)),
        "E": bbox_to_list(pad_bbox((e[0], e[1], e[0], e[1]), 3.0, width=ctx.width, height=ctx.height)),
        "F": bbox_to_list(pad_bbox((f[0], f[1], f[0], f[1]), 3.0, width=ctx.width, height=ctx.height)),
        "P": bbox_to_list(pad_bbox((p[0], p[1], p[0], p[1]), 3.0, width=ctx.width, height=ctx.height)),
    }
    annotation_bboxes = {
        "target_angle_cue": target_bbox,
        "given_angle_label": known_angle_bbox,
    }
    return RenderedPaperFoldScene(
        image=ctx.image,
        answer=float(plan.answer),
        annotation_bboxes=dict(annotation_bboxes),
        scene_entities=scene_entities,
        render_map={
            "target_bbox": bbox_to_list(target_bbox),
            "support_bboxes": [bbox_to_list(known_angle_bbox)],
            "paper_bbox": bbox_to_list(paper_bbox),
            "folded_flap_bbox": bbox_to_list(folded_bbox),
            "original_corner_bbox": bbox_to_list(original_fold_bbox),
            "crease_bbox": bbox_to_list(crease_bbox),
            "point_bboxes": point_map,
            "coord_space": "pixel",
        },
        witness={
            "formula_family": str(plan.params["formula_family"]),
            "height_units": float(height),
            "folded_offset_units": float(offset),
            "upper_segment_units": float(round1(geometry.upper_segment_units)),
            "lower_segment_units": float(round1(geometry.lower_segment_units)),
            "half_angle_degrees": float(round1(geometry.half_angle_degrees)),
            "total_angle_degrees": float(round1(geometry.total_angle_degrees)),
            "given_angle_degrees": float(round1(plan.params["given_angle_degrees"])),
            "known_angle_degrees": float(round1(plan.params["known_angle_degrees"])),
            "target_role": "half_angle_x",
        },
        reasoning_steps=int(plan.params["reasoning_steps"]),
    )


def render_paper_fold_segment_scene(ctx: RenderContext, plan: FoldSegmentPlan) -> RenderedPaperFoldScene:
    """Render a folded-corner side-length diagram after final scene transform."""

    geometry = plan.geometry
    margin_x = 94.0
    margin_y = 72.0
    scale = min(
        (float(ctx.width) - 2.0 * margin_x) / geometry.width_units,
        (float(ctx.height) - 170.0) / geometry.height_units,
    )
    paper_w = geometry.width_units * scale
    x0 = (float(ctx.width) - paper_w) / 2.0
    y0 = margin_y

    def pt(x_units: float, y_units: float) -> Point:
        return (x0 + float(x_units) * scale, y0 + float(y_units) * scale)

    a = pt(0.0, 0.0)
    b = pt(geometry.width_units, 0.0)
    c = pt(geometry.width_units, geometry.height_units)
    d = pt(0.0, geometry.height_units)
    e = pt(0.0, float(geometry.leg_ae))
    f = pt(float(geometry.leg_af), 0.0)
    p = pt(float(geometry.folded_point_units[0]), float(geometry.folded_point_units[1]))
    ctx.scene_transform.resolve((a, b, c, d, e, f, p))
    a, b, c, d, e, f, p = ctx.scene_transform.points((a, b, c, d, e, f, p))

    paper_bbox = bbox_from_points((a, b, c, d), width=ctx.width, height=ctx.height, pad=0.0)
    ctx.draw.polygon((a, b, c, d), fill=ctx.paper_fill_color)
    ctx.draw.line((a, b, c, d, a), fill=ctx.line_color, width=ctx.line_width)

    overlay = Image.new("RGBA", ctx.image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    folded_fill = (*ctx.folded_fill_color[:3], 88)
    overlay_draw.polygon([e, f, p], fill=folded_fill)
    ctx.image.paste(Image.alpha_composite(ctx.image.convert("RGBA"), overlay).convert("RGB"))
    ctx.draw = ImageDraw.Draw(ctx.image)

    _draw_dashed_line(ctx.draw, a, e, fill=ctx.dashed_color, width=max(2, ctx.line_width - 1))
    _draw_dashed_line(ctx.draw, a, f, fill=ctx.dashed_color, width=max(2, ctx.line_width - 1))
    ctx.draw.line([e, p, f], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.line([e, f], fill=ctx.crease_color, width=ctx.line_width + 1)
    _draw_right_angle_marker(ctx, a, e, f)

    target_points = (e, p) if plan.case.target_segment == "EP" else (f, p)
    ctx.draw.line(target_points, fill=ctx.crease_color, width=ctx.line_width + 2)

    for point in (a, b, c, d, e, f, p):
        radius = 4.0
        ctx.draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=ctx.label_color,
            outline=ctx.label_stroke_color,
            width=1,
        )

    _draw_point_label(ctx, "A", a, (-14.0, -16.0))
    _draw_point_label(ctx, "B", b, (14.0, -16.0))
    _draw_point_label(ctx, "C", c, (14.0, 16.0))
    _draw_point_label(ctx, "D", d, (-15.0, 17.0))
    _draw_point_label(ctx, "E", e, (-18.0, -2.0))
    _draw_point_label(ctx, "F", f, (0.0, -18.0))
    _draw_point_label(ctx, "P", p, (0.0, 20.0))

    if plan.case.known_leg_segment == "AE":
        known_leg_bbox = _draw_segment_label(
            ctx,
            f"AE={geometry.leg_ae}",
            a,
            e,
            offset=-28.0,
        )
    else:
        known_leg_bbox = _draw_segment_label(
            ctx,
            f"AF={geometry.leg_af}",
            a,
            f,
            offset=-28.0,
        )
    crease_label_bbox = _draw_segment_label(
        ctx,
        f"EF={geometry.crease_ef}",
        e,
        f,
        offset=26.0,
    )
    target_label_bbox = _draw_segment_label(
        ctx,
        f"{plan.case.target_segment}=?",
        target_points[0],
        target_points[1],
        offset=-30.0,
    )

    folded_bbox = bbox_from_points((e, f, p), width=ctx.width, height=ctx.height, pad=8.0)
    original_fold_bbox = bbox_from_points((a, e, f), width=ctx.width, height=ctx.height, pad=8.0)
    crease_bbox = bbox_from_points((e, f), width=ctx.width, height=ctx.height, pad=8.0)
    target_bbox = bbox_from_points(target_points, width=ctx.width, height=ctx.height, pad=8.0)
    point_map = {
        "A": bbox_to_list(pad_bbox((a[0], a[1], a[0], a[1]), 3.0, width=ctx.width, height=ctx.height)),
        "E": bbox_to_list(pad_bbox((e[0], e[1], e[0], e[1]), 3.0, width=ctx.width, height=ctx.height)),
        "F": bbox_to_list(pad_bbox((f[0], f[1], f[0], f[1]), 3.0, width=ctx.width, height=ctx.height)),
        "P": bbox_to_list(pad_bbox((p[0], p[1], p[0], p[1]), 3.0, width=ctx.width, height=ctx.height)),
    }
    scene_entities = (
        {
            "entity_id": "paper_rectangle",
            "entity_type": "paper_rectangle",
            "bbox": bbox_to_list(paper_bbox),
            "height_units": float(geometry.height_units),
            "width_units": float(geometry.width_units),
        },
        {
            "entity_id": "original_corner",
            "entity_type": "dashed_original_fold_region",
            "bbox": bbox_to_list(original_fold_bbox),
        },
        {
            "entity_id": "folded_corner",
            "entity_type": "folded_flap",
            "bbox": bbox_to_list(folded_bbox),
        },
        {
            "entity_id": "fold_crease",
            "entity_type": "fold_crease",
            "bbox": bbox_to_list(crease_bbox),
        },
        {
            "entity_id": "target_folded_segment",
            "entity_type": "target_segment",
            "bbox": bbox_to_list(target_bbox),
            "segment_name": str(plan.case.target_segment),
        },
    )
    return RenderedPaperFoldScene(
        image=ctx.image,
        answer=float(plan.answer),
        annotation_bboxes={},
        annotation_segment=(target_points[0], target_points[1]),
        scene_entities=scene_entities,
        render_map={
            "target_segment": [
                [float(target_points[0][0]), float(target_points[0][1])],
                [float(target_points[1][0]), float(target_points[1][1])],
            ],
            "target_bbox": bbox_to_list(target_bbox),
            "known_leg_label_bbox": bbox_to_list(known_leg_bbox),
            "crease_label_bbox": bbox_to_list(crease_label_bbox),
            "target_label_bbox": bbox_to_list(target_label_bbox),
            "paper_bbox": bbox_to_list(paper_bbox),
            "folded_flap_bbox": bbox_to_list(folded_bbox),
            "original_corner_bbox": bbox_to_list(original_fold_bbox),
            "crease_bbox": bbox_to_list(crease_bbox),
            "point_bboxes": point_map,
            "coord_space": "pixel",
        },
        witness={
            "formula_family": str(plan.params["formula_family"]),
            "leg_ae": int(geometry.leg_ae),
            "leg_af": int(geometry.leg_af),
            "crease_ef": int(geometry.crease_ef),
            "known_leg_segment": str(plan.case.known_leg_segment),
            "target_segment": str(plan.case.target_segment),
            "target_role": "folded_segment_length",
            "pythagorean_unknown_original_segment": str(plan.params["pythagorean_unknown_original_segment"]),
        },
        reasoning_steps=int(plan.params["reasoning_steps"]),
    )


__all__ = ["make_render_context", "render_paper_fold_scene", "render_paper_fold_segment_scene"]
