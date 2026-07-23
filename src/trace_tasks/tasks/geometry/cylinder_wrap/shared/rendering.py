"""Identity-free rendering primitives for cylinder-wrap diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.geometry.shared.diagram_style import (
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, bbox_to_list, pad_bbox

from .defaults import MARKER_STYLE_IDS, SCENE_ID, WRAP_STYLE_IDS
from .state import BBox, Color, Point, RenderContext, RenderedCylinderWrapScene, SurfacePathProblem, WrappedMarkProblem


def make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderContext, Dict[str, Any]]:
    """Create one deterministic scene drawing context."""

    width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 820)))
    height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 580)))
    protected = ((205, 70, 52), (30, 126, 185), (38, 150, 95), (238, 182, 47))
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
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"geometry.{SCENE_ID}.font_family",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    font_size = int(params.get("label_font_size", group_default(render_defaults, "label_font_size", 22)))
    small_font_size = int(params.get("small_label_font_size", group_default(render_defaults, "small_label_font_size", 18)))
    tiny_font_size = int(params.get("tiny_label_font_size", group_default(render_defaults, "tiny_label_font_size", 14)))
    style_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_ID}.wrap_style",
    )
    marker_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_ID}.marker_style",
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
        font=load_font(max(12, font_size), bold=True, font_family=font_family),
        small_font=load_font(max(10, small_font_size), bold=True, font_family=font_family),
        tiny_font=load_font(max(8, tiny_font_size), bold=True, font_family=font_family),
        font_family=str(font_family),
        wrap_style_id=str(WRAP_STYLE_IDS[int(style_index) % len(WRAP_STYLE_IDS)]),
        marker_style_id=str(MARKER_STYLE_IDS[int(marker_index) % len(MARKER_STYLE_IDS)]),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(style_meta),
        "font_asset_version": font_asset_version(),
        "font_family": font_record.to_trace(),
        "wrap_style_id": str(ctx.wrap_style_id),
        "marker_style_id": str(ctx.marker_style_id),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "tiny_label_font_size": int(tiny_font_size),
    }
    return ctx, render_meta


def _draw_text(
    ctx: RenderContext,
    text: str,
    center: Point,
    *,
    font: Any | None = None,
    fill: Color | None = None,
    stroke_width: int = 1,
) -> BBox:
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


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    dash: float = 8.0,
    gap: float = 6.0,
) -> None:
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


def _draw_arrow_line(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    arrow_size: float = 10.0,
) -> None:
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    draw.line([(sx, sy), (ex, ey)], fill=fill, width=max(1, int(width)))
    angle = math.atan2(ey - sy, ex - sx)
    for sign in (-1.0, 1.0):
        head_angle = angle + (sign * math.radians(150.0))
        hx = ex + (float(arrow_size) * math.cos(head_angle))
        hy = ey + (float(arrow_size) * math.sin(head_angle))
        draw.line([(ex, ey), (hx, hy)], fill=fill, width=max(1, int(width)))


def _draw_arc_arrow(
    ctx: RenderContext,
    center: Point,
    *,
    radius: float,
    start_angle_degrees: float,
    end_angle_degrees: float,
    fill: Color,
    width: int,
) -> None:
    points: list[Point] = []
    for step in range(20):
        t = float(step) / 19.0
        angle = math.radians(float(start_angle_degrees) + ((float(end_angle_degrees) - float(start_angle_degrees)) * t))
        points.append(
            (
                float(center[0]) + (float(radius) * math.cos(angle)),
                float(center[1]) - (float(radius) * math.sin(angle)),
            )
        )
    ctx.draw.line(points, fill=fill, width=max(1, int(width)), joint="curve")
    if len(points) >= 2:
        _draw_arrow_line(ctx.draw, points[-2], points[-1], fill=fill, width=max(1, int(width)), arrow_size=9.0)


def _draw_marker(ctx: RenderContext, center: Point, *, radius: float, color: Color) -> BBox:
    x, y = float(center[0]), float(center[1])
    r = float(radius)
    if ctx.marker_style_id == "diamond":
        points = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        ctx.draw.polygon(points, fill=ctx.panel_fill, outline=color)
        ctx.draw.line(points + [points[0]], fill=color, width=max(2, ctx.line_width))
    elif ctx.marker_style_id == "square":
        ctx.draw.rectangle(
            (x - r, y - r, x + r, y + r),
            fill=ctx.panel_fill,
            outline=color,
            width=max(2, ctx.line_width),
        )
    else:
        ctx.draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=ctx.panel_fill,
            outline=color,
            width=max(2, ctx.line_width),
        )
        if ctx.marker_style_id == "target":
            ctx.draw.ellipse(
                (x - (r * 0.45), y - (r * 0.45), x + (r * 0.45), y + (r * 0.45)),
                outline=color,
                width=2,
            )
    ctx.draw.line([(x - r * 0.6, y), (x + r * 0.6, y)], fill=color, width=2)
    ctx.draw.line([(x, y - r * 0.6), (x, y + r * 0.6)], fill=color, width=2)
    return pad_bbox((x - r, y - r, x + r, y + r), 4.0, width=ctx.width, height=ctx.height)


def _draw_panel(ctx: RenderContext, bbox: BBox, *, fill: Color | None = None, width: int = 3) -> BBox:
    ctx.draw.rounded_rectangle(
        tuple(float(v) for v in bbox),
        radius=8,
        fill=fill or ctx.panel_alt_fill,
        outline=ctx.panel_border,
        width=max(1, int(width)),
    )
    return pad_bbox(bbox, 2.0, width=ctx.width, height=ctx.height)


def _union_bbox(*bboxes: Sequence[float]) -> BBox:
    valid = [tuple(float(v) for v in bbox) for bbox in bboxes if len(tuple(bbox)) == 4]
    if not valid:
        raise ValueError("cannot union empty bbox set")
    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def _draw_cylinder_illustration(ctx: RenderContext, bbox: BBox) -> BBox:
    left, top, right, bottom = (float(v) for v in bbox)
    cx = (left + right) / 2.0
    rx = (right - left) / 2.0
    ry = max(22.0, (bottom - top) * 0.105)
    body_top = top + ry
    body_bottom = bottom - ry
    ctx.draw.rectangle((left, body_top, right, body_bottom), fill=ctx.panel_alt_fill)
    ctx.draw.line([(left, body_top), (left, body_bottom)], fill=ctx.secondary_color, width=3)
    ctx.draw.line([(right, body_top), (right, body_bottom)], fill=ctx.secondary_color, width=3)
    ctx.draw.ellipse((left, top, right, top + (2.0 * ry)), fill=ctx.panel_fill, outline=ctx.line_color, width=3)
    ctx.draw.arc((left, bottom - (2.0 * ry), right, bottom), start=0, end=180, fill=ctx.line_color, width=3)
    _draw_dashed_line(ctx.draw, (cx, body_top), (cx, body_bottom), fill=ctx.guide_color, width=2, dash=7, gap=6)
    path_points: list[Point] = []
    for step in range(72):
        t = float(step) / 71.0
        y = body_top + (body_bottom - body_top) * t
        x = cx + (rx * 0.72 * math.sin((2.0 * math.pi * t) - 0.45))
        path_points.append((x, y))
    ctx.draw.line(path_points, fill=ctx.accent_color, width=max(4, ctx.line_width + 1), joint="curve")
    for point in (path_points[0], path_points[-1]):
        ctx.draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=ctx.accent_color)
    return pad_bbox((left, top, right, bottom), 5.0, width=ctx.width, height=ctx.height)


def render_surface_path_scene(
    ctx: RenderContext,
    problem: SurfacePathProblem,
    *,
    layout_seed: int,
) -> RenderedCylinderWrapScene:
    """Render an unwrapped cylinder side with a marked diagonal path."""

    layout_rng = spawn_rng(int(layout_seed), f"{SCENE_ID}.surface_layout")
    jitter_x = float(layout_rng.randint(-18, 18))
    jitter_y = float(layout_rng.randint(-10, 12))
    cylinder_bbox = (
        74.0 + jitter_x,
        118.0 + jitter_y,
        294.0 + jitter_x,
        458.0 + jitter_y,
    )
    net_left = 366.0 + jitter_x
    net_top = 128.0 + jitter_y
    net_width = 350.0
    net_height = 278.0
    if ctx.wrap_style_id in {"drafting_strip", "workshop_sheet"}:
        net_width = 328.0
        net_height = 292.0
    net_bbox = (net_left, net_top, net_left + net_width, net_top + net_height)
    cylinder_visible_bbox = _draw_cylinder_illustration(ctx, cylinder_bbox)
    panel_bbox = _draw_panel(ctx, net_bbox, fill=ctx.panel_fill)
    left, top, right, bottom = net_bbox
    seam_count = 4
    for idx in range(1, seam_count):
        x = left + (net_width * float(idx) / float(seam_count))
        _draw_dashed_line(ctx.draw, (x, top), (x, bottom), fill=ctx.guide_color, width=1, dash=7, gap=8)
    start = (left + 28.0, bottom - 28.0)
    end = (right - 28.0, top + 28.0)
    ctx.draw.line([start, end], fill=ctx.accent_color, width=max(5, ctx.line_width + 2))
    for point in (start, end):
        ctx.draw.ellipse((point[0] - 7.0, point[1] - 7.0, point[0] + 7.0, point[1] + 7.0), fill=ctx.accent_color)
    path_label_bbox = _draw_text(
        ctx,
        "?",
        ((start[0] + end[0]) / 2.0 + 18.0, (start[1] + end[1]) / 2.0 - 20.0),
        font=ctx.font,
        fill=ctx.secondary_accent_color,
    )
    path_bbox = bbox_from_points((start, end), width=ctx.width, height=ctx.height, pad=20.0)
    bottom_dim_y = bottom + 38.0
    ctx.draw.line([(left + 28.0, bottom_dim_y), (right - 28.0, bottom_dim_y)], fill=ctx.secondary_color, width=2)
    ctx.draw.line([(left + 28.0, bottom_dim_y - 8.0), (left + 28.0, bottom_dim_y + 8.0)], fill=ctx.secondary_color, width=2)
    ctx.draw.line([(right - 28.0, bottom_dim_y - 8.0), (right - 28.0, bottom_dim_y + 8.0)], fill=ctx.secondary_color, width=2)
    circumference_bbox = _draw_text(
        ctx,
        f"C={int(problem.circumference)}",
        ((left + right) / 2.0, bottom_dim_y + 22.0),
        font=ctx.small_font,
    )
    circumference_dimension_bbox = _union_bbox(
        bbox_from_points(
            (
                (left + 28.0, bottom_dim_y - 8.0),
                (right - 28.0, bottom_dim_y + 8.0),
            ),
            width=ctx.width,
            height=ctx.height,
            pad=4.0,
        ),
        circumference_bbox,
    )
    right_dim_x = right + 34.0
    ctx.draw.line([(right_dim_x, top + 28.0), (right_dim_x, bottom - 28.0)], fill=ctx.secondary_color, width=2)
    ctx.draw.line([(right_dim_x - 8.0, top + 28.0), (right_dim_x + 8.0, top + 28.0)], fill=ctx.secondary_color, width=2)
    ctx.draw.line([(right_dim_x - 8.0, bottom - 28.0), (right_dim_x + 8.0, bottom - 28.0)], fill=ctx.secondary_color, width=2)
    height_bbox = _draw_text(
        ctx,
        f"h={int(problem.height)}",
        (right_dim_x + 31.0, (top + bottom) / 2.0),
        font=ctx.small_font,
    )
    height_dimension_bbox = _union_bbox(
        bbox_from_points(
            (
                (right_dim_x - 8.0, top + 28.0),
                (right_dim_x + 8.0, bottom - 28.0),
            ),
            width=ctx.width,
            height=ctx.height,
            pad=4.0,
        ),
        height_bbox,
    )
    scene_entities = (
        {
            "entity_id": "cylinder",
            "entity_type": "cylinder",
            "bbox": bbox_to_list(cylinder_visible_bbox),
        },
        {
            "entity_id": "unwrapped_side_net",
            "entity_type": "rectangle_net",
            "circumference": int(problem.circumference),
            "height": int(problem.height),
            "bbox": bbox_to_list(panel_bbox),
        },
        {
            "entity_id": "marked_surface_path",
            "entity_type": "surface_path",
            "path_length": int(problem.path_length),
            "bbox": bbox_to_list(path_bbox),
        },
    )
    return RenderedCylinderWrapScene(
        image=ctx.image,
        answer=int(problem.path_length),
        answer_type="integer",
        annotation_type="bbox_map",
        annotation_value={
            "marked_surface_path": bbox_to_list(path_bbox),
            "circumference_dimension": bbox_to_list(circumference_dimension_bbox),
            "height_dimension": bbox_to_list(height_dimension_bbox),
        },
        annotation_roles=("marked_surface_path", "circumference_dimension", "height_dimension"),
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "cylinder_bbox": bbox_to_list(cylinder_visible_bbox),
            "net_bbox": bbox_to_list(panel_bbox),
            "surface_path_endpoints_px": [
                [round(start[0], 3), round(start[1], 3)],
                [round(end[0], 3), round(end[1], 3)],
            ],
            "path_label_bbox": bbox_to_list(path_label_bbox),
            "circumference_label_bbox": bbox_to_list(circumference_bbox),
            "height_label_bbox": bbox_to_list(height_bbox),
        },
        witness={
            "geometry_kind": "cylinder_side_unwrap",
            "circumference": int(problem.circumference),
            "height": int(problem.height),
            "path_length": int(problem.path_length),
            "answer_value": int(problem.path_length),
        },
    )


def _candidate_point(center: Point, radius: float, index: int, option_count: int) -> Point:
    angle = (2.0 * math.pi * (float(index) + 0.5)) / float(option_count)
    return (
        float(center[0]) + (float(radius) * math.cos(angle)),
        float(center[1]) - (float(radius) * math.sin(angle)),
    )


def render_wrapped_mark_scene(
    ctx: RenderContext,
    problem: WrappedMarkProblem,
    *,
    layout_seed: int,
) -> RenderedCylinderWrapScene:
    """Render an unwrapped strip and top-view rim candidate positions."""

    layout_rng = spawn_rng(int(layout_seed), f"{SCENE_ID}.mark_layout")
    jitter_x = float(layout_rng.randint(-20, 18))
    jitter_y = float(layout_rng.randint(-12, 12))
    strip_left = 82.0 + jitter_x
    strip_top = 88.0 + jitter_y
    strip_width = 656.0
    strip_height = 132.0
    strip_bbox = (strip_left, strip_top, strip_left + strip_width, strip_top + strip_height)
    strip_panel_bbox = _draw_panel(ctx, strip_bbox, fill=ctx.panel_fill)
    seam_x = strip_left
    _draw_dashed_line(
        ctx.draw,
        (seam_x, strip_top - 10.0),
        (seam_x, strip_top + strip_height + 10.0),
        fill=ctx.secondary_color,
        width=2,
    )
    _draw_text(ctx, "seam", (seam_x + 32.0, strip_top - 18.0), font=ctx.tiny_font, stroke_width=1)
    _draw_arrow_line(
        ctx.draw,
        (strip_left + 112.0, strip_top - 20.0),
        (strip_left + 232.0, strip_top - 20.0),
        fill=ctx.secondary_color,
        width=2,
        arrow_size=8.0,
    )
    for idx in range(1, int(problem.option_count)):
        x = strip_left + (strip_width * float(idx) / float(problem.option_count))
        _draw_dashed_line(ctx.draw, (x, strip_top), (x, strip_top + strip_height), fill=ctx.guide_color, width=1, dash=6, gap=8)
    target_fraction = (float(problem.target_index) + 0.5) / float(problem.option_count)
    mark_x = strip_left + (strip_width * target_fraction)
    mark_y = strip_top + (strip_height * 0.52)
    source_bbox = _draw_marker(ctx, (mark_x, mark_y), radius=15.0, color=ctx.accent_color)
    _draw_text(ctx, "mark", (mark_x, strip_top + strip_height + 22.0), font=ctx.tiny_font, fill=ctx.accent_color, stroke_width=1)

    circle_center = (float(ctx.width) / 2.0 + jitter_x * 0.15, 392.0 + jitter_y)
    radius = 118.0
    circle_bbox = (
        circle_center[0] - radius,
        circle_center[1] - radius,
        circle_center[0] + radius,
        circle_center[1] + radius,
    )
    ctx.draw.ellipse(circle_bbox, fill=ctx.panel_alt_fill, outline=ctx.line_color, width=3)
    seam_point = (circle_center[0] + radius, circle_center[1])
    ctx.draw.line([circle_center, seam_point], fill=ctx.secondary_color, width=2)
    ctx.draw.ellipse((seam_point[0] - 5, seam_point[1] - 5, seam_point[0] + 5, seam_point[1] + 5), fill=ctx.secondary_color)
    _draw_text(ctx, "seam", (seam_point[0] + 36.0, seam_point[1]), font=ctx.tiny_font, stroke_width=1)
    _draw_arc_arrow(
        ctx,
        circle_center,
        radius=radius + 22.0,
        start_angle_degrees=8.0,
        end_angle_degrees=62.0,
        fill=ctx.secondary_color,
        width=2,
    )
    option_entities: list[Dict[str, Any]] = []
    selected_center: Point | None = None
    for idx, label in enumerate(problem.option_labels):
        point = _candidate_point(circle_center, radius, idx, int(problem.option_count))
        option_bbox = _draw_marker(ctx, point, radius=12.0, color=ctx.secondary_accent_color)
        label_point = _candidate_point(circle_center, radius + 35.0, idx, int(problem.option_count))
        label_bbox = _draw_text(ctx, str(label), label_point, font=ctx.small_font, stroke_width=1)
        combined_bbox = (
            min(option_bbox[0], label_bbox[0]),
            min(option_bbox[1], label_bbox[1]),
            max(option_bbox[2], label_bbox[2]),
            max(option_bbox[3], label_bbox[3]),
        )
        option_entities.append(
            {
                "entity_id": f"candidate_{idx}",
                "entity_type": "rim_candidate",
                "candidate_index": int(idx),
                "label": str(label),
                "center_px": [round(point[0], 3), round(point[1], 3)],
                "bbox": bbox_to_list(combined_bbox),
            }
        )
        if int(idx) == int(problem.target_index):
            selected_center = point
    if selected_center is None:
        raise ValueError("selected rim candidate was not rendered")
    circle_visible_bbox = pad_bbox(circle_bbox, 55.0, width=ctx.width, height=ctx.height)
    scene_entities = (
        {
            "entity_id": "wrapper_strip",
            "entity_type": "unwrapped_strip",
            "option_count": int(problem.option_count),
            "bbox": bbox_to_list(strip_panel_bbox),
        },
        {
            "entity_id": "source_mark",
            "entity_type": "mark_on_strip",
            "target_index": int(problem.target_index),
            "bbox": bbox_to_list(source_bbox),
        },
        {
            "entity_id": "rim_view",
            "entity_type": "cylinder_top_view",
            "bbox": bbox_to_list(circle_visible_bbox),
        },
        *tuple(option_entities),
    )
    return RenderedCylinderWrapScene(
        image=ctx.image,
        answer=str(problem.answer_label),
        answer_type="option_letter",
        annotation_type="point_map",
        annotation_value={
            "source_strip_mark": [round(mark_x, 3), round(mark_y, 3)],
            "matching_rim_candidate": [round(selected_center[0], 3), round(selected_center[1], 3)],
        },
        annotation_roles=("source_strip_mark", "matching_rim_candidate"),
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "strip_bbox": bbox_to_list(strip_panel_bbox),
            "rim_bbox": bbox_to_list(circle_visible_bbox),
            "source_mark_center_px": [round(mark_x, 3), round(mark_y, 3)],
            "selected_candidate_center_px": [round(selected_center[0], 3), round(selected_center[1], 3)],
        },
        witness={
            "geometry_kind": "wrapped_mark_position",
            "option_count": int(problem.option_count),
            "target_index": int(problem.target_index),
            "option_labels": list(problem.option_labels),
            "answer_value": str(problem.answer_label),
        },
    )


__all__ = [
    "make_render_context",
    "render_surface_path_scene",
    "render_wrapped_mark_scene",
]
