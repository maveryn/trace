"""Scene-local rendering helpers for bearing-route diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, bbox_to_list, pad_bbox

from .construction import candidate_unit_points
from .measurements import bearing_label
from .projection import fit_points_to_box, project_point
from .spatial_primitives import route_unit_points
from .state import BBox, Color, Point, RenderContext, RenderedBearingScene, RouteCase, SCENE_ID

_ROUTE_STYLE_IDS: Tuple[str, ...] = ("survey_sheet", "navigation_plot", "field_notebook", "compass_card")
_MARKER_STYLES: Tuple[str, ...] = ("ring", "target", "square", "pin")


def make_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    style_namespace: str,
) -> tuple[RenderContext, Dict[str, Any]]:
    """Resolve scene-level style state shared by bearing-route renderers.

    This helper owns only canvas, style, font, and marker choices; objective
    semantics stay in public task files and route construction helpers.
    """
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
        style_profile=GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
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
    route_style_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{style_namespace}.route_style_id",
    )
    marker_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{style_namespace}.marker_style",
    )
    line_width = int(params.get("line_width", group_default(render_defaults, "line_width", 4)))
    ctx = RenderContext(
        image=image,
        draw=ImageDraw.Draw(image),
        width=int(width),
        height=int(height),
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
        line_width=max(2, int(line_width)),
        font=load_font(max(12, font_size), bold=False, font_family=font_family),
        small_font=load_font(max(10, small_font_size), bold=False, font_family=font_family),
        tiny_font=load_font(max(8, tiny_font_size), bold=False, font_family=font_family),
        font_family=str(font_family),
        route_style_id=str(_ROUTE_STYLE_IDS[int(route_style_index) % len(_ROUTE_STYLE_IDS)]),
        marker_style=str(_MARKER_STYLES[int(marker_index) % len(_MARKER_STYLES)]),
    )
    render_meta = {
        "background_style": dict(background_meta),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(style_meta),
        "font_asset_version": font_asset_version(),
        "font_family": font_record.to_trace(),
        "route_style_id": str(ctx.route_style_id),
        "marker_style": str(ctx.marker_style),
        "line_width": int(ctx.line_width),
        "label_font_size": int(font_size),
        "small_label_font_size": int(small_font_size),
        "tiny_label_font_size": int(tiny_font_size),
        "label_stroke_width": 0,
        "font_bold": False,
        "style_namespace": str(style_namespace),
    }
    return ctx, render_meta


def _draw_text(
    ctx: RenderContext,
    text: str,
    center: Point,
    *,
    font: Any | None = None,
    fill: Color | None = None,
    stroke_width: int = 0,
) -> BBox:
    active_font = font or ctx.small_font
    active_fill = fill or ctx.label_color
    bbox = ctx.draw.textbbox((0, 0), str(text), font=active_font, stroke_width=stroke_width)
    text_w = float(bbox[2] - bbox[0])
    text_h = float(bbox[3] - bbox[1])
    left = float(center[0]) - (text_w / 2.0)
    top = float(center[1]) - (text_h / 2.0)
    draw_text_traced(ctx.draw,
        (left, top),
        str(text),
        font=active_font,
        fill=active_fill,
        stroke_width=int(stroke_width),
        stroke_fill=ctx.label_stroke_color,
     role="readout", required=False,)
    return pad_bbox((left, top, left + text_w, top + text_h), 3.0, width=ctx.width, height=ctx.height)


def _draw_panel(ctx: RenderContext, bbox: BBox, *, fill: Color | None = None, radius: int = 8) -> BBox:
    ctx.draw.rounded_rectangle(
        tuple(float(v) for v in bbox),
        radius=int(radius),
        fill=fill or ctx.panel_fill,
        outline=ctx.panel_border,
        width=3,
    )
    return pad_bbox(bbox, 2.0, width=ctx.width, height=ctx.height)


def _resolve_route_projection(
    ctx: RenderContext,
    panel_bbox: BBox,
    unit_points: Tuple[Point, ...],
) -> tuple[float, Point, Dict[str, Any]]:
    """Return a route-unit projection and draw its visible graph-paper grid."""
    if not unit_points:
        raise ValueError("route projection requires at least one point")

    xs = [round(float(point[0])) for point in unit_points]
    ys = [round(float(point[1])) for point in unit_points]
    grid_min_x = int(min(xs)) - 1
    grid_max_x = int(max(xs)) + 1
    grid_min_y = int(min(ys)) - 1
    grid_max_y = int(max(ys)) + 1
    span_x = max(1, grid_max_x - grid_min_x)
    span_y = max(1, grid_max_y - grid_min_y)

    inner = (
        float(panel_bbox[0]) + 46.0,
        float(panel_bbox[1]) + 42.0,
        float(panel_bbox[2]) - 42.0,
        float(panel_bbox[3]) - 42.0,
    )
    cell = min((inner[2] - inner[0]) / float(span_x), (inner[3] - inner[1]) / float(span_y), 42.0)
    grid_w = float(span_x) * float(cell)
    grid_h = float(span_y) * float(cell)
    left = ((inner[0] + inner[2]) / 2.0) - (grid_w / 2.0)
    top = ((inner[1] + inner[3]) / 2.0) - (grid_h / 2.0)
    origin = (left - (float(grid_min_x) * float(cell)), top - (float(grid_min_y) * float(cell)))

    for unit_x in range(int(grid_min_x), int(grid_max_x) + 1):
        x = float(origin[0]) + (float(unit_x) * float(cell))
        ctx.draw.line((x, top, x, top + grid_h), fill=ctx.guide_color, width=1)
    for unit_y in range(int(grid_min_y), int(grid_max_y) + 1):
        y = float(origin[1]) + (float(unit_y) * float(cell))
        ctx.draw.line((left, y, left + grid_w, y), fill=ctx.guide_color, width=1)

    projection_meta = {
        "unit_bounds": {
            "min_x": int(grid_min_x),
            "max_x": int(grid_max_x),
            "min_y": int(grid_min_y),
            "max_y": int(grid_max_y),
        },
        "unit_scale_px": round(float(cell), 3),
        "unit_origin_px": [round(float(origin[0]), 3), round(float(origin[1]), 3)],
        "projection_bbox": bbox_to_list((left, top, left + grid_w, top + grid_h)),
        "visible_grid": True,
        "grid_unit": "one_square_equals_one_step",
        "grid_cell_px": round(float(cell), 3),
        "unit_role": "graph_paper_route_step_layout",
    }
    return float(cell), origin, projection_meta


def _draw_arrow_line(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    *,
    fill: Color,
    width: int,
    arrow_size: float = 12.0,
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


def _draw_marker(ctx: RenderContext, center: Point, *, radius: float, color: Color, fill: Color | None = None) -> BBox:
    x, y = float(center[0]), float(center[1])
    r = float(radius)
    marker_fill = fill or ctx.panel_alt_fill
    if ctx.marker_style == "square":
        ctx.draw.rectangle((x - r, y - r, x + r, y + r), fill=marker_fill, outline=color, width=max(2, ctx.line_width))
    elif ctx.marker_style == "pin":
        ctx.draw.ellipse((x - r, y - r, x + r, y + r), fill=marker_fill, outline=color, width=max(2, ctx.line_width))
        ctx.draw.polygon([(x, y + r + 8), (x - r * 0.55, y + r * 0.1), (x + r * 0.55, y + r * 0.1)], fill=color)
    else:
        ctx.draw.ellipse((x - r, y - r, x + r, y + r), fill=marker_fill, outline=color, width=max(2, ctx.line_width))
        if ctx.marker_style == "target":
            ctx.draw.ellipse((x - (r * 0.45), y - (r * 0.45), x + (r * 0.45), y + (r * 0.45)), outline=color, width=2)
    ctx.draw.line([(x - r * 0.55, y), (x + r * 0.55, y)], fill=color, width=2)
    ctx.draw.line([(x, y - r * 0.55), (x, y + r * 0.55)], fill=color, width=2)
    return pad_bbox((x - r, y - r, x + r, y + r + (8.0 if ctx.marker_style == "pin" else 0.0)), 4.0, width=ctx.width, height=ctx.height)


def _draw_compass_rose(ctx: RenderContext, center: Point, *, radius: float) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    ctx.draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ctx.secondary_color, width=2)
    _draw_arrow_line(ctx.draw, (cx, cy + r * 0.65), (cx, cy - r * 0.78), fill=ctx.secondary_color, width=2, arrow_size=8)
    ctx.draw.line([(cx - r * 0.65, cy), (cx + r * 0.65, cy)], fill=ctx.guide_color, width=2)
    _draw_text(ctx, "N", (cx, cy - r - 15), font=ctx.tiny_font)
    _draw_text(ctx, "E", (cx + r + 15, cy), font=ctx.tiny_font)
    _draw_text(ctx, "S", (cx, cy + r + 16), font=ctx.tiny_font)
    _draw_text(ctx, "W", (cx - r - 15, cy), font=ctx.tiny_font)
    return pad_bbox((cx - r - 24, cy - r - 26, cx + r + 26, cy + r + 28), 2.0, width=ctx.width, height=ctx.height)


def _draw_bearing_option_strip(
    ctx: RenderContext,
    *,
    labels: Sequence[str],
    values: Sequence[int],
) -> tuple[BBox, tuple[Dict[str, Any], ...]]:
    if len(labels) != len(values):
        raise ValueError("bearing option labels and values must have the same length")
    if len(labels) != 6:
        raise ValueError("final bearing MCQ requires exactly six options")

    panel_bbox = _draw_panel(ctx, (56.0, 512.0, 784.0, 568.0), fill=ctx.panel_fill, radius=8)
    left, top, right, bottom = 68.0, 522.0, 772.0, 558.0
    cell_w = (right - left) / float(len(labels))
    option_entities: list[Dict[str, Any]] = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x0 = left + (float(idx) * cell_w)
        x1 = left + (float(idx + 1) * cell_w) - 8.0
        option_bbox = (x0, top, x1, bottom)
        ctx.draw.rounded_rectangle(
            tuple(float(v) for v in option_bbox),
            radius=6,
            fill=ctx.panel_alt_fill,
            outline=ctx.panel_border,
            width=2,
        )
        text_bbox = _draw_text(
            ctx,
            f"{str(label)}: {bearing_label(int(value))}",
            ((x0 + x1) / 2.0, (top + bottom) / 2.0),
            font=ctx.tiny_font,
        )
        full_bbox = _bbox_union(option_bbox, text_bbox)
        option_entities.append(
            {
                "entity_id": f"bearing_option_{idx}",
                "entity_type": "bearing_option",
                "label": str(label),
                "bearing_degrees": int(value),
                "bbox": bbox_to_list(full_bbox),
                "text_bbox": bbox_to_list(text_bbox),
            }
        )
    return panel_bbox, tuple(option_entities)


def _offset_label_point(start: Point, end: Point, amount: float) -> Point:
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    length = max(1.0, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length
    return ((sx + ex) / 2.0 + nx * float(amount), (sy + ey) / 2.0 + ny * float(amount))


def render_final_bearing_scene(ctx: RenderContext, route_case: RouteCase) -> RenderedBearingScene:
    """Render the two-leg route used by the final-bearing objective.

    The route case is already semantically resolved; this function projects its
    points into the final layout and returns keyed start/finish annotations
    after all geometry placement is fixed.
    """
    if route_case.final_bearing is None:
        raise ValueError("final bearing scene requires final_bearing")
    if route_case.option_count != 6 or len(route_case.option_labels) != 6 or len(route_case.option_values) != 6:
        raise ValueError("final bearing scene requires six rendered MCQ options")
    if route_case.target_index is None:
        raise ValueError("final bearing scene requires a target option index")
    p0, p1, p2 = route_unit_points(route_case)
    panel_bbox = (60.0, 70.0, 628.0, 494.0)
    route_panel_bbox = _draw_panel(ctx, panel_bbox, fill=ctx.panel_alt_fill)
    scale, origin = fit_points_to_box((p0, p1, p2), (110.0, 128.0, 560.0, 438.0), min_scale=8.0)
    start = project_point(p0, scale=scale, origin=origin)
    mid = project_point(p1, scale=scale, origin=origin)
    end = project_point(p2, scale=scale, origin=origin)

    _draw_arrow_line(ctx.draw, start, mid, fill=ctx.accent_color, width=ctx.line_width + 1, arrow_size=14)
    _draw_arrow_line(ctx.draw, mid, end, fill=ctx.accent_color, width=ctx.line_width + 1, arrow_size=14)
    north_end = (float(start[0]), float(start[1]) - 58.0)
    _draw_arrow_line(ctx.draw, start, north_end, fill=ctx.guide_color, width=max(2, ctx.line_width - 2), arrow_size=9)
    north_label_bbox = _draw_text(ctx, "N", (north_end[0], north_end[1] - 18.0), font=ctx.tiny_font)
    start_bbox = _draw_marker(ctx, start, radius=10, color=ctx.secondary_color)
    end_bbox = _draw_marker(ctx, end, radius=10, color=ctx.secondary_accent_color)
    start_label_bbox = _draw_text(ctx, "S", (start[0] - 20.0, start[1] + 20.0), font=ctx.small_font)
    end_label_bbox = _draw_text(ctx, "F", (end[0] + 20.0, end[1] - 20.0), font=ctx.small_font)
    label1_bbox = _draw_text(ctx, bearing_label(route_case.bearing_a), _offset_label_point(start, mid, 28.0), font=ctx.small_font)
    label2_bbox = _draw_text(ctx, bearing_label(route_case.bearing_b), _offset_label_point(mid, end, 28.0), font=ctx.small_font)
    first_leg_bbox = _bbox_union(bbox_from_points((start, mid), width=ctx.width, height=ctx.height, pad=16.0), label1_bbox)
    second_leg_bbox = _bbox_union(bbox_from_points((mid, end), width=ctx.width, height=ctx.height, pad=16.0), label2_bbox)
    answer_relation_bbox = bbox_from_points((start, end), width=ctx.width, height=ctx.height, pad=22.0)
    answer_relation_bbox = (
        min(answer_relation_bbox[0], start_bbox[0], end_bbox[0], start_label_bbox[0], end_label_bbox[0], north_label_bbox[0]),
        min(answer_relation_bbox[1], start_bbox[1], end_bbox[1], start_label_bbox[1], end_label_bbox[1], north_label_bbox[1]),
        max(answer_relation_bbox[2], start_bbox[2], end_bbox[2], start_label_bbox[2], end_label_bbox[2], north_label_bbox[2]),
        max(answer_relation_bbox[3], start_bbox[3], end_bbox[3], start_label_bbox[3], end_label_bbox[3], north_label_bbox[3]),
    )
    compass_bbox = _draw_compass_rose(ctx, (720.0, 148.0), radius=42.0)
    note_bbox = _draw_panel(ctx, (650.0, 248.0, 780.0, 402.0), fill=ctx.panel_fill)
    _draw_text(ctx, "bearing", (715.0, 288.0), font=ctx.tiny_font)
    _draw_text(ctx, "clockwise", (715.0, 322.0), font=ctx.tiny_font)
    _draw_text(ctx, "from N", (715.0, 356.0), font=ctx.tiny_font)
    options_bbox, option_entities = _draw_bearing_option_strip(
        ctx,
        labels=route_case.option_labels,
        values=route_case.option_values,
    )
    turn_bbox = pad_bbox(
        (mid[0] - 6.0, mid[1] - 6.0, mid[0] + 6.0, mid[1] + 6.0),
        4.0,
        width=ctx.width,
        height=ctx.height,
    )
    scene_entities = (
        {
            "entity_id": "route_path",
            "entity_type": "bearing_route",
            "points_px": [[round(start[0], 3), round(start[1], 3)], [round(mid[0], 3), round(mid[1], 3)], [round(end[0], 3), round(end[1], 3)]],
            "bbox": bbox_to_list(route_panel_bbox),
        },
        {
            "entity_id": "final_bearing_segment",
            "entity_type": "implicit_bearing_relation",
            "visible": False,
            "bearing_degrees": int(route_case.final_bearing),
            "bbox": bbox_to_list(answer_relation_bbox),
        },
        {
            "entity_id": "compass_rose",
            "entity_type": "compass",
            "bbox": bbox_to_list(compass_bbox),
        },
        {
            "entity_id": "bearing_options",
            "entity_type": "bearing_option_panel",
            "bbox": bbox_to_list(options_bbox),
            "option_count": int(route_case.option_count),
        },
        *option_entities,
    )
    return RenderedBearingScene(
        image=ctx.image,
        answer=str(route_case.option_labels[int(route_case.target_index)]),
        answer_type="option_letter",
        annotation_bboxes=(start_bbox, end_bbox),
        annotation_roles=("S", "F"),
        annotation_points=(start, end),
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "route_panel_bbox": bbox_to_list(route_panel_bbox),
            "route_points_px": [[round(start[0], 3), round(start[1], 3)], [round(mid[0], 3), round(mid[1], 3)], [round(end[0], 3), round(end[1], 3)]],
            "north_reference_line_px": [[round(start[0], 3), round(start[1], 3)], [round(north_end[0], 3), round(north_end[1], 3)]],
            "compass_bbox": bbox_to_list(compass_bbox),
            "bearing_note_bbox": bbox_to_list(note_bbox),
            "bearing_options_bbox": bbox_to_list(options_bbox),
            "bearing_options": [dict(entity) for entity in option_entities],
            "route_leg_annotation_bboxes": {
                "first_route_leg": bbox_to_list(label1_bbox),
                "second_route_leg": bbox_to_list(label2_bbox),
            },
        },
        witness={
            "geometry_kind": "bearing_route_final_bearing",
            "leg_a": int(route_case.leg_a),
            "leg_b": int(route_case.leg_b),
            "bearing_a": int(route_case.bearing_a),
            "bearing_b": int(route_case.bearing_b),
            "turn_direction": str(route_case.turn_direction),
            "displacement": int(route_case.displacement),
            "final_bearing": int(route_case.final_bearing),
            "option_count": int(route_case.option_count),
            "target_index": int(route_case.target_index),
            "option_labels": list(route_case.option_labels),
            "option_values": list(route_case.option_values),
            "correct_option_label": str(route_case.option_labels[int(route_case.target_index)]),
            "correct_option_value": int(route_case.final_bearing),
            "answer_value": str(route_case.option_labels[int(route_case.target_index)]),
        },
    )


def render_endpoint_label_scene(ctx: RenderContext, route_case: RouteCase) -> RenderedBearingScene:
    """Render endpoint candidates for the option-label objective.

    The route case supplies the correct candidate and distractors; this function
    owns only final layout projection, visible candidate labels, and keyed
    annotation coordinates for the selected endpoint.
    """
    if route_case.target_index is None:
        raise ValueError("endpoint label scene requires target_index")
    base_candidates = candidate_unit_points(route_case)
    correct = base_candidates[0]
    distractors = base_candidates[1:]
    target_index = int(route_case.target_index)
    ordered_candidates = list(distractors)
    ordered_candidates.insert(target_index, correct)
    ordered_candidates = ordered_candidates[: int(route_case.option_count)]
    all_points = [(0.0, 0.0)] + [point for _, point in ordered_candidates]
    plot_panel = (60.0, 84.0, 552.0, 510.0)
    plot_panel_bbox = _draw_panel(ctx, plot_panel, fill=ctx.panel_alt_fill)
    scale, origin, projection_meta = _resolve_route_projection(ctx, plot_panel, tuple(all_points))
    start = project_point((0.0, 0.0), scale=scale, origin=origin)
    start_bbox = _draw_marker(ctx, start, radius=10.0, color=ctx.secondary_color)
    _draw_text(ctx, "S", (start[0] - 20.0, start[1] + 20.0), font=ctx.small_font)

    option_entities: list[Dict[str, Any]] = []
    selected_bbox: BBox | None = None
    selected_label_bbox: BBox | None = None
    selected_center: Point | None = None
    for idx, (candidate_kind, unit_point) in enumerate(ordered_candidates):
        label = str(route_case.option_labels[idx])
        point = project_point(unit_point, scale=scale, origin=origin)
        point_bbox = _draw_marker(ctx, point, radius=11.0, color=ctx.secondary_accent_color)
        label_dx = 20.0 if point[0] >= start[0] else -20.0
        label_dy = -20.0 if point[1] <= start[1] else 20.0
        label_bbox = _draw_text(ctx, label, (point[0] + label_dx, point[1] + label_dy), font=ctx.small_font)
        combined_bbox = (
            min(point_bbox[0], label_bbox[0]),
            min(point_bbox[1], label_bbox[1]),
            max(point_bbox[2], label_bbox[2]),
            max(point_bbox[3], label_bbox[3]),
        )
        option_entities.append(
            {
                "entity_id": f"candidate_{idx}",
                "entity_type": "route_endpoint_candidate",
                "candidate_kind": str(candidate_kind),
                "candidate_index": int(idx),
                "label": str(label),
                "center_px": [round(point[0], 3), round(point[1], 3)],
                "marker_bbox": bbox_to_list(point_bbox),
                "label_bbox": bbox_to_list(label_bbox),
                "bbox": bbox_to_list(combined_bbox),
            }
        )
        if idx == target_index:
            selected_bbox = point_bbox
            selected_label_bbox = label_bbox
            selected_center = point
    if selected_bbox is None or selected_label_bbox is None or selected_center is None:
        raise ValueError("selected endpoint candidate was not rendered")

    instruction_panel = (586.0, 92.0, 782.0, 350.0)
    instruction_bbox = _draw_panel(ctx, instruction_panel, fill=ctx.panel_fill)
    _draw_text(ctx, "route", (684.0, 126.0), font=ctx.small_font)
    instr1 = _draw_text(ctx, f"1: {bearing_label(route_case.bearing_a)}", (684.0, 158.0), font=ctx.tiny_font)
    instr1_steps = _draw_text(ctx, f"{int(route_case.leg_a)} steps", (684.0, 180.0), font=ctx.tiny_font)
    instr2 = _draw_text(ctx, f"2: {bearing_label(route_case.bearing_b)}", (684.0, 214.0), font=ctx.tiny_font)
    instr2_steps = _draw_text(ctx, f"{int(route_case.leg_b)} steps", (684.0, 236.0), font=ctx.tiny_font)
    _draw_compass_rose(ctx, (684.0, 304.0), radius=28.0)
    instruction_annotation_bbox = (
        min(instruction_bbox[0], instr1[0], instr1_steps[0], instr2[0], instr2_steps[0]),
        min(instruction_bbox[1], instr1[1], instr1_steps[1], instr2[1], instr2_steps[1]),
        max(instruction_bbox[2], instr1[2], instr1_steps[2], instr2[2], instr2_steps[2]),
        max(instruction_bbox[3], instr1[3], instr1_steps[3], instr2[3], instr2_steps[3]),
    )
    scene_entities = (
        {
            "entity_id": "route_start",
            "entity_type": "route_start_point",
            "center_px": [round(start[0], 3), round(start[1], 3)],
            "bbox": bbox_to_list(start_bbox),
        },
        {
            "entity_id": "instruction_panel",
            "entity_type": "route_instruction_panel",
            "bbox": bbox_to_list(instruction_annotation_bbox),
        },
        {
            "entity_id": "candidate_panel",
            "entity_type": "endpoint_candidate_panel",
            "bbox": bbox_to_list(plot_panel_bbox),
        },
        *tuple(option_entities),
    )
    return RenderedBearingScene(
        image=ctx.image,
        answer=str(route_case.option_labels[int(route_case.target_index)]),
        answer_type="option_letter",
        annotation_bboxes=(start_bbox, selected_bbox),
        annotation_roles=("S", str(route_case.option_labels[int(route_case.target_index)])),
        annotation_points=(start, selected_center),
        scene_entities=scene_entities,
        render_map={
            "coord_space": "pixel",
            "candidate_panel_bbox": bbox_to_list(plot_panel_bbox),
            "candidate_projection": dict(projection_meta),
            "candidate_graph_paper": {
                "grid_unit": "one_square_equals_one_step",
                "grid_cell_px": round(float(scale), 3),
                "projection_bbox": list(projection_meta["projection_bbox"]),
                "unit_bounds": dict(projection_meta["unit_bounds"]),
            },
            "instruction_panel_bbox": bbox_to_list(instruction_annotation_bbox),
            "start_center_px": [round(start[0], 3), round(start[1], 3)],
            "selected_candidate_center_px": [round(selected_center[0], 3), round(selected_center[1], 3)],
            "selected_candidate_label_bbox": bbox_to_list(selected_label_bbox),
        },
        witness={
            "geometry_kind": "bearing_route_endpoint",
            "leg_a": int(route_case.leg_a),
            "leg_b": int(route_case.leg_b),
            "bearing_a": int(route_case.bearing_a),
            "bearing_b": int(route_case.bearing_b),
            "turn_direction": str(route_case.turn_direction),
            "displacement": int(route_case.displacement),
            "option_count": int(route_case.option_count),
            "target_index": int(route_case.target_index),
            "option_labels": list(route_case.option_labels),
            "answer_value": str(route_case.option_labels[int(route_case.target_index)]),
        },
    )


def _bbox_union(*bboxes: BBox) -> BBox:
    return (
        min(float(bbox[0]) for bbox in bboxes),
        min(float(bbox[1]) for bbox in bboxes),
        max(float(bbox[2]) for bbox in bboxes),
        max(float(bbox[3]) for bbox in bboxes),
    )


__all__ = [
    "make_render_context",
    "render_endpoint_label_scene",
    "render_final_bearing_scene",
]
