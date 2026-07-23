"""Rendering primitives for graph-paper geometry scenes."""

from __future__ import annotations

from math import atan2, ceil, cos, degrees, floor, radians, sin, sqrt
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.geometry.shared.shape_style import (
    extract_background_anchor_colors,
    sample_geometry_shape_style,
)
from trace_tasks.tasks.geometry.shared.single_object_scene import (
    finalize_graph_scene_image,
    make_graph_scene_canvas,
    resolve_graph_scene_context,
)
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import int_default
from .state import BBox, Color, GraphObject, GraphPaperContext, Point, POINT_LABELS, SCENE_ID

_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)


def _coerce_color(value: Any, fallback: Color) -> Color:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            max(0, min(255, int(value[0]))),
            max(0, min(255, int(value[1]))),
            max(0, min(255, int(value[2]))),
        )
    return tuple(int(channel) for channel in fallback)


def _blend_color(color_a: Color, color_b: Color, ratio_b: float) -> Color:
    ratio = max(0.0, min(1.0, float(ratio_b)))
    return (
        int(round(float(color_a[0]) * (1.0 - ratio) + float(color_b[0]) * ratio)),
        int(round(float(color_a[1]) * (1.0 - ratio) + float(color_b[1]) * ratio)),
        int(round(float(color_a[2]) * (1.0 - ratio) + float(color_b[2]) * ratio)),
    )


def _relative_luminance(color: Color) -> float:
    """Return approximate perceived brightness for theme-aware stroke colors."""

    red, green, blue = (float(color[0]), float(color[1]), float(color[2]))
    return (0.299 * red) + (0.587 * green) + (0.114 * blue)


def _object_palette(panel_fill: Color) -> tuple[Color, ...]:
    """Return distinct non-semantic object strokes for all graph-paper labels."""

    light_panel_palette: tuple[Color, ...] = (
        (28, 105, 205),
        (200, 50, 65),
        (25, 135, 80),
        (120, 80, 205),
        (190, 95, 20),
        (0, 135, 155),
        (200, 65, 145),
        (115, 120, 30),
        (65, 80, 170),
        (150, 70, 45),
        (0, 120, 115),
        (120, 70, 120),
    )
    dark_panel_palette: tuple[Color, ...] = (
        (65, 220, 80),
        (120, 120, 250),
        (240, 55, 135),
        (245, 170, 35),
        (30, 210, 225),
        (190, 110, 255),
        (255, 95, 55),
        (165, 235, 70),
        (255, 125, 195),
        (80, 175, 255),
        (230, 220, 55),
        (135, 255, 195),
    )
    if _relative_luminance(panel_fill) < 128.0:
        return dark_panel_palette
    return light_panel_palette


def make_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    theme_index: int = 0,
) -> GraphPaperContext:
    """Create a graph-paper canvas through the shared geometry grid style layer."""

    canvas_size = int_default(params, defaults, "canvas_size", 768)
    graph_min = int_default(params, defaults, "graph_cells_min", 14)
    graph_max = int_default(params, defaults, "graph_cells_max", 20)
    style_seed = int(instance_seed) + (int(theme_index) * 1009)
    rng = spawn_rng(style_seed, "geometry.graph_paper.shared_context")
    shared_context = resolve_graph_scene_context(
        rng,
        instance_seed=int(style_seed),
        scene_id=SCENE_ID,
        params=params,
        render_defaults=defaults,
        background_defaults=_BACKGROUND_DEFAULTS,
        fallback_canvas_min=int(canvas_size),
        fallback_canvas_max=int(canvas_size),
        fallback_cells_min=int(graph_min),
        fallback_cells_max=int(graph_max),
        require_graph_paper_background=True,
        graph_style_overrides={
            "origin_fraction_x": 0.5,
            "origin_fraction_y": 0.5,
            "axis_enabled": True,
            "axis_scale_labels_enabled": True,
            "origin_label_enabled": False,
        },
    )
    image, _draw, background_meta = make_graph_scene_canvas(
        instance_seed=int(style_seed),
        context=shared_context,
        background_defaults=_BACKGROUND_DEFAULTS,
        require_graph_paper=True,
    )
    image, background_meta, post_noise_meta = finalize_graph_scene_image(
        image,
        instance_seed=int(style_seed),
        context=shared_context,
        background_meta=background_meta,
        noise_defaults=_NOISE_DEFAULTS,
    )
    draw = ImageDraw.Draw(image)
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    style_spec = background_meta.get("style_spec", {}) if isinstance(background_meta, Mapping) else {}
    style_spec = style_spec if isinstance(style_spec, Mapping) else {}
    panel_fill = _coerce_color(style_spec.get("base_color"), (248, 250, 252))
    axis_color = _coerce_color(style_spec.get("axis_color"), (112, 124, 144))
    grid_color = _coerce_color(style_spec.get("line_color"), (222, 228, 236))
    label_color = tuple(int(channel) for channel in shape_style.label_color)
    object_colors = _object_palette(panel_fill)
    shape_fill = _blend_color(panel_fill, object_colors[0], 0.18)
    layout = shared_context.graph_panel_layout
    panel_box = tuple(float(value) for value in layout.panel_bbox_px)
    content_box = tuple(float(value) for value in layout.content_bbox_px)
    origin = (
        float(shared_context.graph_origin[0]),
        float(shared_context.graph_origin[1]),
    )
    return GraphPaperContext(
        image=image,
        draw=draw,
        canvas_size=int(shared_context.canvas_size),
        graph_cells=int(shared_context.graph_cells),
        panel_box=panel_box,
        content_box=content_box,
        origin_px=origin,
        spacing_px=float(shared_context.graph_spacing),
        graph_half_range=int(shared_context.graph_cells // 2),
        ink_color=object_colors[0],
        accent_color=object_colors[1],
        grid_color=grid_color,
        axis_color=axis_color,
        label_color=label_color,
        label_stroke_color=tuple(
            int(channel) for channel in shape_style.label_stroke_color
        ),
        shape_fill_color=shape_fill,
        object_colors=object_colors,
        shape_style_meta=shape_style.to_trace_dict(),
        graph_layout_meta={
            **dict(shared_context.graph_layout_metadata),
            "graph_coordinate_frame": dict(shared_context.graph_frame),
            "graph_paper_grid": graph_paper_grid_from_frame(shared_context.graph_frame),
        },
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def object_color(ctx: GraphPaperContext, index: int) -> Color:
    """Return a deterministic non-semantic object stroke color by draw order."""

    palette = tuple(ctx.object_colors) or (ctx.ink_color,)
    return tuple(int(channel) for channel in palette[int(index) % len(palette)])


def project(ctx: GraphPaperContext, point: Point) -> Point:
    """Project graph-unit coordinates into final image pixels."""

    return (
        float(ctx.origin_px[0]) + float(point[0]) * float(ctx.spacing_px),
        float(ctx.origin_px[1]) - float(point[1]) * float(ctx.spacing_px),
    )


def _axis_offset(rng: Any, lower: float, upper: float, *, step: float) -> float:
    """Sample one quantized graph-unit offset inside an allowed interval."""

    resolved_step = max(0.25, float(step))
    low_index = int(ceil(float(lower) / resolved_step))
    high_index = int(floor(float(upper) / resolved_step))
    if low_index > high_index:
        return (float(lower) + float(upper)) / 2.0
    return float(rng.randint(low_index, high_index)) * resolved_step


def random_shift_points(
    ctx: GraphPaperContext,
    points: Sequence[Point],
    rng: Any,
    *,
    margin_units: float = 1.0,
    step: float = 0.5,
) -> tuple[Point, ...]:
    """Translate graph-unit points by one random in-bounds offset."""

    if not points:
        return tuple()
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    limit = max(1.0, float(ctx.graph_half_range) - float(margin_units))
    dx = _axis_offset(rng, -limit - min(xs), limit - max(xs), step=float(step))
    dy = _axis_offset(rng, -limit - min(ys), limit - max(ys), step=float(step))
    return tuple((float(point[0]) + dx, float(point[1]) + dy) for point in points)


def random_center_for_radii(
    ctx: GraphPaperContext,
    radius_x: float,
    radius_y: float,
    rng: Any,
    *,
    margin_units: float = 1.0,
    step: float = 0.5,
) -> Point:
    """Sample a graph-unit center that keeps an ellipse/circle in bounds."""

    limit = max(1.0, float(ctx.graph_half_range) - float(margin_units))
    center_x = _axis_offset(
        rng,
        -limit + float(radius_x),
        limit - float(radius_x),
        step=float(step),
    )
    center_y = _axis_offset(
        rng,
        -limit + float(radius_y),
        limit - float(radius_y),
        step=float(step),
    )
    return (float(center_x), float(center_y))


def draw_measurement_guide(
    ctx: GraphPaperContext,
    start: Point,
    end: Point,
    *,
    color: Color | None = None,
) -> None:
    """Draw a subtle unlabeled guide segment between graph lattice points."""

    start_px = project(ctx, start)
    end_px = project(ctx, end)
    ctx.draw.line([start_px, end_px], fill=color or ctx.axis_color, width=2)


def graph_bbox(
    ctx: GraphPaperContext, points: Sequence[Point], *, pad_px: float = 8.0
) -> BBox:
    """Return a padded pixel bbox around graph-unit points."""

    projected = [project(ctx, point) for point in points]
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    return (min(xs) - pad_px, min(ys) - pad_px, max(xs) + pad_px, max(ys) + pad_px)


def pixel_bbox(points: Sequence[Point], *, pad_px: float = 8.0) -> BBox:
    """Return a padded pixel bbox around already projected points."""

    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs) - pad_px, min(ys) - pad_px, max(xs) + pad_px, max(ys) + pad_px)


def draw_label(
    ctx: GraphPaperContext, text: str, point: Point, *, anchor: str = "mm"
) -> None:
    """Draw one compact object label near a rendered object."""

    font = load_font(22, bold=True)
    x, y = float(point[0]), float(point[1])
    bbox = ctx.draw.textbbox((0, 0), str(text), font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if anchor == "mm":
        loc = (x - width / 2.0, y - height / 2.0)
    elif anchor == "above":
        loc = (x - width / 2.0, y - height - 8)
    else:
        loc = (x + 6, y - height / 2.0)
    ctx.draw.text(
        loc,
        str(text),
        fill=ctx.label_color,
        font=font,
        stroke_width=1,
        stroke_fill=ctx.label_stroke_color,
    )


def _point_inside_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Return whether a graph-space point lies inside a simple polygon."""

    x, y = float(point[0]), float(point[1])
    inside = False
    pts = tuple((float(px), float(py)) for px, py in polygon)
    for index, first in enumerate(pts):
        second = pts[(index + 1) % len(pts)]
        y_crosses = (first[1] > y) != (second[1] > y)
        if y_crosses:
            x_at_y = first[0] + (y - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if x < x_at_y:
                inside = not inside
    return inside


def _outside_label_point(
    polygon: Sequence[Point], vertex: Point, *, offset_units: float
) -> Point:
    """Choose a nearby graph point outside the polygon for a vertex label."""

    pts = tuple((float(x), float(y)) for x, y in polygon)
    centroid = (
        sum(point[0] for point in pts) / max(1, len(pts)),
        sum(point[1] for point in pts) / max(1, len(pts)),
    )
    away = (float(vertex[0]) - centroid[0], float(vertex[1]) - centroid[1])
    away_length = max(1e-6, sqrt((away[0] * away[0]) + (away[1] * away[1])))
    candidate_dirs = [
        (away[0] / away_length, away[1] / away_length),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.707, 0.707),
        (-0.707, 0.707),
        (0.707, -0.707),
        (-0.707, -0.707),
    ]
    scored_dirs = sorted(
        candidate_dirs,
        key=lambda direction: (
            (direction[0] * away[0]) + (direction[1] * away[1]),
            -abs(direction[0]),
            -abs(direction[1]),
        ),
        reverse=True,
    )
    for dx, dy in scored_dirs:
        candidate = (
            float(vertex[0]) + float(dx) * float(offset_units),
            float(vertex[1]) + float(dy) * float(offset_units),
        )
        if not _point_inside_polygon(candidate, pts):
            return candidate
    return (
        float(vertex[0]) + (away[0] / away_length) * float(offset_units),
        float(vertex[1]) + (away[1] / away_length) * float(offset_units),
    )


def draw_vertex_labels(
    ctx: GraphPaperContext,
    points: Sequence[Point],
    *,
    labels: Sequence[str] = POINT_LABELS,
    offset_units: float = 0.72,
) -> dict[str, Point]:
    """Draw A/B/C... labels just outside polygon vertices and return pixels."""

    label_points: dict[str, Point] = {}
    for label, vertex in zip(labels, points, strict=False):
        label_graph_point = _outside_label_point(
            points, vertex, offset_units=float(offset_units)
        )
        label_px = project(ctx, label_graph_point)
        draw_label(ctx, str(label), label_px, anchor="mm")
        label_points[str(label)] = label_px
    return label_points


def draw_point_marker(
    ctx: GraphPaperContext, label: str, point: Point, *, color: Color | None = None
) -> Point:
    """Draw one labeled graph point and return its projected pixel point."""

    px = project(ctx, point)
    stroke = color or ctx.accent_color
    radius = 5
    ctx.draw.ellipse(
        (px[0] - radius, px[1] - radius, px[0] + radius, px[1] + radius), fill=stroke
    )
    draw_label(ctx, label, (px[0] + 14, px[1] - 12), anchor="left")
    return px


def draw_segment(
    ctx: GraphPaperContext,
    label: str,
    start: Point,
    end: Point,
    *,
    color: Color | None = None,
) -> GraphObject:
    """Draw a labeled line segment and return its rendered object record."""

    start_px = project(ctx, start)
    end_px = project(ctx, end)
    stroke = color or ctx.ink_color
    ctx.draw.line([start_px, end_px], fill=stroke, width=4)
    mid = ((start_px[0] + end_px[0]) / 2.0, (start_px[1] + end_px[1]) / 2.0)
    if str(label):
        draw_label(ctx, label, (mid[0], mid[1] - 18), anchor="mm")
    return GraphObject(
        label=str(label),
        kind="segment",
        points_px=(start_px, end_px),
        bbox_px=pixel_bbox((start_px, end_px), pad_px=10),
        metric_value=0.0,
        class_name="segment",
        graph_points=(start, end),
    )


def angle_points(
    center: Point, degrees: float, *, radius: float = 2.8
) -> tuple[Point, Point, Point]:
    """Return endpoint-center-endpoint graph points for one angle."""

    left = radians(180.0)
    right = radians(180.0 - float(degrees))
    first = (
        float(center[0]) + radius * cos(left),
        float(center[1]) + radius * sin(left),
    )
    second = (
        float(center[0]) + radius * cos(right),
        float(center[1]) + radius * sin(right),
    )
    return first, center, second


def draw_angle(
    ctx: GraphPaperContext,
    label: str,
    points: Sequence[Point],
    *,
    color: Color | None = None,
) -> GraphObject:
    """Draw one labeled angle from graph points ordered endpoint, vertex, endpoint."""

    endpoint_a, vertex, endpoint_b = tuple(points)
    px_a, px_v, px_b = (
        project(ctx, endpoint_a),
        project(ctx, vertex),
        project(ctx, endpoint_b),
    )
    stroke = color or ctx.ink_color
    ctx.draw.line([px_v, px_a], fill=stroke, width=4)
    ctx.draw.line([px_v, px_b], fill=stroke, width=4)
    angle_a = degrees(atan2(px_a[1] - px_v[1], px_a[0] - px_v[0]))
    angle_b = degrees(atan2(px_b[1] - px_v[1], px_b[0] - px_v[0]))
    while angle_b < angle_a:
        angle_b += 360.0
    if angle_b - angle_a > 180.0:
        angle_a, angle_b = angle_b, angle_a + 360.0
    ctx.draw.arc(
        (px_v[0] - 28, px_v[1] - 28, px_v[0] + 28, px_v[1] + 28),
        start=float(angle_a),
        end=float(angle_b),
        fill=stroke,
        width=3,
    )
    if str(label):
        draw_label(ctx, label, (px_v[0], px_v[1] - 34), anchor="mm")
    return GraphObject(
        label=str(label),
        kind="angle",
        points_px=(px_a, px_v, px_b),
        bbox_px=pixel_bbox((px_a, px_v, px_b), pad_px=14),
        metric_value=0.0,
        class_name="angle",
        graph_points=(endpoint_a, vertex, endpoint_b),
    )


def draw_polygon(
    ctx: GraphPaperContext,
    label: str,
    points: Sequence[Point],
    *,
    class_name: str = "polygon",
    color: Color | None = None,
    fill: Color | None = None,
    filled: bool = True,
) -> GraphObject:
    """Draw one labeled polygon."""

    pts = tuple(project(ctx, point) for point in points)
    fill_color = (fill or ctx.shape_fill_color) if bool(filled) else None
    ctx.draw.polygon(pts, fill=fill_color, outline=color or ctx.ink_color)
    ctx.draw.line([*pts, pts[0]], fill=color or ctx.ink_color, width=4)
    bbox = pixel_bbox(pts, pad_px=8)
    if str(label):
        draw_label(ctx, label, ((bbox[0] + bbox[2]) / 2.0, bbox[1] - 16), anchor="mm")
    return GraphObject(
        label=str(label),
        kind="polygon",
        points_px=pts,
        bbox_px=bbox,
        metric_value=0.0,
        class_name=str(class_name),
        graph_points=tuple(points),
    )


def draw_ellipse_or_circle(
    ctx: GraphPaperContext,
    label: str,
    center: Point,
    radius_x: float,
    radius_y: float,
    *,
    class_name: str,
    color: Color | None = None,
    filled: bool = True,
) -> GraphObject:
    """Draw one labeled circle or ellipse with graph-unit radii."""

    center_px = project(ctx, center)
    rx = float(radius_x) * float(ctx.spacing_px)
    ry = float(radius_y) * float(ctx.spacing_px)
    bbox = (center_px[0] - rx, center_px[1] - ry, center_px[0] + rx, center_px[1] + ry)
    fill_color = ctx.shape_fill_color if bool(filled) else None
    ctx.draw.ellipse(bbox, outline=color or ctx.ink_color, width=4, fill=fill_color)
    if str(label):
        draw_label(ctx, label, (center_px[0], bbox[1] - 16), anchor="mm")
    return GraphObject(
        label=str(label),
        kind=str(class_name),
        points_px=(center_px,),
        bbox_px=bbox,
        metric_value=0.0,
        class_name=str(class_name),
        graph_points=(center,),
        extra={"radius_x": float(radius_x), "radius_y": float(radius_y)},
    )


def slot_centers(
    ctx: GraphPaperContext,
    count: int,
    *,
    rng: Any | None = None,
    footprint_units: float = 1.2,
) -> list[Point]:
    """Return spread-out graph-unit slot centers for multi-object scenes.

    The graph-paper scene shows independent objects, so the slot layout should
    use the available grid area instead of clustering every sample near the
    origin. Slots are sampled from a broad in-bounds lattice with a minimum
    separation, then shuffled deterministically for visual variety.
    """

    resolved_count = max(0, int(count))
    if resolved_count == 0:
        return []
    limit = max(2.0, float(ctx.graph_half_range) - max(1.6, float(footprint_units)))
    step = 0.5
    grid_values = [
        round((-limit + (index * step)), 3)
        for index in range(int(floor((2.0 * limit) / step)) + 1)
    ]
    candidates = [
        (float(x), float(y))
        for x in grid_values
        for y in grid_values
        if -limit <= float(x) <= limit and -limit <= float(y) <= limit
    ]
    if rng is not None:
        rng.shuffle(candidates)
    else:
        # Keep deterministic fallback broad rather than centered.
        candidates.sort(
            key=lambda point: (
                abs(point[0]) + abs(point[1]),
                point[1],
                point[0],
            ),
            reverse=True,
        )

    min_distance = max(2.2, float(footprint_units) * 1.75)
    selected: list[Point] = []
    for candidate in candidates:
        if all(_distance(candidate, existing) >= min_distance for existing in selected):
            selected.append(candidate)
            if len(selected) == resolved_count:
                return selected

    # If a dense sample cannot satisfy the ideal separation, keep the broadest
    # remaining candidates rather than falling back to a centered grid.
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == resolved_count:
                break
    return selected


def _distance(first: Point, second: Point) -> float:
    """Return Euclidean distance in graph units."""

    return sqrt(
        (float(first[0]) - float(second[0])) ** 2
        + (float(first[1]) - float(second[1])) ** 2
    )


def render_metadata(ctx: GraphPaperContext) -> dict[str, Any]:
    """Build graph-paper render metadata for trace payloads."""

    return {
        "canvas_size": [int(ctx.canvas_size), int(ctx.canvas_size)],
        "coord_space": "pixel",
        "scene_bbox_px": [round(float(v), 3) for v in ctx.panel_box],
        "layout_placement": {
            "mode": "fractional_free_area",
            "origin_fraction_x": 0.5,
            "origin_fraction_y": 0.5,
        },
        "background_style": dict(ctx.background_meta),
        "post_image_noise": dict(ctx.post_noise_meta),
        "text_style": {"draw_object_labels": True},
        "shape_style": dict(ctx.shape_style_meta),
        "object_colors": [[int(channel) for channel in color] for color in ctx.object_colors],
        "graph_cells": int(ctx.graph_cells),
        "graph_panel_bbox_px": [round(float(v), 3) for v in ctx.panel_box],
        "graph_content_bbox_px": [round(float(v), 3) for v in ctx.content_box],
        "graph_origin_px": [round(float(v), 3) for v in ctx.origin_px],
        "graph_coordinate_frame": {
            "origin_pixel": [round(float(v), 3) for v in ctx.origin_px],
            "spacing_px": round(float(ctx.spacing_px), 3),
            "x_positive": "right",
            "y_positive": "up",
        },
        "graph_paper_grid": {
            "spacing_px": round(float(ctx.spacing_px), 3),
            "cells_per_side": int(ctx.graph_cells),
        },
        **dict(ctx.graph_layout_meta),
    }
