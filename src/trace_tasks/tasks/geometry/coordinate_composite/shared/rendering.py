"""Rendering primitives for coordinate-composite diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame, graph_units_to_pixel, scale_point
from trace_tasks.tasks.geometry.shared.shape_style import extract_background_anchor_colors, sample_geometry_shape_style
from trace_tasks.tasks.geometry.shared.single_object_scene import finalize_graph_scene_image, make_graph_scene_canvas, resolve_graph_scene_context
from trace_tasks.tasks.geometry.shared.vector2d import point_to_list
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.text_rendering import (
    draw_text_centered,
    load_font,
    resolve_scene_label_font_size_px,
    resolve_text_label_center,
)

from .relations import filtered_intersections, object_to_trace
from .state import CircleObject, Color, LineObject, PairFilter, PolygonObject, RenderedScene, SceneObject


def _draw_object(
    draw: ImageDraw.ImageDraw,
    obj: SceneObject,
    *,
    color: Color,
    context: Any,
    width_px: int,
) -> Dict[str, Any]:
    """Draw one graph-space object and return its pixel-space render metadata."""

    scale = int(context.scene_scale)
    width_render = max(1, int(width_px) * int(scale))
    if isinstance(obj, LineObject):
        p0 = scale_point(
            graph_units_to_pixel(obj.p0, origin=context.graph_origin, spacing=int(context.graph_spacing)),
            int(scale),
        )
        p1 = scale_point(
            graph_units_to_pixel(obj.p1, origin=context.graph_origin, spacing=int(context.graph_spacing)),
            int(scale),
        )
        draw.line([p0, p1], fill=color, width=width_render)
        return {"id": str(obj.object_id), "kind": "line_segment", "p0_px": point_to_list(p0), "p1_px": point_to_list(p1)}
    if isinstance(obj, CircleObject):
        center_px = scale_point(
            graph_units_to_pixel(obj.center, origin=context.graph_origin, spacing=int(context.graph_spacing)),
            int(scale),
        )
        radius_px = float(obj.radius) * float(context.graph_spacing) * float(scale)
        bbox = [
            float(center_px[0]) - radius_px,
            float(center_px[1]) - radius_px,
            float(center_px[0]) + radius_px,
            float(center_px[1]) + radius_px,
        ]
        draw.ellipse(bbox, outline=color, width=width_render)
        return {
            "id": str(obj.object_id),
            "kind": "circle",
            "center_px": point_to_list(center_px),
            "radius_px": round(float(radius_px), 3),
        }
    vertices = [
        scale_point(
            graph_units_to_pixel(point, origin=context.graph_origin, spacing=int(context.graph_spacing)),
            int(scale),
        )
        for point in obj.vertices
    ]
    draw.line([*vertices, vertices[0]], fill=color, width=width_render, joint="curve")
    return {"id": str(obj.object_id), "kind": "polygon", "vertices_px": [point_to_list(point) for point in vertices]}


def _sample_object_colors(rng: Any, *, shape_color: Color) -> Tuple[Color, ...]:
    base = tuple(int(value) for value in shape_color)
    palette: Tuple[Color, ...] = (
        base,
        (max(24, min(210, base[2] + 35)), max(24, min(175, base[0] + 12)), max(24, min(190, base[1] - 18))),
        (max(24, min(190, base[1] + 24)), max(24, min(190, base[2] - 8)), max(24, min(190, base[0] + 40))),
        (max(24, min(200, base[0] - 20)), max(24, min(190, base[1] + 30)), max(24, min(200, base[2] + 20))),
    )
    offset = int(rng.randrange(len(palette)))
    return tuple(palette[(offset + index) % len(palette)] for index in range(len(palette)))


def _scale_bbox_down(bbox: Sequence[float], *, scale: int) -> List[float]:
    factor = float(max(1, int(scale)))
    return [round(float(value) / factor, 3) for value in bbox]


def _draw_candidate_points(
    draw: ImageDraw.ImageDraw,
    *,
    candidate_points: Sequence[Tuple[str, Tuple[float, float]]],
    context: Any,
) -> Dict[str, Any]:
    """Draw labeled candidate points and return canonical pixel metadata."""

    if not candidate_points:
        return {
            "candidate_points_graph": [],
            "candidate_points_px": [],
            "candidate_point_labels": [],
            "candidate_marker_bboxes": {},
            "candidate_label_bboxes": {},
        }
    scale = int(context.scene_scale)
    label_font_size = resolve_scene_label_font_size_px(
        canvas_size=int(context.canvas_size),
        graph_spacing=int(context.graph_spacing),
        scene_scale=int(scale),
        min_px=16,
        max_px=24,
    )
    font = load_font(int(label_font_size), bold=True)
    stroke_width = max(1, int(round(1.3 * float(scale))))
    marker_radius = max(5.0 * float(scale), 0.18 * float(context.graph_spacing) * float(scale))
    marker_fill = (214, 54, 64)
    marker_outline = (255, 255, 255)
    label_fill = (20, 29, 43)
    label_stroke = (255, 255, 255)
    graph_points = [(label, (float(point[0]), float(point[1]))) for label, point in candidate_points]
    canonical_points = {
        str(label): graph_units_to_pixel(
            point,
            origin=context.graph_origin,
            spacing=int(context.graph_spacing),
        )
        for label, point in graph_points
    }
    render_points = {
        str(label): scale_point(point, int(scale))
        for label, point in canonical_points.items()
    }
    centroid_x = sum(float(point[0]) for point in render_points.values()) / float(len(render_points))
    centroid_y = sum(float(point[1]) for point in render_points.values()) / float(len(render_points))
    occupied_boxes: List[Tuple[float, float, float, float]] = []
    marker_bboxes: Dict[str, List[float]] = {}
    label_bboxes: Dict[str, List[float]] = {}
    blocked_points = list(render_points.values())
    canvas_size = int(context.canvas_size) * int(scale)

    for label, _graph_point in graph_points:
        point = render_points[str(label)]
        px, py = float(point[0]), float(point[1])
        marker_bbox = [
            float(px - marker_radius),
            float(py - marker_radius),
            float(px + marker_radius),
            float(py + marker_radius),
        ]
        draw.ellipse(marker_bbox, fill=marker_outline)
        inset = max(1.0, float(scale))
        draw.ellipse(
            [
                marker_bbox[0] + inset,
                marker_bbox[1] + inset,
                marker_bbox[2] - inset,
                marker_bbox[3] - inset,
            ],
            fill=marker_fill,
        )
        marker_bboxes[str(label)] = _scale_bbox_down(marker_bbox, scale=int(scale))
        base_direction = (float(px - centroid_x), float(py - centroid_y))
        center, label_bbox = resolve_text_label_center(
            draw,
            text=str(label),
            anchor=(float(px), float(py)),
            base_direction=base_direction,
            offset_px=max(22.0 * float(scale), 0.70 * float(context.graph_spacing) * float(scale)),
            font=font,
            blocked_points=blocked_points,
            occupied_boxes=occupied_boxes,
            stroke_width=int(stroke_width),
            point_clearance_px=float(marker_radius + (8.0 * float(scale))),
            canvas_size=int(canvas_size),
        )
        draw_text_centered(
            draw,
            text=str(label),
            center=(float(center[0]), float(center[1])),
            font=font,
            fill=label_fill,
            stroke_fill=label_stroke,
            stroke_width=int(stroke_width),
        )
        occupied_boxes.append(label_bbox)
        label_bboxes[str(label)] = _scale_bbox_down(label_bbox, scale=int(scale))

    labels = tuple(str(label) for label, _point in graph_points)
    return {
        "candidate_points_graph": [
            {"label": str(label), "point": point_to_list(point)}
            for label, point in graph_points
        ],
        "candidate_points_px": [
            {"label": str(label), "point": point_to_list(canonical_points[str(label)])}
            for label in labels
        ],
        "candidate_point_labels": list(labels),
        "candidate_marker_bboxes": dict(marker_bboxes),
        "candidate_label_bboxes": dict(label_bboxes),
    }


def render_coordinate_composite_scene(
    *,
    instance_seed: int,
    objects: Tuple[SceneObject, ...],
    pair_filter: PairFilter,
    transform: str,
    expected_count: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    random_namespace: str,
    candidate_points: Sequence[Tuple[str, Tuple[float, float]]] | None = None,
) -> RenderedScene:
    """Render a coordinate diagram after public task code selects the objects."""

    rng = spawn_rng(int(instance_seed), str(random_namespace))
    context = resolve_graph_scene_context(
        rng,
        instance_seed=int(instance_seed),
        scene_id="coordinate_composite",
        params=params,
        render_defaults=render_defaults,
        background_defaults=background_defaults,
        fallback_canvas_min=int(group_default(render_defaults, "coordinate_composite_canvas_size_min", 660)),
        fallback_canvas_max=int(group_default(render_defaults, "coordinate_composite_canvas_size_max", 740)),
        fallback_cells_min=int(group_default(render_defaults, "coordinate_composite_graph_cells_min", 18)),
        fallback_cells_max=int(group_default(render_defaults, "coordinate_composite_graph_cells_max", 20)),
        graph_style_overrides={
            "axis_scale_labels_enabled": False,
            "origin_label_enabled": False,
            "axis_arrows_enabled": True,
        },
    )
    image, draw, background_meta = make_graph_scene_canvas(
        instance_seed=int(instance_seed),
        context=context,
        background_defaults=background_defaults,
        require_graph_paper=True,
    )
    shape_style = sample_geometry_shape_style(
        rng,
        params=params,
        render_defaults=render_defaults,
        anchor_colors=extract_background_anchor_colors(background_meta),
    )
    intersections = filtered_intersections(objects, pair_filter)
    if len(intersections) != int(expected_count):
        raise RuntimeError(
            f"coordinate-composite case expected {int(expected_count)} intersections, got {len(intersections)}"
        )

    line_width = int(
        rng.randint(
            int(group_default(render_defaults, "coordinate_composite_line_width_min", 3)),
            int(group_default(render_defaults, "coordinate_composite_line_width_max", 5)),
        )
    )
    object_colors = _sample_object_colors(rng, shape_color=shape_style.line_color)
    drawn_objects: List[Dict[str, Any]] = []
    for index, obj in enumerate(objects):
        object_color = tuple(int(value) for value in object_colors[int(index) % len(object_colors)])
        drawn = _draw_object(
            draw,
            obj,
            color=object_color,
            context=context,
            width_px=int(line_width),
        )
        drawn["graph"] = object_to_trace(obj)
        drawn["color"] = [int(value) for value in object_color]
        drawn_objects.append(drawn)

    candidate_meta = _draw_candidate_points(
        draw,
        candidate_points=tuple(candidate_points or ()),
        context=context,
    )

    intersections_px = tuple(
        graph_units_to_pixel(point, origin=context.graph_origin, spacing=int(context.graph_spacing))
        for point in intersections
    )
    final_image, final_background_meta, post_noise_meta = finalize_graph_scene_image(
        image,
        instance_seed=int(instance_seed),
        context=context,
        background_meta=background_meta,
        noise_defaults=noise_defaults,
    )
    render_spec_extra = {
        "graph_coordinate_frame": dict(context.graph_frame),
        "graph_paper_grid": graph_paper_grid_from_frame(context.graph_frame),
        "graph_layout": dict(context.graph_layout_metadata),
        "shape_style": dict(shape_style.to_trace_dict()),
        "object_colors": [[int(channel) for channel in color] for color in object_colors],
        "line_width_px": int(line_width),
    }
    render_map = {
        "objects": [dict(obj) for obj in drawn_objects],
        "intersection_points_graph": [point_to_list(point) for point in intersections],
        "intersection_points_px": [point_to_list(point) for point in intersections_px],
        "transform": str(transform),
        **dict(candidate_meta),
    }
    return RenderedScene(
        image=final_image,
        intersection_points_px=tuple((float(point[0]), float(point[1])) for point in intersections_px),
        intersection_points_graph=tuple((float(point[0]), float(point[1])) for point in intersections),
        object_specs=tuple(object_to_trace(obj) for obj in objects),
        render_map=render_map,
        background_meta=dict(final_background_meta),
        post_noise_meta=dict(post_noise_meta),
        render_spec_extra=render_spec_extra,
        candidate_points_px=tuple(
            tuple(float(value) for value in item["point"])
            for item in candidate_meta["candidate_points_px"]
        ),
        candidate_points_graph=tuple(
            tuple(float(value) for value in item["point"])
            for item in candidate_meta["candidate_points_graph"]
        ),
        candidate_point_labels=tuple(str(label) for label in candidate_meta["candidate_point_labels"]),
        candidate_marker_bboxes=dict(candidate_meta["candidate_marker_bboxes"]),
        candidate_label_bboxes=dict(candidate_meta["candidate_label_bboxes"]),
    )


__all__ = ["render_coordinate_composite_scene"]
