"""Road shell and floor rendering for street-intersection scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ...shared.camera_projection import (
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    polygon_axis_line_segment as _polygon_axis_line_segment,
    project_screen as _project_screen,
    project_xy as _project_xy,
    screen_to_floor_xy as _screen_to_floor_xy,
)
from ...shared.color_variation import resolve_three_d_object_fill_rgb
from ...shared.object_resources import (
    BUILDING_STYLE_BASE_COLORS,
    BUILDING_STYLE_DIMENSION_FACTORS,
    BUILDING_STYLE_DISPLAY_NAMES,
    STREET_OBJECT_BASE_DIMENSIONS,
    STREET_OBJECT_COLORS,
    STREET_OBJECT_NAMES,
    STREET_RADIAL_OBJECT_TYPES,
    STREET_VEHICLE_OBJECT_TYPES,
)
from ...shared.object_scene_rendering import (
    _bbox_union,
    _draw_box_object,
    _draw_box_parts_object,
    _draw_cone_object,
    _draw_cylinder_object,
    _draw_line,
    _draw_sphere_object,
    _shade,
    _sub_box_spec,
    _tint,
)
from ...shared.object_scene import _object_screen_bbox

from ...shared.street_object_rendering_common import *  # noqa: F403

def _world_polygon(
    points: Sequence[Sequence[float]],
    *,
    camera,
    frame,
) -> List[Tuple[float, float]]:
    return [_project_xy(point, camera, frame) for point in points]

def _draw_world_rect(
    draw: ImageDraw.ImageDraw,
    *,
    camera,
    frame,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int] | None = None,
    width: int = 1,
) -> List[float]:
    points = _world_polygon(
        [
            (float(x0), float(y0), float(z)),
            (float(x1), float(y0), float(z)),
            (float(x1), float(y1), float(z)),
            (float(x0), float(y1), float(z)),
        ],
        camera=camera,
        frame=frame,
    )
    draw.polygon(points, fill=fill)
    if outline is not None:
        draw.line(points + [points[0]], fill=outline, width=max(1, int(width)))
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in points])

def _floor_polygon_area_xy(polygon_xy: Sequence[Tuple[float, float]]) -> float:
    area = 0.0
    if len(polygon_xy) < 3:
        return 0.0
    for index, point_a in enumerate(polygon_xy):
        next_index = index + 1
        if next_index >= len(polygon_xy):
            next_index = 0
        point_b = polygon_xy[next_index]
        area += float(point_a[0]) * float(point_b[1]) - float(point_b[0]) * float(point_a[1])
    return float(area * 0.5)

def _line_intersection_xy(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> Tuple[float, float]:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(q1[0]), float(q1[1])
    x4, y4 = float(q2[0]), float(q2[1])
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return (float(x2), float(y2))
    det_p = x1 * y2 - y1 * x2
    det_q = x3 * y4 - y3 * x4
    return (
        float((det_p * (x3 - x4) - (x1 - x2) * det_q) / denom),
        float((det_p * (y3 - y4) - (y1 - y2) * det_q) / denom),
    )

def _dedupe_polygon_points_xy(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    deduped: List[Tuple[float, float]] = []
    for point in points:
        candidate = (float(point[0]), float(point[1]))
        if not deduped or math.hypot(candidate[0] - deduped[-1][0], candidate[1] - deduped[-1][1]) > 1e-7:
            deduped.append(candidate)
    if len(deduped) > 1 and math.hypot(deduped[0][0] - deduped[-1][0], deduped[0][1] - deduped[-1][1]) <= 1e-7:
        deduped.pop()
    return deduped

def _clip_polygon_to_convex_floor(
    subject_polygon_xy: Sequence[Tuple[float, float]],
    clip_polygon_xy: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if len(subject_polygon_xy) < 3 or len(clip_polygon_xy) < 3:
        return []
    orientation = 1.0 if _floor_polygon_area_xy(clip_polygon_xy) >= 0.0 else -1.0

    def inside(point: Tuple[float, float], edge_a: Tuple[float, float], edge_b: Tuple[float, float]) -> bool:
        cross = (
            (float(edge_b[0]) - float(edge_a[0])) * (float(point[1]) - float(edge_a[1]))
            - (float(edge_b[1]) - float(edge_a[1])) * (float(point[0]) - float(edge_a[0]))
        )
        return cross >= -1e-8 if orientation > 0.0 else cross <= 1e-8

    output = _dedupe_polygon_points_xy(subject_polygon_xy)
    for index, edge_start in enumerate(clip_polygon_xy):
        next_index = index + 1
        if next_index >= len(clip_polygon_xy):
            next_index = 0
        edge_end = clip_polygon_xy[next_index]
        if not output:
            break
        input_points = list(output)
        output = []
        previous = input_points[-1]
        previous_inside = inside(previous, edge_start, edge_end)
        for current in input_points:
            current_inside = inside(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection_xy(previous, current, edge_start, edge_end))
                output.append((float(current[0]), float(current[1])))
            elif previous_inside:
                output.append(_line_intersection_xy(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
        output = _dedupe_polygon_points_xy(output)
    return _dedupe_polygon_points_xy(output)

def _fallback_floor_polygon_xy(render_params: Any) -> List[Tuple[float, float]]:
    extent = float(render_params.street_extent) * float(STREET_FULL_BLEED_FALLBACK_EXTENT_MULTIPLIER)
    return [
        (-extent, -extent),
        (extent, -extent),
        (extent, extent),
        (-extent, extent),
    ]

def _visible_floor_polygon_xy(
    *,
    camera,
    frame,
    render_params: Any,
) -> Tuple[List[Tuple[float, float]], str]:
    floor_polygon = _canvas_floor_polygon_xy(camera=camera, frame=frame, render_params=render_params)
    if len(floor_polygon) >= 3:
        return ([(float(x), float(y)) for x, y in floor_polygon], "canvas_ray_polygon")

    width = float(render_params.canvas_width)
    height = float(render_params.canvas_height)
    sample_hits: List[Tuple[float, float]] = []
    for screen_x in (0.0, width * 0.5, width):
        for screen_y in (0.0, height * 0.5, height):
            floor_xy = _screen_to_floor_xy(screen_x, screen_y, camera=camera, frame=frame)
            if floor_xy is not None:
                sample_hits.append((float(floor_xy[0]), float(floor_xy[1])))
    if len(sample_hits) >= 3:
        return (_fallback_floor_polygon_xy(render_params), "expanded_fallback_square")
    return (_fallback_floor_polygon_xy(render_params), "expanded_fallback_square")

def _canvas_floor_polygon_available(
    *,
    camera,
    frame,
    render_params: Any,
) -> bool:
    return len(_canvas_floor_polygon_xy(camera=camera, frame=frame, render_params=render_params)) >= 3

def _floor_bounds_xy(floor_polygon_xy: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    return (
        min(float(point[0]) for point in floor_polygon_xy),
        max(float(point[0]) for point in floor_polygon_xy),
        min(float(point[1]) for point in floor_polygon_xy),
        max(float(point[1]) for point in floor_polygon_xy),
    )

def _draw_clipped_floor_rect(
    draw: ImageDraw.ImageDraw,
    *,
    camera,
    frame,
    floor_polygon_xy: Sequence[Tuple[float, float]],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    z: float,
    fill: Tuple[int, int, int],
) -> List[float] | None:
    low_x, high_x = sorted((float(x0), float(x1)))
    low_y, high_y = sorted((float(y0), float(y1)))
    subject = [
        (low_x, low_y),
        (high_x, low_y),
        (high_x, high_y),
        (low_x, high_y),
    ]
    clipped = _clip_polygon_to_convex_floor(subject, floor_polygon_xy)
    if len(clipped) < 3:
        return None
    points = [_project_xy((float(x), float(y), float(z)), camera, frame) for x, y in clipped]
    draw.polygon(points, fill=fill)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in points])

def _draw_crosswalks(
    draw: ImageDraw.ImageDraw,
    *,
    camera,
    frame,
    render_params: Any,
    intersection_center_xy: Sequence[float],
    intersection_layout: str,
) -> None:
    """Draw visible crosswalk stripes only for roads present in layout."""

    road = float(render_params.road_half_width)
    cx, cy = float(intersection_center_xy[0]), float(intersection_center_xy[1])
    z = 0.018
    stripe = 0.075
    gap = 0.12
    span = road * 1.82
    crosswalk_offset = road + 0.34
    horizontal_crosswalks = []
    if _arm_is_present(str(intersection_layout), "west"):
        horizontal_crosswalks.append(cx - crosswalk_offset)
    if _arm_is_present(str(intersection_layout), "east"):
        horizontal_crosswalks.append(cx + crosswalk_offset)
    for center_x in horizontal_crosswalks:
        start_x = float(center_x - 0.33)
        for index in range(6):
            x0 = start_x + index * gap
            _draw_world_rect(
                draw,
                camera=camera,
                frame=frame,
                x0=x0,
                y0=cy - span * 0.5,
                x1=x0 + stripe,
                y1=cy + span * 0.5,
                z=z,
                fill=render_params.crosswalk_rgb,
            )
    vertical_crosswalks = []
    if _arm_is_present(str(intersection_layout), "south"):
        vertical_crosswalks.append(cy - crosswalk_offset)
    if _arm_is_present(str(intersection_layout), "north"):
        vertical_crosswalks.append(cy + crosswalk_offset)
    for center_y in vertical_crosswalks:
        start_y = float(center_y - 0.33)
        for index in range(6):
            y0 = start_y + index * gap
            _draw_world_rect(
                draw,
                camera=camera,
                frame=frame,
                x0=cx - span * 0.5,
                y0=y0,
                x1=cx + span * 0.5,
                y1=y0 + stripe,
                z=z,
                fill=render_params.crosswalk_rgb,
            )

def _draw_lane_markings(
    draw: ImageDraw.ImageDraw,
    *,
    camera,
    frame,
    render_params: Any,
    floor_polygon_xy: Sequence[Tuple[float, float]],
    intersection_center_xy: Sequence[float],
    intersection_layout: str,
) -> None:
    """Draw clipped lane dashes along each present road arm."""

    road = float(render_params.road_half_width)
    cx, cy = float(intersection_center_xy[0]), float(intersection_center_xy[1])
    z = 0.022
    dash_len = 0.42
    gap = 0.28
    mark_w = 0.035
    intersection_gap = road * 1.32

    def draw_horizontal_segment(x0: float, x1: float) -> None:
        if float(x1) <= float(x0):
            return
        x = float(x0)
        while x < float(x1):
            dash_x1 = min(float(x1), x + dash_len)
            if dash_x1 - x > 0.08:
                _draw_clipped_floor_rect(
                    draw,
                    camera=camera,
                    frame=frame,
                    floor_polygon_xy=floor_polygon_xy,
                    x0=x,
                    y0=cy - mark_w,
                    x1=dash_x1,
                    y1=cy + mark_w,
                    z=z,
                    fill=render_params.road_mark_rgb,
                )
            x += dash_len + gap

    def draw_vertical_segment(y0: float, y1: float) -> None:
        if float(y1) <= float(y0):
            return
        y = float(y0)
        while y < float(y1):
            dash_y1 = min(float(y1), y + dash_len)
            if dash_y1 - y > 0.08:
                _draw_clipped_floor_rect(
                    draw,
                    camera=camera,
                    frame=frame,
                    floor_polygon_xy=floor_polygon_xy,
                    x0=cx - mark_w,
                    y0=y,
                    x1=cx + mark_w,
                    y1=dash_y1,
                    z=z,
                    fill=render_params.road_mark_rgb,
                )
            y += dash_len + gap

    horizontal_segment = _polygon_axis_line_segment(floor_polygon_xy, axis="y", value=cy)
    if horizontal_segment is None:
        x_min, x_max, _y_min, _y_max = _floor_bounds_xy(floor_polygon_xy)
    else:
        x_min = min(float(horizontal_segment[0][0]), float(horizontal_segment[1][0]))
        x_max = max(float(horizontal_segment[0][0]), float(horizontal_segment[1][0]))
    vertical_segment = _polygon_axis_line_segment(floor_polygon_xy, axis="x", value=cx)
    if vertical_segment is None:
        _x_min, _x_max, y_min, y_max = _floor_bounds_xy(floor_polygon_xy)
    else:
        y_min = min(float(vertical_segment[0][1]), float(vertical_segment[1][1]))
        y_max = max(float(vertical_segment[0][1]), float(vertical_segment[1][1]))

    if _arm_is_present(str(intersection_layout), "west"):
        draw_horizontal_segment(float(x_min), cx - intersection_gap)
    if _arm_is_present(str(intersection_layout), "east"):
        draw_horizontal_segment(cx + intersection_gap, float(x_max))
    if _arm_is_present(str(intersection_layout), "south"):
        draw_vertical_segment(float(y_min), cy - intersection_gap)
    if _arm_is_present(str(intersection_layout), "north"):
        draw_vertical_segment(cy + intersection_gap, float(y_max))

def _road_rects_for_layout(
    *,
    intersection_center_xy: Sequence[float],
    intersection_layout: str,
    floor_bounds_xy: Tuple[float, float, float, float],
    road_half_width: float,
) -> Dict[str, Tuple[float, float, float, float]]:
    cx, cy = float(intersection_center_xy[0]), float(intersection_center_xy[1])
    road = float(road_half_width)
    x_min, x_max, y_min, y_max = (float(value) for value in floor_bounds_xy)
    horizontal_x0 = x_min if _arm_is_present(str(intersection_layout), "west") else cx - road
    horizontal_x1 = x_max if _arm_is_present(str(intersection_layout), "east") else cx + road
    vertical_y0 = y_min if _arm_is_present(str(intersection_layout), "south") else cy - road
    vertical_y1 = y_max if _arm_is_present(str(intersection_layout), "north") else cy + road
    return {
        "horizontal": (horizontal_x0, cy - road, horizontal_x1, cy + road),
        "vertical": (cx - road, vertical_y0, cx + road, vertical_y1),
        "center": (cx - road, cy - road, cx + road, cy + road),
    }

def _draw_street_shell(
    draw: ImageDraw.ImageDraw,
    *,
    camera,
    frame,
    render_params: Any,
    scene_variant: str,
    intersection_center_xy: Sequence[float],
    intersection_layout: str,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    """Render the full-bleed street surface and return its grounding entity."""

    extent = float(render_params.street_extent)
    road = float(render_params.road_half_width)
    cx, cy = float(intersection_center_xy[0]), float(intersection_center_xy[1])
    sidewalk_fill = render_params.sidewalk_rgb
    if str(scene_variant) == "neighborhood_intersection":
        sidewalk_fill = _tint(sidewalk_fill, 0.04)
    elif str(scene_variant) == "transit_intersection":
        sidewalk_fill = _tint(sidewalk_fill, 0.025)
    floor_polygon_xy, floor_polygon_mode = _visible_floor_polygon_xy(
        camera=camera,
        frame=frame,
        render_params=render_params,
    )
    floor_bounds = _floor_bounds_xy(floor_polygon_xy)
    draw.rectangle(
        (0, 0, int(render_params.canvas_width), int(render_params.canvas_height)),
        fill=sidewalk_fill,
    )
    road_rects = _road_rects_for_layout(
        intersection_center_xy=intersection_center_xy,
        intersection_layout=str(intersection_layout),
        floor_bounds_xy=floor_bounds,
        road_half_width=float(road),
    )
    h_x0, h_y0, h_x1, h_y1 = road_rects["horizontal"]
    _draw_clipped_floor_rect(
        draw,
        camera=camera,
        frame=frame,
        floor_polygon_xy=floor_polygon_xy,
        x0=h_x0,
        y0=h_y0,
        x1=h_x1,
        y1=h_y1,
        z=0.006,
        fill=render_params.asphalt_rgb,
    )
    v_x0, v_y0, v_x1, v_y1 = road_rects["vertical"]
    _draw_clipped_floor_rect(
        draw,
        camera=camera,
        frame=frame,
        floor_polygon_xy=floor_polygon_xy,
        x0=v_x0,
        y0=v_y0,
        x1=v_x1,
        y1=v_y1,
        z=0.008,
        fill=_tint(render_params.asphalt_rgb, 0.03),
    )
    c_x0, c_y0, c_x1, c_y1 = road_rects["center"]
    _draw_clipped_floor_rect(
        draw,
        camera=camera,
        frame=frame,
        floor_polygon_xy=floor_polygon_xy,
        x0=c_x0,
        y0=c_y0,
        x1=c_x1,
        y1=c_y1,
        z=0.012,
        fill=_shade(render_params.asphalt_rgb, 0.92),
    )
    _draw_lane_markings(
        draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        floor_polygon_xy=floor_polygon_xy,
        intersection_center_xy=intersection_center_xy,
        intersection_layout=str(intersection_layout),
    )
    _draw_crosswalks(
        draw,
        camera=camera,
        frame=frame,
        render_params=render_params,
        intersection_center_xy=intersection_center_xy,
        intersection_layout=str(intersection_layout),
    )
    street_bbox = [
        0.0,
        0.0,
        float(render_params.canvas_width),
        float(render_params.canvas_height),
    ]
    return list(street_bbox), [
        {
            "entity_id": "street_intersection_surface",
            "entity_type": "three_d_street_intersection_surface",
            "bbox_px": list(street_bbox),
            "attrs": {
                "scene_variant": str(scene_variant),
                "street_extent": round(float(extent), 4),
                "semantic_street_extent": round(float(extent), 4),
                "render_full_bleed_surface": True,
                "floor_polygon_mode": str(floor_polygon_mode),
                "floor_bounds_xy": [round(float(value), 4) for value in floor_bounds],
                "road_half_width": round(float(road), 4),
                "intersection_center_xy": [round(float(cx), 4), round(float(cy), 4)],
                "intersection_layout": str(intersection_layout),
                "missing_road_arm": _missing_arm_for_layout(str(intersection_layout)),
            },
        }
    ]


__all__ = [
    '_world_polygon',
    '_draw_world_rect',
    '_floor_polygon_area_xy',
    '_line_intersection_xy',
    '_dedupe_polygon_points_xy',
    '_clip_polygon_to_convex_floor',
    '_fallback_floor_polygon_xy',
    '_visible_floor_polygon_xy',
    '_canvas_floor_polygon_available',
    '_floor_bounds_xy',
    '_draw_clipped_floor_rect',
    '_draw_crosswalks',
    '_draw_lane_markings',
    '_road_rects_for_layout',
    '_draw_street_shell',
]
