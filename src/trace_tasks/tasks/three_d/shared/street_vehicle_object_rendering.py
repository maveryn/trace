"""Vehicle and micromobility rendering for street-intersection scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .camera_projection import (
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    polygon_axis_line_segment as _polygon_axis_line_segment,
    project_screen as _project_screen,
    project_xy as _project_xy,
    screen_to_floor_xy as _screen_to_floor_xy,
)
from .color_variation import resolve_three_d_object_fill_rgb
from .object_resources import (
    BUILDING_STYLE_BASE_COLORS,
    BUILDING_STYLE_DIMENSION_FACTORS,
    BUILDING_STYLE_DISPLAY_NAMES,
    STREET_OBJECT_BASE_DIMENSIONS,
    STREET_OBJECT_COLORS,
    STREET_OBJECT_NAMES,
    STREET_RADIAL_OBJECT_TYPES,
    STREET_VEHICLE_OBJECT_TYPES,
)
from .object_scene_rendering import (
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
from .projected_object_geometry import _object_screen_bbox

from .street_object_rendering_common import *  # noqa: F403

def _draw_vehicle_projected_details(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    bbox: Sequence[float],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[List[float]]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    screen_width = max(1.0, x1 - x0)
    screen_height = max(1.0, y1 - y0)
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    cx, cy, base_z = (float(value) for value in spec["base_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    vehicle_type = str(spec.get("object_type"))
    glass = (154, 190, 205)
    outline = (21, 27, 35)
    bboxes: List[List[float]] = []

    def side_quad(long0: float, long1: float, z0: float, z1: float) -> List[Tuple[float, float]]:
        if axis == "x":
            side_sign = 1.0 if float(camera.camera_position[1]) >= cy else -1.0
            y = cy + side_sign * depth * 0.506
            x_start = cx - width * 0.5 + width * float(long0)
            x_end = cx - width * 0.5 + width * float(long1)
            world_points = [
                (x_start, y, base_z + height * float(z0)),
                (x_end, y, base_z + height * float(z0)),
                (x_end, y, base_z + height * float(z1)),
                (x_start, y, base_z + height * float(z1)),
            ]
        else:
            side_sign = 1.0 if float(camera.camera_position[0]) >= cx else -1.0
            x = cx + side_sign * width * 0.506
            y_start = cy - depth * 0.5 + depth * float(long0)
            y_end = cy - depth * 0.5 + depth * float(long1)
            world_points = [
                (x, y_start, base_z + height * float(z0)),
                (x, y_end, base_z + height * float(z0)),
                (x, y_end, base_z + height * float(z1)),
                (x, y_start, base_z + height * float(z1)),
            ]
        return [_project_xy(point, camera, frame) for point in world_points]

    def draw_panel(points: List[Tuple[float, float]], color: Tuple[int, int, int], *, edge: Tuple[int, int, int] = outline) -> None:
        draw.polygon(points, fill=color)
        for index in range(len(points)):
            next_index = index + 1
            if next_index >= len(points):
                next_index = 0
            _draw_line(draw, points[index], points[next_index], fill=edge, width=1)
        bboxes.append(_screen_points_bbox(points, pad_px=1.0))

    if vehicle_type == "bus":
        for start, end in ((0.12, 0.32), (0.38, 0.58), (0.64, 0.84)):
            draw_panel(side_quad(start, end, 0.47, 0.72), glass)
    elif vehicle_type == "delivery_truck":
        draw_panel(side_quad(0.08, 0.34, 0.46, 0.72), glass)
        draw_panel(side_quad(0.48, 0.88, 0.30, 0.62), _tint(fill, 0.20), edge=_shade(fill, 0.58))
    elif vehicle_type == "pickup_truck":
        draw_panel(side_quad(0.12, 0.48, 0.48, 0.73), glass)
        draw_panel(side_quad(0.56, 0.88, 0.28, 0.55), _shade(fill, 0.76), edge=_shade(fill, 0.52))
    elif vehicle_type == "van":
        draw_panel(side_quad(0.18, 0.72, 0.50, 0.74), glass)
    else:
        draw_panel(side_quad(0.30, 0.70, 0.50, 0.73), glass)

    light_y = y1 - screen_height * 0.24
    if str(axis) == "x":
        light_points = (
            [x0 + screen_width * 0.09, light_y, x0 + screen_width * 0.17, light_y + screen_height * 0.06],
            [x1 - screen_width * 0.17, light_y, x1 - screen_width * 0.09, light_y + screen_height * 0.06],
        )
    else:
        light_points = (
            [x0 + screen_width * 0.17, light_y, x0 + screen_width * 0.25, light_y + screen_height * 0.06],
            [x1 - screen_width * 0.25, light_y, x1 - screen_width * 0.17, light_y + screen_height * 0.06],
        )
    for light in light_points:
        draw.rectangle(tuple(light), fill=(244, 224, 116), outline=(88, 72, 35), width=1)
        bboxes.append(light)
    return bboxes

def _draw_vehicle_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    vehicle_type = str(spec.get("object_type"))
    body_h = height * 0.58
    roof_h = height * 0.34
    wheel_h = max(0.055, height * 0.12)
    if axis == "x":
        roof_dims = (width * 0.44, depth * 0.74, roof_h)
        roof_offset = (width * 0.02, 0.0, wheel_h + body_h)
    else:
        roof_dims = (width * 0.74, depth * 0.44, roof_h)
        roof_offset = (0.0, depth * 0.02, wheel_h + body_h)
    parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, 0.0, wheel_h), dimensions_xyz=(width, depth, body_h)),
        _sub_box_spec(spec, offset_xyz=roof_offset, dimensions_xyz=roof_dims),
    ]
    if vehicle_type == "bus":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, wheel_h), dimensions_xyz=(width, depth, height * 0.78)),
        ]
    elif vehicle_type == "van":
        if axis == "x":
            roof_dims = (width * 0.72, depth * 0.80, height * 0.30)
            roof_offset = (width * 0.04, 0.0, wheel_h + height * 0.50)
        else:
            roof_dims = (width * 0.80, depth * 0.72, height * 0.30)
            roof_offset = (0.0, depth * 0.04, wheel_h + height * 0.50)
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, wheel_h), dimensions_xyz=(width, depth, height * 0.62)),
            _sub_box_spec(spec, offset_xyz=roof_offset, dimensions_xyz=roof_dims),
        ]
    elif vehicle_type in {"delivery_truck", "pickup_truck"}:
        if axis == "x":
            cab_offset = (-width * 0.30, 0.0, wheel_h + body_h * 0.06)
            cargo_offset = (width * 0.18, 0.0, wheel_h + body_h * 0.04)
            cab_dims = (width * 0.36, depth * 0.92, body_h * 1.16)
            cargo_dims = (width * 0.58, depth, body_h * (0.78 if vehicle_type == "pickup_truck" else 1.24))
        else:
            cab_offset = (0.0, -depth * 0.30, wheel_h + body_h * 0.06)
            cargo_offset = (0.0, depth * 0.18, wheel_h + body_h * 0.04)
            cab_dims = (width * 0.92, depth * 0.36, body_h * 1.16)
            cargo_dims = (width, depth * 0.58, body_h * (0.78 if vehicle_type == "pickup_truck" else 1.24))
        parts = [
            _sub_box_spec(spec, offset_xyz=cab_offset, dimensions_xyz=cab_dims),
            _sub_box_spec(spec, offset_xyz=cargo_offset, dimensions_xyz=cargo_dims),
        ]
    bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
    x, y, _z = (float(value) for value in spec["base_xyz"])
    wheel_points = []
    if axis == "x":
        for dx in (-width * 0.34, width * 0.34):
            wheel_points.append((x + dx, y + depth * 0.53, wheel_h * 0.62))
    else:
        for dy in (-depth * 0.34, depth * 0.34):
            wheel_points.append((x + width * 0.53, y + dy, wheel_h * 0.62))
    wheel_bboxes: List[List[float]] = []
    for point in wheel_points:
        px, py = _project_xy(point, camera, frame)
        radius = max(4.0, min(11.0, 6.0 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.4))
        wheel_bbox = [px - radius, py - radius * 0.66, px + radius, py + radius * 0.66]
        draw.ellipse(wheel_bbox, fill=(28, 31, 36), outline=(10, 12, 16), width=1)
        wheel_bboxes.append([round(float(value), 3) for value in wheel_bbox])
    if vehicle_type == "taxi":
        top = _project_xy((float(spec["base_xyz"][0]), float(spec["base_xyz"][1]), height + 0.07), camera, frame)
        draw.rectangle((top[0] - 9, top[1] - 5, top[0] + 9, top[1] + 5), fill=(248, 241, 153), outline=(34, 35, 38), width=1)
        wheel_bboxes.append([top[0] - 9, top[1] - 5, top[0] + 9, top[1] + 5])
    detail_bboxes = _draw_vehicle_projected_details(draw, spec, bbox, camera=camera, frame=frame, fill=fill)
    return _bbox_union(bbox, *wheel_bboxes, *detail_bboxes) if wheel_bboxes or detail_bboxes else list(bbox)

def _draw_scooter_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    deck_dims = (width, depth * 0.45, height * 0.22) if axis == "x" else (width * 0.45, depth, height * 0.22)
    deck = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.12), dimensions_xyz=deck_dims)
    bbox = _draw_box_object(draw, deck, camera=camera, frame=frame, fill=fill)
    x, y, _z = (float(value) for value in spec["base_xyz"])
    endpoints = (
        [(x - width * 0.42, y, height * 0.12), (x + width * 0.42, y, height * 0.12)]
        if axis == "x"
        else [(x, y - depth * 0.42, height * 0.12), (x, y + depth * 0.42, height * 0.12)]
    )
    wheel_bboxes = []
    for point in endpoints:
        px, py = _project_xy(point, camera, frame)
        radius = max(5.0, min(11.0, 7.0 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.45))
        wheel_bbox = [px - radius, py - radius, px + radius, py + radius]
        draw.ellipse(wheel_bbox, fill=(24, 27, 33), outline=(8, 9, 12), width=1)
        wheel_bboxes.append(wheel_bbox)
    handle_base = endpoints[-1]
    handle_top = (handle_base[0], handle_base[1], height)
    deck_start = _project_xy(endpoints[0], camera, frame)
    deck_end = _project_xy(endpoints[-1], camera, frame)
    _draw_line(draw, deck_start, deck_end, fill=(245, 241, 221), width=5)
    _draw_line(draw, deck_start, deck_end, fill=fill, width=3)
    _draw_line(
        draw,
        _project_xy(handle_base, camera, frame),
        _project_xy(handle_top, camera, frame),
        fill=(30, 34, 42),
        width=2,
    )
    handle_top_px = _project_xy(handle_top, camera, frame)
    handle_width = max(13.0, min(22.0, 15.0 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.35))
    _draw_line(
        draw,
        (handle_top_px[0] - handle_width * 0.5, handle_top_px[1]),
        (handle_top_px[0] + handle_width * 0.5, handle_top_px[1]),
        fill=(30, 34, 42),
        width=3,
    )
    handle_bbox = _bbox_union(
        [_project_xy(handle_base, camera, frame)[0], _project_xy(handle_base, camera, frame)[1], _project_xy(handle_base, camera, frame)[0], _project_xy(handle_base, camera, frame)[1]],
        [_project_xy(handle_top, camera, frame)[0], _project_xy(handle_top, camera, frame)[1], _project_xy(handle_top, camera, frame)[0], _project_xy(handle_top, camera, frame)[1]],
    )
    handlebar_bbox = [handle_top_px[0] - handle_width * 0.5, handle_top_px[1] - 2.0, handle_top_px[0] + handle_width * 0.5, handle_top_px[1] + 2.0]
    return _bbox_union(bbox, handle_bbox, handlebar_bbox, _screen_line_bbox(deck_start, deck_end, pad_px=4.0), *wheel_bboxes)

def _draw_bicycle_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    x, y, _z = (float(value) for value in spec["base_xyz"])
    if axis == "x":
        wheel_world = [(x - width * 0.40, y, height * 0.18), (x + width * 0.40, y, height * 0.18)]
        frame_world = [(x - width * 0.40, y, height * 0.18), (x, y, height * 0.50), (x + width * 0.40, y, height * 0.18), (x, y, height * 0.18)]
    else:
        wheel_world = [(x, y - depth * 0.40, height * 0.18), (x, y + depth * 0.40, height * 0.18)]
        frame_world = [(x, y - depth * 0.40, height * 0.18), (x, y, height * 0.50), (x, y + depth * 0.40, height * 0.18), (x, y, height * 0.18)]
    bboxes: List[List[float]] = []
    for point in wheel_world:
        px, py = _project_xy(point, camera, frame)
        radius = max(7.0, min(13.0, 8.5 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.45))
        bbox = [px - radius, py - radius, px + radius, py + radius]
        draw.ellipse(bbox, outline=(238, 238, 226), width=4)
        draw.ellipse(bbox, outline=(18, 22, 27), width=2)
        hub_r = max(2.0, radius * 0.14)
        draw.ellipse((px - hub_r, py - hub_r, px + hub_r, py + hub_r), fill=(238, 238, 226), outline=(18, 22, 27), width=1)
        bboxes.append(bbox)
    projected = [_project_xy(point, camera, frame) for point in frame_world]
    for wheel_center in [_project_xy(point, camera, frame) for point in wheel_world]:
        for frame_point in projected:
            if math.hypot(frame_point[0] - wheel_center[0], frame_point[1] - wheel_center[1]) < 32.0:
                _draw_line(draw, wheel_center, frame_point, fill=(238, 238, 226), width=1)
    frame_path = [projected[0], projected[1], projected[2], projected[3], projected[0]]
    draw.line(frame_path, fill=(238, 238, 226), width=5, joint="curve")
    draw.line(frame_path, fill=(36, 112, 190), width=3, joint="curve")
    seat = _project_xy((x, y, height * 0.60), camera, frame)
    handle = _project_xy((x + width * 0.48, y, height * 0.52) if axis == "x" else (x, y + depth * 0.48, height * 0.52), camera, frame)
    _draw_line(draw, (seat[0] - 5.0, seat[1]), (seat[0] + 6.0, seat[1]), fill=(18, 22, 27), width=3)
    _draw_line(draw, handle, (handle[0] + 7.0, handle[1] - 2.0), fill=(18, 22, 27), width=3)
    frame_bbox = _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected])
    return _bbox_union(frame_bbox, *bboxes)

def _draw_motorcycle_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    x, y, _z = (float(value) for value in spec["base_xyz"])
    if axis == "x":
        wheel_world = [(x - width * 0.38, y, height * 0.20), (x + width * 0.38, y, height * 0.20)]
        body_dims = (width * 0.50, depth * 0.72, height * 0.24)
        seat_dims = (width * 0.34, depth * 0.50, height * 0.12)
        handle_base = (x + width * 0.34, y, height * 0.42)
        handle_top = (x + width * 0.47, y, height * 0.66)
        fork_base = (x + width * 0.38, y, height * 0.20)
    else:
        wheel_world = [(x, y - depth * 0.38, height * 0.20), (x, y + depth * 0.38, height * 0.20)]
        body_dims = (width * 0.72, depth * 0.50, height * 0.24)
        seat_dims = (width * 0.50, depth * 0.34, height * 0.12)
        handle_base = (x, y + depth * 0.34, height * 0.42)
        handle_top = (x, y + depth * 0.47, height * 0.66)
        fork_base = (x, y + depth * 0.38, height * 0.20)
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.28), dimensions_xyz=body_dims)
    seat = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.53), dimensions_xyz=seat_dims)
    bboxes: List[List[float]] = [
        _draw_box_object(draw, body, camera=camera, frame=frame, fill=fill),
        _draw_box_object(draw, seat, camera=camera, frame=frame, fill=(34, 38, 43)),
    ]
    for point in wheel_world:
        px, py = _project_xy(point, camera, frame)
        radius = max(8.0, min(16.0, 10.5 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.45))
        wheel_bbox = [px - radius, py - radius, px + radius, py + radius]
        draw.ellipse(wheel_bbox, fill=(25, 28, 32), outline=(8, 9, 12), width=2)
        inner = [px - radius * 0.42, py - radius * 0.42, px + radius * 0.42, py + radius * 0.42]
        draw.ellipse(inner, fill=(101, 111, 120), outline=(35, 39, 44), width=1)
        bboxes.append(wheel_bbox)
    wheel_centers = [_project_xy(point, camera, frame) for point in wheel_world]
    body_center = _project_xy((x, y, height * 0.42), camera, frame)
    tank_top = _project_xy((x, y, height * 0.58), camera, frame)
    if len(wheel_centers) >= 2:
        body_poly = [
            (wheel_centers[0][0], wheel_centers[0][1] - 4.0),
            (body_center[0] - 6.0, body_center[1] - 12.0),
            (tank_top[0] + 10.0, tank_top[1]),
            (wheel_centers[1][0], wheel_centers[1][1] - 4.0),
            (body_center[0] + 7.0, body_center[1] + 6.0),
            (body_center[0] - 8.0, body_center[1] + 6.0),
        ]
        draw.polygon(body_poly, fill=fill, outline=(18, 22, 27))
        bboxes.append(_screen_points_bbox(body_poly, pad_px=2.0))
        seat_line = [
            (body_center[0] - 10.0, body_center[1] - 14.0),
            (body_center[0] + 10.0, body_center[1] - 12.0),
        ]
        _draw_line(draw, seat_line[0], seat_line[1], fill=(18, 22, 27), width=4)
        bboxes.append(_screen_points_bbox(seat_line, pad_px=4.0))
    handle_base_px = _project_xy(handle_base, camera, frame)
    handle_top_px = _project_xy(handle_top, camera, frame)
    fork_base_px = _project_xy(fork_base, camera, frame)
    _draw_line(draw, fork_base_px, handle_base_px, fill=(31, 35, 40), width=3)
    _draw_line(draw, handle_base_px, handle_top_px, fill=(31, 35, 40), width=3)
    handle_w = max(12.0, min(22.0, 15.0 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.35))
    draw.line(
        (handle_top_px[0] - handle_w * 0.5, handle_top_px[1], handle_top_px[0] + handle_w * 0.5, handle_top_px[1]),
        fill=(31, 35, 40),
        width=3,
    )
    headlight_r = max(3.0, min(6.0, 4.2 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.35))
    draw.ellipse(
        (
            handle_base_px[0] - headlight_r,
            handle_base_px[1] - headlight_r,
            handle_base_px[0] + headlight_r,
            handle_base_px[1] + headlight_r,
        ),
        fill=(241, 225, 135),
        outline=(31, 35, 40),
        width=1,
    )
    bboxes.extend(
        [
            _screen_line_bbox(fork_base_px, handle_base_px, pad_px=4.0),
            _screen_line_bbox(handle_base_px, handle_top_px, pad_px=4.0),
            [handle_top_px[0] - handle_w * 0.5, handle_top_px[1] - 3.0, handle_top_px[0] + handle_w * 0.5, handle_top_px[1] + 3.0],
            [handle_base_px[0] - headlight_r, handle_base_px[1] - headlight_r, handle_base_px[0] + headlight_r, handle_base_px[1] + headlight_r],
        ]
    )
    return _bbox_union(*bboxes)


__all__ = [
    '_draw_vehicle_projected_details',
    '_draw_vehicle_object',
    '_draw_scooter_object',
    '_draw_bicycle_object',
    '_draw_motorcycle_object',
]
