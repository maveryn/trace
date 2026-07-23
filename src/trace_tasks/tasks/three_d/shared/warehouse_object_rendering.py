"""Reusable warehouse object rendering helpers for projected 3D scenes."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ...shared.color_distance import coerce_rgb as _rgb
from .color_variation import resolve_three_d_object_fill_rgb
from .object_resources import (
    WAREHOUSE_OBJECT_COLORS,
    WAREHOUSE_ROBOT_ACCENT_COLORS,
    WAREHOUSE_ROBOT_HEADINGS,
)
from .object_scene import _CameraSpec, _ProjectionFrame, _project_screen, _project_xy
from .object_scene_rendering import (
    _bbox_union,
    _draw_box_object,
    _draw_box_parts_object,
    _draw_cylinder_object,
    _draw_line,
    _draw_wedge_object,
    _shade,
    _sub_box_spec,
    _tint,
)


OBJECT_COLORS = WAREHOUSE_OBJECT_COLORS
ROBOT_ACCENT_COLORS = WAREHOUSE_ROBOT_ACCENT_COLORS
SUPPORTED_ROBOT_HEADINGS = WAREHOUSE_ROBOT_HEADINGS


def _heading_vector(robot_heading: str) -> Tuple[float, float]:
    return {
        "east": (1.0, 0.0),
        "west": (-1.0, 0.0),
        "north": (0.0, 1.0),
        "south": (0.0, -1.0),
    }[str(robot_heading)]


def _screen_points_bbox(points: Sequence[Sequence[float]], *, pad_px: float = 0.0) -> List[float]:
    return [
        round(float(min(point[0] for point in points) - float(pad_px)), 3),
        round(float(min(point[1] for point in points) - float(pad_px)), 3),
        round(float(max(point[0] for point in points) + float(pad_px)), 3),
        round(float(max(point[1] for point in points) + float(pad_px)), 3),
    ]


def _screen_line_bbox(p1: Sequence[float], p2: Sequence[float], *, pad_px: float) -> List[float]:
    return [
        round(float(min(float(p1[0]), float(p2[0])) - float(pad_px)), 3),
        round(float(min(float(p1[1]), float(p2[1])) - float(pad_px)), 3),
        round(float(max(float(p1[0]), float(p2[0])) + float(pad_px)), 3),
        round(float(max(float(p1[1]), float(p2[1])) + float(pad_px)), 3),
    ]


def _upright_screen_basis(spec: Mapping[str, Any], camera: _CameraSpec, frame: _ProjectionFrame) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
    x, y, _base_z = (float(value) for value in spec["base_xyz"])
    height = float(spec["dimensions_xyz"][2])
    base = _project_xy((x, y, 0.0), camera, frame)
    top = _project_xy((x, y, height), camera, frame)
    up = (float(top[0]) - float(base[0]), float(top[1]) - float(base[1]))
    height_px = max(1.0, math.hypot(up[0], up[1]))
    up_unit = (up[0] / height_px, up[1] / height_px)
    side_unit = (-up_unit[1], up_unit[0])
    return (float(base[0]), float(base[1])), up_unit, side_unit, height_px


def _draw_warehouse_traffic_cone_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base, up_unit, side_unit, height_px = _upright_screen_basis(spec, camera, frame)
    cone_h = max(42.0, min(78.0, height_px * 1.08))

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    outline = (103, 58, 34)
    base_w = cone_h * 0.62
    foot = [p(-base_w * 0.58, cone_h * 0.03), p(base_w * 0.58, cone_h * 0.03), p(base_w * 0.48, -cone_h * 0.06), p(-base_w * 0.48, -cone_h * 0.06)]
    cone = [p(-base_w * 0.36, cone_h * 0.12), p(0.0, cone_h * 0.94), p(base_w * 0.36, cone_h * 0.12)]
    draw.polygon(foot, fill=_shade(fill, 0.66), outline=outline)
    draw.polygon(cone, fill=fill, outline=outline)
    bboxes = [_screen_points_bbox(foot), _screen_points_bbox(cone)]
    for lower, upper, scale in ((0.31, 0.40, 0.52), (0.57, 0.65, 0.28)):
        band = [p(-base_w * scale, cone_h * lower), p(base_w * scale, cone_h * lower), p(base_w * scale * 0.78, cone_h * upper), p(-base_w * scale * 0.78, cone_h * upper)]
        draw.polygon(band, fill=(246, 238, 211), outline=(150, 86, 43))
        bboxes.append(_screen_points_bbox(band))
    return _bbox_union(*bboxes)


def _draw_warehouse_fire_extinguisher_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base, up_unit, side_unit, height_px = _upright_screen_basis(spec, camera, frame)
    extinguisher_h = max(46.0, min(82.0, height_px * 1.06))
    body_w = max(7, min(13, int(round(extinguisher_h * 0.17))))

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    outline = (82, 28, 28)
    bboxes: List[List[float]] = []
    body_bottom = p(0.0, extinguisher_h * 0.10)
    body_top = p(0.0, extinguisher_h * 0.72)
    _draw_line(draw, body_bottom, body_top, fill=outline, width=body_w + 4)
    _draw_line(draw, body_bottom, body_top, fill=fill, width=body_w)
    bboxes.append(_screen_line_bbox(body_bottom, body_top, pad_px=float(body_w + 3)))
    label_center = p(0.0, extinguisher_h * 0.43)
    label_bbox = [
        label_center[0] - body_w * 0.55,
        label_center[1] - extinguisher_h * 0.055,
        label_center[0] + body_w * 0.55,
        label_center[1] + extinguisher_h * 0.055,
    ]
    draw.rectangle(tuple(label_bbox), fill=(246, 233, 184), outline=(99, 46, 35), width=1)
    bboxes.append(label_bbox)
    neck_bottom = p(0.0, extinguisher_h * 0.70)
    neck_top = p(0.0, extinguisher_h * 0.84)
    _draw_line(draw, neck_bottom, neck_top, fill=(51, 51, 54), width=max(3, body_w // 2))
    bboxes.append(_screen_line_bbox(neck_bottom, neck_top, pad_px=4.0))
    handle_left = p(-body_w * 0.85, extinguisher_h * 0.86)
    handle_right = p(body_w * 0.85, extinguisher_h * 0.86)
    _draw_line(draw, handle_left, handle_right, fill=(32, 34, 37), width=4)
    bboxes.append(_screen_line_bbox(handle_left, handle_right, pad_px=4.0))
    hose_mid = p(body_w * 1.25, extinguisher_h * 0.70)
    nozzle = p(body_w * 1.75, extinguisher_h * 0.61)
    _draw_line(draw, neck_top, hose_mid, fill=(28, 29, 32), width=3)
    _draw_line(draw, hose_mid, nozzle, fill=(28, 29, 32), width=3)
    bboxes.extend([_screen_line_bbox(neck_top, hose_mid, pad_px=3.0), _screen_line_bbox(hose_mid, nozzle, pad_px=3.0)])
    return _bbox_union(*bboxes)


def _draw_warehouse_bollard_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base, up_unit, side_unit, height_px = _upright_screen_basis(spec, camera, frame)
    bollard_h = max(48.0, min(88.0, height_px * 1.05))
    post_w = max(8, min(15, int(round(bollard_h * 0.16))))

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    bottom = p(0.0, bollard_h * 0.06)
    top = p(0.0, bollard_h * 0.88)
    _draw_line(draw, bottom, top, fill=(53, 48, 34), width=post_w + 4)
    _draw_line(draw, bottom, top, fill=fill, width=post_w)
    bboxes = [_screen_line_bbox(bottom, top, pad_px=float(post_w + 3))]
    for frac in (0.28, 0.50, 0.72):
        left = p(-post_w * 0.50, bollard_h * frac)
        right = p(post_w * 0.50, bollard_h * (frac + 0.055))
        _draw_line(draw, left, right, fill=(34, 36, 38), width=4)
        bboxes.append(_screen_line_bbox(left, right, pad_px=4.0))
    foot = p(0.0, bollard_h * 0.05)
    foot_bbox = [foot[0] - post_w * 1.1, foot[1] - 3.0, foot[0] + post_w * 1.1, foot[1] + 3.0]
    draw.ellipse(tuple(foot_bbox), fill=(61, 62, 58), outline=(34, 34, 33), width=1)
    bboxes.append(foot_bbox)
    return _bbox_union(*bboxes)


def _draw_warehouse_floor_sign_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base, up_unit, side_unit, height_px = _upright_screen_basis(spec, camera, frame)
    sign_h = max(40.0, min(68.0, height_px * 1.10))
    sign_w = sign_h * 0.58

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    panel = [p(-sign_w * 0.50, sign_h * 0.05), p(0.0, sign_h * 0.92), p(sign_w * 0.50, sign_h * 0.05), p(sign_w * 0.35, -sign_h * 0.03), p(-sign_w * 0.35, -sign_h * 0.03)]
    draw.polygon(panel, fill=fill, outline=(84, 68, 30))
    bboxes = [_screen_points_bbox(panel)]
    warning = [p(-sign_w * 0.18, sign_h * 0.32), p(0.0, sign_h * 0.58), p(sign_w * 0.18, sign_h * 0.32)]
    draw.polygon(warning, fill=(32, 33, 35))
    bboxes.append(_screen_points_bbox(warning))
    dot = p(0.0, sign_h * 0.27)
    draw.ellipse((dot[0] - 1.8, dot[1] - 1.8, dot[0] + 1.8, dot[1] + 1.8), fill=(32, 33, 35))
    _draw_line(draw, p(0.0, sign_h * 0.36), p(0.0, sign_h * 0.47), fill=fill, width=2)
    return _bbox_union(*bboxes)


def _draw_warehouse_ladder_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base, up_unit, side_unit, height_px = _upright_screen_basis(spec, camera, frame)
    ladder_h = max(56.0, min(100.0, height_px * 1.08))
    ladder_w = ladder_h * 0.42

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    rail_left_bottom = p(-ladder_w * 0.42, 0.0)
    rail_left_top = p(-ladder_w * 0.32, ladder_h * 0.96)
    rail_right_bottom = p(ladder_w * 0.42, 0.0)
    rail_right_top = p(ladder_w * 0.32, ladder_h * 0.96)
    bboxes: List[List[float]] = []
    for start, end in ((rail_left_bottom, rail_left_top), (rail_right_bottom, rail_right_top)):
        _draw_line(draw, start, end, fill=(53, 62, 70), width=5)
        _draw_line(draw, start, end, fill=fill, width=3)
        bboxes.append(_screen_line_bbox(start, end, pad_px=5.0))
    for frac in (0.16, 0.32, 0.48, 0.64, 0.80):
        left = p(-ladder_w * (0.42 - frac * 0.10), ladder_h * frac)
        right = p(ladder_w * (0.42 - frac * 0.10), ladder_h * frac)
        _draw_line(draw, left, right, fill=(224, 228, 224), width=4)
        _draw_line(draw, left, right, fill=fill, width=2)
        bboxes.append(_screen_line_bbox(left, right, pad_px=4.0))
    return _bbox_union(*bboxes)


def _draw_warehouse_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    object_type = str(spec["object_type"])
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    if object_type == "warehouse_robot":
        accent = _rgb(spec.get("robot_accent_rgb"), ROBOT_ACCENT_COLORS[0])
        robot_design = str(spec.get("robot_design", "low_cart"))
        if robot_design == "sensor_tower":
            parts = [
                _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.04), dimensions_xyz=(width, depth, height * 0.34)),
                _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.34), dimensions_xyz=(width * 0.36, depth * 0.40, height * 0.46)),
                _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.78), dimensions_xyz=(width * 0.56, depth * 0.50, height * 0.20)),
            ]
            accent_indices = {2}
        elif robot_design == "stacker_bot":
            parts = [
                _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.04), dimensions_xyz=(width, depth, height * 0.32)),
                _sub_box_spec(spec, offset_xyz=(-width * 0.28, -depth * 0.28, height * 0.24), dimensions_xyz=(width * 0.10, depth * 0.10, height * 0.72)),
                _sub_box_spec(spec, offset_xyz=(width * 0.28, -depth * 0.28, height * 0.24), dimensions_xyz=(width * 0.10, depth * 0.10, height * 0.72)),
                _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.28, height * 0.78), dimensions_xyz=(width * 0.72, depth * 0.08, height * 0.12)),
                _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.22, height * 0.30), dimensions_xyz=(width * 0.56, depth * 0.12, height * 0.10)),
            ]
            accent_indices = {3, 4}
        else:
            parts = [
                _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.04), dimensions_xyz=(width, depth, height * 0.50)),
                _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.48), dimensions_xyz=(width * 0.58, depth * 0.54, height * 0.30)),
                _sub_box_spec(spec, offset_xyz=(width * 0.24, 0.0, height * 0.74), dimensions_xyz=(width * 0.16, depth * 0.18, height * 0.20)),
            ]
            accent_indices = {2}
        heading = str(spec.get("robot_heading", "east" if str(spec.get("orientation_axis")) == "x" else "north"))
        heading_xy = _heading_vector(heading) if heading in set(SUPPORTED_ROBOT_HEADINGS) else (1.0, 0.0)
        fx, fy = float(heading_xy[0]), float(heading_xy[1])
        arm_base_z = height * 0.50
        if abs(fx) > 0.0:
            arm_dims = (width * 0.46, depth * 0.13, height * 0.12)
            arm_offset = (fx * width * 0.43, 0.0, arm_base_z)
            finger_dims = (width * 0.24, depth * 0.065, height * 0.10)
            finger_forward = fx * width * 0.74
            finger_offsets = [
                (finger_forward, -depth * 0.16, arm_base_z - height * 0.01),
                (finger_forward, depth * 0.16, arm_base_z - height * 0.01),
            ]
        else:
            arm_dims = (width * 0.13, depth * 0.46, height * 0.12)
            arm_offset = (0.0, fy * depth * 0.43, arm_base_z)
            finger_dims = (width * 0.065, depth * 0.24, height * 0.10)
            finger_forward = fy * depth * 0.74
            finger_offsets = [
                (-width * 0.16, finger_forward, arm_base_z - height * 0.01),
                (width * 0.16, finger_forward, arm_base_z - height * 0.01),
            ]
        parts.append(_sub_box_spec(spec, offset_xyz=arm_offset, dimensions_xyz=arm_dims))
        arm_index = len(parts) - 1
        for finger_offset in finger_offsets:
            parts.append(_sub_box_spec(spec, offset_xyz=finger_offset, dimensions_xyz=finger_dims))
        accent_indices = set(accent_indices) | {arm_index, arm_index + 1, arm_index + 2}
        bboxes: List[List[float]] = []
        for index, part in enumerate(sorted(enumerate(parts), key=lambda item: float(_project_screen(item[1]["world_xyz"], camera, frame)[7]), reverse=True)):
            part_index, part_spec = part
            part_fill = accent if part_index in accent_indices else (_tint(fill, 0.05) if index % 2 == 0 else _shade(fill, 0.92))
            bboxes.append(_draw_box_object(draw, part_spec, camera=camera, frame=frame, fill=part_fill))
        x, y, _base_z = (float(value) for value in spec["base_xyz"])
        robot_bbox = _bbox_union(*bboxes)
        x0, y0, x1, y1 = (float(value) for value in robot_bbox)
        panel = [x0 + (x1 - x0) * 0.32, y0 + (y1 - y0) * 0.24, x1 - (x1 - x0) * 0.32, y0 + (y1 - y0) * 0.42]
        draw.rectangle(tuple(panel), fill=accent, outline=(31, 36, 44), width=1)
        bboxes.append(panel)
        if abs(fx) > 0.0:
            near_y = y + (1.0 if float(camera.camera_position[1]) >= y else -1.0) * depth * 0.45
            wheel_points = [
                (x - width * 0.32, near_y, height * 0.09),
                (x + width * 0.32, near_y, height * 0.09),
            ]
        else:
            near_x = x + (1.0 if float(camera.camera_position[0]) >= x else -1.0) * width * 0.45
            wheel_points = [
                (near_x, y - depth * 0.32, height * 0.09),
                (near_x, y + depth * 0.32, height * 0.09),
            ]
        wheel_radius = max(2.5, min(4.8, 3.3 * (7.0 / max(2.4, float(spec["camera_distance"]))) ** 0.30))
        for point in wheel_points:
            px, py = _project_xy(point, camera, frame)
            wheel_bbox = [px - wheel_radius, py - wheel_radius * 0.68, px + wheel_radius, py + wheel_radius * 0.68]
            draw.ellipse(tuple(wheel_bbox), fill=(24, 26, 30), outline=(7, 8, 10), width=1)
            bboxes.append([round(float(value), 3) for value in wheel_bbox])
        sensor = _project_xy((x + fx * width * 0.42, y + fy * depth * 0.42, height * 0.48), camera, frame)
        sensor_radius = max(2.5, min(4.6, wheel_radius * 0.92))
        sensor_bbox = [sensor[0] - sensor_radius, sensor[1] - sensor_radius, sensor[0] + sensor_radius, sensor[1] + sensor_radius]
        draw.ellipse(tuple(sensor_bbox), fill=(231, 221, 126), outline=(31, 35, 42), width=1)
        bboxes.append(sensor_bbox)
        return _bbox_union(*bboxes)
    if object_type == "shelf_rack":
        raise ValueError("warehouse shelf_rack is scene-local support, not a reusable loose object")
    if object_type in {"crate_stack", "box_stack"}:
        parts = [
            _sub_box_spec(spec, offset_xyz=(-width * 0.21, -depth * 0.16, 0.0), dimensions_xyz=(width * 0.52, depth * 0.52, height * 0.50)),
            _sub_box_spec(spec, offset_xyz=(width * 0.19, depth * 0.14, 0.0), dimensions_xyz=(width * 0.52, depth * 0.52, height * 0.48)),
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.47), dimensions_xyz=(width * 0.58, depth * 0.54, height * 0.52)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        if object_type == "crate_stack":
            for left, right in (((x0 + 4.0, y0 + 5.0), (x1 - 4.0, y1 - 5.0)), ((x1 - 4.0, y0 + 5.0), (x0 + 4.0, y1 - 5.0))):
                _draw_line(draw, left, right, fill=(98, 62, 35), width=2)
                bboxes.append(_screen_line_bbox(left, right, pad_px=2.0))
            for frac in (0.36, 0.64):
                y_line = y0 + (y1 - y0) * frac
                _draw_line(draw, (x0 + 4.0, y_line), (x1 - 4.0, y_line), fill=(118, 73, 39), width=1)
                bboxes.append(_screen_line_bbox((x0 + 4.0, y_line), (x1 - 4.0, y_line), pad_px=1.0))
        else:
            stripe_x = x0 + (x1 - x0) * 0.50
            stripe_y = y0 + (y1 - y0) * 0.42
            _draw_line(draw, (stripe_x, y0 + 3.0), (stripe_x, y1 - 3.0), fill=(222, 203, 148), width=2)
            _draw_line(draw, (x0 + 4.0, stripe_y), (x1 - 4.0, stripe_y), fill=(222, 203, 148), width=2)
            bboxes.extend([
                _screen_line_bbox((stripe_x, y0 + 3.0), (stripe_x, y1 - 3.0), pad_px=2.0),
                _screen_line_bbox((x0 + 4.0, stripe_y), (x1 - 4.0, stripe_y), pad_px=2.0),
            ])
            for frac in (0.28, 0.68):
                y_line = y0 + (y1 - y0) * frac
                _draw_line(draw, (x0 + 5.0, y_line), (x1 - 5.0, y_line), fill=(123, 93, 54), width=1)
                bboxes.append(_screen_line_bbox((x0 + 5.0, y_line), (x1 - 5.0, y_line), pad_px=1.0))
        return _bbox_union(*bboxes)
    if object_type == "pallet_load":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height * 0.18)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.18, 0.0, height * 0.15), dimensions_xyz=(width * 0.52, depth * 0.80, height * 0.46)),
            _sub_box_spec(spec, offset_xyz=(width * 0.22, -depth * 0.08, height * 0.15), dimensions_xyz=(width * 0.42, depth * 0.58, height * 0.50)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        for frac in (0.28, 0.50, 0.72):
            x_line = x0 + (x1 - x0) * frac
            _draw_line(draw, (x_line, y1 - (y1 - y0) * 0.22), (x_line, y1 - 2.0), fill=(93, 66, 42), width=2)
            bboxes.append(_screen_line_bbox((x_line, y1 - (y1 - y0) * 0.22), (x_line, y1 - 2.0), pad_px=2.0))
        strap_y = y0 + (y1 - y0) * 0.42
        _draw_line(draw, (x0 + 4.0, strap_y), (x1 - 4.0, strap_y), fill=(56, 58, 62), width=3)
        bboxes.append(_screen_line_bbox((x0 + 4.0, strap_y), (x1 - 4.0, strap_y), pad_px=3.0))
        top_strap_x = x0 + (x1 - x0) * 0.58
        _draw_line(draw, (top_strap_x, y0 + 4.0), (top_strap_x, y1 - 5.0), fill=(56, 58, 62), width=2)
        bboxes.append(_screen_line_bbox((top_strap_x, y0 + 4.0), (top_strap_x, y1 - 5.0), pad_px=2.0))
        return _bbox_union(*bboxes)
    if object_type == "tool_cart":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.05), dimensions_xyz=(width, depth, height * 0.72)),
            _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.45, height * 0.46), dimensions_xyz=(width * 0.88, 0.08, height * 0.12)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        for frac in (0.38, 0.58):
            y_line = y0 + (y1 - y0) * frac
            _draw_line(draw, (x0 + 5.0, y_line), (x1 - 5.0, y_line), fill=(34, 70, 66), width=2)
            bboxes.append(_screen_line_bbox((x0 + 5.0, y_line), (x1 - 5.0, y_line), pad_px=2.0))
        for frac in (0.22, 0.78):
            wheel_x = x0 + (x1 - x0) * frac
            wheel_y = y1 - (y1 - y0) * 0.06
            wheel_bbox = [wheel_x - 4.0, wheel_y - 3.0, wheel_x + 4.0, wheel_y + 3.0]
            draw.ellipse(tuple(wheel_bbox), fill=(26, 29, 32), outline=(8, 9, 11), width=1)
            bboxes.append(wheel_bbox)
        handle_y = y0 + (y1 - y0) * 0.22
        _draw_line(draw, (x1 - 3.0, handle_y), (x1 + 9.0, handle_y - 4.0), fill=(36, 43, 45), width=3)
        bboxes.append(_screen_line_bbox((x1 - 3.0, handle_y), (x1 + 9.0, handle_y - 4.0), pad_px=3.0))
        return _bbox_union(*bboxes)
    if object_type == "workbench":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.58), dimensions_xyz=(width, depth, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.38, -depth * 0.34, 0.0), dimensions_xyz=(0.09, 0.09, height * 0.74)),
            _sub_box_spec(spec, offset_xyz=(width * 0.38, -depth * 0.34, 0.0), dimensions_xyz=(0.09, 0.09, height * 0.74)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.38, depth * 0.34, 0.0), dimensions_xyz=(0.09, 0.09, height * 0.74)),
            _sub_box_spec(spec, offset_xyz=(width * 0.38, depth * 0.34, 0.0), dimensions_xyz=(0.09, 0.09, height * 0.74)),
        ]
        return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
    if object_type == "rolling_bin":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.08), dimensions_xyz=(width, depth, height * 0.80)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.30, -depth * 0.28, 0.0), dimensions_xyz=(0.12, 0.12, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(width * 0.30, -depth * 0.28, 0.0), dimensions_xyz=(0.12, 0.12, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.30, depth * 0.28, 0.0), dimensions_xyz=(0.12, 0.12, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(width * 0.30, depth * 0.28, 0.0), dimensions_xyz=(0.12, 0.12, height * 0.16)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        lid_y = y0 + (y1 - y0) * 0.18
        _draw_line(draw, (x0 + 5.0, lid_y), (x1 - 5.0, lid_y), fill=(32, 56, 65), width=3)
        bboxes.append(_screen_line_bbox((x0 + 5.0, lid_y), (x1 - 5.0, lid_y), pad_px=3.0))
        handle_bbox = [x0 + (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.34, x1 - (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.44]
        draw.rectangle(tuple(handle_bbox), fill=(205, 226, 230), outline=(31, 55, 60), width=1)
        bboxes.append(handle_bbox)
        return _bbox_union(*bboxes)
    if object_type == "pallet_jack":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.02), dimensions_xyz=(width * 0.72, depth * 0.34, height * 0.32)),
            _sub_box_spec(spec, offset_xyz=(width * 0.28, -depth * 0.24, 0.02), dimensions_xyz=(width * 0.46, depth * 0.12, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(width * 0.28, depth * 0.24, 0.02), dimensions_xyz=(width * 0.46, depth * 0.12, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.38, 0.0, height * 0.20), dimensions_xyz=(0.09, 0.10, height * 0.82)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x, y, _z = (float(value) for value in spec["base_xyz"])
        handle_bottom = _project_xy((x - width * 0.38, y, height * 0.42), camera, frame)
        handle_top = _project_xy((x - width * 0.50, y, height * 0.96), camera, frame)
        _draw_line(draw, handle_bottom, handle_top, fill=(72, 38, 38), width=5)
        _draw_line(draw, handle_bottom, handle_top, fill=fill, width=3)
        wheel_bboxes: List[List[float]] = [_screen_line_bbox(handle_bottom, handle_top, pad_px=5.0)]
        for point in ((x + width * 0.50, y - depth * 0.24, height * 0.05), (x + width * 0.50, y + depth * 0.24, height * 0.05), (x - width * 0.28, y, height * 0.05)):
            px, py = _project_xy(point, camera, frame)
            wheel_bbox = [px - 3.0, py - 2.3, px + 3.0, py + 2.3]
            draw.ellipse(tuple(wheel_bbox), fill=(24, 26, 28), outline=(8, 9, 10), width=1)
            wheel_bboxes.append(wheel_bbox)
        fork_left_start = _project_xy((x + width * 0.18, y - depth * 0.25, height * 0.12), camera, frame)
        fork_left_end = _project_xy((x + width * 0.56, y - depth * 0.25, height * 0.12), camera, frame)
        fork_right_start = _project_xy((x + width * 0.18, y + depth * 0.25, height * 0.12), camera, frame)
        fork_right_end = _project_xy((x + width * 0.56, y + depth * 0.25, height * 0.12), camera, frame)
        for start, end in ((fork_left_start, fork_left_end), (fork_right_start, fork_right_end)):
            _draw_line(draw, start, end, fill=(92, 38, 35), width=4)
            _draw_line(draw, start, end, fill=fill, width=2)
            wheel_bboxes.append(_screen_line_bbox(start, end, pad_px=4.0))
        return _bbox_union(bbox, *wheel_bboxes)
    if object_type == "forklift":
        parts = [
            _sub_box_spec(spec, offset_xyz=(-width * 0.10, 0.0, 0.0), dimensions_xyz=(width * 0.62, depth, height * 0.58)),
            _sub_box_spec(spec, offset_xyz=(width * 0.34, 0.0, 0.0), dimensions_xyz=(0.10, depth * 0.92, height)),
            _sub_box_spec(spec, offset_xyz=(width * 0.52, -depth * 0.24, 0.02), dimensions_xyz=(width * 0.44, depth * 0.12, height * 0.10)),
            _sub_box_spec(spec, offset_xyz=(width * 0.52, depth * 0.24, 0.02), dimensions_xyz=(width * 0.44, depth * 0.12, height * 0.10)),
            _sub_box_spec(spec, offset_xyz=(-width * 0.16, 0.0, height * 0.64), dimensions_xyz=(width * 0.40, depth * 0.68, height * 0.12)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x, y, _z = (float(value) for value in spec["base_xyz"])
        bboxes = [list(bbox)]
        x0, y0, x1, y1 = (float(value) for value in bbox)
        cabin = [x0 + (x1 - x0) * 0.22, y0 + (y1 - y0) * 0.16, x0 + (x1 - x0) * 0.47, y0 + (y1 - y0) * 0.42]
        draw.rectangle(tuple(cabin), fill=(188, 214, 223), outline=(46, 62, 72), width=1)
        bboxes.append(cabin)
        for point in ((x - width * 0.34, y - depth * 0.44, height * 0.08), (x + width * 0.24, y - depth * 0.44, height * 0.08)):
            px, py = _project_xy(point, camera, frame)
            wheel_bbox = [px - 6.0, py - 4.0, px + 6.0, py + 4.0]
            draw.ellipse(tuple(wheel_bbox), fill=(24, 25, 28), outline=(8, 9, 10), width=1)
            bboxes.append(wheel_bbox)
        return _bbox_union(*bboxes)
    if object_type == "wrapped_bundle":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height)),
            _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.18, height * 0.34), dimensions_xyz=(width * 1.04, 0.06, height * 0.14)),
            _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.18, height * 0.34), dimensions_xyz=(width * 1.04, 0.06, height * 0.14)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        for frac in (0.32, 0.68):
            x_line = x0 + (x1 - x0) * frac
            _draw_line(draw, (x_line, y0 + 4.0), (x_line, y1 - 4.0), fill=(93, 82, 70), width=2)
            bboxes.append(_screen_line_bbox((x_line, y0 + 4.0), (x_line, y1 - 4.0), pad_px=2.0))
        label = [x0 + (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.38, x1 - (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.54]
        draw.rectangle(tuple(label), fill=(235, 228, 204), outline=(102, 91, 76), width=1)
        bboxes.append(label)
        return _bbox_union(*bboxes)
    if object_type == "stacked_pipes":
        parts = [
            _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.25, 0.0), dimensions_xyz=(width, depth * 0.24, height * 0.30)),
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.24), dimensions_xyz=(width * 0.96, depth * 0.24, height * 0.30)),
            _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.25, 0.0), dimensions_xyz=(width, depth * 0.24, height * 0.30)),
        ]
        bbox = _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        for frac in (0.25, 0.50, 0.75):
            cx = x0 + (x1 - x0) * 0.17
            cy = y0 + (y1 - y0) * frac
            pipe_end = [cx - 5.5, cy - 4.2, cx + 5.5, cy + 4.2]
            draw.ellipse(tuple(pipe_end), fill=(212, 222, 225), outline=(70, 81, 90), width=1)
            inner = [cx - 2.4, cy - 1.8, cx + 2.4, cy + 1.8]
            draw.ellipse(tuple(inner), fill=(82, 94, 104))
            bboxes.extend([pipe_end, inner])
            rx = x1 - (x1 - x0) * 0.16
            far_end = [rx - 4.0, cy - 3.2, rx + 4.0, cy + 3.2]
            draw.ellipse(tuple(far_end), outline=(83, 95, 106), width=1)
            bboxes.append(far_end)
        return _bbox_union(*bboxes)
    if object_type == "ladder":
        return _draw_warehouse_ladder_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "barrel":
        bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=(154, 99, 48))
        x0, y0, x1, y1 = (float(value) for value in bbox)
        width_px = max(1.0, x1 - x0)
        height_px = max(1.0, y1 - y0)
        hoop_rgb = (58, 66, 74)
        wood_line = (92, 55, 31)
        for frac in (0.20, 0.50, 0.80):
            y = y0 + height_px * frac
            band = [x0 + width_px * 0.10, y - height_px * 0.035, x1 - width_px * 0.10, y + height_px * 0.035]
            draw.rectangle(tuple(band), fill=hoop_rgb, outline=(31, 36, 43), width=1)
        for frac in (0.32, 0.50, 0.68):
            x_line = x0 + width_px * frac
            _draw_line(draw, (x_line, y0 + height_px * 0.14), (x_line, y1 - height_px * 0.12), fill=wood_line, width=1)
        tap = [
            x0 + width_px * 0.58,
            y0 + height_px * 0.42,
            x0 + width_px * 0.75,
            y0 + height_px * 0.52,
        ]
        draw.rectangle(tuple(tap), fill=(202, 157, 76), outline=(44, 35, 26), width=1)
        return _bbox_union(bbox, tap)
    if object_type == "tire_stack":
        bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        for frac in (0.30, 0.52, 0.74):
            y = y0 + (y1 - y0) * frac
            _draw_line(draw, (x0 + 4.0, y), (x1 - 4.0, y), fill=(17, 19, 22), width=3)
            bboxes.append(_screen_line_bbox((x0 + 4.0, y), (x1 - 4.0, y), pad_px=3.0))
        inner = [x0 + (x1 - x0) * 0.32, y0 + (y1 - y0) * 0.18, x1 - (x1 - x0) * 0.32, y0 + (y1 - y0) * 0.35]
        draw.ellipse(tuple(inner), fill=(33, 36, 41), outline=(12, 13, 15), width=1)
        bboxes.append(inner)
        return _bbox_union(*bboxes)
    if object_type == "trash_can":
        bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        lid_y = y0 + (y1 - y0) * 0.18
        _draw_line(draw, (x0 + 4.0, lid_y), (x1 - 4.0, lid_y), fill=(32, 39, 38), width=3)
        bboxes.append(_screen_line_bbox((x0 + 4.0, lid_y), (x1 - 4.0, lid_y), pad_px=3.0))
        lid = [x0 + (x1 - x0) * 0.12, y0 + (y1 - y0) * 0.08, x1 - (x1 - x0) * 0.12, y0 + (y1 - y0) * 0.22]
        draw.ellipse(tuple(lid), fill=(61, 73, 70), outline=(28, 34, 33), width=1)
        bboxes.append(lid)
        for frac in (0.35, 0.50, 0.65):
            x_line = x0 + (x1 - x0) * frac
            _draw_line(draw, (x_line, y0 + (y1 - y0) * 0.28), (x_line, y1 - 3.0), fill=(48, 58, 56), width=2)
            bboxes.append(_screen_line_bbox((x_line, y0 + (y1 - y0) * 0.28), (x_line, y1 - 3.0), pad_px=2.0))
        handle = [x0 + (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.26, x1 - (x1 - x0) * 0.36, y0 + (y1 - y0) * 0.34]
        draw.rectangle(tuple(handle), fill=(198, 207, 202), outline=(48, 58, 56), width=1)
        bboxes.append(handle)
        return _bbox_union(*bboxes)
    if object_type == "warning_bollard":
        return _draw_warehouse_bollard_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "fire_extinguisher":
        return _draw_warehouse_fire_extinguisher_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "traffic_cone":
        return _draw_warehouse_traffic_cone_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "floor_sign":
        return _draw_warehouse_floor_sign_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "charging_dock":
        bbox = _draw_wedge_object(draw, spec, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        light = [x0 + (x1 - x0) * 0.18, y0 + (y1 - y0) * 0.34, x0 + (x1 - x0) * 0.28, y0 + (y1 - y0) * 0.45]
        draw.ellipse(tuple(light), fill=(78, 204, 120), outline=(24, 78, 50), width=1)
        plug_y = y0 + (y1 - y0) * 0.60
        _draw_line(draw, (x0 + (x1 - x0) * 0.35, plug_y), (x1 - (x1 - x0) * 0.18, plug_y), fill=(224, 229, 224), width=3)
        bboxes.extend([light, _screen_line_bbox((x0 + (x1 - x0) * 0.35, plug_y), (x1 - (x1 - x0) * 0.18, plug_y), pad_px=3.0)])
        cable = [(x0 + (x1 - x0) * 0.48, y0 + (y1 - y0) * 0.70), (x0 + (x1 - x0) * 0.62, y1 - 3.0), (x1 - 3.0, y1 - 6.0)]
        _draw_line(draw, cable[0], cable[1], fill=(32, 36, 40), width=2)
        _draw_line(draw, cable[1], cable[2], fill=(32, 36, 40), width=2)
        bboxes.extend([_screen_line_bbox(cable[0], cable[1], pad_px=2.0), _screen_line_bbox(cable[1], cable[2], pad_px=2.0)])
        return _bbox_union(*bboxes)
    if object_type == "conveyor":
        bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        belt = [x0 + (x1 - x0) * 0.08, y0 + (y1 - y0) * 0.28, x1 - (x1 - x0) * 0.08, y0 + (y1 - y0) * 0.52]
        draw.rectangle(tuple(belt), fill=(46, 53, 61), outline=(22, 25, 29), width=1)
        bboxes.append(belt)
        for frac in (0.20, 0.36, 0.52, 0.68, 0.84):
            x_line = x0 + (x1 - x0) * frac
            _draw_line(draw, (x_line, belt[1]), (x_line, belt[3]), fill=(146, 155, 162), width=2)
            bboxes.append(_screen_line_bbox((x_line, belt[1]), (x_line, belt[3]), pad_px=2.0))
        return _bbox_union(*bboxes)
    if object_type == "pallet":
        bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        bboxes = [list(bbox)]
        for frac in (0.24, 0.50, 0.76):
            x_line = x0 + (x1 - x0) * frac
            _draw_line(draw, (x_line, y0 + 3.0), (x_line, y1 - 2.0), fill=(87, 61, 40), width=2)
            bboxes.append(_screen_line_bbox((x_line, y0 + 3.0), (x_line, y1 - 2.0), pad_px=2.0))
        for frac in (0.34, 0.66):
            y_line = y0 + (y1 - y0) * frac
            _draw_line(draw, (x0 + 4.0, y_line), (x1 - 4.0, y_line), fill=(101, 71, 45), width=2)
            bboxes.append(_screen_line_bbox((x0 + 4.0, y_line), (x1 - 4.0, y_line), pad_px=2.0))
        return _bbox_union(*bboxes)
    if object_type == "storage_bin":
        bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        lid_y = y0 + (y1 - y0) * 0.22
        label = [x0 + (x1 - x0) * 0.28, y0 + (y1 - y0) * 0.42, x1 - (x1 - x0) * 0.28, y0 + (y1 - y0) * 0.56]
        _draw_line(draw, (x0 + 4.0, lid_y), (x1 - 4.0, lid_y), fill=(31, 73, 47), width=3)
        draw.rectangle(tuple(label), fill=(224, 236, 214), outline=(38, 86, 49), width=1)
        return _bbox_union(bbox, _screen_line_bbox((x0 + 4.0, lid_y), (x1 - 4.0, lid_y), pad_px=3.0), label)
    return _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)


def _fill_for_object(spec: Mapping[str, Any], *, scene_variant: str) -> Tuple[int, int, int]:
    if str(spec.get("object_type")) == "warehouse_robot" and isinstance(spec.get("robot_base_rgb"), Sequence):
        return _rgb(spec.get("robot_base_rgb"), OBJECT_COLORS["warehouse_robot"])
    if str(spec.get("object_type")) == "shelf_rack" and isinstance(spec.get("shelf_frame_rgb"), Sequence):
        return _rgb(spec.get("shelf_frame_rgb"), OBJECT_COLORS["shelf_rack"])
    base_rgb = OBJECT_COLORS.get(str(spec["object_type"]), (126, 136, 146))
    variation_strength = 0.24 if bool(spec.get("is_answer_candidate", False)) else 0.18
    return resolve_three_d_object_fill_rgb(
        spec,
        base_rgb=base_rgb,
        salt=f"{scene_variant}.warehouse.{spec['object_type']}",
        variation_strength=variation_strength,
    )


def _draw_ground_shadow(draw: ImageDraw.ImageDraw, spec: Mapping[str, Any], *, camera: _CameraSpec, frame: _ProjectionFrame) -> None:
    return None


__all__ = [
    "_draw_ground_shadow",
    "_draw_warehouse_object",
    "_fill_for_object",
]
