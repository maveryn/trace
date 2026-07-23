"""Low-level drawing primitives for shared three_d object scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .camera_projection import (
    CameraSpec as _CameraSpec,
    ProjectionFrame as _ProjectionFrame,
    distance as _distance,
    project_xy as _project_xy,
)
from .projected_object_geometry import _oriented_offset_xy


def _bbox_union(*bboxes: Sequence[float]) -> List[float]:
    return [
        round(float(min(float(bbox[0]) for bbox in bboxes)), 3),
        round(float(min(float(bbox[1]) for bbox in bboxes)), 3),
        round(float(max(float(bbox[2]) for bbox in bboxes)), 3),
        round(float(max(float(bbox[3]) for bbox in bboxes)), 3),
    ]


def _draw_line(draw: ImageDraw.ImageDraw, p1: Sequence[float], p2: Sequence[float], *, fill: Tuple[int, int, int], width: int) -> None:
    draw.line((float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])), fill=fill, width=max(1, int(width)))


def _shade(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(float(channel) * float(factor))))) for channel in rgb)


def _tint(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(float(channel) + (255.0 - float(channel)) * float(factor))))) for channel in rgb)


def _object_vertices(spec: Mapping[str, Any]) -> Dict[str, Tuple[float, float, float]]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    vertices: Dict[str, Tuple[float, float, float]] = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            ox, oy = _oriented_offset_xy(spec, sx * width * 0.5, sy * depth * 0.5)
            for top in (0, 1):
                vertices[f"{sx}{sy}{top}"] = (x + ox, y + oy, base_z + (height if top else 0.0))
    return vertices


def _draw_polyline(draw: ImageDraw.ImageDraw, points: Sequence[Sequence[float]], *, fill: Tuple[int, int, int], width: int = 2) -> None:
    for index in range(len(points)):
        start = points[index]
        next_index = index + 1
        if next_index >= len(points):
            next_index = 0
        end = points[next_index]
        _draw_line(draw, start, end, fill=fill, width=width)


def _bbox_from_screen_points(points: Sequence[Sequence[float]]) -> List[float]:
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in points])


def _padded_bbox_from_screen_points(points: Sequence[Sequence[float]], *, pad_px: float = 1.0) -> List[float]:
    bbox = _bbox_from_screen_points(points)
    return [
        round(float(bbox[0]) - float(pad_px), 3),
        round(float(bbox[1]) - float(pad_px), 3),
        round(float(bbox[2]) + float(pad_px), 3),
        round(float(bbox[3]) + float(pad_px), 3),
    ]


def _diagonal_ground_axis_basis(
    spec: Mapping[str, Any],
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    *,
    center_height_frac: float = 0.58,
    length_scale: float = 0.78,
    min_length_px: float = 42.0,
    max_length_px: float = 104.0,
    axis_xy: Tuple[float, float] = (1.0, 0.0),
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
    """Return a screen-space basis for objects whose neutral pose is diagonal."""

    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    center_z = base_z + height * float(center_height_frac)
    center = _project_xy((x, y, center_z), camera, frame)
    axis_norm = max(1e-6, math.hypot(float(axis_xy[0]), float(axis_xy[1])))
    ux = float(axis_xy[0]) / axis_norm
    uy = float(axis_xy[1]) / axis_norm
    ux, uy = _oriented_offset_xy(spec, ux, uy)
    world_length = max(float(width), float(depth))
    projected_start = _project_xy((x - ux * world_length * 0.5, y - uy * world_length * 0.5, center_z), camera, frame)
    projected_end = _project_xy((x + ux * world_length * 0.5, y + uy * world_length * 0.5, center_z), camera, frame)
    axis_dx = float(projected_end[0]) - float(projected_start[0])
    axis_dy = float(projected_end[1]) - float(projected_start[1])
    axis_len = math.hypot(axis_dx, axis_dy)
    if axis_len < 1e-6:
        inv_sqrt2 = math.sqrt(0.5)
        direction = (inv_sqrt2, -inv_sqrt2)
    else:
        direction = (axis_dx / axis_len, axis_dy / axis_len)
    normal = (-direction[1], direction[0])
    length_px = max(float(min_length_px), min(float(max_length_px), float(frame.scale) * world_length * float(length_scale)))
    return center, direction, normal, length_px


def _project_face(face: Sequence[Sequence[float]], camera: _CameraSpec, frame: _ProjectionFrame) -> List[Tuple[float, float]]:
    return [_project_xy(point, camera, frame) for point in face]


def _camera_facing_local_signs(spec: Mapping[str, Any], camera: _CameraSpec) -> Tuple[int, int]:
    """Return the local x/y sides of an oriented box that face the camera."""

    x, y, _z = (float(value) for value in spec["world_xyz"])
    camera_dx = float(camera.camera_position[0]) - x
    camera_dy = float(camera.camera_position[1]) - y
    local_x_axis = _oriented_offset_xy(spec, 1.0, 0.0)
    local_y_axis = _oriented_offset_xy(spec, 0.0, 1.0)
    local_x_dot = camera_dx * float(local_x_axis[0]) + camera_dy * float(local_x_axis[1])
    local_y_dot = camera_dx * float(local_y_axis[0]) + camera_dy * float(local_y_axis[1])
    sx = 1 if local_x_dot >= 0.0 else -1
    sy = 1 if local_y_dot >= 0.0 else -1
    return sx, sy


def _project_local_xy_point(
    spec: Mapping[str, Any],
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    *,
    u: float,
    v: float,
    z_frac: float = 1.0,
) -> Tuple[float, float]:
    """Project a point from one object's local xy footprint onto the screen."""

    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    ox, oy = _oriented_offset_xy(spec, (float(u) - 0.5) * width, (float(v) - 0.5) * depth)
    return _project_xy((x + ox, y + oy, base_z + height * float(z_frac)), camera, frame)


def _project_local_xy_rect(
    spec: Mapping[str, Any],
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    *,
    u0: float,
    v0: float,
    u1: float,
    v1: float,
    z_frac: float = 1.0,
) -> List[Tuple[float, float]]:
    """Project a local object-footprint rectangle while preserving object yaw."""

    return [
        _project_local_xy_point(spec, camera, frame, u=float(u0), v=float(v0), z_frac=float(z_frac)),
        _project_local_xy_point(spec, camera, frame, u=float(u1), v=float(v0), z_frac=float(z_frac)),
        _project_local_xy_point(spec, camera, frame, u=float(u1), v=float(v1), z_frac=float(z_frac)),
        _project_local_xy_point(spec, camera, frame, u=float(u0), v=float(v1), z_frac=float(z_frac)),
    ]


def _face_distance(face: Sequence[Sequence[float]], camera: _CameraSpec) -> float:
    center = tuple(sum(float(point[index]) for point in face) / float(len(face)) for index in range(3))
    return _distance(center, camera.camera_position)


def _draw_box_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    vertices = _object_vertices(spec)
    sx, sy = _camera_facing_local_signs(spec, camera)
    faces = [
        (
            [vertices[f"{-sx}{-sy}1"], vertices[f"{sx}{-sy}1"], vertices[f"{sx}{sy}1"], vertices[f"{-sx}{sy}1"]],
            _tint(fill, 0.25),
        ),
        (
            [vertices[f"{sx}{-sy}0"], vertices[f"{sx}{sy}0"], vertices[f"{sx}{sy}1"], vertices[f"{sx}{-sy}1"]],
            _shade(fill, 0.86),
        ),
        (
            [vertices[f"{-sx}{sy}0"], vertices[f"{sx}{sy}0"], vertices[f"{sx}{sy}1"], vertices[f"{-sx}{sy}1"]],
            _shade(fill, 0.72),
        ),
    ]
    projected_points: List[Tuple[float, float]] = []
    for face, color in sorted(faces, key=lambda item: _face_distance(item[0], camera), reverse=True):
        projected = _project_face(face, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=(28, 35, 45), width=2)
        projected_points.extend(projected)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected_points])


def _sub_box_spec(
    spec: Mapping[str, Any],
    *,
    offset_xyz: Tuple[float, float, float],
    dimensions_xyz: Tuple[float, float, float],
) -> Dict[str, Any]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    parent_base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in dimensions_xyz)
    ox, oy = _oriented_offset_xy(spec, float(offset_xyz[0]), float(offset_xyz[1]))
    cx = float(x + ox)
    cy = float(y + oy)
    base_z = float(parent_base_z) + float(offset_xyz[2])
    return {
        **dict(spec),
        "world_xyz": [round(cx, 4), round(cy, 4), round(base_z + height * 0.5, 4)],
        "base_xyz": [round(cx, 4), round(cy, 4), round(base_z, 4)],
        "dimensions_xyz": [round(float(width), 4), round(float(depth), 4), round(float(height), 4)],
    }


def _draw_box_parts_object(
    draw: ImageDraw.ImageDraw,
    parts: Sequence[Mapping[str, Any]],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bboxes: List[List[float]] = []
    for index, part in enumerate(sorted(parts, key=lambda item: _distance(item["world_xyz"], camera.camera_position), reverse=True)):
        part_fill = _tint(fill, 0.06) if index % 2 == 0 else _shade(fill, 0.95)
        bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=part_fill))
    return _bbox_union(*bboxes)


def _draw_footprint_prism_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    footprint_xy: Sequence[Tuple[float, float]],
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    base = [
        (x + ox, y + oy, base_z)
        for ox, oy in (_oriented_offset_xy(spec, px * width * 0.5, py * depth * 0.5) for px, py in footprint_xy)
    ]
    top = [(point[0], point[1], base_z + height) for point in base]
    faces: List[Tuple[List[Tuple[float, float, float]], Tuple[int, int, int]]] = [
        (list(top), _tint(fill, 0.22)),
    ]
    for index in range(len(base)):
        next_index = index + 1
        if next_index >= len(base):
            next_index = 0
        shade_factor = 0.70 + 0.18 * ((index % 3) / 2.0)
        faces.append(
            (
                [base[index], base[next_index], top[next_index], top[index]],
                _shade(fill, shade_factor),
            )
        )
    projected_points: List[Tuple[float, float]] = []
    for face, color in sorted(faces, key=lambda item: _face_distance(item[0], camera), reverse=True):
        projected = _project_face(face, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=(28, 35, 45), width=2)
        projected_points.extend(projected)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected_points])


def _star_footprint_points() -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(10):
        angle = -math.pi * 0.5 + index * math.pi / 5.0
        radius = 1.0 if index % 2 == 0 else 0.46
        points.append((math.cos(angle) * radius, math.sin(angle) * radius))
    return points


def _hexagon_footprint_points() -> List[Tuple[float, float]]:
    return [
        (math.cos(-math.pi * 0.5 + index * math.pi / 3.0), math.sin(-math.pi * 0.5 + index * math.pi / 3.0))
        for index in range(6)
    ]


def _arrow_footprint_points() -> List[Tuple[float, float]]:
    return [
        (0.0, -1.0),
        (0.74, -0.08),
        (0.25, -0.08),
        (0.25, 1.0),
        (-0.25, 1.0),
        (-0.25, -0.08),
        (-0.74, -0.08),
    ]


def _gear_footprint_points() -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(24):
        angle = -math.pi * 0.5 + index * math.pi / 12.0
        radius = 1.0 if index % 2 == 0 else 0.76
        points.append((math.cos(angle) * radius, math.sin(angle) * radius))
    return points


def _draw_half_cylinder_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    radius_y = depth * 0.5
    profile = [
        (-radius_y, base_z),
        *[
            (
                math.cos(math.pi - step * math.pi / 8.0) * radius_y,
                base_z + math.sin(math.pi - step * math.pi / 8.0) * height,
            )
            for step in range(1, 8)
        ],
        (radius_y, base_z),
    ]
    left_face = []
    right_face = []
    for py, pz in profile:
        left_ox, left_oy = _oriented_offset_xy(spec, -width * 0.5, py)
        right_ox, right_oy = _oriented_offset_xy(spec, width * 0.5, py)
        left_face.append((x + left_ox, y + left_oy, pz))
        right_face.append((x + right_ox, y + right_oy, pz))
    faces: List[Tuple[List[Tuple[float, float, float]], Tuple[int, int, int]]] = [
        (left_face, _shade(fill, 0.76)),
        (right_face, _tint(fill, 0.16)),
    ]
    for index in range(len(profile) - 1):
        faces.append(
            (
                [left_face[index], right_face[index], right_face[index + 1], left_face[index + 1]],
                _shade(fill, 0.74 + 0.16 * (index / max(1, len(profile) - 2))),
            )
        )
    projected_points: List[Tuple[float, float]] = []
    for face, color in sorted(faces, key=lambda item: _face_distance(item[0], camera), reverse=True):
        projected = _project_face(face, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=(28, 35, 45), width=2)
        projected_points.extend(projected)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected_points])


def _draw_pyramid_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    base = []
    for dx, dy in (
        (-width * 0.5, -depth * 0.5),
        (width * 0.5, -depth * 0.5),
        (width * 0.5, depth * 0.5),
        (-width * 0.5, depth * 0.5),
    ):
        ox, oy = _oriented_offset_xy(spec, dx, dy)
        base.append((x + ox, y + oy, base_z))
    apex = (x, y, base_z + height)
    faces = [
        ([base[0], base[1], apex], _tint(fill, 0.16)),
        ([base[1], base[2], apex], _shade(fill, 0.88)),
        ([base[2], base[3], apex], _shade(fill, 0.76)),
        ([base[3], base[0], apex], _shade(fill, 0.68)),
    ]
    projected_points: List[Tuple[float, float]] = []
    for face, color in sorted(faces, key=lambda item: _face_distance(item[0], camera), reverse=True):
        projected = _project_face(face, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=(28, 35, 45), width=2)
        projected_points.extend(projected)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected_points])


def _draw_wedge_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    vertices = []
    for dx, dy, dz in (
        (-width * 0.5, -depth * 0.5, 0.0),
        (width * 0.5, -depth * 0.5, 0.0),
        (width * 0.5, depth * 0.5, 0.0),
        (-width * 0.5, depth * 0.5, 0.0),
        (-width * 0.5, -depth * 0.5, height),
        (-width * 0.5, depth * 0.5, height),
    ):
        ox, oy = _oriented_offset_xy(spec, dx, dy)
        vertices.append((x + ox, y + oy, base_z + dz))
    faces = [
        ([vertices[0], vertices[1], vertices[2], vertices[3]], _shade(fill, 0.70)),
        ([vertices[0], vertices[1], vertices[4]], _tint(fill, 0.16)),
        ([vertices[3], vertices[2], vertices[5]], _shade(fill, 0.76)),
        ([vertices[0], vertices[3], vertices[5], vertices[4]], _shade(fill, 0.86)),
        ([vertices[1], vertices[2], vertices[5], vertices[4]], _tint(fill, 0.24)),
    ]
    projected_points: List[Tuple[float, float]] = []
    for face, color in sorted(faces, key=lambda item: _face_distance(item[0], camera), reverse=True):
        projected = _project_face(face, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=(28, 35, 45), width=2)
        projected_points.extend(projected)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected_points])


def _upright_profile_world_points(
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    profile_xz: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float, float]]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    return [
        (
            x + float(camera.right[0]) * float(px) * width * 0.5,
            y + float(camera.right[1]) * float(px) * width * 0.5,
            base_z + (float(pz) + 1.0) * height * 0.5,
        )
        for px, pz in profile_xz
    ]


def _draw_upright_profile_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    profile_xz: Sequence[Tuple[float, float]],
    inset_scale: float = 0.70,
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    _width, depth, _height = (float(value) for value in spec["dimensions_xyz"])
    back_dx = -float(camera.forward[0]) * float(depth) * 0.55
    back_dy = -float(camera.forward[1]) * float(depth) * 0.55
    shadow_spec = {
        **dict(spec),
        "world_xyz": [x + back_dx, y + back_dy, spec["world_xyz"][2]],
        "base_xyz": [x + back_dx, y + back_dy, spec["base_xyz"][2]],
    }
    shadow_world = _upright_profile_world_points(shadow_spec, camera=camera, profile_xz=profile_xz)
    face_world = _upright_profile_world_points(spec, camera=camera, profile_xz=profile_xz)
    shadow = _project_face(shadow_world, camera, frame)
    face = _project_face(face_world, camera, frame)
    draw.polygon(shadow, fill=_shade(fill, 0.60))
    draw.polygon(face, fill=_tint(fill, 0.12))
    _draw_polyline(draw, face, fill=(28, 35, 45), width=2)
    if 0.0 < float(inset_scale) < 1.0:
        center_x = sum(point[0] for point in face) / float(len(face))
        center_y = sum(point[1] for point in face) / float(len(face))
        inset = [
            (
                center_x + (point[0] - center_x) * float(inset_scale),
                center_y + (point[1] - center_y) * float(inset_scale),
            )
            for point in face
        ]
        _draw_polyline(draw, inset, fill=_shade(fill, 0.78), width=2)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in [*shadow, *face]])


def _heart_profile_points() -> List[Tuple[float, float]]:
    raw: List[Tuple[float, float]] = []
    for index in range(40):
        t = (2.0 * math.pi * index) / 40.0
        x = 16.0 * math.sin(t) ** 3
        y = 13.0 * math.cos(t) - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t)
        raw.append((x, y))
    min_y = min(point[1] for point in raw)
    max_y = max(point[1] for point in raw)
    return [
        (float(x) / 17.0, ((float(y) - min_y) / max(1e-6, max_y - min_y)) * 2.0 - 1.0)
        for x, y in raw
    ]

def _oval_profile_points(count: int = 32, *, x_scale: float = 1.0, z_scale: float = 1.0) -> List[Tuple[float, float]]:
    return [
        (
            math.cos(2.0 * math.pi * index / float(count)) * float(x_scale),
            math.sin(2.0 * math.pi * index / float(count)) * float(z_scale),
        )
        for index in range(int(count))
    ]

def _upright_screen_points(
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    profile_xz: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=profile_xz), camera, frame)

def _radius_px_for_object(spec: Mapping[str, Any], camera: _CameraSpec, frame: _ProjectionFrame) -> float:
    x, y, z = (float(value) for value in spec["world_xyz"])
    width, depth, _height = (float(value) for value in spec["dimensions_xyz"])
    center = _project_xy((x, y, z), camera, frame)
    offsets = []
    for dx, dy in ((width * 0.5, 0.0), (-width * 0.5, 0.0), (0.0, depth * 0.5), (0.0, -depth * 0.5)):
        ox, oy = _oriented_offset_xy(spec, dx, dy)
        offsets.append(_project_xy((x + ox, y + oy, z), camera, frame))
    return max(18.0, max(math.hypot(point[0] - center[0], point[1] - center[1]) for point in offsets))


def _draw_sphere_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, z = (float(value) for value in spec["world_xyz"])
    cx, cy = _project_xy((x, y, z), camera, frame)
    radius = _radius_px_for_object(spec, camera, frame)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    for step in range(8, 0, -1):
        factor = step / 8.0
        inset = radius * (1.0 - factor)
        color = _tint(fill, 0.10 + (1.0 - factor) * 0.32)
        draw.ellipse((bbox[0] + inset, bbox[1] + inset, bbox[2] - inset, bbox[3] - inset), fill=color)
    draw.ellipse(bbox, outline=(28, 35, 45), width=2)
    highlight = radius * 0.18
    draw.ellipse((cx - radius * 0.38 - highlight, cy - radius * 0.42 - highlight, cx - radius * 0.38 + highlight, cy - radius * 0.42 + highlight), fill=(255, 255, 255))
    return [round(float(value), 3) for value in bbox]


def _draw_cylinder_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    base = _project_xy((x, y, base_z), camera, frame)
    top = _project_xy((x, y, base_z + height), camera, frame)
    radius = _radius_px_for_object({**dict(spec), "world_xyz": [x, y, base_z + height * 0.5]}, camera, frame)
    ellipse_h = max(9.0, radius * 0.42)
    side = [(top[0] - radius, top[1]), (top[0] + radius, top[1]), (base[0] + radius, base[1]), (base[0] - radius, base[1])]
    draw.polygon(side, fill=_shade(fill, 0.82))
    draw.ellipse((base[0] - radius, base[1] - ellipse_h, base[0] + radius, base[1] + ellipse_h), fill=_shade(fill, 0.70), outline=(28, 35, 45), width=2)
    draw.ellipse((top[0] - radius, top[1] - ellipse_h, top[0] + radius, top[1] + ellipse_h), fill=_tint(fill, 0.20), outline=(28, 35, 45), width=2)
    _draw_line(draw, (top[0] - radius, top[1]), (base[0] - radius, base[1]), fill=(28, 35, 45), width=2)
    _draw_line(draw, (top[0] + radius, top[1]), (base[0] + radius, base[1]), fill=(28, 35, 45), width=2)
    return _bbox_union(
        [top[0] - radius, top[1] - ellipse_h, top[0] + radius, top[1] + ellipse_h],
        [base[0] - radius, base[1] - ellipse_h, base[0] + radius, base[1] + ellipse_h],
    )


def _draw_cone_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    base = _project_xy((x, y, base_z), camera, frame)
    apex = _project_xy((x, y, base_z + height), camera, frame)
    radius = _radius_px_for_object({**dict(spec), "world_xyz": [x, y, base_z + height * 0.25]}, camera, frame)
    ellipse_h = max(8.0, radius * 0.38)
    left = (base[0] - radius, base[1])
    right = (base[0] + radius, base[1])
    draw.polygon([left, right, apex], fill=_shade(fill, 0.84))
    draw.polygon([left, base, apex], fill=_tint(fill, 0.18))
    draw.ellipse((base[0] - radius, base[1] - ellipse_h, base[0] + radius, base[1] + ellipse_h), fill=_shade(fill, 0.72), outline=(28, 35, 45), width=2)
    _draw_line(draw, left, apex, fill=(28, 35, 45), width=2)
    _draw_line(draw, right, apex, fill=(28, 35, 45), width=2)
    return _bbox_union([base[0] - radius, base[1] - ellipse_h, base[0] + radius, base[1] + ellipse_h], [apex[0], apex[1], apex[0], apex[1]])


def _draw_torus_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    x, y, z = (float(value) for value in spec["world_xyz"])
    cx, cy = _project_xy((x, y, z), camera, frame)
    radius = _radius_px_for_object(spec, camera, frame)
    outer_h = max(18.0, radius * 0.64)
    inner_w = radius * 0.48
    inner_h = outer_h * 0.42
    outer = [cx - radius, cy - outer_h, cx + radius, cy + outer_h]
    inner = [cx - inner_w, cy - inner_h, cx + inner_w, cy + inner_h]
    draw.ellipse(outer, fill=_tint(fill, 0.10), outline=(28, 35, 45), width=2)
    draw.ellipse((outer[0] + radius * 0.10, outer[1] + outer_h * 0.18, outer[2] - radius * 0.10, outer[3] - outer_h * 0.20), outline=_shade(fill, 0.74), width=5)
    draw.ellipse(inner, fill=floor_rgb, outline=(28, 35, 45), width=2)
    draw.arc(outer, start=205, end=330, fill=_shade(fill, 0.72), width=5)
    return [round(float(value), 3) for value in outer]


bbox_from_screen_points = _bbox_from_screen_points
bbox_union = _bbox_union
draw_box_object = _draw_box_object
draw_box_parts_object = _draw_box_parts_object
draw_cone_object = _draw_cone_object
draw_cylinder_object = _draw_cylinder_object
draw_footprint_prism_object = _draw_footprint_prism_object
draw_half_cylinder_object = _draw_half_cylinder_object
draw_line = _draw_line
draw_polyline = _draw_polyline
draw_pyramid_object = _draw_pyramid_object
draw_sphere_object = _draw_sphere_object
draw_torus_object = _draw_torus_object
draw_upright_profile_object = _draw_upright_profile_object
draw_wedge_object = _draw_wedge_object
project_local_xy_point = _project_local_xy_point
project_local_xy_rect = _project_local_xy_rect
project_face = _project_face
shade_rgb = _shade
tint_rgb = _tint


__all__ = [
    "bbox_from_screen_points",
    "bbox_union",
    "draw_box_object",
    "draw_box_parts_object",
    "draw_cone_object",
    "draw_cylinder_object",
    "draw_footprint_prism_object",
    "draw_half_cylinder_object",
    "draw_line",
    "draw_polyline",
    "draw_pyramid_object",
    "draw_sphere_object",
    "draw_torus_object",
    "draw_upright_profile_object",
    "draw_wedge_object",
    "project_local_xy_point",
    "project_local_xy_rect",
    "project_face",
    "shade_rgb",
    "tint_rgb",
    "_arrow_footprint_points",
    "_bbox_from_screen_points",
    "_bbox_union",
    "_diagonal_ground_axis_basis",
    "_draw_box_object",
    "_draw_box_parts_object",
    "_draw_cone_object",
    "_draw_cylinder_object",
    "_draw_footprint_prism_object",
    "_draw_half_cylinder_object",
    "_draw_line",
    "_draw_polyline",
    "_draw_pyramid_object",
    "_draw_sphere_object",
    "_draw_torus_object",
    "_draw_upright_profile_object",
    "_draw_wedge_object",
    "_face_distance",
    "_gear_footprint_points",
    "_heart_profile_points",
    "_hexagon_footprint_points",
    "_object_vertices",
    "_oval_profile_points",
    "_padded_bbox_from_screen_points",
    "_project_face",
    "_project_local_xy_point",
    "_project_local_xy_rect",
    "_radius_px_for_object",
    "_shade",
    "_star_footprint_points",
    "_sub_box_spec",
    "_tint",
    "_upright_profile_world_points",
    "_upright_screen_points",
]
