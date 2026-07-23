"""Symbolic/game-style object glyphs for shared three_d object scenes."""

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
from .object_scene_primitives import (
    _arrow_footprint_points,
    _bbox_from_screen_points,
    _bbox_union,
    _draw_box_object,
    _draw_box_parts_object,
    _draw_cone_object,
    _draw_cylinder_object,
    _draw_footprint_prism_object,
    _draw_half_cylinder_object,
    _draw_line,
    _draw_polyline,
    _draw_pyramid_object,
    _draw_sphere_object,
    _draw_torus_object,
    _draw_upright_profile_object,
    _draw_wedge_object,
    _face_distance,
    _gear_footprint_points,
    _heart_profile_points,
    _hexagon_footprint_points,
    _object_vertices,
    _oval_profile_points,
    _project_face,
    _radius_px_for_object,
    _shade,
    _star_footprint_points,
    _sub_box_spec,
    _tint,
    _upright_profile_world_points,
    _upright_screen_points,
)


def _draw_shield_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.78, 1.0),
        (0.78, 1.0),
        (0.92, 0.36),
        (0.54, -0.66),
        (0.0, -1.0),
        (-0.54, -0.66),
        (-0.92, 0.36),
    ]
    return _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=fill,
        profile_xz=profile,
        inset_scale=0.68,
    )


def _draw_heart_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    return _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=fill,
        profile_xz=_heart_profile_points(),
        inset_scale=0.0,
    )


def _draw_diamond_object(
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
    waist_z = base_z + height * 0.52
    top = (x, y, base_z + height)
    bottom = (x, y, base_z)
    waist = [
        (x - width * 0.5, y, waist_z),
        (x, y - depth * 0.5, waist_z),
        (x + width * 0.5, y, waist_z),
        (x, y + depth * 0.5, waist_z),
    ]
    faces: List[Tuple[List[Tuple[float, float, float]], Tuple[int, int, int]]] = []
    for index in range(4):
        faces.append(([top, waist[index], waist[(index + 1) % 4]], _tint(fill, 0.10 + 0.08 * (index % 2))))
        faces.append(([bottom, waist[(index + 1) % 4], waist[index]], _shade(fill, 0.62 + 0.08 * (index % 2))))
    projected_points: List[Tuple[float, float]] = []
    for face, color in sorted(faces, key=lambda item: _face_distance(item[0], camera), reverse=True):
        projected = _project_face(face, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=(28, 35, 45), width=2)
        projected_points.extend(projected)
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected_points])


def _draw_sword_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, depth * 0.42, 0.0),
                dimensions_xyz=(width * 0.22, depth * 1.16, height * 0.64),
            ),
            (198, 207, 214),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, -depth * 0.20, 0.0),
                dimensions_xyz=(width * 1.22, depth * 0.10, height * 0.70),
            ),
            _tint(fill, 0.08),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, -depth * 0.36, 0.0),
                dimensions_xyz=(width * 0.34, depth * 0.22, height * 0.58),
            ),
            (106, 75, 48),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, -depth * 0.45, 0.0),
                dimensions_xyz=(width * 0.58, depth * 0.10, height * 0.62),
            ),
            _shade(fill, 0.72),
        ),
    ]
    bboxes: List[List[float]] = []
    for part, color in sorted(parts, key=lambda item: _distance(item[0]["world_xyz"], camera.camera_position), reverse=True):
        bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=color))
    return _bbox_union(*bboxes)


def _draw_key_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    key_fill = _tint(fill, 0.08)
    bow = _sub_box_spec(
        spec,
        offset_xyz=(0.0, -depth * 0.36, 0.0),
        dimensions_xyz=(width * 0.76, width * 0.76, height * 0.36),
    )
    parts = [
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, depth * 0.12, 0.0),
                dimensions_xyz=(width * 0.16, depth * 0.88, height * 0.52),
            ),
            _tint(key_fill, 0.08),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(width * 0.16, depth * 0.46, 0.0),
                dimensions_xyz=(width * 0.36, depth * 0.12, height * 0.54),
            ),
            _shade(key_fill, 0.86),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(-width * 0.13, depth * 0.56, 0.0),
                dimensions_xyz=(width * 0.30, depth * 0.11, height * 0.54),
            ),
            _shade(key_fill, 0.78),
        ),
    ]
    bboxes: List[List[float]] = [
        _draw_torus_object(draw, bow, camera=camera, frame=frame, fill=key_fill, floor_rgb=floor_rgb)
    ]
    for part, color in sorted(parts, key=lambda item: _distance(item[0]["world_xyz"], camera.camera_position), reverse=True):
        bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=color))
    return _bbox_union(*bboxes)


def _draw_crown_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.90, -1.0),
        (0.90, -1.0),
        (0.82, 0.18),
        (0.45, -0.05),
        (0.25, 0.96),
        (0.0, 0.20),
        (-0.25, 0.96),
        (-0.45, -0.05),
        (-0.82, 0.18),
    ]
    return _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=_tint(fill, 0.10),
        profile_xz=profile,
        inset_scale=0.78,
    )


def _draw_hourglass_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    glass_profile = [
        (-0.64, 0.82),
        (0.64, 0.82),
        (0.32, 0.22),
        (0.08, 0.02),
        (0.32, -0.22),
        (0.64, -0.82),
        (-0.64, -0.82),
        (-0.32, -0.22),
        (-0.08, 0.02),
        (-0.32, 0.22),
    ]
    glass_world = _upright_profile_world_points(spec, camera=camera, profile_xz=glass_profile)
    glass = _project_face(glass_world, camera, frame)
    draw.polygon(glass, fill=(218, 230, 226))
    _draw_polyline(draw, glass, fill=(50, 60, 64), width=2)

    sand_fill = (210, 164, 82)
    sand_profiles = [
        [(-0.44, 0.62), (0.44, 0.62), (0.12, 0.10), (-0.12, 0.10)],
        [(-0.44, -0.70), (0.44, -0.70), (0.24, -0.36), (0.0, -0.18), (-0.24, -0.36)],
    ]
    bboxes: List[List[float]] = [_bbox_union(*[[point[0], point[1], point[0], point[1]] for point in glass])]
    for profile in sand_profiles:
        projected = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=profile), camera, frame)
        draw.polygon(projected, fill=sand_fill)
        _draw_polyline(draw, projected, fill=(128, 92, 48), width=1)
        bboxes.append(_bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected]))

    frame_parts = [
        (
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.88), dimensions_xyz=(width * 0.96, depth * 0.30, height * 0.10)),
            (116, 80, 48),
        ),
        (
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.96, depth * 0.30, height * 0.10)),
            (116, 80, 48),
        ),
        (
            _sub_box_spec(spec, offset_xyz=(-width * 0.40, 0.0, height * 0.08), dimensions_xyz=(width * 0.10, depth * 0.24, height * 0.84)),
            (96, 66, 42),
        ),
        (
            _sub_box_spec(spec, offset_xyz=(width * 0.40, 0.0, height * 0.08), dimensions_xyz=(width * 0.10, depth * 0.24, height * 0.84)),
            (96, 66, 42),
        ),
    ]
    for part, color in sorted(frame_parts, key=lambda item: _distance(item[0]["world_xyz"], camera.camera_position), reverse=True):
        bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=color))
    waist_top = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.0, 0.10)])[0], camera, frame)
    waist_bottom = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.0, -0.18)])[0], camera, frame)
    _draw_line(draw, waist_top, waist_bottom, fill=sand_fill, width=2)
    bboxes.append(
        [
            min(waist_top[0], waist_bottom[0]),
            min(waist_top[1], waist_bottom[1]),
            max(waist_top[0], waist_bottom[0]),
            max(waist_top[1], waist_bottom[1]),
        ]
    )
    return _bbox_union(*bboxes)


def _draw_anchor_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (0.0, 1.0),
        (0.20, 0.86),
        (0.12, 0.70),
        (0.12, 0.24),
        (0.74, 0.24),
        (0.74, 0.04),
        (0.20, 0.04),
        (0.20, -0.36),
        (0.42, -0.20),
        (0.66, -0.18),
        (0.90, -0.34),
        (0.78, -0.56),
        (0.56, -0.76),
        (0.30, -0.90),
        (0.0, -0.98),
        (-0.30, -0.90),
        (-0.56, -0.76),
        (-0.78, -0.56),
        (-0.90, -0.34),
        (-0.66, -0.18),
        (-0.42, -0.20),
        (-0.20, -0.36),
        (-0.20, 0.04),
        (-0.74, 0.04),
        (-0.74, 0.24),
        (-0.12, 0.24),
        (-0.12, 0.70),
        (-0.20, 0.86),
    ]
    return _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=fill,
        profile_xz=profile,
        inset_scale=0.0,
    )


def _draw_horseshoe_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.88, 0.88),
        (-0.50, 0.88),
        (-0.50, -0.20),
        (-0.34, -0.56),
        (0.0, -0.70),
        (0.34, -0.56),
        (0.50, -0.20),
        (0.50, 0.88),
        (0.88, 0.88),
        (0.88, -0.32),
        (0.64, -0.74),
        (0.32, -0.96),
        (0.0, -1.0),
        (-0.32, -0.96),
        (-0.64, -0.74),
        (-0.88, -0.32),
    ]
    return _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=_tint(fill, 0.04),
        profile_xz=profile,
        inset_scale=0.0,
    )


def _draw_hammer_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, -depth * 0.10, 0.0),
                dimensions_xyz=(width * 0.18, depth * 0.92, height * 0.60),
            ),
            (118, 82, 48),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, depth * 0.36, 0.0),
                dimensions_xyz=(width * 1.08, depth * 0.22, height * 0.76),
            ),
            (174, 181, 188),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(-width * 0.38, depth * 0.38, 0.0),
                dimensions_xyz=(width * 0.34, depth * 0.15, height * 0.68),
            ),
            (152, 160, 168),
        ),
    ]
    bboxes: List[List[float]] = []
    for part, color in sorted(parts, key=lambda item: _distance(item[0]["world_xyz"], camera.camera_position), reverse=True):
        bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=color))
    return _bbox_union(*bboxes)


def _draw_bell_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.86, -1.0),
        (0.86, -1.0),
        (0.94, -0.76),
        (0.78, -0.52),
        (0.62, 0.16),
        (0.36, 0.72),
        (0.0, 0.96),
        (-0.36, 0.72),
        (-0.62, 0.16),
        (-0.78, -0.52),
        (-0.94, -0.76),
    ]
    body_bbox = _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=_tint(fill, 0.10),
        profile_xz=profile,
        inset_scale=0.76,
    )
    rim = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.70, -0.80), (0.70, -0.80)])
    clapper_center = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.72)])[0]
    clapper_radius = max(3.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.035)
    detail = _shade(fill, 0.50)
    draw.line(rim, fill=detail, width=3)
    draw.ellipse(
        (
            clapper_center[0] - clapper_radius,
            clapper_center[1] - clapper_radius,
            clapper_center[0] + clapper_radius,
            clapper_center[1] + clapper_radius,
        ),
        fill=detail,
        outline=_shade(fill, 0.34),
        width=1,
    )
    return _bbox_union(
        body_bbox,
        _bbox_from_screen_points(rim),
        [
            clapper_center[0] - clapper_radius,
            clapper_center[1] - clapper_radius,
            clapper_center[0] + clapper_radius,
            clapper_center[1] + clapper_radius,
        ],
    )


def _draw_trophy_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.90, 0.72),
        (-0.74, 0.96),
        (0.74, 0.96),
        (0.90, 0.72),
        (0.72, 0.30),
        (0.48, 0.06),
        (0.22, -0.08),
        (0.16, -0.52),
        (0.56, -0.52),
        (0.56, -0.74),
        (0.80, -0.74),
        (0.80, -1.0),
        (-0.80, -1.0),
        (-0.80, -0.74),
        (-0.56, -0.74),
        (-0.56, -0.52),
        (-0.16, -0.52),
        (-0.22, -0.08),
        (-0.48, 0.06),
        (-0.72, 0.30),
    ]
    return _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=_tint(fill, 0.10),
        profile_xz=profile,
        inset_scale=0.76,
    )


def _draw_open_book_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
) -> List[float]:
    bboxes: List[List[float]] = []
    faces = [
        ([(-1.02, -0.88), (-0.08, -1.00), (0.00, 0.84), (-0.92, 1.00)], (112, 72, 56), (72, 48, 38)),
        ([(0.08, -1.00), (1.02, -0.88), (0.92, 1.00), (0.00, 0.84)], (112, 72, 56), (72, 48, 38)),
        ([(-0.88, -0.74), (-0.08, -0.88), (-0.02, 0.70), (-0.80, 0.84)], (237, 230, 204), (140, 116, 86)),
        ([(0.08, -0.88), (0.88, -0.74), (0.80, 0.84), (0.02, 0.70)], (246, 238, 213), (140, 116, 86)),
    ]
    for profile, face_fill, outline in faces:
        projected = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=profile)
        draw.polygon(projected, fill=face_fill)
        _draw_polyline(draw, projected, fill=outline, width=2)
        bboxes.append(_bbox_from_screen_points(projected))
    spine = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.90), (0.0, 0.78)])
    draw.line(spine, fill=(92, 62, 48), width=3)
    bboxes.append(_bbox_from_screen_points(spine))
    for pz in (-0.48, -0.20, 0.08, 0.36):
        left_line = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.16, pz - 0.04), (-0.68, pz + 0.04)])
        right_line = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.16, pz - 0.04), (0.68, pz + 0.04)])
        draw.line(left_line, fill=(176, 162, 132), width=1)
        draw.line(right_line, fill=(176, 162, 132), width=1)
        bboxes.extend([_bbox_from_screen_points(left_line), _bbox_from_screen_points(right_line)])
    return _bbox_union(*bboxes)


def _draw_dumbbell_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    shaft = _sub_box_spec(
        spec,
        offset_xyz=(0.0, 0.0, height * 0.36),
        dimensions_xyz=(width * 0.78, depth * 0.18, height * 0.16),
    )
    left_outer = _sub_box_spec(
        spec,
        offset_xyz=(-width * 0.43, 0.0, 0.0),
        dimensions_xyz=(width * 0.18, depth * 0.78, height * 0.92),
    )
    left_inner = _sub_box_spec(
        spec,
        offset_xyz=(-width * 0.30, 0.0, height * 0.06),
        dimensions_xyz=(width * 0.14, depth * 0.64, height * 0.78),
    )
    right_inner = _sub_box_spec(
        spec,
        offset_xyz=(width * 0.30, 0.0, height * 0.06),
        dimensions_xyz=(width * 0.14, depth * 0.64, height * 0.78),
    )
    right_outer = _sub_box_spec(
        spec,
        offset_xyz=(width * 0.43, 0.0, 0.0),
        dimensions_xyz=(width * 0.18, depth * 0.78, height * 0.92),
    )
    left_collar = _sub_box_spec(
        spec,
        offset_xyz=(-width * 0.19, 0.0, height * 0.30),
        dimensions_xyz=(width * 0.07, depth * 0.32, height * 0.28),
    )
    right_collar = _sub_box_spec(
        spec,
        offset_xyz=(width * 0.19, 0.0, height * 0.30),
        dimensions_xyz=(width * 0.07, depth * 0.32, height * 0.28),
    )
    parts = [
        (shaft, _shade(fill, 0.74), "box"),
        (left_outer, _tint(fill, 0.10), "cylinder"),
        (left_inner, _shade(fill, 0.88), "cylinder"),
        (right_inner, _shade(fill, 0.88), "cylinder"),
        (right_outer, _tint(fill, 0.10), "cylinder"),
        (left_collar, (42, 48, 56), "box"),
        (right_collar, (42, 48, 56), "box"),
    ]
    bboxes: List[List[float]] = []
    for part, color, kind in sorted(parts, key=lambda item: _distance(item[0]["world_xyz"], camera.camera_position), reverse=True):
        if str(kind) == "cylinder":
            bboxes.append(_draw_cylinder_object(draw, part, camera=camera, frame=frame, fill=color))
        else:
            bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=color))
    handle_line = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.58, -0.02), (0.58, -0.02)])
    draw.line(handle_line, fill=(34, 38, 44), width=5)
    draw.line(handle_line, fill=(105, 112, 122), width=2)
    bboxes.append(_bbox_from_screen_points(handle_line))
    return _bbox_union(*bboxes)


def _draw_mushroom_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    stem_profile = [(-0.26, -1.0), (0.26, -1.0), (0.34, -0.18), (0.20, 0.10), (-0.20, 0.10), (-0.34, -0.18)]
    cap_profile = [
        (-0.96, -0.10),
        (-0.80, 0.34),
        (-0.48, 0.66),
        (-0.12, 0.80),
        (0.22, 0.78),
        (0.56, 0.58),
        (0.84, 0.26),
        (0.96, -0.10),
        (0.58, -0.22),
        (0.18, -0.27),
        (-0.22, -0.27),
        (-0.60, -0.22),
    ]
    stem = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=stem_profile), camera, frame)
    cap = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=cap_profile), camera, frame)
    draw.polygon(stem, fill=(232, 218, 181))
    _draw_polyline(draw, stem, fill=(105, 82, 58), width=2)
    draw.polygon(cap, fill=(178, 79, 54))
    _draw_polyline(draw, cap, fill=(78, 47, 36), width=2)
    gill = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.60, -0.18), (-0.28, -0.24), (0.12, -0.25), (0.56, -0.18)])
    highlight = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.46, 0.34), (-0.10, 0.56), (0.34, 0.50)])
    draw.line(gill, fill=(112, 76, 52), width=2)
    draw.line(highlight, fill=(209, 117, 87), width=2)
    bboxes = [
        _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in stem]),
        _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in cap]),
        _bbox_from_screen_points(gill),
        _bbox_from_screen_points(highlight),
    ]
    return _bbox_union(*bboxes)


def _draw_lantern_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body_profile = [
        (-0.66, -0.90),
        (0.66, -0.90),
        (0.58, -0.60),
        (0.48, 0.46),
        (0.26, 0.66),
        (0.18, 0.82),
        (-0.18, 0.82),
        (-0.26, 0.66),
        (-0.48, 0.46),
        (-0.58, -0.60),
    ]
    bbox = _draw_upright_profile_object(
        draw,
        spec,
        camera=camera,
        frame=frame,
        fill=(94, 74, 50),
        profile_xz=body_profile,
        inset_scale=0.86,
    )
    width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    glass_spec = _sub_box_spec(
        spec,
        offset_xyz=(0.0, 0.0, height * 0.16),
        dimensions_xyz=(
            width * 0.52,
            float(spec["dimensions_xyz"][1]) * 0.20,
            height * 0.52,
        ),
    )
    glass_world = _upright_profile_world_points(
        glass_spec,
        camera=camera,
        profile_xz=[(-0.62, -0.74), (0.62, -0.74), (0.62, 0.74), (-0.62, 0.74)],
    )
    glass = _project_face(glass_world, camera, frame)
    draw.polygon(glass, fill=(244, 205, 108))
    _draw_polyline(draw, glass, fill=(52, 43, 34), width=2)
    for px in (-0.30, 0.30):
        bottom = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(px, -0.72)])[0], camera, frame)
        top = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(px * 0.72, 0.62)])[0], camera, frame)
        _draw_line(draw, bottom, top, fill=(42, 35, 30), width=2)
    handle_points = []
    for index in range(13):
        t = math.pi * index / 12.0
        px = math.cos(t) * 0.42
        pz = 0.82 + math.sin(t) * 0.42
        handle_points.append(_project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(px, pz)])[0], camera, frame))
    _draw_polyline(draw, handle_points, fill=(42, 35, 30), width=3)
    flame = [
        (0.0, 0.26),
        (0.20, -0.12),
        (0.0, -0.36),
        (-0.20, -0.12),
    ]
    flame_projected = _project_face(_upright_profile_world_points(glass_spec, camera=camera, profile_xz=flame), camera, frame)
    draw.polygon(flame_projected, fill=(255, 238, 156))
    return _bbox_union(
        bbox,
        *[[point[0], point[1], point[0], point[1]] for point in [*glass, *handle_points, *flame_projected]],
    )


def _draw_wrench_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    metal = (171, 181, 188)
    parts = [
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, -depth * 0.14, 0.0),
                dimensions_xyz=(width * 0.20, depth * 0.96, height * 0.64),
            ),
            _shade(metal, 0.96),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, depth * 0.34, 0.0),
                dimensions_xyz=(width * 0.58, depth * 0.18, height * 0.70),
            ),
            _tint(metal, 0.08),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(-width * 0.26, depth * 0.52, 0.0),
                dimensions_xyz=(width * 0.20, depth * 0.36, height * 0.72),
            ),
            _tint(metal, 0.12),
        ),
        (
            _sub_box_spec(
                spec,
                offset_xyz=(width * 0.26, depth * 0.52, 0.0),
                dimensions_xyz=(width * 0.20, depth * 0.36, height * 0.72),
            ),
            _shade(metal, 0.88),
        ),
    ]
    bboxes: List[List[float]] = []
    for part, color in sorted(parts, key=lambda item: _distance(item[0]["world_xyz"], camera.camera_position), reverse=True):
        bboxes.append(_draw_box_object(draw, part, camera=camera, frame=frame, fill=color))
    return _bbox_union(*bboxes)


__all__ = [
    "_draw_shield_object",
    "_draw_heart_object",
    "_draw_diamond_object",
    "_draw_sword_object",
    "_draw_key_object",
    "_draw_crown_object",
    "_draw_hourglass_object",
    "_draw_anchor_object",
    "_draw_horseshoe_object",
    "_draw_hammer_object",
    "_draw_bell_object",
    "_draw_trophy_object",
    "_draw_open_book_object",
    "_draw_dumbbell_object",
    "_draw_mushroom_object",
    "_draw_lantern_object",
    "_draw_wrench_object",
]
