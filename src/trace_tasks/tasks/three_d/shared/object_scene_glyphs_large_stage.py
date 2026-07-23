"""Large stage and storage glyphs for shared three_d object scenes."""

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


def _draw_arch_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    post_w = width * 0.22
    lintel_h = height * 0.24
    parts = [
        _sub_box_spec(spec, offset_xyz=(-width * 0.34, 0.0, 0.0), dimensions_xyz=(post_w, depth, height)),
        _sub_box_spec(spec, offset_xyz=(width * 0.34, 0.0, 0.0), dimensions_xyz=(post_w, depth, height)),
        _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height - lintel_h), dimensions_xyz=(width, depth, lintel_h)),
    ]
    return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)


def _draw_table_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    top_h = height * 0.18
    leg_w = min(width, depth) * 0.16
    leg_h = height - top_h
    parts = [
        _sub_box_spec(spec, offset_xyz=(sx * width * 0.34, sy * depth * 0.34, 0.0), dimensions_xyz=(leg_w, leg_w, leg_h))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    parts.append(_sub_box_spec(spec, offset_xyz=(0.0, 0.0, leg_h), dimensions_xyz=(width, depth, top_h)))
    return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)


def _draw_shelf_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    post_w = width * 0.12
    shelf_h = height * 0.10
    parts = [
        _sub_box_spec(spec, offset_xyz=(-width * 0.42, 0.0, 0.0), dimensions_xyz=(post_w, depth, height)),
        _sub_box_spec(spec, offset_xyz=(width * 0.42, 0.0, 0.0), dimensions_xyz=(post_w, depth, height)),
    ]
    for base_z in (0.0, height * 0.43, height * 0.86):
        parts.append(_sub_box_spec(spec, offset_xyz=(0.0, 0.0, base_z), dimensions_xyz=(width, depth, shelf_h)))
    return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)


def _draw_open_box_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    wall = min(width, depth) * 0.12
    base_h = height * 0.18
    wall_h = (height - base_h) * 0.58
    front_h = (height - base_h) * 0.34
    parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, base_h)),
        _sub_box_spec(spec, offset_xyz=(-width * 0.5 + wall * 0.5, 0.0, base_h), dimensions_xyz=(wall, depth, wall_h)),
        _sub_box_spec(spec, offset_xyz=(width * 0.5 - wall * 0.5, 0.0, base_h), dimensions_xyz=(wall, depth, wall_h)),
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.5 - wall * 0.5, base_h), dimensions_xyz=(width, wall, wall_h)),
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.5 + wall * 0.5, base_h), dimensions_xyz=(width, wall, front_h)),
    ]
    return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)


__all__ = [
    "_draw_arch_object",
    "_draw_table_object",
    "_draw_shelf_object",
    "_draw_open_box_object",
]
