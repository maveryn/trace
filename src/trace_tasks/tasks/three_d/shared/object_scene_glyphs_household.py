"""Household/science object glyphs for shared three_d object scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .object_scene_glyphs_large_stage import _draw_open_box_object
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
    _diagonal_ground_axis_basis,
    _face_distance,
    _gear_footprint_points,
    _heart_profile_points,
    _hexagon_footprint_points,
    _object_vertices,
    _oval_profile_points,
    _padded_bbox_from_screen_points,
    _project_face,
    _radius_px_for_object,
    _shade,
    _star_footprint_points,
    _sub_box_spec,
    _tint,
    _upright_profile_world_points,
    _upright_screen_points,
)


def _draw_padlock_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.92, depth, height * 0.62))
    body_bbox = _draw_box_object(draw, body, camera=camera, frame=frame, fill=(188, 142, 55))
    shackle_points = _project_face(
        _upright_profile_world_points(
            spec,
            camera=camera,
            profile_xz=[(-0.44, -0.10), (-0.44, 0.50), (-0.26, 0.82), (0.0, 0.94), (0.26, 0.82), (0.44, 0.50), (0.44, -0.10)],
        ),
        camera,
        frame,
    )
    draw.line(shackle_points, fill=(166, 174, 181), width=5, joint="curve")
    draw.line(shackle_points, fill=(64, 73, 82), width=2, joint="curve")
    center = _project_xy(_upright_profile_world_points(body, camera=camera, profile_xz=[(0.0, -0.12)])[0], camera, frame)
    bottom = _project_xy(_upright_profile_world_points(body, camera=camera, profile_xz=[(0.0, -0.38)])[0], camera, frame)
    radius = max(3.0, float(frame.scale) * width * 0.035)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=(48, 38, 28))
    _draw_line(draw, center, bottom, fill=(48, 38, 28), width=2)
    return _bbox_union(body_bbox, _bbox_from_screen_points(shackle_points), [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], [bottom[0], bottom[1], bottom[0], bottom[1]])


def _draw_magnifying_glass_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    lens = _sub_box_spec(
        spec,
        offset_xyz=(-width * 0.18, -depth * 0.18, 0.0),
        dimensions_xyz=(width * 0.64, width * 0.64, height * 0.90),
    )
    handle = _sub_box_spec(
        spec,
        offset_xyz=(width * 0.18, depth * 0.24, 0.0),
        dimensions_xyz=(width * 0.15, depth * 0.68, height * 0.72),
    )
    lens_bbox = _draw_torus_object(draw, lens, camera=camera, frame=frame, fill=(110, 141, 164), floor_rgb=floor_rgb)
    cx, cy = _project_xy(lens["world_xyz"], camera, frame)
    radius = max(7.0, _radius_px_for_object(lens, camera, frame) * 0.40)
    draw.ellipse((cx - radius, cy - radius * 0.62, cx + radius, cy + radius * 0.62), fill=(202, 226, 235), outline=(80, 102, 120), width=1)
    handle_bbox = _draw_box_object(draw, handle, camera=camera, frame=frame, fill=(91, 69, 48))
    return _bbox_union(lens_bbox, handle_bbox)


def _draw_candle_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    wax = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, -height * 0.12), dimensions_xyz=(width * 0.42, depth * 0.42, height * 0.56))
    wax_bbox = _draw_box_object(draw, wax, camera=camera, frame=frame, fill=_tint(fill, 0.08))
    top = _project_xy((float(wax["world_xyz"][0]), float(wax["world_xyz"][1]), float(wax["base_xyz"][2]) + float(wax["dimensions_xyz"][2])), camera, frame)
    top_radius = max(4.0, _radius_px_for_object(wax, camera, frame) * 0.45)
    draw.ellipse(
        (
            top[0] - top_radius,
            top[1] - top_radius * 0.38,
            top[0] + top_radius,
            top[1] + top_radius * 0.38,
        ),
        fill=_tint(fill, 0.24),
        outline=(28, 35, 45),
        width=1,
    )
    flame_profile = [(0.0, 1.0), (0.30, 0.24), (0.12, -0.42), (0.0, -0.76), (-0.12, -0.42), (-0.30, 0.24)]
    flame_spec = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.36), dimensions_xyz=(width * 0.26, depth * 0.12, height * 0.34))
    flame = _project_face(_upright_profile_world_points(flame_spec, camera=camera, profile_xz=flame_profile), camera, frame)
    draw.polygon(flame, fill=(244, 132, 46))
    _draw_polyline(draw, flame, fill=(135, 71, 32), width=1)
    inner_flame = _project_face(_upright_profile_world_points(flame_spec, camera=camera, profile_xz=[(0.0, 0.58), (0.12, 0.02), (0.0, -0.42), (-0.12, 0.02)]), camera, frame)
    draw.polygon(inner_flame, fill=(255, 213, 88))
    wick_top = _project_xy(_upright_profile_world_points(wax, camera=camera, profile_xz=[(0.0, 0.54)])[0], camera, frame)
    wick_bottom = _project_xy(_upright_profile_world_points(wax, camera=camera, profile_xz=[(0.0, 0.38)])[0], camera, frame)
    _draw_line(draw, wick_bottom, wick_top, fill=(42, 35, 28), width=2)
    return _bbox_union(wax_bbox, [top[0] - top_radius, top[1] - top_radius * 0.38, top[0] + top_radius, top[1] + top_radius * 0.38], _bbox_from_screen_points(flame), _bbox_from_screen_points(inner_flame), _bbox_from_screen_points([wick_top, wick_bottom]))


def _draw_scroll_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    sheet = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.10), dimensions_xyz=(width * 0.78, depth * 0.48, height * 0.42))
    left_roll = _sub_box_spec(spec, offset_xyz=(-width * 0.42, 0.0, 0.0), dimensions_xyz=(width * 0.22, depth * 0.64, height * 0.86))
    right_roll = _sub_box_spec(spec, offset_xyz=(width * 0.42, 0.0, 0.0), dimensions_xyz=(width * 0.22, depth * 0.64, height * 0.86))
    bboxes = [
        _draw_box_object(draw, sheet, camera=camera, frame=frame, fill=(232, 216, 176)),
        _draw_cylinder_object(draw, left_roll, camera=camera, frame=frame, fill=(206, 184, 137)),
        _draw_cylinder_object(draw, right_roll, camera=camera, frame=frame, fill=(206, 184, 137)),
    ]
    for offset in (-0.18, 0.0, 0.18):
        p1 = _project_xy((float(spec["world_xyz"][0]) - width * 0.25, float(spec["world_xyz"][1]) + offset * depth, height * 0.55), camera, frame)
        p2 = _project_xy((float(spec["world_xyz"][0]) + width * 0.25, float(spec["world_xyz"][1]) + offset * depth, height * 0.55), camera, frame)
        _draw_line(draw, p1, p2, fill=(151, 120, 78), width=1)
        bboxes.append(_bbox_from_screen_points([p1, p2]))
    return _bbox_union(*bboxes)


def _draw_paint_brush_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    center, direction, normal, length_px = _diagonal_ground_axis_basis(
        spec,
        camera,
        frame,
        center_height_frac=0.60,
        length_scale=0.88,
        min_length_px=68.0,
        max_length_px=112.0,
    )
    handle_radius = max(4.0, min(6.5, length_px * 0.055))
    ferrule_radius = max(6.0, min(9.5, length_px * 0.085))
    bristle_radius = max(8.0, min(13.0, length_px * 0.115))

    def point_at(axis_frac: float, offset_px: float = 0.0) -> Tuple[float, float]:
        return (
            center[0] + direction[0] * length_px * float(axis_frac) + normal[0] * float(offset_px),
            center[1] + direction[1] * length_px * float(axis_frac) + normal[1] * float(offset_px),
        )

    def strip_between(a: Sequence[float], b: Sequence[float], ra: float, rb: float | None = None) -> List[Tuple[float, float]]:
        rb = float(ra if rb is None else rb)
        return [
            (float(a[0]) + normal[0] * float(ra), float(a[1]) + normal[1] * float(ra)),
            (float(b[0]) + normal[0] * rb, float(b[1]) + normal[1] * rb),
            (float(b[0]) - normal[0] * rb, float(b[1]) - normal[1] * rb),
            (float(a[0]) - normal[0] * float(ra), float(a[1]) - normal[1] * float(ra)),
        ]

    handle_start = point_at(-0.52)
    handle_end = point_at(0.08)
    ferrule_end = point_at(0.28)
    bristle_end = point_at(0.52)
    handle = strip_between(handle_start, handle_end, handle_radius)
    ferrule = strip_between(handle_end, ferrule_end, ferrule_radius)
    bristles = strip_between(ferrule_end, bristle_end, bristle_radius, bristle_radius * 0.58)
    bboxes: List[List[float]] = []
    for poly, color, outline in (
        (handle, (119, 77, 42), (64, 42, 28)),
        (ferrule, (176, 184, 190), (77, 86, 96)),
        (bristles, _shade(fill, 0.76), _shade(fill, 0.48)),
    ):
        draw.polygon(poly, fill=color, outline=outline)
        bboxes.append(_bbox_from_screen_points(poly))
    for offset in (-0.38, 0.0, 0.38):
        line = [
            point_at(0.10, offset * ferrule_radius),
            point_at(0.27, offset * ferrule_radius * 0.88),
        ]
        _draw_line(draw, line[0], line[1], fill=(92, 102, 112), width=1)
        bboxes.append(_padded_bbox_from_screen_points(line, pad_px=1.0))
    for offset in (-0.36, 0.0, 0.36):
        line = [
            point_at(0.32, offset * bristle_radius * 0.76),
            point_at(0.50, offset * bristle_radius * 0.42),
        ]
        _draw_line(draw, line[0], line[1], fill=_shade(fill, 0.48), width=1)
        bboxes.append(_padded_bbox_from_screen_points(line, pad_px=1.0))
    return _bbox_union(*bboxes)


def _draw_paint_palette_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.90, -0.28), (-0.76, 0.42), (-0.24, 0.90), (0.46, 0.78), (0.92, 0.24), (0.72, -0.42), (0.16, -0.82), (-0.50, -0.72)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(184, 132, 75), profile_xz=profile, inset_scale=0.0)
    points_for_bbox = []
    for px, pz, color in [(-0.38, 0.18, (214, 65, 58)), (0.04, 0.40, (64, 137, 205)), (0.32, 0.04, (77, 158, 91)), (-0.10, -0.30, (230, 190, 60))]:
        center = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(px, pz)])[0], camera, frame)
        radius = max(3.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.035)
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=color, outline=(44, 46, 50), width=1)
        points_for_bbox.extend([(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)])
    hole = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.46, 0.46)])[0], camera, frame)
    radius = max(4.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.045)
    draw.ellipse((hole[0] - radius, hole[1] - radius, hole[0] + radius, hole[1] + radius), fill=(246, 248, 247), outline=(70, 54, 42), width=1)
    points_for_bbox.extend([(hole[0] - radius, hole[1] - radius), (hole[0] + radius, hole[1] + radius)])
    return _bbox_union(bbox, _bbox_from_screen_points(points_for_bbox))


def _draw_goblet_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.68, 0.92), (0.68, 0.92), (0.50, 0.18), (0.20, -0.08), (0.12, -0.58), (0.54, -0.72), (0.54, -0.96), (-0.54, -0.96), (-0.54, -0.72), (-0.12, -0.58), (-0.20, -0.08), (-0.50, 0.18)]
    return _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(204, 163, 65), profile_xz=profile, inset_scale=0.76)


def _draw_teapot_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.74, depth * 0.86, height * 0.78))
    lid = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.58), dimensions_xyz=(width * 0.38, depth * 0.42, height * 0.16))
    spout = _sub_box_spec(spec, offset_xyz=(width * 0.43, -depth * 0.04, height * 0.22), dimensions_xyz=(width * 0.38, depth * 0.14, height * 0.22))
    bboxes = [
        _draw_sphere_object(draw, body, camera=camera, frame=frame, fill=_tint(fill, 0.10)),
        _draw_cylinder_object(draw, lid, camera=camera, frame=frame, fill=_shade(fill, 0.82)),
        _draw_wedge_object(draw, spout, camera=camera, frame=frame, fill=_shade(fill, 0.90)),
    ]
    handle_points = _project_face(
        _upright_profile_world_points(
            spec,
            camera=camera,
            profile_xz=[(-0.48, 0.22), (-0.74, 0.12), (-0.82, -0.18), (-0.66, -0.46), (-0.42, -0.38)],
        ),
        camera,
        frame,
    )
    draw.line(handle_points, fill=_shade(fill, 0.68), width=5, joint="curve")
    draw.line(handle_points, fill=(42, 48, 54), width=2, joint="curve")
    bboxes.append(_bbox_from_screen_points(handle_points))
    return _bbox_union(*bboxes)


def _draw_watering_can_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(-width * 0.06, 0.0, 0.0), dimensions_xyz=(width * 0.66, depth * 0.76, height * 0.62))
    spout = _sub_box_spec(spec, offset_xyz=(width * 0.42, depth * 0.18, height * 0.20), dimensions_xyz=(width * 0.44, depth * 0.14, height * 0.18))
    top = _sub_box_spec(spec, offset_xyz=(-width * 0.10, 0.0, height * 0.52), dimensions_xyz=(width * 0.32, depth * 0.38, height * 0.16))
    bboxes = [
        _draw_cylinder_object(draw, body, camera=camera, frame=frame, fill=_tint(fill, 0.08)),
        _draw_wedge_object(draw, spout, camera=camera, frame=frame, fill=_shade(fill, 0.90)),
        _draw_cylinder_object(draw, top, camera=camera, frame=frame, fill=_shade(fill, 0.82)),
    ]
    handle = _project_face(
        _upright_profile_world_points(spec, camera=camera, profile_xz=[(-0.52, -0.22), (-0.80, 0.06), (-0.66, 0.54), (-0.20, 0.70), (0.22, 0.54)]),
        camera,
        frame,
    )
    draw.line(handle, fill=_shade(fill, 0.70), width=5, joint="curve")
    draw.line(handle, fill=(42, 50, 54), width=2, joint="curve")
    bboxes.append(_bbox_from_screen_points(handle))
    return _bbox_union(*bboxes)


def _draw_basket_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_open_box_object(draw, spec, camera=camera, frame=frame, fill=(170, 116, 58))
    handle = _project_face(
        _upright_profile_world_points(spec, camera=camera, profile_xz=[(-0.72, -0.05), (-0.46, 0.68), (0.0, 0.94), (0.46, 0.68), (0.72, -0.05)]),
        camera,
        frame,
    )
    draw.line(handle, fill=(120, 80, 46), width=4, joint="curve")
    draw.line(handle, fill=(54, 40, 30), width=1, joint="curve")
    for px in (-0.34, 0.0, 0.34):
        top = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(px, -0.12)])[0], camera, frame)
        bottom = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(px, -0.72)])[0], camera, frame)
        _draw_line(draw, top, bottom, fill=(124, 83, 48), width=1)
    return _bbox_union(bbox, _bbox_from_screen_points(handle))


def _draw_mail_envelope_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.94, -0.66), (0.94, -0.66), (0.94, 0.66), (-0.94, 0.66)]
    envelope_fill = _tint(fill, 0.10)
    line_fill = (248, 250, 245) if sum(envelope_fill) < 420 else _shade(envelope_fill, 0.52)
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=envelope_fill, profile_xz=profile, inset_scale=0.0)
    line_profiles = [
        [(-0.90, 0.58), (0.0, -0.08), (0.90, 0.58)],
        [(-0.90, -0.58), (0.0, -0.08), (0.90, -0.58)],
    ]
    line_points = []
    for profile_points in line_profiles:
        projected = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=profile_points), camera, frame)
        draw.line(projected, fill=line_fill, width=2)
        line_points.extend(projected)
    return _bbox_union(bbox, _bbox_from_screen_points(line_points))


def _draw_camera_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height * 0.72))
    lens = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.20, height * 0.02), dimensions_xyz=(width * 0.48, depth * 0.40, height * 0.52))
    top = _sub_box_spec(spec, offset_xyz=(-width * 0.24, 0.0, height * 0.52), dimensions_xyz=(width * 0.28, depth * 0.38, height * 0.18))
    bboxes = [
        _draw_box_object(draw, body, camera=camera, frame=frame, fill=(53, 63, 76)),
        _draw_cylinder_object(draw, lens, camera=camera, frame=frame, fill=(42, 48, 58)),
        _draw_box_object(draw, top, camera=camera, frame=frame, fill=(78, 91, 108)),
    ]
    cx, cy = _project_xy(lens["world_xyz"], camera, frame)
    radius = max(6.0, _radius_px_for_object(lens, camera, frame) * 0.46)
    outer = (cx - radius, cy - radius * 0.76, cx + radius, cy + radius * 0.76)
    inner_radius = radius * 0.56
    inner = (cx - inner_radius, cy - inner_radius * 0.72, cx + inner_radius, cy + inner_radius * 0.72)
    draw.ellipse(outer, fill=(26, 32, 40), outline=(8, 12, 18), width=2)
    draw.ellipse(inner, fill=(94, 133, 163), outline=(155, 183, 202), width=1)
    flash = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.48, 0.42), (0.70, 0.42), (0.70, 0.64), (0.48, 0.64)])
    viewfinder = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.64, 0.32), (-0.38, 0.32), (-0.38, 0.54), (-0.64, 0.54)])
    draw.polygon(flash, fill=(226, 229, 213), outline=(23, 30, 38))
    draw.polygon(viewfinder, fill=(82, 105, 124), outline=(23, 30, 38))
    return _bbox_union(*bboxes, list(outer), _bbox_from_screen_points(flash), _bbox_from_screen_points(viewfinder))


def _draw_compass_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(225, 209, 154), profile_xz=_oval_profile_points(36), inset_scale=0.82)
    needle = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.0, 0.70), (0.12, -0.06), (0.0, -0.72), (-0.12, -0.06)]), camera, frame)
    draw.polygon(needle[:2] + [needle[3]], fill=(202, 56, 52))
    draw.polygon([needle[1], needle[2], needle[3]], fill=(48, 68, 96))
    _draw_polyline(draw, needle, fill=(28, 35, 45), width=1)
    return _bbox_union(bbox, _bbox_from_screen_points(needle))


def _draw_flask_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.26, 0.92), (0.26, 0.92), (0.22, 0.10), (0.78, -0.74), (0.60, -0.98), (-0.60, -0.98), (-0.78, -0.74), (-0.22, 0.10)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(196, 221, 229), profile_xz=profile, inset_scale=0.0)
    liquid = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=[(-0.56, -0.72), (0.56, -0.72), (0.44, -0.42), (-0.44, -0.42)]), camera, frame)
    draw.polygon(liquid, fill=_tint(fill, 0.18))
    _draw_polyline(draw, liquid, fill=_shade(fill, 0.66), width=1)
    return _bbox_union(bbox, _bbox_from_screen_points(liquid))


def _draw_test_tube_rack_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    rack = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth * 0.72, height * 0.34))
    bboxes = [_draw_box_object(draw, rack, camera=camera, frame=frame, fill=(127, 94, 63))]
    liquid_colors = [(90, 153, 205), (213, 103, 89), (92, 169, 113)]
    for index, px in enumerate((-0.30, 0.0, 0.30)):
        tube = _sub_box_spec(spec, offset_xyz=(width * px, 0.0, height * 0.18), dimensions_xyz=(width * 0.14, depth * 0.24, height * 0.74))
        bboxes.append(_draw_cylinder_object(draw, tube, camera=camera, frame=frame, fill=(207, 226, 232)))
        liquid = _sub_box_spec(spec, offset_xyz=(width * px, 0.0, height * 0.20), dimensions_xyz=(width * 0.12, depth * 0.20, height * 0.30))
        bboxes.append(_draw_cylinder_object(draw, liquid, camera=camera, frame=frame, fill=liquid_colors[index]))
    return _bbox_union(*bboxes)


def _draw_scroll_map_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
) -> List[float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    z = base_z + height
    sheet_world = [(x - width * 0.50, y - depth * 0.48, z), (x + width * 0.50, y - depth * 0.44, z), (x + width * 0.44, y + depth * 0.50, z), (x - width * 0.46, y + depth * 0.42, z)]
    sheet = _project_face(sheet_world, camera, frame)
    draw.polygon(sheet, fill=(229, 217, 178))
    _draw_polyline(draw, sheet, fill=(118, 90, 60), width=2)
    route_world = [(x - width * 0.30, y - depth * 0.24, z + height * 0.05), (x - width * 0.05, y + depth * 0.04, z + height * 0.05), (x + width * 0.18, y - depth * 0.02, z + height * 0.05), (x + width * 0.32, y + depth * 0.24, z + height * 0.05)]
    route = _project_face(route_world, camera, frame)
    draw.line(route, fill=(81, 118, 154), width=3)
    mark = _project_xy((x + width * 0.34, y + depth * 0.26, z + height * 0.05), camera, frame)
    radius = max(4.0, float(frame.scale) * width * 0.025)
    _draw_line(draw, (mark[0] - radius, mark[1] - radius), (mark[0] + radius, mark[1] + radius), fill=(180, 56, 52), width=2)
    _draw_line(draw, (mark[0] - radius, mark[1] + radius), (mark[0] + radius, mark[1] - radius), fill=(180, 56, 52), width=2)
    return _bbox_union(_bbox_from_screen_points(sheet), _bbox_from_screen_points(route), [mark[0] - radius, mark[1] - radius, mark[0] + radius, mark[1] + radius])


def _draw_microphone_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    head = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.48), dimensions_xyz=(width * 0.70, depth * 0.72, height * 0.42))
    handle = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.22, depth * 0.28, height * 0.58))
    base = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.72, depth * 0.48, height * 0.10))
    bboxes = [
        _draw_sphere_object(draw, head, camera=camera, frame=frame, fill=(74, 83, 94)),
        _draw_cylinder_object(draw, handle, camera=camera, frame=frame, fill=_shade(fill, 0.70)),
        _draw_cylinder_object(draw, base, camera=camera, frame=frame, fill=(50, 58, 68)),
    ]
    for pz in (0.58, 0.72, 0.86):
        p1 = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(-0.30, pz)])[0], camera, frame)
        p2 = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.30, pz)])[0], camera, frame)
        _draw_line(draw, p1, p2, fill=(150, 160, 170), width=1)
        bboxes.append(_bbox_from_screen_points([p1, p2]))
    return _bbox_union(*bboxes)


def _draw_stopwatch_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(221, 206, 150), profile_xz=_oval_profile_points(36, z_scale=0.88), inset_scale=0.82)
    button = _project_face(_upright_profile_world_points(spec, camera=camera, profile_xz=[(-0.18, 0.86), (0.18, 0.86), (0.18, 1.08), (-0.18, 1.08)]), camera, frame)
    draw.polygon(button, fill=(120, 128, 136))
    _draw_polyline(draw, button, fill=(38, 43, 50), width=1)
    center = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.0, 0.0)])[0], camera, frame)
    hand1 = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.0, 0.46)])[0], camera, frame)
    hand2 = _project_xy(_upright_profile_world_points(spec, camera=camera, profile_xz=[(0.34, -0.18)])[0], camera, frame)
    _draw_line(draw, center, hand1, fill=(43, 50, 60), width=2)
    _draw_line(draw, center, hand2, fill=(43, 50, 60), width=2)
    return _bbox_union(body_bbox, _bbox_from_screen_points(button), _bbox_from_screen_points([center, hand1, hand2]))


__all__ = [
    "_draw_padlock_object",
    "_draw_magnifying_glass_object",
    "_draw_candle_object",
    "_draw_scroll_object",
    "_draw_paint_brush_object",
    "_draw_paint_palette_object",
    "_draw_goblet_object",
    "_draw_teapot_object",
    "_draw_watering_can_object",
    "_draw_basket_object",
    "_draw_mail_envelope_object",
    "_draw_camera_object",
    "_draw_compass_object",
    "_draw_flask_object",
    "_draw_test_tube_rack_object",
    "_draw_scroll_map_object",
    "_draw_microphone_object",
    "_draw_stopwatch_object",
]
