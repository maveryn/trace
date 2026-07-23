"""Nature, music, food, and miscellaneous glyphs for shared three_d object scenes."""

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


def _draw_guitar_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.26, 0.0), dimensions_xyz=(width * 0.64, depth * 0.34, height * 1.10))
    neck = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.18, 0.0), dimensions_xyz=(width * 0.14, depth * 0.68, height * 0.58))
    head = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.54, 0.0), dimensions_xyz=(width * 0.32, depth * 0.18, height * 0.62))
    bboxes = [
        _draw_sphere_object(draw, body, camera=camera, frame=frame, fill=(168, 101, 55)),
        _draw_box_object(draw, neck, camera=camera, frame=frame, fill=(92, 57, 38)),
        _draw_box_object(draw, head, camera=camera, frame=frame, fill=(76, 48, 34)),
    ]
    hole = _project_xy(body["world_xyz"], camera, frame)
    radius = max(4.0, _radius_px_for_object(body, camera, frame) * 0.18)
    draw.ellipse((hole[0] - radius, hole[1] - radius * 0.66, hole[0] + radius, hole[1] + radius * 0.66), fill=(42, 30, 24))
    return _bbox_union(*bboxes, [hole[0] - radius, hole[1] - radius, hole[0] + radius, hole[1] + radius])


def _draw_drum_object(
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
    width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    base = _project_xy((x, y, base_z), camera, frame)
    top = _project_xy((x, y, base_z + height), camera, frame)
    radius = _radius_px_for_object({**dict(spec), "world_xyz": [x, y, base_z + height * 0.5]}, camera, frame)
    ellipse_h = max(9.0, radius * 0.42)
    outline = (28, 35, 45)
    rim = _shade(fill, 0.58)
    head = (238, 231, 210)

    side = [(top[0] - radius, top[1]), (top[0] + radius, top[1]), (base[0] + radius, base[1]), (base[0] - radius, base[1])]
    draw.polygon(side, fill=_shade(fill, 0.84))
    _draw_line(draw, (top[0] - radius, top[1]), (base[0] - radius, base[1]), fill=outline, width=2)
    _draw_line(draw, (top[0] + radius, top[1]), (base[0] + radius, base[1]), fill=outline, width=2)

    base_ellipse = [base[0] - radius, base[1] - ellipse_h, base[0] + radius, base[1] + ellipse_h]
    top_ellipse = [top[0] - radius, top[1] - ellipse_h, top[0] + radius, top[1] + ellipse_h]
    draw.ellipse(tuple(base_ellipse), fill=_shade(fill, 0.68), outline=outline, width=2)
    draw.ellipse(tuple(top_ellipse), fill=head, outline=outline, width=2)
    draw.arc(tuple(top_ellipse), start=0, end=180, fill=rim, width=4)
    draw.arc(tuple(base_ellipse), start=0, end=180, fill=rim, width=4)

    bboxes = [top_ellipse, base_ellipse]
    for px in (-0.42, -0.14, 0.14, 0.42):
        rod = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px, -0.40), (px * 0.86, 0.38)])
        draw.line(rod, fill=_tint(rim, 0.18), width=2)
        bboxes.append(_bbox_from_screen_points(rod))
        lug = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px * 0.92, 0.02)])[0]
        lug_radius = max(2.0, float(frame.scale) * width * 0.010)
        lug_box = [lug[0] - lug_radius, lug[1] - lug_radius, lug[0] + lug_radius, lug[1] + lug_radius]
        draw.ellipse(tuple(lug_box), fill=(226, 213, 177), outline=(61, 54, 45), width=1)
        bboxes.append(lug_box)
    return _bbox_union(*bboxes)


def _draw_pliers_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    pivot = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.02)])[0]
    left_handle = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.10, -0.08), (-0.58, -0.88)])
    right_handle = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.10, -0.08), (0.58, -0.88)])
    left_jaw = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.04, 0.08), (-0.46, 0.78)])
    right_jaw = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.04, 0.08), (0.46, 0.78)])
    draw.line(left_handle, fill=(177, 70, 62), width=6)
    draw.line(right_handle, fill=(177, 70, 62), width=6)
    draw.line(left_jaw, fill=(148, 158, 166), width=5)
    draw.line(right_jaw, fill=(148, 158, 166), width=5)
    radius = max(3.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.025)
    draw.ellipse((pivot[0] - radius, pivot[1] - radius, pivot[0] + radius, pivot[1] + radius), fill=(76, 84, 92))
    return _bbox_union(_bbox_from_screen_points(left_handle), _bbox_from_screen_points(right_handle), _bbox_from_screen_points(left_jaw), _bbox_from_screen_points(right_jaw), [pivot[0] - radius, pivot[1] - radius, pivot[0] + radius, pivot[1] + radius])


def _draw_telescope_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    tube = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.46, depth * 0.82, height * 0.70))
    lens = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.38, 0.0), dimensions_xyz=(width * 0.60, depth * 0.18, height * 0.82))
    eyepiece = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.42, 0.0), dimensions_xyz=(width * 0.32, depth * 0.14, height * 0.58))
    return _bbox_union(
        _draw_cylinder_object(draw, tube, camera=camera, frame=frame, fill=(65, 87, 115)),
        _draw_cylinder_object(draw, lens, camera=camera, frame=frame, fill=(92, 123, 150)),
        _draw_cylinder_object(draw, eyepiece, camera=camera, frame=frame, fill=(44, 55, 70)),
    )


def _draw_ruler_object(
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
        center_height_frac=0.62,
        length_scale=0.88,
        min_length_px=54.0,
        max_length_px=94.0,
    )
    half_width = max(12.5, min(16.0, length_px * 0.110))

    def point_at(axis_frac: float, offset_px: float = 0.0) -> Tuple[float, float]:
        return (
            center[0] + direction[0] * length_px * float(axis_frac) + normal[0] * float(offset_px),
            center[1] + direction[1] * length_px * float(axis_frac) + normal[1] * float(offset_px),
        )

    body = [
        point_at(-0.52, -half_width),
        point_at(0.52, -half_width),
        point_at(0.52, half_width),
        point_at(-0.52, half_width),
    ]
    outline = _shade(fill, 0.45)
    detail = _shade(fill, 0.56)
    draw.polygon(body, fill=_tint(fill, 0.12), outline=outline)
    bboxes = [_bbox_from_screen_points(body)]
    edge = [point_at(-0.48, -half_width * 0.62), point_at(0.48, -half_width * 0.62)]
    draw.line(edge, fill=detail, width=2)
    bboxes.append(_padded_bbox_from_screen_points(edge, pad_px=1.0))
    for index in range(13):
        axis_frac = -0.43 + 0.86 * float(index) / 12.0
        tick_length = half_width * (1.60 if index % 6 == 0 else 1.24 if index % 3 == 0 else 0.84)
        tick = [
            point_at(axis_frac, -half_width * 0.58),
            point_at(axis_frac, -half_width * 0.58 + tick_length),
        ]
        draw.line(tick, fill=detail, width=1)
        bboxes.append(_padded_bbox_from_screen_points(tick, pad_px=1.0))
    return _bbox_union(*bboxes)


def _draw_pickaxe_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    handle = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.18, 0.0), dimensions_xyz=(width * 0.14, depth * 0.72, height * 0.42))
    head = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.38, height * 0.06), dimensions_xyz=(width * 0.96, depth * 0.18, height * 0.44))
    points = _upright_screen_points(head, camera=camera, frame=frame, profile_xz=[(-0.96, 0.14), (-0.26, 0.34), (0.0, 0.06), (0.26, 0.34), (0.96, 0.14), (0.18, -0.16), (0.0, -0.08), (-0.18, -0.16)])
    draw.polygon(points, fill=(137, 148, 158))
    _draw_polyline(draw, points, fill=(53, 62, 70), width=2)
    return _bbox_union(_draw_box_object(draw, handle, camera=camera, frame=frame, fill=(102, 69, 43)), _bbox_from_screen_points(points))


def _draw_paint_roller_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    handle = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.30, 0.0), dimensions_xyz=(width * 0.16, depth * 0.42, height * 0.44))
    roller = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.36, height * 0.04), dimensions_xyz=(width * 0.80, depth * 0.22, height * 0.72))
    wire = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.28), (0.34, 0.10), (0.34, 0.46)])
    draw.line(wire, fill=(88, 98, 106), width=3)
    return _bbox_union(
        _draw_box_object(draw, handle, camera=camera, frame=frame, fill=(87, 65, 47)),
        _draw_cylinder_object(draw, roller, camera=camera, frame=frame, fill=_tint(fill, 0.12)),
        _bbox_from_screen_points(wire),
    )


def _draw_tape_measure_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(223, 177, 50), profile_xz=[(-0.78, -0.62), (0.54, -0.62), (0.76, -0.20), (0.54, 0.56), (-0.52, 0.66), (-0.78, 0.20)], inset_scale=0.0)
    tape = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.50, -0.06), (1.02, -0.06)])
    draw.line(tape, fill=(235, 229, 175), width=5)
    draw.line(tape, fill=(66, 58, 35), width=1)
    return _bbox_union(body, _bbox_from_screen_points(tape))


def _draw_remote_control_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(54, 62, 74), profile_xz=[(-0.52, -1.0), (0.52, -1.0), (0.52, 1.0), (-0.52, 1.0)], inset_scale=0.0)
    bboxes = [bbox]
    for px, pz, color in [(0.0, 0.66, (196, 55, 55)), (-0.22, 0.26, (216, 221, 224)), (0.22, 0.26, (216, 221, 224)), (-0.22, -0.12, (216, 221, 224)), (0.22, -0.12, (216, 221, 224)), (0.0, -0.52, (91, 148, 184))]:
        center = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px, pz)])[0]
        radius = max(1.5, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.013)
        button = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
        draw.ellipse(button, fill=color, outline=(22, 27, 33), width=1)
        bboxes.append(button)
    return _bbox_union(*bboxes)


def _draw_plug_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body_profile = [(-0.62, -0.58), (0.26, -0.58), (0.56, -0.34), (0.56, 0.34), (0.26, 0.58), (-0.62, 0.58)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(214, 218, 211), profile_xz=body_profile, inset_scale=0.0)
    face = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.04, -0.38), (0.42, -0.26), (0.42, 0.26), (0.04, 0.38)])
    draw.polygon(face, fill=(236, 238, 232))
    _draw_polyline(draw, face, fill=(20, 28, 38), width=1)
    prong_top = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.48, 0.20), (1.08, 0.20)])
    prong_bottom = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.48, -0.20), (1.08, -0.20)])
    cord = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.58, 0.00), (-0.98, -0.02), (-1.18, -0.30)])
    for prong in (prong_top, prong_bottom):
        draw.line(prong, fill=(38, 45, 52), width=7)
        draw.line(prong, fill=(207, 213, 218), width=4)
    draw.line(cord, fill=(28, 33, 39), width=5)
    return _bbox_union(
        bbox,
        _bbox_from_screen_points(face),
        _bbox_from_screen_points(prong_top),
        _bbox_from_screen_points(prong_bottom),
        _bbox_from_screen_points(cord),
    )


def _draw_wallet_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(112, 74, 48), profile_xz=[(-0.90, -0.58), (0.90, -0.58), (0.90, 0.58), (-0.90, 0.58)], inset_scale=0.0)
    flap = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.82, 0.18), (-0.18, -0.12), (0.82, 0.18)])
    draw.line(flap, fill=(66, 45, 34), width=2)
    return _bbox_union(bbox, _bbox_from_screen_points(flap))


def _draw_purse_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.72, -0.80), (0.72, -0.80), (0.58, 0.50), (0.24, 0.72), (-0.24, 0.72), (-0.58, 0.50)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(142, 78, 128), profile_xz=profile, inset_scale=0.0)
    handle = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.34, 0.50), (-0.18, 1.02), (0.18, 1.02), (0.34, 0.50)])
    draw.line(handle, fill=(74, 43, 68), width=4, joint="curve")
    return _bbox_union(bbox, _bbox_from_screen_points(handle))


def _draw_sunglasses_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    left = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.46, 0.0)])[0]
    right = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.46, 0.0)])[0]
    radius_x = max(8.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.050)
    radius_y = max(5.0, radius_x * 0.58)
    bboxes = []
    for center in (left, right):
        bbox = [center[0] - radius_x, center[1] - radius_y, center[0] + radius_x, center[1] + radius_y]
        draw.ellipse(bbox, fill=(40, 52, 66), outline=(17, 23, 30), width=3)
        bboxes.append(bbox)
    bridge = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.18, 0.02), (0.18, 0.02)])
    arm_l = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.76, 0.00), (-1.00, 0.34)])
    arm_r = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.76, 0.00), (1.00, 0.34)])
    draw.line(bridge, fill=(17, 23, 30), width=3)
    draw.line(arm_l, fill=(17, 23, 30), width=2)
    draw.line(arm_r, fill=(17, 23, 30), width=2)
    return _bbox_union(*bboxes, _bbox_from_screen_points(bridge), _bbox_from_screen_points(arm_l), _bbox_from_screen_points(arm_r))


def _draw_violin_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.20, 0.0), dimensions_xyz=(width * 0.66, depth * 0.42, height * 1.06))
    neck = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.28, 0.0), dimensions_xyz=(width * 0.12, depth * 0.54, height * 0.56))
    body_bbox = _draw_upright_profile_object(draw, body, camera=camera, frame=frame, fill=(157, 83, 43), profile_xz=[(-0.58, 0.64), (-0.30, 0.88), (0.0, 0.62), (0.30, 0.88), (0.58, 0.64), (0.34, 0.10), (0.58, -0.54), (0.18, -0.90), (0.0, -0.62), (-0.18, -0.90), (-0.58, -0.54), (-0.34, 0.10)], inset_scale=0.0)
    strings = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.72), (0.0, 0.98)])
    draw.line(strings, fill=(236, 220, 174), width=2)
    return _bbox_union(body_bbox, _draw_box_object(draw, neck, camera=camera, frame=frame, fill=(75, 48, 34)), _bbox_from_screen_points(strings))


def _draw_trumpet_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    tube = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.08, 0.0), dimensions_xyz=(width * 0.20, depth * 0.70, height * 0.44))
    bell = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.38, 0.0), dimensions_xyz=(width * 0.72, depth * 0.26, height * 0.70))
    mouth = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.50, 0.0), dimensions_xyz=(width * 0.32, depth * 0.12, height * 0.38))
    bboxes = [
        _draw_cylinder_object(draw, tube, camera=camera, frame=frame, fill=(207, 162, 50)),
        _draw_cone_object(draw, bell, camera=camera, frame=frame, fill=(224, 180, 62)),
        _draw_cylinder_object(draw, mouth, camera=camera, frame=frame, fill=(183, 138, 42)),
    ]
    for px in (-0.22, 0.0, 0.22):
        valve = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px, 0.00), (px, 0.28)])
        draw.line(valve, fill=(138, 98, 30), width=2)
        bboxes.append(_bbox_from_screen_points(valve))
    return _bbox_union(*bboxes)


def _draw_donut_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    return _draw_torus_object(draw, spec, camera=camera, frame=frame, fill=(194, 125, 64), floor_rgb=floor_rgb)


def _draw_pretzel_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    left = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.72, 0.06), (-0.58, 0.48), (-0.16, 0.42), (-0.18, 0.02), (-0.54, -0.26), (-0.82, -0.06)])
    right = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.72, 0.06), (0.58, 0.48), (0.16, 0.42), (0.18, 0.02), (0.54, -0.26), (0.82, -0.06)])
    cross = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.54, -0.46), (0.0, 0.10), (0.54, -0.46)])
    for path in (left, right, cross):
        draw.line(path, fill=(154, 93, 45), width=6, joint="curve")
        draw.line(path, fill=(92, 58, 36), width=1, joint="curve")
    return _bbox_union(_bbox_from_screen_points(left), _bbox_from_screen_points(right), _bbox_from_screen_points(cross))


def _draw_lollipop_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    stick = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.12, depth * 0.12, height * 0.62))
    candy = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.44), dimensions_xyz=(width * 0.88, depth * 0.88, height * 0.46))
    swirl = _upright_screen_points(candy, camera=camera, frame=frame, profile_xz=[(-0.34, 0.04), (-0.10, 0.26), (0.20, 0.14), (0.34, -0.12), (0.02, -0.28), (-0.24, -0.10)])
    draw.line(swirl, fill=(242, 238, 244), width=3, joint="curve")
    return _bbox_union(
        _draw_cylinder_object(draw, stick, camera=camera, frame=frame, fill=(232, 226, 190)),
        _draw_sphere_object(draw, candy, camera=camera, frame=frame, fill=(205, 64, 126)),
        _bbox_from_screen_points(swirl),
    )


def _draw_ice_cream_cone_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    cone = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.66, depth * 0.66, height * 0.58))
    scoop = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.46), dimensions_xyz=(width * 0.86, depth * 0.86, height * 0.44))
    hatch1 = _upright_screen_points(cone, camera=camera, frame=frame, profile_xz=[(-0.34, -0.34), (0.34, 0.34)])
    hatch2 = _upright_screen_points(cone, camera=camera, frame=frame, profile_xz=[(0.34, -0.34), (-0.34, 0.34)])
    bboxes = [
        _draw_cone_object(draw, cone, camera=camera, frame=frame, fill=(198, 145, 72)),
        _draw_sphere_object(draw, scoop, camera=camera, frame=frame, fill=(232, 206, 146)),
    ]
    draw.line(hatch1, fill=(125, 82, 42), width=1)
    draw.line(hatch2, fill=(125, 82, 42), width=1)
    return _bbox_union(*bboxes, _bbox_from_screen_points(hatch1), _bbox_from_screen_points(hatch2))


def _draw_soap_bar_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(136, 190, 204), profile_xz=[(-0.84, -0.46), (0.84, -0.46), (0.96, -0.18), (0.74, 0.46), (-0.74, 0.46), (-0.96, -0.18)], inset_scale=0.76)
    shine = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.42, 0.10), (-0.10, 0.24)])
    draw.line(shine, fill=(213, 236, 240), width=2)
    return _bbox_union(bbox, _bbox_from_screen_points(shine))


def _draw_clock_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(224, 207, 149), profile_xz=_oval_profile_points(48, z_scale=0.92), inset_scale=0.74)
    outer_rim = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=_oval_profile_points(56, z_scale=0.92))
    inner_face = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(x * 0.74, z * 0.74) for x, z in _oval_profile_points(56, z_scale=0.92)])
    _draw_polyline(draw, outer_rim + [outer_rim[0]], fill=(42, 49, 56), width=2)
    _draw_polyline(draw, inner_face + [inner_face[0]], fill=(126, 112, 82), width=1)
    center = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.0)])[0]
    hand1 = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.0), (0.0, 0.46)])
    hand2 = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.0), (0.36, -0.16)])
    _draw_line(draw, hand1[0], hand1[1], fill=(45, 52, 60), width=2)
    _draw_line(draw, hand2[0], hand2[1], fill=(45, 52, 60), width=2)
    tick_bboxes = [_bbox_from_screen_points(outer_rim), _bbox_from_screen_points(inner_face)]
    for index in range(12):
        angle = math.pi * 0.5 - index * math.tau / 12.0
        outer_x = math.cos(angle) * 0.66
        outer_z = math.sin(angle) * 0.61
        inner_scale = 0.82 if index % 3 == 0 else 0.90
        tick = _upright_screen_points(
            spec,
            camera=camera,
            frame=frame,
            profile_xz=[(outer_x * inner_scale, outer_z * inner_scale), (outer_x, outer_z)],
        )
        draw.line(tick, fill=(62, 70, 78), width=2 if index % 3 == 0 else 1)
        tick_bboxes.append(_bbox_from_screen_points(tick))
    center_radius = max(2.4, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.018)
    draw.ellipse(
        (
            center[0] - center_radius,
            center[1] - center_radius,
            center[0] + center_radius,
            center[1] + center_radius,
        ),
        fill=(45, 52, 60),
    )
    return _bbox_union(
        body,
        _bbox_from_screen_points([center, hand1[1], hand2[1]]),
        [center[0] - center_radius, center[1] - center_radius, center[0] + center_radius, center[1] + center_radius],
        *tick_bboxes,
    )


__all__ = [
    "_draw_guitar_object",
    "_draw_drum_object",
    "_draw_pliers_object",
    "_draw_telescope_object",
    "_draw_ruler_object",
    "_draw_pickaxe_object",
    "_draw_paint_roller_object",
    "_draw_tape_measure_object",
    "_draw_remote_control_object",
    "_draw_plug_object",
    "_draw_wallet_object",
    "_draw_purse_object",
    "_draw_sunglasses_object",
    "_draw_violin_object",
    "_draw_trumpet_object",
    "_draw_donut_object",
    "_draw_pretzel_object",
    "_draw_lollipop_object",
    "_draw_ice_cream_cone_object",
    "_draw_soap_bar_object",
    "_draw_clock_object",
]
