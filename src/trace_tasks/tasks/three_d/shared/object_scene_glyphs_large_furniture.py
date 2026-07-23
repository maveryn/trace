"""Large furniture glyphs for shared three_d object scenes."""

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


def _draw_bench_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    wood = _shade((158, 102, 55), 0.88 + (sum(fill) % 20) / 160.0)
    metal = (58, 66, 74)
    seat_z = height * 0.40
    plank_h = height * 0.10
    plank_depth = depth * 0.18
    leg_h = height * 0.42
    leg_w = min(width, depth) * 0.10
    parts: List[Mapping[str, Any]] = []
    for y_offset in (-depth * 0.24, 0.0, depth * 0.24):
        parts.append(
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, y_offset, seat_z),
                dimensions_xyz=(width * 0.92, plank_depth, plank_h),
            )
        )
    parts.extend(
        [
            _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.36, height * 0.62), dimensions_xyz=(width * 0.92, depth * 0.10, height * 0.16)),
            _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.40, height * 0.82), dimensions_xyz=(width * 0.92, depth * 0.10, height * 0.16)),
        ]
    )
    bboxes = [_draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=wood)]

    support_parts: List[Mapping[str, Any]] = []
    for sx in (-0.38, 0.38):
        for sy in (-0.24, 0.24):
            support_parts.append(
                _sub_box_spec(
                    spec,
                    offset_xyz=(sx * width, sy * depth, 0.0),
                    dimensions_xyz=(leg_w, leg_w, leg_h),
                )
            )
        support_parts.append(
            _sub_box_spec(
                spec,
                offset_xyz=(sx * width, depth * 0.39, leg_h),
                dimensions_xyz=(leg_w, leg_w, height * 0.48),
            )
        )
    bboxes.append(_draw_box_parts_object(draw, support_parts, camera=camera, frame=frame, fill=metal))

    rail_parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.28, leg_h * 0.56), dimensions_xyz=(width * 0.76, leg_w * 0.55, leg_w * 0.55)),
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.28, leg_h * 0.56), dimensions_xyz=(width * 0.76, leg_w * 0.55, leg_w * 0.55)),
    ]
    bboxes.append(_draw_box_parts_object(draw, rail_parts, camera=camera, frame=frame, fill=metal))
    return _bbox_union(*bboxes)


def _draw_piano_object(
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
    front_sign = -1.0 if float(camera.camera_position[1]) <= y else 1.0
    keyboard_y = y + front_sign * depth * 0.52
    back_sign = -front_sign
    black = (35, 30, 28)
    glossy = (54, 45, 41)
    inner_wood = (182, 119, 58)
    outline = (24, 21, 19)
    leg_h = height * 0.38
    case_h = height * 0.32
    case_top_z = base_z + leg_h + case_h
    bboxes: List[List[float]] = []

    leg_parts: List[Mapping[str, Any]] = []
    for lx, ly in ((-0.38, 0.34), (0.36, 0.28), (0.00, -0.42)):
        leg_parts.append(
            _sub_box_spec(
                spec,
                offset_xyz=(lx * width, ly * depth, 0.0),
                dimensions_xyz=(width * 0.055, depth * 0.055, leg_h),
            )
        )
    bboxes.append(_draw_box_parts_object(draw, leg_parts, camera=camera, frame=frame, fill=black))

    body_spec = _sub_box_spec(
        spec,
        offset_xyz=(0.0, 0.0, leg_h),
        dimensions_xyz=(width, depth, case_h),
    )
    body_footprint = [
        (-0.88, front_sign * 0.82),
        (0.22, front_sign * 0.86),
        (0.76, front_sign * 0.50),
        (0.94, front_sign * 0.05),
        (0.72, back_sign * 0.54),
        (0.10, back_sign * 0.88),
        (-0.72, back_sign * 0.72),
        (-0.94, back_sign * 0.12),
    ]
    bboxes.append(_draw_footprint_prism_object(draw, body_spec, camera=camera, frame=frame, fill=glossy, footprint_xy=body_footprint))

    soundboard = _project_face(
        [
            (x - width * 0.46, y + front_sign * depth * 0.22, case_top_z + height * 0.01),
            (x + width * 0.18, y + front_sign * depth * 0.24, case_top_z + height * 0.01),
            (x + width * 0.54, y + back_sign * depth * 0.22, case_top_z + height * 0.01),
            (x + width * 0.12, y + back_sign * depth * 0.52, case_top_z + height * 0.01),
            (x - width * 0.54, y + back_sign * depth * 0.38, case_top_z + height * 0.01),
        ],
        camera,
        frame,
    )
    draw.polygon(soundboard, fill=inner_wood)
    _draw_polyline(draw, soundboard, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(soundboard))
    for frac in (0.18, 0.30, 0.42, 0.54, 0.66, 0.78):
        string = [
            _project_xy((x - width * 0.42 + width * 0.34 * frac, y + front_sign * depth * 0.24, case_top_z + height * 0.025), camera, frame),
            _project_xy((x - width * 0.20 + width * 0.68 * frac, y + back_sign * depth * 0.48, case_top_z + height * 0.025), camera, frame),
        ]
        _draw_line(draw, string[0], string[1], fill=(218, 176, 82), width=1)

    lid_hinge_a = (x - width * 0.52, y + back_sign * depth * 0.42, case_top_z + height * 0.02)
    lid_hinge_b = (x + width * 0.18, y + back_sign * depth * 0.50, case_top_z + height * 0.02)
    lid_tip_a = (x - width * 0.42, y + back_sign * depth * 0.66, base_z + height * 0.98)
    lid_tip_b = (x + width * 0.28, y + back_sign * depth * 0.72, base_z + height * 0.88)
    lid = _project_face([lid_hinge_a, lid_hinge_b, lid_tip_b, lid_tip_a], camera, frame)
    draw.polygon(lid, fill=(28, 25, 24))
    _draw_polyline(draw, lid, fill=(10, 10, 10), width=2)
    bboxes.append(_bbox_from_screen_points(lid))
    prop_line = [
        _project_xy((x + width * 0.02, y + back_sign * depth * 0.28, case_top_z + height * 0.02), camera, frame),
        _project_xy((x + width * 0.18, y + back_sign * depth * 0.62, base_z + height * 0.86), camera, frame),
    ]
    _draw_line(draw, prop_line[0], prop_line[1], fill=(190, 149, 70), width=3)
    bboxes.append(_bbox_from_screen_points(prop_line))

    keyboard_shelf = _sub_box_spec(
        spec,
        offset_xyz=(-width * 0.20, front_sign * depth * 0.50, leg_h + case_h * 0.06),
        dimensions_xyz=(width * 0.86, depth * 0.24, height * 0.12),
    )
    bboxes.append(_draw_box_object(draw, keyboard_shelf, camera=camera, frame=frame, fill=black))
    key_left = x - width * 0.60
    key_right = x + width * 0.20
    key_y_front = y + front_sign * depth * 0.63
    key_y_back = y + front_sign * depth * 0.39
    key_z = case_top_z + height * 0.05
    fallboard = _project_face(
        [
            (key_left - width * 0.02, key_y_back - front_sign * depth * 0.05, key_z + height * 0.02),
            (key_right + width * 0.02, key_y_back - front_sign * depth * 0.05, key_z + height * 0.02),
            (key_right + width * 0.02, key_y_back, key_z + height * 0.02),
            (key_left - width * 0.02, key_y_back, key_z + height * 0.02),
        ],
        camera,
        frame,
    )
    draw.polygon(fallboard, fill=(20, 19, 18))
    bboxes.append(_bbox_from_screen_points(fallboard))
    keyboard_face = _project_face(
        [
            (key_left, key_y_front, key_z),
            (key_right, key_y_front, key_z),
            (key_right, key_y_back, key_z),
            (key_left, key_y_back, key_z),
        ],
        camera,
        frame,
    )
    draw.polygon(keyboard_face, fill=(238, 232, 213))
    _draw_polyline(draw, keyboard_face, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(keyboard_face))
    white_key_count = 16
    for index in range(1, white_key_count):
        key_x = key_left + (key_right - key_left) * index / white_key_count
        line = [
            _project_xy((key_x, key_y_front, key_z + height * 0.004), camera, frame),
            _project_xy((key_x, key_y_back, key_z + height * 0.004), camera, frame),
        ]
        _draw_line(draw, line[0], line[1], fill=(118, 105, 88), width=2)
    black_pattern = {0, 1, 3, 4, 5}
    for index in range(white_key_count - 1):
        if index % 7 not in black_pattern:
            continue
        bx0 = key_left + (key_right - key_left) * (index + 0.66) / white_key_count
        bx1 = key_left + (key_right - key_left) * (index + 0.98) / white_key_count
        black_key = _project_face(
            [
                (bx0, y + front_sign * depth * 0.53, key_z + height * 0.015),
                (bx1, y + front_sign * depth * 0.53, key_z + height * 0.015),
                (bx1, y + front_sign * depth * 0.40, key_z + height * 0.015),
                (bx0, y + front_sign * depth * 0.40, key_z + height * 0.015),
            ],
            camera,
            frame,
        )
        draw.polygon(black_key, fill=(28, 29, 30))
        _draw_polyline(draw, black_key, fill=(8, 8, 8), width=1)
        bboxes.append(_bbox_from_screen_points(black_key))

    lyre = [
        _project_xy((x - width * 0.22, key_y_front, base_z + height * 0.12), camera, frame),
        _project_xy((x - width * 0.22, key_y_front, base_z + leg_h * 0.84), camera, frame),
    ]
    _draw_line(draw, lyre[0], lyre[1], fill=black, width=4)
    bboxes.append(_bbox_from_screen_points(lyre))
    for pedal_x in (-0.24, -0.21, -0.18):
        pedal = [
            _project_xy((x + width * pedal_x, key_y_front, base_z + height * 0.06), camera, frame),
            _project_xy((x + width * pedal_x, key_y_front, base_z + height * 0.16), camera, frame),
        ]
        _draw_line(draw, pedal[0], pedal[1], fill=(205, 158, 67), width=3)
        bboxes.append(_bbox_from_screen_points(pedal))
    return _bbox_union(*bboxes)


def _draw_locker_object(
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
    face_y = y - depth * 0.515 if float(camera.camera_position[1]) <= y else y + depth * 0.515
    steel = _shade((112, 143, 159), 0.88 + (sum(fill) % 22) / 180.0)
    door_left = _shade(steel, 0.92)
    door_right = _tint(steel, 0.06)
    outline = (32, 45, 54)
    left_x = x - width * 0.40
    right_x = x + width * 0.40
    mid_x = x
    bboxes: List[List[float]] = [
        _draw_box_object(
            draw,
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height)),
            camera=camera,
            frame=frame,
            fill=steel,
        )
    ]
    left_door = _project_face(
        [
            (left_x, face_y, base_z + height * 0.08),
            (mid_x, face_y, base_z + height * 0.08),
            (mid_x, face_y, base_z + height * 0.94),
            (left_x, face_y, base_z + height * 0.94),
        ],
        camera,
        frame,
    )
    right_door = _project_face(
        [
            (mid_x, face_y, base_z + height * 0.08),
            (right_x, face_y, base_z + height * 0.08),
            (right_x, face_y, base_z + height * 0.94),
            (mid_x, face_y, base_z + height * 0.94),
        ],
        camera,
        frame,
    )
    for door, color in ((left_door, door_left), (right_door, door_right)):
        draw.polygon(door, fill=color)
        _draw_polyline(draw, door, fill=outline, width=2)
        bboxes.append(_bbox_from_screen_points(door))

    seam = [
        _project_xy((mid_x, face_y, base_z + height * 0.08), camera, frame),
        _project_xy((mid_x, face_y, base_z + height * 0.94), camera, frame),
    ]
    _draw_line(draw, seam[0], seam[1], fill=outline, width=3)
    bboxes.append(_bbox_from_screen_points(seam))

    vent_ranges = ((0.76, 0.88), (0.18, 0.30))
    for x0, x1 in ((left_x + width * 0.07, mid_x - width * 0.07), (mid_x + width * 0.07, right_x - width * 0.07)):
        for z0, z1 in vent_ranges:
            for row in range(4):
                z = base_z + height * (z0 + (z1 - z0) * (row + 0.5) / 4.0)
                line = [
                    _project_xy((x0, face_y, z), camera, frame),
                    _project_xy((x1, face_y, z), camera, frame),
                ]
                _draw_line(draw, line[0], line[1], fill=(45, 62, 72), width=2)
    for handle_x in (mid_x - width * 0.045, mid_x + width * 0.045):
        handle = [
            _project_xy((handle_x, face_y, base_z + height * 0.43), camera, frame),
            _project_xy((handle_x, face_y, base_z + height * 0.56), camera, frame),
        ]
        _draw_line(draw, handle[0], handle[1], fill=(230, 235, 238), width=4)
        bboxes.append(_bbox_from_screen_points(handle))
    name_plate = _project_face(
        [
            (left_x + width * 0.08, face_y, base_z + height * 0.64),
            (left_x + width * 0.24, face_y, base_z + height * 0.64),
            (left_x + width * 0.24, face_y, base_z + height * 0.69),
            (left_x + width * 0.08, face_y, base_z + height * 0.69),
        ],
        camera,
        frame,
    )
    draw.polygon(name_plate, fill=(235, 235, 208))
    _draw_polyline(draw, name_plate, fill=outline, width=1)
    bboxes.append(_bbox_from_screen_points(name_plate))
    return _bbox_union(*bboxes)


def _draw_cabinet_object(
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
    wood = _shade(fill, 0.86)
    bboxes: List[List[float]] = [
        _draw_box_object(
            draw,
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height)),
            camera=camera,
            frame=frame,
            fill=wood,
        )
    ]
    face_y = y - depth * 0.515 if float(camera.camera_position[1]) <= y else y + depth * 0.515
    x_left = x - width * 0.39
    x_right = x + width * 0.39
    outline = (50, 42, 36)
    handle_rgb = (210, 176, 92)
    for index, lower_frac in enumerate((0.12, 0.31, 0.50, 0.69)):
        drawer_bottom = base_z + height * float(lower_frac)
        drawer_top = drawer_bottom + height * 0.145
        drawer_face = [
            (x_left, face_y, drawer_bottom),
            (x_right, face_y, drawer_bottom),
            (x_right, face_y, drawer_top),
            (x_left, face_y, drawer_top),
        ]
        projected = _project_face(drawer_face, camera, frame)
        draw.polygon(projected, fill=_tint(wood, 0.08 + 0.035 * (index % 2)))
        _draw_polyline(draw, projected, fill=outline, width=2)
        handle_z = (drawer_bottom + drawer_top) * 0.5
        handle = [
            _project_xy((x - width * 0.12, face_y, handle_z), camera, frame),
            _project_xy((x + width * 0.12, face_y, handle_z), camera, frame),
        ]
        _draw_line(draw, handle[0], handle[1], fill=handle_rgb, width=4)
        bboxes.append(_bbox_union(*[[point[0], point[1], point[0], point[1]] for point in [*projected, *handle]]))
    return _bbox_union(*bboxes)


def _draw_sofa_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.08, 0.0), dimensions_xyz=(width * 0.82, depth * 0.66, height * 0.34)),
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.36, height * 0.28), dimensions_xyz=(width * 0.90, depth * 0.18, height * 0.66)),
        _sub_box_spec(spec, offset_xyz=(-width * 0.47, -depth * 0.04, 0.0), dimensions_xyz=(width * 0.13, depth * 0.74, height * 0.58)),
        _sub_box_spec(spec, offset_xyz=(width * 0.47, -depth * 0.04, 0.0), dimensions_xyz=(width * 0.13, depth * 0.74, height * 0.58)),
        _sub_box_spec(spec, offset_xyz=(-width * 0.20, -depth * 0.12, height * 0.18), dimensions_xyz=(width * 0.34, depth * 0.52, height * 0.20)),
        _sub_box_spec(spec, offset_xyz=(width * 0.20, -depth * 0.12, height * 0.18), dimensions_xyz=(width * 0.34, depth * 0.52, height * 0.20)),
    ]
    return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)


def _draw_barrel_object(
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
    mid = _project_xy((x, y, base_z + height * 0.50), camera, frame)
    top = _project_xy((x, y, base_z + height), camera, frame)
    radius = _radius_px_for_object({**dict(spec), "world_xyz": [x, y, base_z + height * 0.5]}, camera, frame)
    top_radius = radius * 0.86
    bulge_radius = radius * 1.08
    ellipse_h = max(10.0, radius * 0.36)
    wood = (154, 99, 48)
    dark_wood = (104, 63, 34)
    hoop = (61, 70, 78)
    outline = (28, 35, 45)
    side = [
        (top[0] - top_radius, top[1]),
        (top[0] + top_radius, top[1]),
        (mid[0] + bulge_radius, mid[1]),
        (base[0] + top_radius, base[1]),
        (base[0] - top_radius, base[1]),
        (mid[0] - bulge_radius, mid[1]),
    ]
    draw.polygon(side, fill=wood)
    draw.polygon(
        [(top[0] - top_radius, top[1]), (mid[0] - bulge_radius, mid[1]), (base[0] - top_radius, base[1]), base, top],
        fill=_shade(wood, 0.78),
    )
    draw.ellipse(
        (base[0] - top_radius, base[1] - ellipse_h, base[0] + top_radius, base[1] + ellipse_h),
        fill=_shade(dark_wood, 0.82),
        outline=outline,
        width=2,
    )
    draw.ellipse(
        (top[0] - top_radius, top[1] - ellipse_h, top[0] + top_radius, top[1] + ellipse_h),
        fill=_tint(wood, 0.18),
        outline=outline,
        width=2,
    )
    for frac in (0.14, 0.30, 0.68, 0.86):
        center = _project_xy((x, y, base_z + height * frac), camera, frame)
        band_radius = radius * (1.05 if 0.28 < frac < 0.72 else 0.96)
        band_h = max(7.0, ellipse_h * 0.62)
        draw.ellipse(
            (center[0] - band_radius, center[1] - band_h, center[0] + band_radius, center[1] + band_h),
            outline=hoop,
            width=6,
        )
    for offset in (-0.52, -0.26, 0.0, 0.26, 0.52):
        top_point = (top[0] + offset * top_radius, top[1] + ellipse_h * 0.20)
        base_point = (base[0] + offset * top_radius, base[1] - ellipse_h * 0.20)
        _draw_line(draw, top_point, base_point, fill=_shade(dark_wood, 0.90), width=2)
    tap_y = float(mid[1] + ellipse_h * 0.08)
    tap_x = float(mid[0] - radius * 0.18)
    tap = [tap_x - radius * 0.10, tap_y - ellipse_h * 0.30, tap_x + radius * 0.24, tap_y + ellipse_h * 0.12]
    draw.rectangle(tap, fill=(201, 154, 75), outline=outline, width=2)
    plug_r = max(3.0, radius * 0.08)
    plug_cx = float(mid[0] + radius * 0.24)
    plug_cy = float(mid[1] - ellipse_h * 0.05)
    draw.ellipse((plug_cx - plug_r, plug_cy - plug_r, plug_cx + plug_r, plug_cy + plug_r), fill=(76, 43, 26), outline=outline, width=1)
    _draw_polyline(draw, side, fill=outline, width=2)
    return _bbox_union(
        [top[0] - top_radius, top[1] - ellipse_h, top[0] + top_radius, top[1] + ellipse_h],
        [mid[0] - bulge_radius, mid[1] - ellipse_h, mid[0] + bulge_radius, mid[1] + ellipse_h],
        [base[0] - top_radius, base[1] - ellipse_h, base[0] + top_radius, base[1] + ellipse_h],
        tap,
    )


def _draw_chair_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    seat_h = height * 0.14
    leg_h = height * 0.42
    leg_w = min(width, depth) * 0.11
    parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.08, leg_h), dimensions_xyz=(width * 0.82, depth * 0.70, seat_h)),
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.35, leg_h + seat_h * 0.15), dimensions_xyz=(width * 0.82, depth * 0.15, height * 0.58)),
    ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            parts.append(
                _sub_box_spec(
                    spec,
                    offset_xyz=(sx * width * 0.32, sy * depth * 0.26, 0.0),
                    dimensions_xyz=(leg_w, leg_w, leg_h),
                )
            )
    return _draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)


__all__ = [
    "_draw_bench_object",
    "_draw_piano_object",
    "_draw_locker_object",
    "_draw_cabinet_object",
    "_draw_sofa_object",
    "_draw_barrel_object",
    "_draw_chair_object",
]
