"""Large appliance glyphs for shared three_d object scenes."""

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


def _draw_refrigerator_object(
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
    body_rgb = (198, 216, 222)
    side_rgb = (158, 181, 190)
    outline = (36, 48, 56)
    bbox = _draw_box_object(
        draw,
        _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height)),
        camera=camera,
        frame=frame,
        fill=body_rgb,
    )
    face_y = y - depth * 0.515 if float(camera.camera_position[1]) <= y else y + depth * 0.515
    left_x = x - width * 0.40
    right_x = x + width * 0.40
    mid_x = x + width * 0.03
    freezer_bottom = base_z + height * 0.64
    door_bottom = base_z + height * 0.08
    door_top = base_z + height * 0.94
    freezer_face = [
        (left_x, face_y, freezer_bottom),
        (right_x, face_y, freezer_bottom),
        (right_x, face_y, door_top),
        (left_x, face_y, door_top),
    ]
    lower_left = [
        (left_x, face_y, door_bottom),
        (mid_x, face_y, door_bottom),
        (mid_x, face_y, freezer_bottom),
        (left_x, face_y, freezer_bottom),
    ]
    lower_right = [
        (mid_x, face_y, door_bottom),
        (right_x, face_y, door_bottom),
        (right_x, face_y, freezer_bottom),
        (mid_x, face_y, freezer_bottom),
    ]
    bboxes = [bbox]
    for points, color in (
        (freezer_face, _tint(body_rgb, 0.08)),
        (lower_left, side_rgb),
        (lower_right, _tint(side_rgb, 0.12)),
    ):
        projected = _project_face(points, camera, frame)
        draw.polygon(projected, fill=color)
        _draw_polyline(draw, projected, fill=outline, width=2)
        bboxes.append(_bbox_from_screen_points(projected))
    seam = [
        _project_xy((mid_x, face_y, door_bottom), camera, frame),
        _project_xy((mid_x, face_y, freezer_bottom), camera, frame),
    ]
    freezer_handle = [
        _project_xy((right_x - width * 0.10, face_y, freezer_bottom + height * 0.08), camera, frame),
        _project_xy((right_x - width * 0.10, face_y, door_top - height * 0.08), camera, frame),
    ]
    lower_handle = [
        _project_xy((mid_x + width * 0.08, face_y, door_bottom + height * 0.10), camera, frame),
        _project_xy((mid_x + width * 0.08, face_y, freezer_bottom - height * 0.10), camera, frame),
    ]
    for line in (seam, freezer_handle, lower_handle):
        _draw_line(draw, line[0], line[1], fill=outline, width=3)
        bboxes.append(_bbox_from_screen_points(line))
    vent = _project_face(
        [
            (left_x + width * 0.05, face_y, door_bottom + height * 0.04),
            (left_x + width * 0.34, face_y, door_bottom + height * 0.04),
            (left_x + width * 0.34, face_y, door_bottom + height * 0.10),
            (left_x + width * 0.05, face_y, door_bottom + height * 0.10),
        ],
        camera,
        frame,
    )
    draw.polygon(vent, fill=(92, 111, 122))
    bboxes.append(_bbox_from_screen_points(vent))
    return _bbox_union(*bboxes)


def _draw_washing_machine_object(
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
    shell = _tint(fill, 0.68)
    trim = (54, 67, 78)
    glass = (86, 145, 175)
    face_y = y - depth * 0.515 if float(camera.camera_position[1]) <= y else y + depth * 0.515
    left_x = x - width * 0.40
    right_x = x + width * 0.40
    bboxes: List[List[float]] = [
        _draw_box_object(
            draw,
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height)),
            camera=camera,
            frame=frame,
            fill=shell,
        )
    ]
    control_panel = _project_face(
        [
            (left_x, face_y, base_z + height * 0.76),
            (right_x, face_y, base_z + height * 0.76),
            (right_x, face_y, base_z + height * 0.92),
            (left_x, face_y, base_z + height * 0.92),
        ],
        camera,
        frame,
    )
    draw.polygon(control_panel, fill=(205, 218, 224))
    _draw_polyline(draw, control_panel, fill=trim, width=2)
    bboxes.append(_bbox_from_screen_points(control_panel))

    door_center = _project_xy((x - width * 0.04, face_y, base_z + height * 0.43), camera, frame)
    door_edge = _project_xy((x + width * 0.24, face_y, base_z + height * 0.43), camera, frame)
    door_radius = max(12.0, min(70.0, math.hypot(door_edge[0] - door_center[0], door_edge[1] - door_center[1])))
    door_outer = [
        door_center[0] - door_radius,
        door_center[1] - door_radius,
        door_center[0] + door_radius,
        door_center[1] + door_radius,
    ]
    inner_radius = door_radius * 0.66
    door_inner = [
        door_center[0] - inner_radius,
        door_center[1] - inner_radius,
        door_center[0] + inner_radius,
        door_center[1] + inner_radius,
    ]
    draw.ellipse(door_outer, fill=(220, 229, 233), outline=trim, width=4)
    draw.ellipse(door_inner, fill=glass, outline=(33, 60, 78), width=3)
    shine = [
        door_center[0] - inner_radius * 0.42,
        door_center[1] - inner_radius * 0.46,
        door_center[0] + inner_radius * 0.10,
        door_center[1] - inner_radius * 0.10,
    ]
    draw.arc(shine, start=190, end=330, fill=(184, 224, 239), width=3)
    bboxes.append(door_outer)

    knob_center = _project_xy((right_x - width * 0.14, face_y, base_z + height * 0.84), camera, frame)
    knob_radius = max(4.0, door_radius * 0.13)
    knob_bbox = [
        knob_center[0] - knob_radius,
        knob_center[1] - knob_radius,
        knob_center[0] + knob_radius,
        knob_center[1] + knob_radius,
    ]
    draw.ellipse(knob_bbox, fill=(89, 105, 116), outline=trim, width=1)
    for offset in (-0.10, 0.02, 0.14):
        button = _project_xy((left_x + width * (0.18 + offset), face_y, base_z + height * 0.84), camera, frame)
        radius = max(2.5, knob_radius * 0.55)
        draw.ellipse(
            (button[0] - radius, button[1] - radius, button[0] + radius, button[1] + radius),
            fill=(73, 91, 101),
            outline=trim,
            width=1,
        )
    bboxes.append(knob_bbox)

    kick_plate = _project_face(
        [
            (left_x, face_y, base_z + height * 0.08),
            (right_x, face_y, base_z + height * 0.08),
            (right_x, face_y, base_z + height * 0.15),
            (left_x, face_y, base_z + height * 0.15),
        ],
        camera,
        frame,
    )
    draw.polygon(kick_plate, fill=(150, 163, 171))
    bboxes.append(_bbox_from_screen_points(kick_plate))
    return _bbox_union(*bboxes)


def _draw_vending_machine_object(
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
    body = _shade(fill, 0.88)
    outline = (34, 42, 52)
    face_y = y - depth * 0.515 if float(camera.camera_position[1]) <= y else y + depth * 0.515
    left_x = x - width * 0.42
    right_x = x + width * 0.42
    bboxes: List[List[float]] = [
        _draw_box_object(
            draw,
            _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height)),
            camera=camera,
            frame=frame,
            fill=body,
        )
    ]

    sign = _project_face(
        [
            (left_x, face_y, base_z + height * 0.86),
            (right_x, face_y, base_z + height * 0.86),
            (right_x, face_y, base_z + height * 0.96),
            (left_x, face_y, base_z + height * 0.96),
        ],
        camera,
        frame,
    )
    draw.polygon(sign, fill=(244, 205, 74))
    _draw_polyline(draw, sign, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(sign))

    glass_left = x - width * 0.34
    glass_right = x + width * 0.08
    glass_bottom = base_z + height * 0.30
    glass_top = base_z + height * 0.82
    glass_face = _project_face(
        [
            (glass_left, face_y, glass_bottom),
            (glass_right, face_y, glass_bottom),
            (glass_right, face_y, glass_top),
            (glass_left, face_y, glass_top),
        ],
        camera,
        frame,
    )
    draw.polygon(glass_face, fill=(53, 89, 112))
    _draw_polyline(draw, glass_face, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(glass_face))

    snack_colors = [(232, 88, 67), (88, 166, 105), (247, 192, 78), (105, 145, 214)]
    for row, z_frac in enumerate((0.38, 0.50, 0.62, 0.74)):
        row_z = base_z + height * z_frac
        shelf = [
            _project_xy((glass_left + width * 0.04, face_y, row_z), camera, frame),
            _project_xy((glass_right - width * 0.04, face_y, row_z), camera, frame),
        ]
        _draw_line(draw, shelf[0], shelf[1], fill=(172, 196, 204), width=2)
        for col, x_frac in enumerate((-0.25, -0.11, 0.03)):
            item_center = _project_xy((x + width * x_frac, face_y, row_z + height * 0.035), camera, frame)
            item_w = max(5.0, abs(_project_xy((x + width * (x_frac + 0.035), face_y, row_z), camera, frame)[0] - item_center[0]))
            item_h = max(7.0, item_w * 1.25)
            packet = [
                item_center[0] - item_w,
                item_center[1] - item_h,
                item_center[0] + item_w,
                item_center[1] + item_h,
            ]
            color_index = int(row + col)
            while color_index >= len(snack_colors):
                color_index -= len(snack_colors)
            draw.rounded_rectangle(packet, radius=max(1, int(item_w * 0.25)), fill=snack_colors[color_index], outline=outline, width=1)

    panel_left = x + width * 0.16
    panel_right = x + width * 0.34
    keypad_face = _project_face(
        [
            (panel_left, face_y, base_z + height * 0.42),
            (panel_right, face_y, base_z + height * 0.42),
            (panel_right, face_y, base_z + height * 0.76),
            (panel_left, face_y, base_z + height * 0.76),
        ],
        camera,
        frame,
    )
    draw.polygon(keypad_face, fill=(231, 236, 240))
    _draw_polyline(draw, keypad_face, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(keypad_face))
    for row in range(3):
        for col in range(2):
            button = _project_xy(
                (panel_left + width * (0.045 + 0.075 * col), face_y, base_z + height * (0.50 + 0.07 * row)),
                camera,
                frame,
            )
            r = max(2.0, float(height) * 1.7)
            draw.ellipse((button[0] - r, button[1] - r, button[0] + r, button[1] + r), fill=(68, 80, 91), outline=outline, width=1)
    coin_slot = [
        _project_xy((panel_left + width * 0.03, face_y, base_z + height * 0.68), camera, frame),
        _project_xy((panel_right - width * 0.03, face_y, base_z + height * 0.68), camera, frame),
    ]
    _draw_line(draw, coin_slot[0], coin_slot[1], fill=(42, 51, 61), width=4)

    dispenser = _project_face(
        [
            (x - width * 0.20, face_y, base_z + height * 0.10),
            (x + width * 0.28, face_y, base_z + height * 0.10),
            (x + width * 0.28, face_y, base_z + height * 0.23),
            (x - width * 0.20, face_y, base_z + height * 0.23),
        ],
        camera,
        frame,
    )
    draw.polygon(dispenser, fill=(56, 64, 72))
    _draw_polyline(draw, dispenser, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(dispenser))
    return _bbox_union(*bboxes)


def _draw_trash_bin_object(
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
    body_rgb = _shade((74, 132, 103), 0.88 + (sum(fill) % 18) / 180.0)
    rim_rgb = _shade(body_rgb, 0.72)
    outline = (28, 39, 43)
    bboxes: List[List[float]] = [
        _draw_box_object(
            draw,
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, 0.0, 0.0),
                dimensions_xyz=(width * 0.92, depth * 0.88, height * 0.82),
            ),
            camera=camera,
            frame=frame,
            fill=body_rgb,
        )
    ]
    rim_h = height * 0.10
    rim_z = height * 0.78
    rail = min(width, depth) * 0.10
    rim_parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.41, rim_z), dimensions_xyz=(width, rail, rim_h)),
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.41, rim_z), dimensions_xyz=(width, rail, rim_h)),
        _sub_box_spec(spec, offset_xyz=(-width * 0.45, 0.0, rim_z), dimensions_xyz=(rail, depth, rim_h)),
        _sub_box_spec(spec, offset_xyz=(width * 0.45, 0.0, rim_z), dimensions_xyz=(rail, depth, rim_h)),
    ]
    bboxes.append(_draw_box_parts_object(draw, rim_parts, camera=camera, frame=frame, fill=rim_rgb))

    top_z = base_z + height * 0.83
    opening = _project_face(
        [
            (x - width * 0.34, y - depth * 0.31, top_z),
            (x + width * 0.34, y - depth * 0.31, top_z),
            (x + width * 0.34, y + depth * 0.31, top_z),
            (x - width * 0.34, y + depth * 0.31, top_z),
        ],
        camera,
        frame,
    )
    draw.polygon(opening, fill=(34, 42, 45))
    _draw_polyline(draw, opening, fill=outline, width=2)
    bboxes.append(_bbox_from_screen_points(opening))

    trash_items = [
        (-0.18, -0.08, 0.89, (229, 222, 202), "paper"),
        (0.06, 0.02, 0.92, (216, 78, 70), "can"),
        (0.18, -0.10, 0.90, (198, 206, 216), "paper"),
        (-0.02, 0.16, 0.91, (72, 133, 188), "can"),
        (-0.24, 0.12, 0.88, (238, 235, 218), "paper"),
    ]
    for dx, dy, z_frac, color, kind in trash_items:
        center = _project_xy((x + width * dx, y + depth * dy, base_z + height * z_frac), camera, frame)
        scale = max(5.0, min(16.0, _radius_px_for_object(spec, camera, frame) * 0.18))
        if kind == "can":
            bbox = [center[0] - scale * 0.72, center[1] - scale, center[0] + scale * 0.72, center[1] + scale]
            draw.rounded_rectangle(bbox, radius=max(1, int(scale * 0.28)), fill=color, outline=outline, width=1)
            draw.line((bbox[0], center[1] - scale * 0.35, bbox[2], center[1] - scale * 0.35), fill=_tint(color, 0.25), width=1)
        else:
            points = [
                (center[0] - scale * 0.95, center[1] - scale * 0.28),
                (center[0] - scale * 0.25, center[1] - scale * 0.86),
                (center[0] + scale * 0.82, center[1] - scale * 0.36),
                (center[0] + scale * 0.46, center[1] + scale * 0.62),
                (center[0] - scale * 0.62, center[1] + scale * 0.54),
            ]
            draw.polygon(points, fill=color, outline=outline)
            bbox = _bbox_from_screen_points(points)
        bboxes.append(list(bbox))
    return _bbox_union(*bboxes)


__all__ = [
    "_draw_refrigerator_object",
    "_draw_washing_machine_object",
    "_draw_vending_machine_object",
    "_draw_trash_bin_object",
]
