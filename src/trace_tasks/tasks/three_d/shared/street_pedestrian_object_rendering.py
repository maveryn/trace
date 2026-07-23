"""Pedestrian rendering helpers for street-intersection scenes."""

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

def _draw_pedestrian_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    x, y, _base_z = (float(value) for value in spec["base_xyz"])
    base = _project_xy((x, y, 0.0), camera, frame)
    top = _project_xy((x, y, height), camera, frame)
    up = (float(top[0]) - float(base[0]), float(top[1]) - float(base[1]))
    height_px = math.hypot(up[0], up[1])
    if height_px < 1.0:
        up_unit = (0.0, -1.0)
    else:
        up_unit = (up[0] / height_px, up[1] / height_px)
    side_unit = (-up_unit[1], up_unit[0])
    person_h = max(38.0, min(70.0, float(height_px) * 0.96))
    base_center = (float(base[0]), float(base[1]))

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base_center[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base_center[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    def line_bbox(points: Sequence[Sequence[float]], *, fill_rgb: Tuple[int, int, int], width_px: int, outline_rgb: Tuple[int, int, int] = (28, 35, 45)) -> List[float]:
        screen_points = [(float(px), float(py)) for px, py in points]
        draw.line(screen_points, fill=outline_rgb, width=max(3, int(width_px) + 3), joint="curve")
        draw.line(screen_points, fill=fill_rgb, width=max(2, int(width_px)), joint="curve")
        radius = max(2.0, float(width_px) * 0.52)
        for px, py in points:
            draw.ellipse(
                (float(px) - radius - 1.0, float(py) - radius - 1.0, float(px) + radius + 1.0, float(py) + radius + 1.0),
                fill=outline_rgb,
            )
            draw.ellipse((float(px) - radius, float(py) - radius, float(px) + radius, float(py) + radius), fill=fill_rgb)
        return [
            round(float(min(px for px, _py in points) - radius), 3),
            round(float(min(py for _px, py in points) - radius), 3),
            round(float(max(px for px, _py in points) + radius), 3),
            round(float(max(py for _px, py in points) + radius), 3),
        ]

    def poly_bbox(points: Sequence[Sequence[float]], *, fill_rgb: Tuple[int, int, int], outline_rgb: Tuple[int, int, int], width_px: int = 2) -> List[float]:
        screen_points = [(float(px), float(py)) for px, py in points]
        draw.polygon(screen_points, fill=fill_rgb)
        for index in range(len(screen_points)):
            next_index = index + 1
            if next_index >= len(screen_points):
                next_index = 0
            _draw_line(draw, screen_points[index], screen_points[next_index], fill=outline_rgb, width=max(1, int(width_px)))
        return [
            round(float(min(px for px, _py in screen_points)), 3),
            round(float(min(py for _px, py in screen_points)), 3),
            round(float(max(px for px, _py in screen_points)), 3),
            round(float(max(py for _px, py in screen_points)), 3),
        ]

    def ellipse_bbox(center: Sequence[float], radius_x: float, radius_y: float, *, fill_rgb: Tuple[int, int, int], outline_rgb: Tuple[int, int, int], width_px: int = 2) -> List[float]:
        bbox = [
            float(center[0]) - float(radius_x),
            float(center[1]) - float(radius_y),
            float(center[0]) + float(radius_x),
            float(center[1]) + float(radius_y),
        ]
        draw.ellipse(tuple(bbox), fill=fill_rgb, outline=outline_rgb, width=max(1, int(width_px)))
        return [round(float(value), 3) for value in bbox]

    limb_width = max(4, min(9, int(round(person_h * 0.085))))
    arm_width = max(3, int(round(limb_width * 0.82)))
    shoe_width = max(4, int(round(limb_width * 1.10)))
    object_type = str(spec.get("object_type", "pedestrian"))
    skin_palette = ((224, 178, 137), (188, 132, 88), (235, 198, 162), (128, 82, 58))
    hair_palette = ((55, 45, 37), (42, 34, 28), (83, 58, 39), (37, 38, 42))
    skin = skin_palette[_stable_palette_index(f"{spec.get('object_id', '')}.skin", len(skin_palette))]
    pants = (47, 62, 96)
    hair = hair_palette[_stable_palette_index(f"{spec.get('object_id', '')}.hair", len(hair_palette))]
    outline = (28, 35, 45)
    gender_id = str(spec.get("pedestrian_gender_id", "male")).lower()
    if object_type == "female_pedestrian":
        gender_id = "female"
    elif object_type == "male_pedestrian":
        gender_id = "male"
    if gender_id not in {"male", "female"}:
        gender_id = "male"
    shirt = _tint(fill, 0.08)
    skirt_rgb = (145, 82, 138)
    if gender_id == "female":
        shirt = (91, 141, 166)
        skirt_rgb = (149, 76, 134)
    else:
        pants = (45, 58, 84)
    bboxes: List[List[float]] = []

    left_foot = p(-person_h * 0.15, person_h * 0.02)
    right_foot = p(person_h * 0.15, person_h * 0.02)
    left_knee = p(-person_h * 0.08, person_h * 0.18)
    right_knee = p(person_h * 0.08, person_h * 0.18)
    if gender_id == "female":
        left_leg_top = p(-person_h * 0.08, person_h * 0.30)
        right_leg_top = p(person_h * 0.08, person_h * 0.30)
    else:
        left_leg_top = p(-person_h * 0.07, person_h * 0.38)
        right_leg_top = p(person_h * 0.07, person_h * 0.38)
    bboxes.append(line_bbox([left_leg_top, left_knee, left_foot], fill_rgb=pants, width_px=limb_width, outline_rgb=outline))
    bboxes.append(line_bbox([right_leg_top, right_knee, right_foot], fill_rgb=pants, width_px=limb_width, outline_rgb=outline))
    bboxes.append(line_bbox([left_foot, p(-person_h * 0.24, person_h * 0.02)], fill_rgb=(31, 35, 40), width_px=shoe_width, outline_rgb=outline))
    bboxes.append(line_bbox([right_foot, p(person_h * 0.24, person_h * 0.02)], fill_rgb=(31, 35, 40), width_px=shoe_width, outline_rgb=outline))

    left_shoulder = p(-person_h * 0.17, person_h * 0.62)
    right_shoulder = p(person_h * 0.17, person_h * 0.62)
    left_hand = p(-person_h * 0.33, person_h * 0.42)
    right_hand = p(person_h * 0.33, person_h * 0.42)
    bboxes.append(line_bbox([left_shoulder, p(-person_h * 0.26, person_h * 0.50), left_hand], fill_rgb=_shade(shirt, 0.88), width_px=arm_width, outline_rgb=outline))
    bboxes.append(line_bbox([right_shoulder, p(person_h * 0.26, person_h * 0.50), right_hand], fill_rgb=_shade(shirt, 0.88), width_px=arm_width, outline_rgb=outline))
    hand_radius = max(2.2, float(arm_width) * 0.72)
    for hand in (left_hand, right_hand):
        bboxes.append(ellipse_bbox(hand, hand_radius, hand_radius, fill_rgb=skin, outline_rgb=outline, width_px=1))

    if gender_id == "female":
        torso_points = [
            left_shoulder,
            right_shoulder,
            p(person_h * 0.12, person_h * 0.46),
            p(-person_h * 0.12, person_h * 0.46),
        ]
        bboxes.append(poly_bbox(torso_points, fill_rgb=shirt, outline_rgb=outline, width_px=2))
        skirt_points = [
            p(-person_h * 0.12, person_h * 0.46),
            p(person_h * 0.12, person_h * 0.46),
            p(person_h * 0.23, person_h * 0.30),
            p(-person_h * 0.23, person_h * 0.30),
        ]
        bboxes.append(poly_bbox(skirt_points, fill_rgb=skirt_rgb, outline_rgb=outline, width_px=2))
    else:
        torso_points = [
            left_shoulder,
            right_shoulder,
            p(person_h * 0.14, person_h * 0.38),
            p(-person_h * 0.14, person_h * 0.38),
        ]
        bboxes.append(poly_bbox(torso_points, fill_rgb=shirt, outline_rgb=outline, width_px=2))
        belt = [p(-person_h * 0.12, person_h * 0.39), p(person_h * 0.12, person_h * 0.39)]
        bboxes.append(line_bbox(belt, fill_rgb=(39, 43, 51), width_px=max(2, arm_width - 1), outline_rgb=outline))

    neck_bottom = p(0.0, person_h * 0.62)
    neck_top = p(0.0, person_h * 0.68)
    bboxes.append(line_bbox([neck_bottom, neck_top], fill_rgb=skin, width_px=max(3, arm_width), outline_rgb=outline))

    head_center = p(0.0, person_h * 0.80)
    head_radius = max(8.0, min(14.0, person_h * 0.13))
    head_bbox = [
        head_center[0] - head_radius,
        head_center[1] - head_radius,
        head_center[0] + head_radius,
        head_center[1] + head_radius,
    ]
    hx0, hy0, hx1, hy1 = (float(value) for value in head_bbox)
    head_w = hx1 - hx0
    head_h = hy1 - hy0
    if gender_id == "female":
        hair_back = [
            hx0 - head_w * 0.18,
            hy0 - head_h * 0.20,
            hx1 + head_w * 0.18,
            hy1 + head_h * 0.56,
        ]
        draw.ellipse(tuple(hair_back), fill=hair, outline=outline, width=2)
        bboxes.append([round(float(value), 3) for value in hair_back])
        left_lock = [
            (hx0 + head_w * 0.10, hy0 + head_h * 0.18),
            (hx0 - head_w * 0.26, hy0 + head_h * 0.38),
            (hx0 - head_w * 0.10, hy1 + head_h * 0.70),
            (hx0 + head_w * 0.24, hy1 + head_h * 0.28),
        ]
        right_lock = [
            (hx1 - head_w * 0.10, hy0 + head_h * 0.18),
            (hx1 + head_w * 0.26, hy0 + head_h * 0.38),
            (hx1 + head_w * 0.10, hy1 + head_h * 0.70),
            (hx1 - head_w * 0.24, hy1 + head_h * 0.28),
        ]
        bboxes.append(poly_bbox(left_lock, fill_rgb=hair, outline_rgb=outline, width_px=1))
        bboxes.append(poly_bbox(right_lock, fill_rgb=hair, outline_rgb=outline, width_px=1))
    bboxes.append(ellipse_bbox(head_center, head_radius, head_radius, fill_rgb=skin, outline_rgb=outline, width_px=2))
    if gender_id == "female":
        cap = [
            hx0 - head_w * 0.10,
            hy0 - head_h * 0.24,
            hx1 + head_w * 0.10,
            hy0 + head_h * 0.44,
        ]
        draw.ellipse(tuple(cap), fill=hair, outline=outline, width=1)
        fringe = [
            (hx0 + head_w * 0.05, hy0 + head_h * 0.20),
            (hx0 + head_w * 0.42, hy0 - head_h * 0.08),
            (hx1 - head_w * 0.08, hy0 + head_h * 0.22),
            (hx0 + head_w * 0.36, hy0 + head_h * 0.42),
        ]
        bboxes.append(poly_bbox(fringe, fill_rgb=hair, outline_rgb=outline, width_px=1))
    else:
        cap = [
            hx0 + head_w * 0.04,
            hy0 - head_h * 0.10,
            hx1 - head_w * 0.04,
            hy0 + head_h * 0.30,
        ]
        draw.rounded_rectangle(tuple(cap), radius=max(2, int(round(head_radius * 0.20))), fill=hair, outline=outline, width=1)
        bboxes.append([round(float(value), 3) for value in cap])
        for sideburn in (
            (hx0 + head_w * 0.02, hy0 + head_h * 0.20, hx0 + head_w * 0.18, hy0 + head_h * 0.50),
            (hx1 - head_w * 0.18, hy0 + head_h * 0.20, hx1 - head_w * 0.02, hy0 + head_h * 0.50),
        ):
            draw.rounded_rectangle(tuple(sideburn), radius=2, fill=hair)
            bboxes.append([round(float(value), 3) for value in sideburn])
    eye_r = max(1.2, head_radius * 0.09)
    for eye in (p(-head_radius * 0.36, person_h * 0.805), p(head_radius * 0.36, person_h * 0.805)):
        bboxes.append(ellipse_bbox(eye, eye_r, eye_r, fill_rgb=(35, 40, 44), outline_rgb=(35, 40, 44), width_px=1))
    mouth = [p(-head_radius * 0.20, person_h * 0.772), p(head_radius * 0.22, person_h * 0.772)]
    bboxes.append(line_bbox(mouth, fill_rgb=(125, 67, 63), width_px=1, outline_rgb=(125, 67, 63)))
    return _bbox_union(*bboxes)


__all__ = [
    '_draw_pedestrian_object',
]
