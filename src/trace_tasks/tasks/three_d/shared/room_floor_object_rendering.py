"""Floor and furniture object renderers for shared room-wall 3D scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .object_scene import CONTEXT_OBJECT_COLORS
from .object_scene_primitives import (
    bbox_union,
    draw_box_object,
    draw_cone_object,
    draw_cylinder_object,
    draw_line,
    draw_pyramid_object,
    draw_sphere_object,
    shade_rgb,
    tint_rgb,
)
from .object_scene_rendering import draw_open_box_object, draw_table_object
from .projected_object_geometry import object_screen_bbox
from .room_wall_rendering_geometry import _draw_screen_scenery, _inset_bbox, _points_bbox, _project_points


def _wrap_color_index(index: int, color_count: int) -> int:
    wrapped = int(index)
    while wrapped < 0:
        wrapped += int(color_count)
    while wrapped >= int(color_count):
        wrapped -= int(color_count)
    return int(wrapped)


def _room_floor_base_z(spec: Mapping[str, Any]) -> float:
    raw_base = spec.get("base_xyz", (0.0, 0.0, 0.0))
    if isinstance(raw_base, Sequence) and len(raw_base) >= 3:
        return float(raw_base[2])
    world_z = float(spec["world_xyz"][2])
    height = float(spec["dimensions_xyz"][2])
    return float(world_z - height * 0.5)


def _room_distance_sq(point: Sequence[float], camera) -> float:
    camera_xyz = tuple(float(value) for value in camera.camera_position)
    return sum((float(point[index]) - camera_xyz[index]) ** 2 for index in range(3))


def _room_sub_box_spec(
    spec: Mapping[str, Any],
    *,
    offset_xyz: Tuple[float, float, float],
    dimensions_xyz: Tuple[float, float, float],
) -> Dict[str, Any]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    base_z = _room_floor_base_z(spec)
    width, depth, height = (float(value) for value in dimensions_xyz)
    center_x = float(x + float(offset_xyz[0]))
    center_y = float(y + float(offset_xyz[1]))
    part_base_z = float(base_z + float(offset_xyz[2]))
    return {
        **dict(spec),
        "shape_type": "rectangular_prism",
        "world_xyz": [round(center_x, 4), round(center_y, 4), round(part_base_z + height * 0.5, 4)],
        "base_xyz": [round(center_x, 4), round(center_y, 4), round(part_base_z, 4)],
        "dimensions_xyz": [round(width, 4), round(depth, 4), round(height, 4)],
    }


def _draw_room_box_parts(
    draw: ImageDraw.ImageDraw,
    parts: Sequence[Tuple[Mapping[str, Any], Tuple[int, int, int]]],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    for part, fill in sorted(parts, key=lambda item: _room_distance_sq(item[0]["world_xyz"], camera), reverse=True):
        bboxes.append(draw_box_object(draw, part, camera=camera, frame=frame, fill=fill))
    return bbox_union(*bboxes)


def _room_point_at_height(spec: Mapping[str, Any], height_frac: float, camera, frame) -> Tuple[float, float]:
    x, y, _z = (float(value) for value in spec["world_xyz"])
    height = float(spec["dimensions_xyz"][2])
    point = (float(x), float(y), float(_room_floor_base_z(spec) + height * float(height_frac)))
    return _project_points([point], camera, frame)[0]


def _draw_clock_face(draw: ImageDraw.ImageDraw, bbox: Sequence[float]) -> None:
    face = _inset_bbox(bbox, 0.05, 0.05)
    draw.ellipse(tuple(face), fill=(245, 239, 205), outline=(45, 54, 62), width=2)
    cx = (float(face[0]) + float(face[2])) * 0.5
    cy = (float(face[1]) + float(face[3])) * 0.5
    radius = min(float(face[2]) - float(face[0]), float(face[3]) - float(face[1])) * 0.30
    if radius <= 1.0:
        return
    for angle_index in range(12):
        angle = math.pi * 0.5 - angle_index * math.tau / 12.0
        tick_outer = (cx + math.cos(angle) * radius * 1.42, cy - math.sin(angle) * radius * 1.42)
        tick_inner = (
            cx + math.cos(angle) * radius * (1.10 if angle_index % 3 == 0 else 1.24),
            cy - math.sin(angle) * radius * (1.10 if angle_index % 3 == 0 else 1.24),
        )
        draw_line(draw, tick_inner, tick_outer, fill=(70, 77, 84), width=2 if angle_index % 3 == 0 else 1)
    draw_line(draw, (cx, cy), (cx, cy - radius), fill=(38, 45, 54), width=2)
    draw_line(draw, (cx, cy), (cx + radius * 0.78, cy + radius * 0.30), fill=(38, 45, 54), width=2)
    draw.ellipse((cx - 2.2, cy - 2.2, cx + 2.2, cy + 2.2), fill=(38, 45, 54))


def _draw_room_table_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    object_type = str(spec.get("object_type"))
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    top_h = max(0.06, height * 0.16)
    leg_h = max(0.05, height - top_h)
    leg_w = min(width, depth) * (0.12 if object_type != "desk" else 0.10)
    parts: List[Tuple[Mapping[str, Any], Tuple[int, int, int]]] = [
        (
            _room_sub_box_spec(spec, offset_xyz=(sx * width * 0.36, sy * depth * 0.34, 0.0), dimensions_xyz=(leg_w, leg_w, leg_h)),
            shade_rgb(fill, 0.78),
        )
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    parts.append(
        (
            _room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, leg_h), dimensions_xyz=(width, depth, top_h)),
            tint_rgb(fill, 0.18),
        )
    )
    if object_type == "desk":
        drawer_h = leg_h * 0.45
        parts.append(
            (
                _room_sub_box_spec(
                    spec,
                    offset_xyz=(width * 0.24, -depth * 0.31, leg_h * 0.23),
                    dimensions_xyz=(width * 0.36, depth * 0.16, drawer_h),
                ),
                shade_rgb(fill, 0.84),
            )
        )
    bbox = _draw_room_box_parts(draw, parts, camera=camera, frame=frame)
    if object_type == "desk":
        panel = _inset_bbox(bbox, 0.55, 0.42)
        panel[0] = float(bbox[0]) + (float(bbox[2]) - float(bbox[0])) * 0.54
        panel[2] = float(bbox[0]) + (float(bbox[2]) - float(bbox[0])) * 0.86
        draw.rectangle(tuple(panel), fill=shade_rgb(fill, 0.72), outline=(62, 50, 40), width=1)
        for frac in (0.38, 0.62):
            y = float(panel[1]) + (float(panel[3]) - float(panel[1])) * frac
            draw_line(draw, (float(panel[0]) + 2.0, y), (float(panel[2]) - 2.0, y), fill=(62, 50, 40), width=1)
    return list(bbox)


def _draw_room_sofa_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (_room_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.08, 0.0), dimensions_xyz=(width * 0.82, depth * 0.66, height * 0.34)), shade_rgb(fill, 0.88)),
        (_room_sub_box_spec(spec, offset_xyz=(0.0, depth * 0.34, height * 0.25), dimensions_xyz=(width * 0.92, depth * 0.20, height * 0.70)), shade_rgb(fill, 0.72)),
        (_room_sub_box_spec(spec, offset_xyz=(-width * 0.47, -depth * 0.02, 0.0), dimensions_xyz=(width * 0.13, depth * 0.76, height * 0.58)), shade_rgb(fill, 0.76)),
        (_room_sub_box_spec(spec, offset_xyz=(width * 0.47, -depth * 0.02, 0.0), dimensions_xyz=(width * 0.13, depth * 0.76, height * 0.58)), shade_rgb(fill, 0.76)),
        (_room_sub_box_spec(spec, offset_xyz=(-width * 0.21, -depth * 0.13, height * 0.18), dimensions_xyz=(width * 0.34, depth * 0.50, height * 0.22)), tint_rgb(fill, 0.15)),
        (_room_sub_box_spec(spec, offset_xyz=(width * 0.21, -depth * 0.13, height * 0.18), dimensions_xyz=(width * 0.34, depth * 0.50, height * 0.22)), tint_rgb(fill, 0.09)),
    ]
    bbox = _draw_room_box_parts(draw, parts, camera=camera, frame=frame)
    seam_x = (float(bbox[0]) + float(bbox[2])) * 0.5
    draw_line(draw, (seam_x, float(bbox[1]) + (float(bbox[3]) - float(bbox[1])) * 0.38), (seam_x, float(bbox[3]) - 3.0), fill=(55, 69, 88), width=1)
    return list(bbox)


def _draw_room_armchair_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (_room_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.08, 0.0), dimensions_xyz=(width * 0.70, depth * 0.62, height * 0.34)), shade_rgb(fill, 0.88)),
        (_room_sub_box_spec(spec, offset_xyz=(0.0, depth * 0.34, height * 0.25), dimensions_xyz=(width * 0.74, depth * 0.20, height * 0.68)), shade_rgb(fill, 0.72)),
        (_room_sub_box_spec(spec, offset_xyz=(-width * 0.42, -depth * 0.04, 0.0), dimensions_xyz=(width * 0.16, depth * 0.70, height * 0.55)), shade_rgb(fill, 0.76)),
        (_room_sub_box_spec(spec, offset_xyz=(width * 0.42, -depth * 0.04, 0.0), dimensions_xyz=(width * 0.16, depth * 0.70, height * 0.55)), shade_rgb(fill, 0.76)),
        (_room_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.14, height * 0.18), dimensions_xyz=(width * 0.52, depth * 0.46, height * 0.20)), tint_rgb(fill, 0.14)),
    ]
    return _draw_room_box_parts(draw, parts, camera=camera, frame=frame)


def _draw_room_bed_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (_room_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.06, 0.0), dimensions_xyz=(width * 0.96, depth * 0.86, height * 0.20)), shade_rgb(fill, 0.70)),
        (_room_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.08, height * 0.20), dimensions_xyz=(width * 0.92, depth * 0.78, height * 0.16)), tint_rgb(fill, 0.35)),
        (_room_sub_box_spec(spec, offset_xyz=(0.0, depth * 0.46, height * 0.05), dimensions_xyz=(width, depth * 0.12, height * 0.90)), shade_rgb(fill, 0.66)),
        (_room_sub_box_spec(spec, offset_xyz=(-width * 0.42, -depth * 0.36, 0.0), dimensions_xyz=(width * 0.06, depth * 0.06, height * 0.18)), shade_rgb(fill, 0.48)),
        (_room_sub_box_spec(spec, offset_xyz=(width * 0.42, -depth * 0.36, 0.0), dimensions_xyz=(width * 0.06, depth * 0.06, height * 0.18)), shade_rgb(fill, 0.48)),
    ]
    bbox = _draw_room_box_parts(draw, parts, camera=camera, frame=frame)
    width_px = float(bbox[2]) - float(bbox[0])
    height_px = float(bbox[3]) - float(bbox[1])
    pillow_left = [float(bbox[0]) + width_px * 0.14, float(bbox[1]) + height_px * 0.20, float(bbox[0]) + width_px * 0.38, float(bbox[1]) + height_px * 0.37]
    pillow_right = [float(bbox[0]) + width_px * 0.41, float(bbox[1]) + height_px * 0.20, float(bbox[0]) + width_px * 0.65, float(bbox[1]) + height_px * 0.37]
    blanket = [float(bbox[0]) + width_px * 0.10, float(bbox[1]) + height_px * 0.47, float(bbox[2]) - width_px * 0.08, float(bbox[3]) - height_px * 0.10]
    foot_fold = [blanket[0], blanket[3] - height_px * 0.12, blanket[2], blanket[3] - height_px * 0.05]
    for pillow in (pillow_left, pillow_right):
        draw.rounded_rectangle(tuple(pillow), radius=max(4, int(min(width_px, height_px) * 0.05)), fill=(238, 234, 222), outline=(116, 107, 136), width=1)
    draw.rounded_rectangle(tuple(blanket), radius=max(5, int(min(width_px, height_px) * 0.05)), fill=(89, 126, 170), outline=(62, 80, 106), width=1)
    draw.rounded_rectangle(tuple(foot_fold), radius=3, fill=(73, 104, 144), outline=(62, 80, 106), width=1)
    return list(bbox_union(bbox, pillow_left, pillow_right, blanket, foot_fold))


def _draw_room_media_console_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (_room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.08), dimensions_xyz=(width, depth * 0.92, height * 0.78)), fill),
        (_room_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.01, height * 0.84), dimensions_xyz=(width * 1.04, depth, height * 0.10)), tint_rgb(fill, 0.12)),
        (_room_sub_box_spec(spec, offset_xyz=(-width * 0.40, -depth * 0.34, 0.0), dimensions_xyz=(width * 0.07, depth * 0.08, height * 0.16)), shade_rgb(fill, 0.48)),
        (_room_sub_box_spec(spec, offset_xyz=(width * 0.40, -depth * 0.34, 0.0), dimensions_xyz=(width * 0.07, depth * 0.08, height * 0.16)), shade_rgb(fill, 0.48)),
    ]
    bbox = _draw_room_box_parts(draw, parts, camera=camera, frame=frame)
    front = _inset_bbox(bbox, 0.08, 0.18)
    panel_w = (float(front[2]) - float(front[0])) / 3.0
    panel_boxes: List[List[float]] = []
    for index in range(3):
        x1 = float(front[0]) + panel_w * index + 1.0
        x2 = float(front[0]) + panel_w * (index + 1) - 1.0
        panel = [x1, float(front[1]), x2, float(front[3])]
        draw.rectangle(tuple(panel), fill=shade_rgb(fill, 0.72 + index * 0.04), outline=(58, 48, 40), width=1)
        shelf_y = float(panel[1]) + (float(panel[3]) - float(panel[1])) * 0.45
        draw_line(draw, (x1 + 2.0, shelf_y), (x2 - 2.0, shelf_y), fill=(58, 48, 40), width=1)
        handle_y = float(panel[1]) + (float(panel[3]) - float(panel[1])) * 0.68
        draw_line(draw, (x1 + panel_w * 0.28, handle_y), (x2 - panel_w * 0.28, handle_y), fill=(215, 181, 95), width=2)
        panel_boxes.append(panel)
    center_opening = [float(front[0]) + panel_w + 3.0, float(front[1]) + 3.0, float(front[0]) + panel_w * 2.0 - 3.0, float(front[1]) + (float(front[3]) - float(front[1])) * 0.42]
    draw.rectangle(tuple(center_opening), fill=(37, 42, 49), outline=(58, 48, 40), width=1)
    return list(bbox_union(bbox, front, center_opening, *panel_boxes))


def _draw_room_tv_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    base_h = height * 0.08
    stand_h = height * 0.16
    panel_h = height - base_h - stand_h * 0.35
    panel_z = height - panel_h
    base = _room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.42, depth * 0.85, base_h))
    stand = _room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, base_h), dimensions_xyz=(width * 0.10, depth * 0.48, stand_h))
    panel = _room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, panel_z), dimensions_xyz=(width, depth, panel_h))
    bboxes = [
        draw_box_object(draw, base, camera=camera, frame=frame, fill=(45, 48, 54)),
        draw_box_object(draw, stand, camera=camera, frame=frame, fill=(35, 38, 44)),
        draw_box_object(draw, panel, camera=camera, frame=frame, fill=fill),
    ]
    panel_bbox = object_screen_bbox(panel, camera, frame, pad_px=0.0)
    screen = _inset_bbox(panel_bbox, 0.12, 0.14)
    draw.rectangle(tuple(screen), fill=(16, 24, 34), outline=(74, 88, 103), width=2)
    shine = [float(screen[0]) + 3.0, float(screen[1]) + 3.0, float(screen[2]) - 3.0, float(screen[1]) + max(5.0, (float(screen[3]) - float(screen[1])) * 0.22)]
    draw.rectangle(tuple(shine), fill=(31, 52, 68))
    return list(bbox_union(*bboxes, screen))


def _draw_room_clock_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    height = float(spec["dimensions_xyz"][2])
    if str(spec.get("mounting")) != "on_furniture" and height >= 0.75:
        bbox = draw_box_object(draw, spec, camera=camera, frame=frame, fill=(138, 94, 56))
        width_px = float(bbox[2]) - float(bbox[0])
        height_px = float(bbox[3]) - float(bbox[1])
        face = [
            float(bbox[0]) + width_px * 0.20,
            float(bbox[1]) + height_px * 0.09,
            float(bbox[2]) - width_px * 0.20,
            float(bbox[1]) + height_px * 0.43,
        ]
        _draw_clock_face(draw, face)
        window = [
            float(bbox[0]) + width_px * 0.28,
            float(bbox[1]) + height_px * 0.50,
            float(bbox[2]) - width_px * 0.28,
            float(bbox[3]) - height_px * 0.12,
        ]
        draw.rectangle(tuple(window), fill=(94, 64, 42), outline=(52, 36, 27), width=1)
        cx = (float(window[0]) + float(window[2])) * 0.5
        draw_line(draw, (cx, float(window[1]) + 2.0), (cx, float(window[3]) - 6.0), fill=(218, 176, 82), width=2)
        draw.ellipse((cx - 4.0, float(window[3]) - 10.0, cx + 4.0, float(window[3]) - 2.0), fill=(218, 176, 82), outline=(64, 48, 25), width=1)
        return list(bbox_union(bbox, face, window))
    bbox = draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    _draw_clock_face(draw, _inset_bbox(bbox, 0.18, 0.14))
    return list(bbox)


def _draw_room_picture_frame_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    mat = _inset_bbox(bbox, 0.10, 0.10)
    draw.rectangle(tuple(mat), fill=(133, 86, 61), outline=(72, 46, 34), width=2)
    inner = _inset_bbox(bbox, 0.18, 0.17)
    _draw_screen_scenery(draw, inner, variant=str(spec.get("scenery_variant", "mountains")), outline=(72, 46, 34))
    frame_lip = _inset_bbox(bbox, 0.045, 0.05)
    draw.rectangle(tuple(frame_lip), outline=(82, 51, 33), width=3)
    shine = [
        (float(inner[0]) + (float(inner[2]) - float(inner[0])) * 0.12, float(inner[1]) + (float(inner[3]) - float(inner[1])) * 0.16),
        (float(inner[0]) + (float(inner[2]) - float(inner[0])) * 0.60, float(inner[1]) + (float(inner[3]) - float(inner[1])) * 0.06),
    ]
    draw_line(draw, shine[0], shine[1], fill=(236, 244, 242), width=2)
    extra = [frame_lip, _points_bbox(shine)]
    if str(spec.get("mounting")) != "on_furniture":
        foot_y = float(bbox[3]) - 2.0
        stand_a = ((float(bbox[0]) + float(bbox[2])) * 0.5, foot_y)
        stand_b = (float(bbox[2]) - 4.0, foot_y + 5.0)
        stand_c = (float(bbox[0]) + 4.0, foot_y + 4.0)
        draw_line(draw, stand_a, stand_b, fill=(72, 46, 34), width=2)
        draw_line(draw, stand_a, stand_c, fill=(72, 46, 34), width=2)
        extra.append(_points_bbox([stand_a, stand_b, stand_c]))
    return list(bbox_union(bbox, mat, inner, *extra))


def _draw_room_mirror_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    inner = _inset_bbox(bbox, 0.16, 0.12)
    draw.rectangle(tuple(inner), fill=(197, 226, 232), outline=(70, 91, 100), width=2)
    draw_line(draw, (inner[0], inner[1]), (inner[2], inner[3]), fill=(229, 244, 247), width=2)
    draw_line(draw, (float(inner[0]) + 5.0, float(inner[1]) + 10.0), (float(inner[2]) - 8.0, float(inner[1]) + 25.0), fill=(237, 250, 252), width=2)
    return list(bbox_union(bbox, inner))


def _draw_room_standing_shelf_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    post_w = width * 0.10
    shelf_h = height * 0.07
    parts: List[Tuple[Mapping[str, Any], Tuple[int, int, int]]] = [
        (_room_sub_box_spec(spec, offset_xyz=(-width * 0.44, 0.0, 0.0), dimensions_xyz=(post_w, depth, height)), shade_rgb(fill, 0.78)),
        (_room_sub_box_spec(spec, offset_xyz=(width * 0.44, 0.0, 0.0), dimensions_xyz=(post_w, depth, height)), shade_rgb(fill, 0.78)),
    ]
    for z_frac in (0.04, 0.34, 0.64, 0.92):
        parts.append(
            (
                _room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * z_frac), dimensions_xyz=(width, depth, shelf_h)),
                tint_rgb(fill, 0.08),
            )
        )
    book_colors = [(180, 72, 76), (79, 127, 168), (229, 177, 78), (88, 145, 94), (128, 93, 150)]
    for shelf_index, z_frac in enumerate((0.13, 0.43, 0.73)):
        for book_index in range(4):
            book_w = width * 0.075
            x_offset = -width * 0.26 + book_index * width * 0.15
            book_h = height * (0.13 + 0.015 * ((book_index + shelf_index) % 2))
            parts.append(
                (
                    _room_sub_box_spec(
                        spec,
                        offset_xyz=(x_offset, -depth * 0.18, height * z_frac),
                        dimensions_xyz=(book_w, depth * 0.20, book_h),
                    ),
                    book_colors[_wrap_color_index(book_index + shelf_index, len(book_colors))],
                )
            )
    return _draw_room_box_parts(draw, parts, camera=camera, frame=frame)


def _draw_room_floor_lamp_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = object_screen_bbox(spec, camera, frame, pad_px=2.0)
    width_px = float(bbox[2]) - float(bbox[0])
    height_px = float(bbox[3]) - float(bbox[1])
    base_center = _room_point_at_height(spec, 0.02, camera, frame)
    shade_center = _room_point_at_height(spec, 0.82, camera, frame)
    top_center = _room_point_at_height(spec, 0.98, camera, frame)
    pole_top = (float(shade_center[0]), float(shade_center[1]) + height_px * 0.06)
    pole_bottom = (float(base_center[0]), float(base_center[1]) - height_px * 0.05)
    draw_line(draw, pole_bottom, pole_top, fill=(91, 70, 44), width=max(2, int(width_px * 0.06)))
    base = [
        float(base_center[0]) - width_px * 0.25,
        float(base_center[1]) - height_px * 0.035,
        float(base_center[0]) + width_px * 0.25,
        float(base_center[1]) + height_px * 0.035,
    ]
    draw.ellipse(tuple(base), fill=(128, 91, 52), outline=(60, 45, 29), width=2)
    shade = [
        (float(top_center[0]) - width_px * 0.23, float(top_center[1]) + height_px * 0.02),
        (float(top_center[0]) + width_px * 0.23, float(top_center[1]) + height_px * 0.02),
        (float(shade_center[0]) + width_px * 0.36, float(shade_center[1]) + height_px * 0.15),
        (float(shade_center[0]) - width_px * 0.36, float(shade_center[1]) + height_px * 0.15),
    ]
    draw.polygon(shade, fill=fill, outline=(88, 62, 35))
    lip_y = float(shade_center[1]) + height_px * 0.15
    draw.ellipse((shade[3][0], lip_y - height_px * 0.035, shade[2][0], lip_y + height_px * 0.035), fill=shade_rgb(fill, 0.78), outline=(88, 62, 35), width=2)
    return list(bbox_union(bbox, base, _points_bbox(shade)))


def _draw_room_plant_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    pot = _room_sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.55, depth * 0.55, height * 0.32))
    pot_bbox = draw_cylinder_object(draw, pot, camera=camera, frame=frame, fill=(154, 98, 64))
    bbox = object_screen_bbox(spec, camera, frame, pad_px=2.0)
    width_px = float(bbox[2]) - float(bbox[0])
    height_px = float(bbox[3]) - float(bbox[1])
    center = _room_point_at_height(spec, 0.66, camera, frame)
    leaf_bboxes: List[List[float]] = []
    for index, (dx, dy, sx, sy, color) in enumerate(
        (
            (-0.22, -0.10, 0.28, 0.26, (61, 128, 82)),
            (0.20, -0.12, 0.30, 0.24, (76, 151, 89)),
            (-0.04, -0.28, 0.26, 0.30, (51, 116, 77)),
            (0.02, 0.08, 0.36, 0.26, (84, 158, 96)),
            (-0.30, 0.08, 0.24, 0.22, (63, 137, 86)),
        )
    ):
        cx = float(center[0]) + width_px * dx
        cy = float(center[1]) + height_px * dy
        rx = max(5.0, width_px * sx)
        ry = max(5.0, height_px * sy)
        leaf = [cx - rx, cy - ry, cx + rx, cy + ry]
        draw.ellipse(tuple(leaf), fill=color, outline=(37, 88, 57), width=1)
        draw_line(draw, (cx, cy + ry * 0.78), (cx, cy - ry * 0.70), fill=(37, 88, 57), width=1)
        leaf_bboxes.append(leaf)
    return list(bbox_union(pot_bbox, bbox, *leaf_bboxes))


def _draw_room_closed_box_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    width_px = float(bbox[2]) - float(bbox[0])
    height_px = float(bbox[3]) - float(bbox[1])
    tape = (219, 189, 122)
    draw_line(draw, (float(bbox[0]) + width_px * 0.50, float(bbox[1]) + height_px * 0.10), (float(bbox[0]) + width_px * 0.50, float(bbox[3]) - height_px * 0.08), fill=tape, width=3)
    draw_line(draw, (float(bbox[0]) + width_px * 0.15, float(bbox[1]) + height_px * 0.30), (float(bbox[2]) - width_px * 0.15, float(bbox[1]) + height_px * 0.30), fill=tape, width=3)
    return list(bbox)


def _draw_room_open_box_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = draw_open_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    inner = _inset_bbox(bbox, 0.22, 0.26)
    inner[3] = float(inner[1]) + max(5.0, (float(bbox[3]) - float(bbox[1])) * 0.22)
    draw.polygon(
        [
            (float(inner[0]), float(inner[3])),
            ((float(inner[0]) + float(inner[2])) * 0.5, float(inner[1])),
            (float(inner[2]), float(inner[3])),
            ((float(inner[0]) + float(inner[2])) * 0.5, float(inner[3]) + 5.0),
        ],
        fill=(93, 65, 42),
        outline=(56, 41, 30),
    )
    return list(bbox_union(bbox, inner))


def _draw_room_floor_fan_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = object_screen_bbox(spec, camera, frame, pad_px=2.0)
    width_px = float(bbox[2]) - float(bbox[0])
    height_px = float(bbox[3]) - float(bbox[1])
    face = [
        float(bbox[0]) + width_px * 0.12,
        float(bbox[1]) + height_px * 0.06,
        float(bbox[2]) - width_px * 0.12,
        float(bbox[1]) + height_px * 0.58,
    ]
    draw.ellipse(tuple(face), fill=(214, 222, 224), outline=(45, 56, 66), width=2)
    cx = (float(face[0]) + float(face[2])) * 0.5
    cy = (float(face[1]) + float(face[3])) * 0.5
    radius = min(float(face[2]) - float(face[0]), float(face[3]) - float(face[1])) * 0.34
    for angle in (math.pi * 0.5, math.pi * 1.17, math.pi * 1.83):
        tip = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
        draw_line(draw, (cx, cy), tip, fill=(96, 113, 124), width=2)
    for angle in (0.0, math.pi * 0.25, math.pi * 0.50, math.pi * 0.75, math.pi, math.pi * 1.25, math.pi * 1.50, math.pi * 1.75):
        tip = (cx + math.cos(angle) * radius * 1.22, cy + math.sin(angle) * radius * 1.22)
        draw_line(draw, (cx, cy), tip, fill=(123, 137, 146), width=1)
    draw.ellipse((cx - 3.0, cy - 3.0, cx + 3.0, cy + 3.0), fill=(48, 58, 68))
    pole_bottom = (cx, float(bbox[3]) - height_px * 0.12)
    draw_line(draw, (cx, float(face[3])), pole_bottom, fill=(45, 56, 66), width=3)
    base = [cx - width_px * 0.24, float(bbox[3]) - height_px * 0.12, cx + width_px * 0.24, float(bbox[3]) - height_px * 0.03]
    draw.ellipse(tuple(base), fill=(106, 119, 127), outline=(45, 56, 66), width=2)
    return list(bbox_union(bbox, face, base))


def _draw_room_portable_ac_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    grille = _inset_bbox(bbox, 0.18, 0.18)
    grille[3] = grille[1] + max(10.0, (float(bbox[3]) - float(bbox[1])) * 0.24)
    draw.rectangle(tuple(grille), fill=(182, 197, 202), outline=(76, 91, 98), width=1)
    for index in range(4):
        y = float(grille[1]) + (index + 1) * (float(grille[3]) - float(grille[1])) / 4.0
        draw_line(draw, (float(grille[0]) + 2.0, y), (float(grille[2]) - 2.0, y), fill=(86, 101, 108), width=1)
    side_vent = [
        float(bbox[2]) - (float(bbox[2]) - float(bbox[0])) * 0.20,
        float(bbox[1]) + (float(bbox[3]) - float(bbox[1])) * 0.22,
        float(bbox[2]) - (float(bbox[2]) - float(bbox[0])) * 0.08,
        float(bbox[1]) + (float(bbox[3]) - float(bbox[1])) * 0.70,
    ]
    draw.rectangle(tuple(side_vent), fill=(119, 137, 145), outline=(60, 72, 78), width=1)
    outlet = _inset_bbox(bbox, 0.20, 0.22)
    outlet[1] = float(bbox[1]) + (float(bbox[3]) - float(bbox[1])) * 0.58
    outlet[3] = float(bbox[1]) + (float(bbox[3]) - float(bbox[1])) * 0.78
    draw.rectangle(tuple(outlet), fill=(233, 238, 238), outline=(76, 91, 98), width=1)
    wheel_y = float(bbox[3]) - 3.0
    draw_line(draw, (float(bbox[0]) + 5.0, wheel_y), (float(bbox[0]) + 14.0, wheel_y), fill=(58, 65, 70), width=2)
    draw_line(draw, (float(bbox[2]) - 14.0, wheel_y), (float(bbox[2]) - 5.0, wheel_y), fill=(58, 65, 70), width=2)
    return list(bbox_union(bbox, grille, side_vent, outlet))


def _draw_room_coat_stand_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = object_screen_bbox(spec, camera, frame, pad_px=2.0)
    stand_top = _room_point_at_height(spec, 0.98, camera, frame)
    stand_mid = _room_point_at_height(spec, 0.58, camera, frame)
    stand_base = _room_point_at_height(spec, 0.02, camera, frame)
    draw_line(draw, stand_base, stand_top, fill=(66, 47, 35), width=3)
    draw_line(draw, (stand_top[0] - 10.0, stand_top[1] + 5.0), (stand_top[0] + 10.0, stand_top[1] + 5.0), fill=(66, 47, 35), width=3)
    draw_line(draw, (stand_base[0] - 14.0, stand_base[1]), (stand_base[0] + 14.0, stand_base[1]), fill=(66, 47, 35), width=3)
    body = _inset_bbox(bbox, 0.14, 0.08)
    sleeve_y = body[1] + (body[3] - body[1]) * 0.30
    draw.polygon(
        [
            (body[0] + (body[2] - body[0]) * 0.15, body[1] + (body[3] - body[1]) * 0.12),
            (body[0] - (body[2] - body[0]) * 0.26, sleeve_y),
            (body[0] - (body[2] - body[0]) * 0.10, sleeve_y + (body[3] - body[1]) * 0.28),
            (body[0] + (body[2] - body[0]) * 0.28, body[1] + (body[3] - body[1]) * 0.40),
        ],
        fill=shade_rgb(fill, 0.84),
        outline=(53, 42, 68),
    )
    draw.polygon(
        [
            (body[2] - (body[2] - body[0]) * 0.15, body[1] + (body[3] - body[1]) * 0.12),
            (body[2] + (body[2] - body[0]) * 0.26, sleeve_y),
            (body[2] + (body[2] - body[0]) * 0.10, sleeve_y + (body[3] - body[1]) * 0.28),
            (body[2] - (body[2] - body[0]) * 0.28, body[1] + (body[3] - body[1]) * 0.40),
        ],
        fill=shade_rgb(fill, 0.92),
        outline=(53, 42, 68),
    )
    draw.polygon(
        [
            ((body[0] + body[2]) * 0.5, body[1]),
            (body[2], body[1] + (body[3] - body[1]) * 0.34),
            (body[2] - (body[2] - body[0]) * 0.12, body[3]),
            (body[0] + (body[2] - body[0]) * 0.12, body[3]),
            (body[0], body[1] + (body[3] - body[1]) * 0.34),
        ],
        fill=fill,
        outline=(53, 42, 68),
    )
    collar = [
        ((body[0] + body[2]) * 0.5, body[1] + (body[3] - body[1]) * 0.20),
        (body[0] + (body[2] - body[0]) * 0.36, body[1] + (body[3] - body[1]) * 0.08),
        (body[0] + (body[2] - body[0]) * 0.46, body[1] + (body[3] - body[1]) * 0.24),
        (body[2] - (body[2] - body[0]) * 0.46, body[1] + (body[3] - body[1]) * 0.24),
        (body[2] - (body[2] - body[0]) * 0.36, body[1] + (body[3] - body[1]) * 0.08),
    ]
    draw.polygon(collar, fill=(226, 218, 202), outline=(53, 42, 68))
    button_x = (float(body[0]) + float(body[2])) * 0.5
    for button_y in (body[1] + (body[3] - body[1]) * 0.44, body[1] + (body[3] - body[1]) * 0.64):
        draw.ellipse((button_x - 2.0, button_y - 2.0, button_x + 2.0, button_y + 2.0), fill=(226, 218, 202))
    return list(bbox_union(bbox, body, _points_bbox([stand_base, stand_mid, stand_top])))


def _draw_floor_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    color_role = str(spec.get("color_role", "furniture"))
    default_color_index = _wrap_color_index(
        sum(ord(char) for char in str(spec["object_id"])),
        len(CONTEXT_OBJECT_COLORS),
    )
    color = {
        "sofa": (163, 112, 92),
        "table": (142, 94, 55),
        "cabinet": (127, 102, 73),
        "chair": (125, 86, 59),
        "floor_barrel": (129, 84, 45),
        "picture_frame_floor": (190, 147, 89),
        "speaker": (58, 62, 70),
        "side_table": (156, 106, 64),
        "plant_stand": (98, 128, 78),
        "fan_floor": (174, 187, 193),
        "ac_floor": (205, 216, 218),
        "coat_floor": (116, 80, 145),
        "plant": (76, 142, 92),
        "lamp": (218, 184, 93),
        "box": (185, 128, 60),
        "toy": (91, 154, 200),
    }.get(color_role, CONTEXT_OBJECT_COLORS[default_color_index])
    object_type = str(spec.get("object_type"))
    if object_type == "sofa":
        return _draw_room_sofa_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "armchair":
        return _draw_room_armchair_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type in {"side_table", "desk"}:
        return _draw_room_table_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "media_console":
        return _draw_room_media_console_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "bed":
        return _draw_room_bed_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "plant":
        return _draw_room_plant_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "floor_lamp":
        return _draw_room_floor_lamp_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "box":
        return _draw_room_closed_box_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "open_box":
        return _draw_room_open_box_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "wall_shelf":
        return _draw_room_standing_shelf_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "tv":
        return _draw_room_tv_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "clock":
        return _draw_room_clock_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "picture_frame":
        return _draw_room_picture_frame_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "mirror":
        return _draw_room_mirror_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "wall_fan":
        return _draw_room_floor_fan_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "air_conditioner":
        return _draw_room_portable_ac_object(draw, spec, camera=camera, frame=frame, fill=color)
    if object_type == "hanging_coat":
        return _draw_room_coat_stand_object(draw, spec, camera=camera, frame=frame, fill=color)
    shape_type = str(spec["shape_type"])
    if shape_type == "table":
        return draw_table_object(draw, spec, camera=camera, frame=frame, fill=color)
    if shape_type == "sphere":
        return draw_sphere_object(draw, spec, camera=camera, frame=frame, fill=color)
    if shape_type == "cylinder":
        return draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=color)
    if shape_type == "cone":
        return draw_cone_object(draw, spec, camera=camera, frame=frame, fill=color)
    if shape_type == "pyramid":
        return draw_pyramid_object(draw, spec, camera=camera, frame=frame, fill=color)
    if shape_type == "open_box":
        return draw_open_box_object(draw, spec, camera=camera, frame=frame, fill=color)
    return draw_box_object(draw, spec, camera=camera, frame=frame, fill=color)

__all__ = ["_draw_floor_object"]
