"""Landscape and bench rendering helpers for street-intersection scenes."""

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

def _draw_street_evergreen_tree_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base, up_unit, side_unit, height_px = _upright_screen_basis(spec, camera, frame)
    tree_h = max(58.0, min(120.0, height_px * 1.12))
    outline = (32, 73, 43)
    trunk = (112, 78, 48)

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    bboxes: List[List[float]] = []
    trunk_bottom = p(0.0, tree_h * 0.02)
    trunk_top = p(0.0, tree_h * 0.30)
    trunk_width = max(5, int(round(tree_h * 0.11)))
    _draw_line(draw, trunk_bottom, trunk_top, fill=(70, 45, 31), width=trunk_width + 3)
    _draw_line(draw, trunk_bottom, trunk_top, fill=trunk, width=trunk_width)
    bboxes.append(_screen_line_bbox(trunk_bottom, trunk_top, pad_px=trunk_width))

    tiers = (
        (0.16, 0.52, 0.48, _shade(fill, 0.82)),
        (0.34, 0.74, 0.38, fill),
        (0.54, 0.96, 0.28, _tint(fill, 0.12)),
    )
    ornament_points: List[Tuple[float, float]] = []
    for base_frac, tip_frac, half_width_frac, tier_fill in tiers:
        points = [
            p(-tree_h * half_width_frac, tree_h * base_frac),
            p(0.0, tree_h * tip_frac),
            p(tree_h * half_width_frac, tree_h * base_frac),
            p(tree_h * half_width_frac * 0.58, tree_h * (base_frac - 0.05)),
            p(0.0, tree_h * (base_frac + 0.03)),
            p(-tree_h * half_width_frac * 0.58, tree_h * (base_frac - 0.05)),
        ]
        draw.polygon(points, fill=tier_fill, outline=outline)
        bboxes.append(_screen_points_bbox(points, pad_px=1.0))
        ornament_points.extend([p(-tree_h * half_width_frac * 0.34, tree_h * (base_frac + 0.10)), p(tree_h * half_width_frac * 0.30, tree_h * (base_frac + 0.14))])

    star_center = p(0.0, tree_h * 1.02)
    star_r = max(4.0, min(8.0, tree_h * 0.065))
    star_points = []
    for index in range(10):
        radius = star_r if index % 2 == 0 else star_r * 0.45
        angle = -math.pi * 0.5 + math.tau * index / 10.0
        star_points.append((star_center[0] + math.cos(angle) * radius, star_center[1] + math.sin(angle) * radius))
    draw.polygon(star_points, fill=(238, 204, 76), outline=(119, 91, 36))
    bboxes.append(_screen_points_bbox(star_points, pad_px=1.0))

    ornament_colors = ((211, 63, 61), (236, 204, 78), (80, 129, 197), (232, 236, 219))
    for index, point in enumerate(ornament_points[:6]):
        radius = max(2.0, min(4.0, tree_h * 0.032))
        ornament_bbox = [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius]
        color_index = int(index)
        while color_index >= len(ornament_colors):
            color_index -= len(ornament_colors)
        draw.ellipse(tuple(ornament_bbox), fill=ornament_colors[color_index], outline=(45, 55, 42), width=1)
        bboxes.append(ornament_bbox)
    return _bbox_union(*bboxes)

def _draw_street_bush_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _object_screen_bbox(spec, camera, frame, pad_px=0.0)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    width = max(24.0, x1 - x0)
    height = max(16.0, y1 - y0)
    cx = (x0 + x1) * 0.5
    base_y = y1 - height * 0.10
    outline = (39, 86, 50)
    lobes = (
        (-0.30, -0.08, 0.34, 0.42, _shade(fill, 0.84)),
        (0.00, -0.20, 0.42, 0.52, fill),
        (0.30, -0.07, 0.34, 0.42, _tint(fill, 0.08)),
        (-0.10, 0.06, 0.40, 0.36, _shade(fill, 0.92)),
        (0.18, 0.10, 0.38, 0.34, _tint(fill, 0.16)),
    )
    bboxes: List[List[float]] = []
    ground = [cx - width * 0.48, base_y - height * 0.08, cx + width * 0.48, base_y + height * 0.06]
    draw.ellipse(tuple(ground), fill=(67, 101, 64), outline=(42, 75, 46), width=1)
    bboxes.append(ground)
    for dx, dy, sx, sy, lobe_fill in lobes:
        lobe = [
            cx + width * dx - width * sx * 0.5,
            base_y + height * dy - height * sy * 0.5,
            cx + width * dx + width * sx * 0.5,
            base_y + height * dy + height * sy * 0.5,
        ]
        draw.ellipse(tuple(lobe), fill=lobe_fill, outline=outline, width=2)
        bboxes.append(lobe)
    for frac in (0.26, 0.44, 0.62, 0.78):
        leaf = [cx - width * 0.45 + width * frac - 2.0, base_y - height * 0.12, cx - width * 0.45 + width * frac + 2.0, base_y - height * 0.12 + 4.0]
        draw.ellipse(tuple(leaf), fill=(219, 232, 147), outline=(62, 103, 55), width=1)
        bboxes.append(leaf)
    return _bbox_union(*bboxes)

def _draw_street_bench_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    length = max(width, depth)
    bench_depth = min(width, depth)

    def part(
        *,
        long_offset: float,
        lateral_offset: float,
        z_offset: float,
        long_dim: float,
        lateral_dim: float,
        z_dim: float,
    ) -> Dict[str, Any]:
        if axis == "y":
            offset_xyz = (float(lateral_offset), float(long_offset), float(z_offset))
            dimensions_xyz = (float(lateral_dim), float(long_dim), float(z_dim))
        else:
            offset_xyz = (float(long_offset), float(lateral_offset), float(z_offset))
            dimensions_xyz = (float(long_dim), float(lateral_dim), float(z_dim))
        return _sub_box_spec(spec, offset_xyz=offset_xyz, dimensions_xyz=dimensions_xyz)

    wood = _tint(fill, 0.14)
    dark_wood = _shade(fill, 0.82)
    metal = (70, 74, 73)
    back_y = bench_depth * 0.36
    front_y = -bench_depth * 0.28
    side_x = length * 0.43
    entries: List[Tuple[Mapping[str, Any], Tuple[int, int, int]]] = []

    for lateral in (-bench_depth * 0.16, bench_depth * 0.02, bench_depth * 0.20):
        entries.append(
            (
                part(
                    long_offset=0.0,
                    lateral_offset=lateral,
                    z_offset=height * 0.24,
                    long_dim=length,
                    lateral_dim=bench_depth * 0.16,
                    z_dim=height * 0.10,
                ),
                wood,
            )
        )
    for z_offset, color in ((height * 0.46, _tint(fill, 0.08)), (height * 0.64, wood)):
        entries.append(
            (
                part(
                    long_offset=0.0,
                    lateral_offset=back_y,
                    z_offset=z_offset,
                    long_dim=length,
                    lateral_dim=bench_depth * 0.13,
                    z_dim=height * 0.11,
                ),
                color,
            )
        )
    for long_offset in (-side_x, side_x):
        entries.extend(
            [
                (
                    part(
                        long_offset=long_offset,
                        lateral_offset=front_y,
                        z_offset=0.0,
                        long_dim=length * 0.07,
                        lateral_dim=bench_depth * 0.12,
                        z_dim=height * 0.30,
                    ),
                    metal,
                ),
                (
                    part(
                        long_offset=long_offset,
                        lateral_offset=back_y,
                        z_offset=0.0,
                        long_dim=length * 0.07,
                        lateral_dim=bench_depth * 0.12,
                        z_dim=height * 0.72,
                    ),
                    metal,
                ),
                (
                    part(
                        long_offset=long_offset,
                        lateral_offset=bench_depth * 0.02,
                        z_offset=height * 0.38,
                        long_dim=length * 0.07,
                        lateral_dim=bench_depth * 0.80,
                        z_dim=height * 0.07,
                    ),
                    dark_wood,
                ),
            ]
        )
    for long_offset in (-length * 0.28, length * 0.28):
        entries.append(
            (
                part(
                    long_offset=long_offset,
                    lateral_offset=front_y,
                    z_offset=0.0,
                    long_dim=length * 0.07,
                    lateral_dim=bench_depth * 0.10,
                    z_dim=height * 0.26,
                ),
                metal,
            )
        )

    def camera_distance_sq(entry: Tuple[Mapping[str, Any], Tuple[int, int, int]]) -> float:
        world_xyz = entry[0]["world_xyz"]
        return sum((float(world_xyz[index]) - float(camera.camera_position[index])) ** 2 for index in range(3))

    bboxes: List[List[float]] = []
    for bench_part, color in sorted(entries, key=camera_distance_sq, reverse=True):
        bboxes.append(_draw_box_object(draw, bench_part, camera=camera, frame=frame, fill=color))
    return _bbox_union(*bboxes)


__all__ = [
    '_draw_street_evergreen_tree_object',
    '_draw_street_bush_object',
    '_draw_street_bench_object',
]
