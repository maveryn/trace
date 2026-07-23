"""Wall-mounted object renderers for shared room-wall 3D scenes."""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .object_scene_primitives import bbox_union, draw_line, shade_rgb, tint_rgb
from .room_wall_rendering_geometry import (
    _draw_poly,
    _draw_screen_scenery,
    _face_camera_distance_sq,
    _inset_bbox,
    _points_bbox,
    _projected_polygon_bbox,
    _project_points,
    _wall_physical_point,
    _wall_plane_point,
    _wall_quad_points,
    _wall_rect_points,
)


def _draw_wall_cuboid(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    normal_near: float,
    normal_far: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> List[float]:
    half_w = float(spec["wall_width"]) * 0.5
    half_h = float(spec["wall_height"]) * 0.5
    back = {
        "lb": _wall_physical_point(spec, -half_w, -half_h, normal_offset=float(normal_near)),
        "rb": _wall_physical_point(spec, half_w, -half_h, normal_offset=float(normal_near)),
        "rt": _wall_physical_point(spec, half_w, half_h, normal_offset=float(normal_near)),
        "lt": _wall_physical_point(spec, -half_w, half_h, normal_offset=float(normal_near)),
    }
    front = {
        "lb": _wall_physical_point(spec, -half_w, -half_h, normal_offset=float(normal_far)),
        "rb": _wall_physical_point(spec, half_w, -half_h, normal_offset=float(normal_far)),
        "rt": _wall_physical_point(spec, half_w, half_h, normal_offset=float(normal_far)),
        "lt": _wall_physical_point(spec, -half_w, half_h, normal_offset=float(normal_far)),
    }
    faces = [
        ([front["lb"], front["rb"], front["rt"], front["lt"]], tint_rgb(fill, 0.10)),
        ([back["lt"], front["lt"], front["rt"], back["rt"]], tint_rgb(fill, 0.22)),
        ([back["lb"], back["rb"], front["rb"], front["lb"]], shade_rgb(fill, 0.76)),
        ([back["rb"], back["rt"], front["rt"], front["rb"]], shade_rgb(fill, 0.84)),
        ([back["lb"], front["lb"], front["lt"], back["lt"]], shade_rgb(fill, 0.68)),
    ]
    bboxes: List[List[float]] = []
    for points, face_fill in sorted(faces, key=lambda item: _face_camera_distance_sq(item[0], camera), reverse=True):
        bboxes.append(_draw_poly(draw, points, camera=camera, frame=frame, fill=face_fill, outline=outline, width=2))
    return bbox_union(*bboxes)


def _draw_wall_tv_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    shadow_points = _wall_quad_points(spec, -0.50, -0.50, 0.50, 0.50, normal_offset=0.012)
    shadow_projected = _project_points(shadow_points, camera, frame)
    shadow_projected = [(x + 3.0, y + 4.0) for x, y in shadow_projected]
    draw.polygon(shadow_projected, fill=(84, 91, 98))
    bboxes: List[List[float]] = [_points_bbox(shadow_projected)]
    bboxes.append(
        _draw_wall_cuboid(
            draw,
            spec,
            camera=camera,
            frame=frame,
            normal_near=0.018,
            normal_far=0.082,
            fill=(34, 39, 47),
            outline=(12, 15, 19),
        )
    )
    bezel_points = _wall_quad_points(spec, -0.48, -0.46, 0.48, 0.46, normal_offset=0.088)
    bboxes.append(
        _draw_poly(
            draw,
            bezel_points,
            camera=camera,
            frame=frame,
            fill=(18, 21, 26),
            outline=(7, 9, 12),
            width=2,
        )
    )
    screen_points = _wall_quad_points(spec, -0.405, -0.335, 0.405, 0.335, normal_offset=0.092)
    bboxes.append(
        _draw_poly(
            draw,
            screen_points,
            camera=camera,
            frame=frame,
            fill=(20, 35, 51),
            outline=(75, 92, 108),
            width=1,
        )
    )
    glow_points = _wall_quad_points(spec, -0.35, -0.27, 0.35, 0.27, normal_offset=0.094)
    bboxes.append(
        _draw_poly(
            draw,
            glow_points,
            camera=camera,
            frame=frame,
            fill=(31, 55, 77),
            outline=(31, 55, 77),
            width=1,
        )
    )
    highlight_points = _wall_quad_points(spec, -0.34, 0.12, 0.20, 0.22, normal_offset=0.096)
    projected_highlight = _project_points(highlight_points, camera, frame)
    draw.polygon(projected_highlight, fill=(69, 94, 113))
    bboxes.append(_points_bbox(projected_highlight))
    status_bar = _wall_quad_points(spec, -0.16, -0.425, 0.16, -0.385, normal_offset=0.097)
    projected_status = _project_points(status_bar, camera, frame)
    draw.polygon(projected_status, fill=(71, 78, 87))
    bboxes.append(_points_bbox(projected_status))
    mount_neck_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 0.12,
        "wall_height": float(spec["wall_height"]) * 0.11,
        "world_xyz": list(_wall_physical_point(spec, 0.0, -float(spec["wall_height"]) * 0.58, normal_offset=0.048)),
    }
    bboxes.append(
        _draw_wall_cuboid(
            draw,
            mount_neck_spec,
            camera=camera,
            frame=frame,
            normal_near=0.0,
            normal_far=0.045,
            fill=(28, 32, 38),
            outline=(12, 15, 19),
        )
    )
    mount_plate_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 0.32,
        "wall_height": float(spec["wall_height"]) * 0.055,
        "world_xyz": list(_wall_physical_point(spec, 0.0, -float(spec["wall_height"]) * 0.66, normal_offset=0.048)),
    }
    bboxes.append(
        _draw_wall_cuboid(
            draw,
            mount_plate_spec,
            camera=camera,
            frame=frame,
            normal_near=0.0,
            normal_far=0.045,
            fill=(26, 30, 35),
            outline=(12, 15, 19),
        )
    )
    return bbox_union(*bboxes)


def _draw_wall_shelf_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    board_bbox = _draw_wall_cuboid(
        draw,
        spec,
        camera=camera,
        frame=frame,
        normal_near=0.035,
        normal_far=0.42,
        fill=(132, 101, 73),
        outline=(67, 51, 39),
    )
    half_w = float(spec["wall_width"]) * 0.5
    half_h = float(spec["wall_height"]) * 0.5
    bracket_bboxes: List[List[float]] = []
    for hpos in (-half_w * 0.36, half_w * 0.36):
        triangle = [
            _wall_physical_point(spec, hpos, -half_h, normal_offset=0.055),
            _wall_physical_point(spec, hpos, -half_h, normal_offset=0.35),
            _wall_physical_point(spec, hpos, -half_h - 0.30, normal_offset=0.055),
        ]
        bracket_bboxes.append(
            _draw_poly(draw, triangle, camera=camera, frame=frame, fill=(103, 77, 55), outline=(67, 51, 39), width=2)
        )
    lip_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 0.96,
        "wall_height": max(0.045, float(spec["wall_height"]) * 0.35),
        "world_xyz": list(_wall_physical_point(spec, 0.0, -half_h - 0.02, normal_offset=0.43)),
    }
    bracket_bboxes.append(
        _draw_wall_cuboid(
            draw,
            lip_spec,
            camera=camera,
            frame=frame,
            normal_near=0.0,
            normal_far=0.035,
            fill=(110, 83, 58),
            outline=(67, 51, 39),
        )
    )
    item_specs = [
        (-0.34, 0.10, 0.10, 0.30, (79, 119, 165)),
        (-0.22, 0.10, 0.08, 0.24, (201, 92, 78)),
        (0.04, 0.10, 0.12, 0.22, (224, 182, 82)),
        (0.28, 0.10, 0.12, 0.18, (75, 138, 92)),
    ]
    for h_center, bottom_v, item_w, item_h, color in item_specs:
        item = [
            _wall_physical_point(spec, h_center - item_w * 0.5, half_h + bottom_v, normal_offset=0.455),
            _wall_physical_point(spec, h_center + item_w * 0.5, half_h + bottom_v, normal_offset=0.455),
            _wall_physical_point(spec, h_center + item_w * 0.5, half_h + bottom_v + item_h, normal_offset=0.455),
            _wall_physical_point(spec, h_center - item_w * 0.5, half_h + bottom_v + item_h, normal_offset=0.455),
        ]
        bracket_bboxes.append(_draw_poly(draw, item, camera=camera, frame=frame, fill=color, outline=(45, 45, 38), width=1))
    return bbox_union(board_bbox, *bracket_bboxes)


def _fill_wall_shape(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> None:
    draw.polygon(_project_points(points, camera, frame), fill=fill)


def _outline_wall_shape(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    camera,
    frame,
    outline: Tuple[int, int, int],
    width: int = 1,
) -> None:
    projected = _project_points(points, camera, frame)
    for index in range(len(projected)):
        next_index = index + 1
        if next_index >= len(projected):
            next_index = 0
        draw_line(draw, projected[index], projected[next_index], fill=outline, width=int(width))


def _draw_wall_disc(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    radius_u: float = 0.46,
    radius_v: float = 0.46,
    normal_offset: float = 0.034,
) -> List[float]:
    points = [
        _wall_plane_point(
            spec,
            math.cos(2.0 * math.pi * index / 28.0) * float(radius_u),
            math.sin(2.0 * math.pi * index / 28.0) * float(radius_v),
            normal_offset=float(normal_offset),
        )
        for index in range(28)
    ]
    projected = _project_points(points, camera, frame)
    draw.polygon(projected, fill=fill)
    for index in range(len(projected)):
        next_index = index + 1
        if next_index >= len(projected):
            next_index = 0
        draw_line(draw, projected[index], projected[next_index], fill=outline, width=2)
    return _points_bbox(projected)


def _draw_wall_fan_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    bboxes.append(
        _draw_wall_disc(
            draw,
            spec,
            camera=camera,
            frame=frame,
            fill=(225, 233, 235),
            outline=(35, 46, 55),
            radius_u=0.50,
            radius_v=0.50,
            normal_offset=0.032,
        )
    )
    blade_fill = (89, 116, 132)
    blade_outline = (42, 56, 66)
    for angle in (math.pi * 0.50, math.pi, math.pi * 1.50, math.tau):
        inner_left = angle - 0.34
        inner_right = angle + 0.34
        outer_left = angle + 0.12
        outer_right = angle + 0.50
        blade = [
            _wall_plane_point(spec, 0.10 * math.cos(inner_left), 0.10 * math.sin(inner_left), normal_offset=0.046),
            _wall_plane_point(spec, 0.42 * math.cos(outer_left), 0.42 * math.sin(outer_left), normal_offset=0.046),
            _wall_plane_point(spec, 0.34 * math.cos(outer_right), 0.34 * math.sin(outer_right), normal_offset=0.046),
            _wall_plane_point(spec, 0.10 * math.cos(inner_right), 0.10 * math.sin(inner_right), normal_offset=0.046),
        ]
        bboxes.append(_draw_poly(draw, blade, camera=camera, frame=frame, fill=blade_fill, outline=blade_outline, width=2))
    hub_spec = {**dict(spec), "wall_width": float(spec["wall_width"]) * 0.24, "wall_height": float(spec["wall_height"]) * 0.24}
    hub_spec["world_xyz"] = list(_wall_plane_point(spec, 0.0, 0.0, normal_offset=0.054))
    bboxes.append(
        _draw_wall_disc(
            draw,
            hub_spec,
            camera=camera,
            frame=frame,
            fill=(55, 69, 80),
            outline=(24, 31, 37),
            radius_u=0.50,
            radius_v=0.50,
            normal_offset=0.056,
        )
    )
    center = _project_points([_wall_plane_point(spec, 0.0, 0.0, normal_offset=0.060)], camera, frame)[0]
    for angle in (0.0, math.pi * 0.25, math.pi * 0.50, math.pi * 0.75, math.pi, math.pi * 1.25, math.pi * 1.50, math.pi * 1.75):
        edge = _project_points([_wall_plane_point(spec, 0.48 * math.cos(angle), 0.48 * math.sin(angle), normal_offset=0.060)], camera, frame)[0]
        draw_line(draw, center, edge, fill=(77, 91, 101), width=2)
        bboxes.append(_points_bbox([center, edge]))
    for radius_u, radius_v, color in ((0.36, 0.36, (104, 118, 128)), (0.47, 0.47, (55, 67, 76))):
        ring_points = [
            _wall_plane_point(spec, math.cos(math.tau * index / 36.0) * radius_u, math.sin(math.tau * index / 36.0) * radius_v, normal_offset=0.061)
            for index in range(36)
        ]
        projected_ring = _project_points(ring_points, camera, frame)
        for index in range(len(projected_ring)):
            next_index = index + 1
            if next_index >= len(projected_ring):
                next_index = 0
            draw_line(draw, projected_ring[index], projected_ring[next_index], fill=color, width=2)
        bboxes.append(_points_bbox(projected_ring))
    mount = _project_points(
        [
            _wall_plane_point(spec, 0.0, -0.50, normal_offset=0.050),
            _wall_plane_point(spec, 0.0, -0.62, normal_offset=0.050),
        ],
        camera,
        frame,
    )
    draw_line(draw, mount[0], mount[1], fill=(45, 54, 61), width=3)
    bboxes.append(_points_bbox(mount))
    return bbox_union(*bboxes)


def _draw_wall_air_conditioner_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bbox = _draw_wall_flat_object(draw, spec, camera=camera, frame=frame, fill=(220, 226, 226), trim=(77, 89, 94))
    bboxes: List[List[float]] = [list(bbox)]
    left_cap = _wall_quad_points(spec, -0.48, -0.36, -0.39, 0.36, normal_offset=0.041)
    right_cap = _wall_quad_points(spec, 0.39, -0.36, 0.48, 0.36, normal_offset=0.041)
    _fill_wall_shape(draw, left_cap, camera=camera, frame=frame, fill=(197, 209, 212))
    _fill_wall_shape(draw, right_cap, camera=camera, frame=frame, fill=(197, 209, 212))
    _outline_wall_shape(draw, left_cap, camera=camera, frame=frame, outline=(77, 89, 94), width=1)
    _outline_wall_shape(draw, right_cap, camera=camera, frame=frame, outline=(77, 89, 94), width=1)
    bboxes.extend([_points_bbox(_project_points(left_cap, camera, frame)), _points_bbox(_project_points(right_cap, camera, frame))])
    grille = _wall_quad_points(spec, -0.42, 0.08, 0.42, 0.30, normal_offset=0.038)
    _fill_wall_shape(draw, grille, camera=camera, frame=frame, fill=(192, 203, 207))
    _outline_wall_shape(draw, grille, camera=camera, frame=frame, outline=(87, 102, 108), width=1)
    bboxes.append(_points_bbox(_project_points(grille, camera, frame)))
    for v in (0.12, 0.17, 0.22, 0.27):
        left = _project_points([_wall_plane_point(spec, -0.37, v, normal_offset=0.043)], camera, frame)[0]
        right = _project_points([_wall_plane_point(spec, 0.37, v, normal_offset=0.043)], camera, frame)[0]
        draw_line(draw, left, right, fill=(92, 108, 115), width=1)
        bboxes.append(_points_bbox([left, right]))
    outlet = _wall_quad_points(spec, -0.38, -0.28, 0.38, -0.11, normal_offset=0.04)
    _fill_wall_shape(draw, outlet, camera=camera, frame=frame, fill=(236, 240, 239))
    _outline_wall_shape(draw, outlet, camera=camera, frame=frame, outline=(88, 103, 108), width=1)
    bboxes.append(_points_bbox(_project_points(outlet, camera, frame)))
    for u in (-0.24, 0.0, 0.24):
        top = _project_points([_wall_plane_point(spec, u, -0.11, normal_offset=0.044)], camera, frame)[0]
        bottom = _project_points([_wall_plane_point(spec, u - 0.08, -0.28, normal_offset=0.044)], camera, frame)[0]
        draw_line(draw, top, bottom, fill=(113, 126, 130), width=1)
        bboxes.append(_points_bbox([top, bottom]))
    indicator = _wall_quad_points(spec, 0.28, -0.02, 0.39, 0.05, normal_offset=0.045)
    _fill_wall_shape(draw, indicator, camera=camera, frame=frame, fill=(88, 151, 174))
    _outline_wall_shape(draw, indicator, camera=camera, frame=frame, outline=(47, 83, 96), width=1)
    bboxes.append(_points_bbox(_project_points(indicator, camera, frame)))
    for u0, v0, u1, v1 in ((-0.30, -0.02, -0.18, -0.02), (-0.10, -0.02, 0.02, -0.02)):
        p1, p2 = _project_points(
            [_wall_plane_point(spec, u0, v0, normal_offset=0.046), _wall_plane_point(spec, u1, v1, normal_offset=0.046)],
            camera,
            frame,
        )
        draw_line(draw, p1, p2, fill=(92, 108, 115), width=1)
        bboxes.append(_points_bbox([p1, p2]))
    return bbox_union(*bboxes)


def _draw_wall_hanging_coat_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    hook = [
        _wall_plane_point(spec, 0.0, 0.48, normal_offset=0.04),
        _wall_plane_point(spec, 0.0, 0.34, normal_offset=0.04),
    ]
    projected_hook = _project_points(hook, camera, frame)
    draw_line(draw, projected_hook[0], projected_hook[1], fill=(53, 55, 58), width=2)
    body = [
        _wall_plane_point(spec, -0.30, 0.30, normal_offset=0.055),
        _wall_plane_point(spec, 0.30, 0.30, normal_offset=0.055),
        _wall_plane_point(spec, 0.39, -0.44, normal_offset=0.055),
        _wall_plane_point(spec, -0.39, -0.44, normal_offset=0.055),
    ]
    left_sleeve = [
        _wall_plane_point(spec, -0.30, 0.22, normal_offset=0.053),
        _wall_plane_point(spec, -0.56, -0.12, normal_offset=0.053),
        _wall_plane_point(spec, -0.42, -0.28, normal_offset=0.053),
        _wall_plane_point(spec, -0.14, 0.10, normal_offset=0.053),
    ]
    right_sleeve = [
        _wall_plane_point(spec, 0.30, 0.22, normal_offset=0.053),
        _wall_plane_point(spec, 0.56, -0.12, normal_offset=0.053),
        _wall_plane_point(spec, 0.42, -0.28, normal_offset=0.053),
        _wall_plane_point(spec, 0.14, 0.10, normal_offset=0.053),
    ]
    color = (116, 80, 145)
    bboxes = [
        _points_bbox(projected_hook),
        _draw_poly(draw, left_sleeve, camera=camera, frame=frame, fill=shade_rgb(color, 0.86), outline=(53, 42, 68), width=2),
        _draw_poly(draw, right_sleeve, camera=camera, frame=frame, fill=shade_rgb(color, 0.92), outline=(53, 42, 68), width=2),
        _draw_poly(draw, body, camera=camera, frame=frame, fill=color, outline=(53, 42, 68), width=2),
    ]
    collar = [
        _wall_plane_point(spec, -0.10, 0.25, normal_offset=0.062),
        _wall_plane_point(spec, 0.0, 0.12, normal_offset=0.062),
        _wall_plane_point(spec, 0.10, 0.25, normal_offset=0.062),
    ]
    bboxes.append(_draw_poly(draw, collar, camera=camera, frame=frame, fill=(232, 225, 210), outline=(53, 42, 68), width=1))
    seam = _project_points(
        [
            _wall_plane_point(spec, 0.0, 0.08, normal_offset=0.064),
            _wall_plane_point(spec, 0.0, -0.38, normal_offset=0.064),
        ],
        camera,
        frame,
    )
    draw_line(draw, seam[0], seam[1], fill=(63, 47, 82), width=2)
    bboxes.append(_points_bbox(seam))
    for v in (-0.04, -0.18, -0.32):
        center = _project_points([_wall_plane_point(spec, 0.05, v, normal_offset=0.066)], camera, frame)[0]
        draw.ellipse((center[0] - 2.2, center[1] - 2.2, center[0] + 2.2, center[1] + 2.2), fill=(226, 218, 202), outline=(53, 42, 68), width=1)
        bboxes.append([center[0] - 2.2, center[1] - 2.2, center[0] + 2.2, center[1] + 2.2])
    return bbox_union(*bboxes)


def _draw_wall_scenery(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    variant: str,
    outline: Tuple[int, int, int],
) -> None:
    if str(variant) == "sunset":
        sky = (238, 170, 118)
        ground = (110, 132, 104)
        accent = (250, 218, 116)
    elif str(variant) == "lake":
        sky = (177, 215, 231)
        ground = (93, 151, 180)
        accent = (236, 244, 248)
    elif str(variant) == "forest":
        sky = (188, 218, 198)
        ground = (75, 131, 91)
        accent = (42, 96, 63)
    elif str(variant) == "city":
        sky = (181, 202, 224)
        ground = (126, 136, 150)
        accent = (78, 87, 102)
    else:
        sky = (175, 210, 230)
        ground = (115, 154, 105)
        accent = (93, 118, 142)

    horizon = -0.06
    _fill_wall_shape(draw, _wall_quad_points(spec, -0.5, horizon, 0.5, 0.5), camera=camera, frame=frame, fill=sky)
    _fill_wall_shape(draw, _wall_quad_points(spec, -0.5, -0.5, 0.5, horizon), camera=camera, frame=frame, fill=ground)

    if str(variant) == "city":
        for index in range(4):
            u_start = -0.38 + index * 0.20
            u_end = u_start + 0.11
            top = horizon + 0.05 + 0.10 * (index % 2)
            _fill_wall_shape(
                draw,
                _wall_quad_points(spec, u_start, horizon, u_end, top),
                camera=camera,
                frame=frame,
                fill=accent,
            )
    elif str(variant) == "forest":
        for index in range(4):
            u = -0.30 + index * 0.20
            tree = [
                _wall_plane_point(spec, u, 0.22),
                _wall_plane_point(spec, u - 0.08, horizon),
                _wall_plane_point(spec, u + 0.08, horizon),
            ]
            _fill_wall_shape(draw, tree, camera=camera, frame=frame, fill=accent)
    else:
        mountains = [
            (
                [
                    _wall_plane_point(spec, -0.45, horizon),
                    _wall_plane_point(spec, -0.22, 0.24),
                    _wall_plane_point(spec, 0.02, horizon),
                ],
                accent,
            ),
            (
                [
                    _wall_plane_point(spec, -0.08, horizon),
                    _wall_plane_point(spec, 0.20, 0.20),
                    _wall_plane_point(spec, 0.48, horizon),
                ],
                shade_rgb(accent, 0.86),
            ),
        ]
        for points, color in mountains:
            _fill_wall_shape(draw, points, camera=camera, frame=frame, fill=color)
        if str(variant) in {"lake", "sunset"}:
            sun_spec = {**dict(spec), "wall_width": float(spec["wall_width"]) * 0.18, "wall_height": float(spec["wall_height"]) * 0.18}
            sun_spec["world_xyz"] = list(_wall_plane_point(spec, 0.30, 0.27, normal_offset=0.033))
            _draw_wall_disc(draw, sun_spec, camera=camera, frame=frame, fill=(247, 221, 111), outline=(196, 154, 75), radius_u=0.5, radius_v=0.5, normal_offset=0.034)

    _outline_wall_shape(
        draw,
        _wall_quad_points(spec, -0.5, -0.5, 0.5, 0.5),
        camera=camera,
        frame=frame,
        outline=outline,
        width=1,
    )

def _draw_wall_flat_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
    trim: Tuple[int, int, int],
) -> List[float]:
    outer = _wall_rect_points(spec, normal_offset=0.024)
    bbox = _draw_poly(draw, outer, camera=camera, frame=frame, fill=fill, outline=trim, width=2)
    inner = _wall_rect_points({**dict(spec), "wall_width": float(spec["wall_width"]) * 0.78, "wall_height": float(spec["wall_height"]) * 0.74}, normal_offset=0.026)
    if str(spec.get("object_type")) in {"picture_frame", "poster"}:
        _draw_poly(draw, inner, camera=camera, frame=frame, fill=tint_rgb(fill, 0.34), outline=shade_rgb(trim, 0.8), width=1)
    elif str(spec.get("object_type")) == "mirror":
        _draw_poly(draw, inner, camera=camera, frame=frame, fill=(199, 226, 232), outline=shade_rgb(trim, 0.9), width=1)
    return list(bbox)


def _draw_wall_speaker_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bbox = _draw_wall_flat_object(draw, spec, camera=camera, frame=frame, fill=(43, 49, 56), trim=(17, 20, 24))
    bboxes = [bbox]
    for v, scale, cone_fill in ((-0.18, 0.52, (74, 84, 96)), (0.28, 0.26, (92, 103, 116))):
        driver_spec = {
            **dict(spec),
            "world_xyz": list(_wall_plane_point(spec, 0.0, v, normal_offset=0.040)),
            "wall_width": float(spec["wall_width"]) * float(scale),
            "wall_height": float(spec["wall_height"]) * float(scale),
        }
        bboxes.append(
            _draw_wall_disc(
                draw,
                driver_spec,
                camera=camera,
                frame=frame,
                fill=cone_fill,
                outline=(12, 15, 18),
                radius_u=0.50,
                radius_v=0.50,
                normal_offset=0.043,
            )
        )
        cap_spec = {
            **driver_spec,
            "wall_width": float(driver_spec["wall_width"]) * 0.38,
            "wall_height": float(driver_spec["wall_height"]) * 0.38,
        }
        bboxes.append(
            _draw_wall_disc(
                draw,
                cap_spec,
                camera=camera,
                frame=frame,
                fill=(25, 29, 34),
                outline=(120, 132, 142),
                radius_u=0.50,
                radius_v=0.50,
                normal_offset=0.047,
            )
        )
    for v in (-0.44, 0.04, 0.48):
        p1, p2 = _project_points(
            [
                _wall_plane_point(spec, -0.36, v, normal_offset=0.048),
                _wall_plane_point(spec, 0.36, v, normal_offset=0.048),
            ],
            camera,
            frame,
        )
        draw_line(draw, p1, p2, fill=(122, 133, 143), width=1)
        bboxes.append([min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])])
    highlight = _wall_quad_points(spec, -0.42, 0.58, 0.42, 0.68, normal_offset=0.049)
    projected = _project_points(highlight, camera, frame)
    draw.polygon(projected, fill=(72, 82, 92))
    bboxes.append(_points_bbox(projected))
    return bbox_union(*bboxes)


def _draw_wall_picture_frame_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    bboxes.append(
        _draw_wall_cuboid(
            draw,
            spec,
            camera=camera,
            frame=frame,
            normal_near=0.012,
            normal_far=0.060,
            fill=(122, 73, 48),
            outline=(64, 39, 28),
        )
    )
    mat = _wall_quad_points(spec, -0.40, -0.36, 0.40, 0.36, normal_offset=0.064)
    bboxes.append(_draw_poly(draw, mat, camera=camera, frame=frame, fill=(232, 219, 194), outline=(83, 53, 36), width=2))
    inner_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 0.66,
        "wall_height": float(spec["wall_height"]) * 0.56,
        "world_xyz": list(_wall_plane_point(spec, 0.0, 0.0, normal_offset=0.069)),
    }
    _draw_wall_scenery(
        draw,
        inner_spec,
        camera=camera,
        frame=frame,
        variant=str(spec.get("scenery_variant", "mountains")),
        outline=(74, 48, 37),
    )
    bboxes.append(_projected_polygon_bbox(_wall_rect_points(inner_spec, normal_offset=0.070), camera, frame))
    return bbox_union(*bboxes)


def _draw_wall_mirror_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    bboxes.append(
        _draw_wall_cuboid(
            draw,
            spec,
            camera=camera,
            frame=frame,
            normal_near=0.012,
            normal_far=0.055,
            fill=(67, 91, 104),
            outline=(38, 55, 64),
        )
    )
    glass = _wall_quad_points(spec, -0.34, -0.40, 0.34, 0.40, normal_offset=0.060)
    bboxes.append(_draw_poly(draw, glass, camera=camera, frame=frame, fill=(178, 218, 229), outline=(43, 74, 87), width=2))
    shine_lines = [
        [(-0.24, 0.26), (0.10, 0.38)],
        [(-0.28, 0.06), (0.28, 0.26)],
        [(-0.18, -0.24), (0.18, -0.10)],
    ]
    for start, end in shine_lines:
        p1, p2 = _project_points(
            [
                _wall_plane_point(spec, start[0], start[1], normal_offset=0.064),
                _wall_plane_point(spec, end[0], end[1], normal_offset=0.064),
            ],
            camera,
            frame,
        )
        draw_line(draw, p1, p2, fill=(232, 249, 252), width=2)
        bboxes.append(_points_bbox([p1, p2]))
    return bbox_union(*bboxes)


def _draw_wall_lamp_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    glow_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 1.12,
        "wall_height": float(spec["wall_height"]) * 1.06,
    }
    bboxes.append(
        _draw_wall_disc(
            draw,
            glow_spec,
            camera=camera,
            frame=frame,
            fill=(252, 231, 137),
            outline=(190, 145, 55),
            radius_u=0.50,
            radius_v=0.48,
            normal_offset=0.026,
        )
    )
    plate_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 0.40,
        "wall_height": float(spec["wall_height"]) * 0.72,
    }
    bboxes.append(
        _draw_wall_disc(
            draw,
            plate_spec,
            camera=camera,
            frame=frame,
            fill=(92, 67, 42),
            outline=(46, 35, 25),
            radius_u=0.42,
            radius_v=0.50,
            normal_offset=0.052,
        )
    )
    arm_points = [
        _wall_plane_point(spec, 0.0, 0.08, normal_offset=0.058),
        _wall_plane_point(spec, 0.12, -0.04, normal_offset=0.105),
        _wall_plane_point(spec, 0.0, -0.20, normal_offset=0.150),
    ]
    projected_arm = _project_points(arm_points, camera, frame)
    draw_line(draw, projected_arm[0], projected_arm[1], fill=(62, 47, 32), width=4)
    draw_line(draw, projected_arm[1], projected_arm[2], fill=(62, 47, 32), width=4)
    bboxes.append(_points_bbox(projected_arm))
    shade = [
        _wall_plane_point(spec, -0.34, -0.04, normal_offset=0.165),
        _wall_plane_point(spec, 0.34, -0.04, normal_offset=0.165),
        _wall_plane_point(spec, 0.48, -0.42, normal_offset=0.165),
        _wall_plane_point(spec, -0.48, -0.42, normal_offset=0.165),
    ]
    bboxes.append(_draw_poly(draw, shade, camera=camera, frame=frame, fill=(226, 171, 62), outline=(82, 61, 35), width=3))
    shade_lip = [
        _wall_plane_point(spec, -0.50, -0.42, normal_offset=0.170),
        _wall_plane_point(spec, 0.50, -0.42, normal_offset=0.170),
        _wall_plane_point(spec, 0.43, -0.50, normal_offset=0.170),
        _wall_plane_point(spec, -0.43, -0.50, normal_offset=0.170),
    ]
    bboxes.append(_draw_poly(draw, shade_lip, camera=camera, frame=frame, fill=(166, 103, 40), outline=(82, 61, 35), width=2))
    bulb_spec = {
        **dict(spec),
        "wall_width": float(spec["wall_width"]) * 0.24,
        "wall_height": float(spec["wall_height"]) * 0.20,
        "world_xyz": list(_wall_plane_point(spec, 0.0, -0.30, normal_offset=0.178)),
    }
    bboxes.append(
        _draw_wall_disc(
            draw,
            bulb_spec,
            camera=camera,
            frame=frame,
            fill=(255, 242, 172),
            outline=(151, 115, 48),
            radius_u=0.50,
            radius_v=0.50,
            normal_offset=0.182,
        )
    )
    for u0, v0, u1, v1 in ((-0.36, -0.54, -0.20, -0.64), (0.0, -0.56, 0.0, -0.68), (0.36, -0.54, 0.20, -0.64)):
        ray = _project_points(
            [
                _wall_plane_point(spec, u0, v0, normal_offset=0.060),
                _wall_plane_point(spec, u1, v1, normal_offset=0.060),
            ],
            camera,
            frame,
        )
        draw_line(draw, ray[0], ray[1], fill=(232, 190, 75), width=2)
        bboxes.append(_points_bbox(ray))
    chain = _project_points(
        [
            _wall_plane_point(spec, 0.18, -0.46, normal_offset=0.176),
            _wall_plane_point(spec, 0.20, -0.58, normal_offset=0.174),
            _wall_plane_point(spec, 0.16, -0.70, normal_offset=0.172),
        ],
        camera,
        frame,
    )
    draw_line(draw, chain[0], chain[1], fill=(90, 70, 44), width=2)
    draw_line(draw, chain[1], chain[2], fill=(90, 70, 44), width=2)
    knob = _project_points([_wall_plane_point(spec, 0.16, -0.73, normal_offset=0.172)], camera, frame)[0]
    draw.ellipse((knob[0] - 2.0, knob[1] - 2.0, knob[0] + 2.0, knob[1] + 2.0), fill=(168, 129, 55), outline=(79, 57, 34), width=1)
    bboxes.append(_points_bbox(chain + [knob]))
    return bbox_union(*bboxes)


def _draw_wall_cabinet_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    bboxes.append(
        _draw_wall_cuboid(
            draw,
            spec,
            camera=camera,
            frame=frame,
            normal_near=0.018,
            normal_far=0.135,
            fill=(126, 113, 96),
            outline=(55, 48, 40),
        )
    )
    left = _wall_quad_points(spec, -0.42, -0.38, -0.02, 0.38, normal_offset=0.142)
    right = _wall_quad_points(spec, 0.02, -0.38, 0.42, 0.38, normal_offset=0.142)
    bboxes.append(_draw_poly(draw, left, camera=camera, frame=frame, fill=(146, 129, 107), outline=(64, 57, 48), width=2))
    bboxes.append(_draw_poly(draw, right, camera=camera, frame=frame, fill=(113, 101, 86), outline=(64, 57, 48), width=2))
    seam = _project_points(
        [
            _wall_plane_point(spec, 0.0, -0.38, normal_offset=0.148),
            _wall_plane_point(spec, 0.0, 0.38, normal_offset=0.148),
        ],
        camera,
        frame,
    )
    draw_line(draw, seam[0], seam[1], fill=(48, 43, 36), width=2)
    bboxes.append(_points_bbox(seam))
    for u0, u1 in ((-0.34, -0.10), (0.10, 0.34)):
        panel = _wall_quad_points(spec, u0, -0.22, u1, 0.26, normal_offset=0.150)
        _outline_wall_shape(draw, panel, camera=camera, frame=frame, outline=(88, 74, 58), width=1)
        bboxes.append(_points_bbox(_project_points(panel, camera, frame)))
    for u in (-0.08, 0.08):
        handle = _project_points(
            [
                _wall_plane_point(spec, u, -0.12, normal_offset=0.152),
                _wall_plane_point(spec, u, 0.12, normal_offset=0.152),
            ],
            camera,
            frame,
        )
        draw_line(draw, handle[0], handle[1], fill=(219, 184, 96), width=3)
        bboxes.append(_points_bbox(handle))
    top_highlight = _project_points(
        [
            _wall_plane_point(spec, -0.38, 0.32, normal_offset=0.152),
            _wall_plane_point(spec, 0.38, 0.32, normal_offset=0.152),
        ],
        camera,
        frame,
    )
    draw_line(draw, top_highlight[0], top_highlight[1], fill=(182, 158, 122), width=2)
    bboxes.append(_points_bbox(top_highlight))
    return bbox_union(*bboxes)


def _draw_wall_poster_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    bboxes: List[List[float]] = []
    outer = _wall_rect_points(spec, normal_offset=0.026)
    bboxes.append(_draw_poly(draw, outer, camera=camera, frame=frame, fill=(196, 106, 105), outline=(96, 54, 52), width=2))
    top_band = _wall_quad_points(spec, -0.40, 0.25, 0.40, 0.39, normal_offset=0.031)
    bottom_band = _wall_quad_points(spec, -0.40, -0.39, 0.40, -0.26, normal_offset=0.031)
    bboxes.append(_draw_poly(draw, top_band, camera=camera, frame=frame, fill=(242, 191, 105), outline=(126, 70, 62), width=1))
    bboxes.append(_draw_poly(draw, bottom_band, camera=camera, frame=frame, fill=(78, 124, 155), outline=(75, 62, 68), width=1))
    diamond = [
        _wall_plane_point(spec, 0.0, 0.18, normal_offset=0.034),
        _wall_plane_point(spec, 0.24, 0.00, normal_offset=0.034),
        _wall_plane_point(spec, 0.0, -0.18, normal_offset=0.034),
        _wall_plane_point(spec, -0.24, 0.00, normal_offset=0.034),
    ]
    bboxes.append(_draw_poly(draw, diamond, camera=camera, frame=frame, fill=(235, 229, 202), outline=(96, 54, 52), width=2))
    return bbox_union(*bboxes)


def _draw_wall_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    object_type = str(spec["object_type"])
    if object_type == "tv":
        return _draw_wall_tv_object(draw, spec, camera=camera, frame=frame)
    if object_type == "clock":
        bbox = _draw_wall_disc(draw, spec, camera=camera, frame=frame, fill=(244, 239, 218), outline=(46, 56, 66))
        center = _project_points([_wall_plane_point(spec, 0.0, 0.0, normal_offset=0.038)], camera, frame)[0]
        hand_up = _project_points([_wall_plane_point(spec, 0.0, 0.24, normal_offset=0.039)], camera, frame)[0]
        hand_right = _project_points([_wall_plane_point(spec, 0.18, -0.10, normal_offset=0.039)], camera, frame)[0]
        for index in range(12):
            angle = math.pi * 0.5 - index * math.tau / 12.0
            outer_u = math.cos(angle) * 0.37
            outer_v = math.sin(angle) * 0.37
            inner_scale = 0.76 if index % 3 == 0 else 0.86
            p1, p2 = _project_points(
                [
                    _wall_plane_point(spec, outer_u * inner_scale, outer_v * inner_scale, normal_offset=0.040),
                    _wall_plane_point(spec, outer_u, outer_v, normal_offset=0.040),
                ],
                camera,
                frame,
            )
            draw_line(draw, p1, p2, fill=(70, 77, 84), width=2 if index % 3 == 0 else 1)
        draw_line(draw, center, hand_up, fill=(38, 45, 54), width=2)
        draw_line(draw, center, hand_right, fill=(38, 45, 54), width=2)
        dot_r = 2.4
        draw.ellipse((center[0] - dot_r, center[1] - dot_r, center[0] + dot_r, center[1] + dot_r), fill=(38, 45, 54))
        return list(bbox)
    if object_type == "picture_frame":
        return _draw_wall_picture_frame_object(draw, spec, camera=camera, frame=frame)
    if object_type == "mirror":
        return _draw_wall_mirror_object(draw, spec, camera=camera, frame=frame)
    if object_type == "wall_shelf":
        return _draw_wall_shelf_object(draw, spec, camera=camera, frame=frame)
    if object_type == "wall_fan":
        return _draw_wall_fan_object(draw, spec, camera=camera, frame=frame)
    if object_type == "air_conditioner":
        return _draw_wall_air_conditioner_object(draw, spec, camera=camera, frame=frame)
    if object_type == "hanging_coat":
        return _draw_wall_hanging_coat_object(draw, spec, camera=camera, frame=frame)
    if object_type == "wall_lamp":
        return _draw_wall_lamp_object(draw, spec, camera=camera, frame=frame)
    if object_type == "speaker":
        return _draw_wall_speaker_object(draw, spec, camera=camera, frame=frame)
    if object_type == "wall_cabinet":
        return _draw_wall_cabinet_object(draw, spec, camera=camera, frame=frame)
    if object_type == "poster":
        return _draw_wall_poster_object(draw, spec, camera=camera, frame=frame)
    return _draw_wall_flat_object(draw, spec, camera=camera, frame=frame, fill=(160, 160, 150), trim=(70, 70, 68))

__all__ = [
    "_draw_wall_cuboid",
    "_draw_wall_tv_object",
    "_draw_wall_shelf_object",
    "_draw_wall_disc",
    "_draw_wall_fan_object",
    "_draw_wall_air_conditioner_object",
    "_draw_wall_hanging_coat_object",
    "_draw_wall_scenery",
    "_draw_wall_flat_object",
    "_draw_wall_speaker_object",
    "_draw_wall_picture_frame_object",
    "_draw_wall_mirror_object",
    "_draw_wall_lamp_object",
    "_draw_wall_cabinet_object",
    "_draw_wall_poster_object",
    "_draw_wall_object",
]
