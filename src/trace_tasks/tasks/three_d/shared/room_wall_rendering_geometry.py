"""Reusable room-wall projection and drawing helpers for 3D object renderers."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .camera_projection import project_screen
from .object_scene_primitives import draw_line, shade_rgb


def _wall_axes(wall: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    if str(wall) == "back":
        return (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)
    if str(wall) == "left":
        return (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    return (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)


def _add_vec(
    center: Sequence[float],
    horizontal_axis: Sequence[float],
    vertical_axis: Sequence[float],
    normal_axis: Sequence[float],
    *,
    hw: float,
    hh: float,
    normal_offset: float = 0.0,
) -> Tuple[float, float, float]:
    return (
        float(center[0])
        + float(horizontal_axis[0]) * float(hw)
        + float(vertical_axis[0]) * float(hh)
        + float(normal_axis[0]) * float(normal_offset),
        float(center[1])
        + float(horizontal_axis[1]) * float(hw)
        + float(vertical_axis[1]) * float(hh)
        + float(normal_axis[1]) * float(normal_offset),
        float(center[2])
        + float(horizontal_axis[2]) * float(hw)
        + float(vertical_axis[2]) * float(hh)
        + float(normal_axis[2]) * float(normal_offset),
    )


def _wall_rect_points(spec: Mapping[str, Any], *, inset: float = 0.0, normal_offset: float = 0.018) -> List[Tuple[float, float, float]]:
    wall = str(spec["wall"])
    width = max(0.05, float(spec["wall_width"]) - float(inset) * 2.0)
    height = max(0.05, float(spec["wall_height"]) - float(inset) * 2.0)
    horizontal_axis, vertical_axis, normal_axis = _wall_axes(wall)
    center = tuple(float(value) for value in spec["world_xyz"])
    return [
        _add_vec(center, horizontal_axis, vertical_axis, normal_axis, hw=-width * 0.5, hh=-height * 0.5, normal_offset=normal_offset),
        _add_vec(center, horizontal_axis, vertical_axis, normal_axis, hw=width * 0.5, hh=-height * 0.5, normal_offset=normal_offset),
        _add_vec(center, horizontal_axis, vertical_axis, normal_axis, hw=width * 0.5, hh=height * 0.5, normal_offset=normal_offset),
        _add_vec(center, horizontal_axis, vertical_axis, normal_axis, hw=-width * 0.5, hh=height * 0.5, normal_offset=normal_offset),
    ]


def _project_points(points: Sequence[Sequence[float]], camera, frame) -> List[Tuple[float, float]]:
    return [(float(project_screen(point, camera, frame)[0]), float(project_screen(point, camera, frame)[1])) for point in points]


def _points_bbox(points: Sequence[Sequence[float]], *, pad_px: float = 0.0) -> List[float]:
    return [
        round(float(min(point[0] for point in points) - pad_px), 3),
        round(float(min(point[1] for point in points) - pad_px), 3),
        round(float(max(point[0] for point in points) + pad_px), 3),
        round(float(max(point[1] for point in points) + pad_px), 3),
    ]


def _projected_polygon_bbox(points: Sequence[Sequence[float]], camera, frame, *, pad_px: float = 0.0) -> List[float]:
    return _points_bbox(_project_points(points, camera, frame), pad_px=float(pad_px))


def _draw_poly(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int] = (58, 68, 78),
    width: int = 2,
) -> List[float]:
    projected = _project_points(points, camera, frame)
    draw.polygon(projected, fill=fill)
    for index in range(len(projected)):
        next_index = index + 1
        if next_index >= len(projected):
            next_index = 0
        draw_line(draw, projected[index], projected[next_index], fill=outline, width=int(width))
    return _points_bbox(projected)


def _draw_poly_fill_only(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    projected = _project_points(points, camera, frame)
    draw.polygon(projected, fill=fill)
    return _points_bbox(projected)


def _inset_bbox(bbox: Sequence[float], x_frac: float, y_frac: float) -> List[float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return [
        round(float(x1 + width * float(x_frac)), 3),
        round(float(y1 + height * float(y_frac)), 3),
        round(float(x2 - width * float(x_frac)), 3),
        round(float(y2 - height * float(y_frac)), 3),
    ]


def _draw_screen_scenery(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    variant: str,
    outline: Tuple[int, int, int] = (74, 48, 37),
) -> None:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 - x1 < 10.0 or y2 - y1 < 10.0:
        draw.rectangle((x1, y1, x2, y2), fill=(204, 217, 210), outline=outline, width=1)
        return
    width = x2 - x1
    height = y2 - y1
    horizon = y1 + height * 0.56
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
    draw.rectangle((x1, y1, x2, horizon), fill=sky)
    draw.rectangle((x1, horizon, x2, y2), fill=ground)
    if str(variant) == "city":
        for index in range(4):
            bx1 = x1 + width * (0.12 + index * 0.18)
            bx2 = bx1 + width * 0.11
            by1 = horizon - height * (0.18 + 0.07 * (index % 2))
            draw.rectangle((bx1, by1, bx2, horizon), fill=accent)
    elif str(variant) == "forest":
        for index in range(4):
            cx = x1 + width * (0.18 + index * 0.18)
            draw.polygon(
                [(cx, horizon - height * 0.28), (cx - width * 0.08, horizon), (cx + width * 0.08, horizon)],
                fill=accent,
            )
    else:
        draw.polygon(
            [(x1 + width * 0.05, horizon), (x1 + width * 0.30, y1 + height * 0.22), (x1 + width * 0.56, horizon)],
            fill=accent,
        )
        draw.polygon(
            [(x1 + width * 0.40, horizon), (x1 + width * 0.68, y1 + height * 0.26), (x1 + width * 0.96, horizon)],
            fill=shade_rgb(accent, 0.86),
        )
        if str(variant) in {"lake", "sunset"}:
            cx = x1 + width * 0.76
            cy = y1 + height * 0.22
            radius = min(width, height) * 0.09
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(247, 221, 111))
    draw.rectangle((x1, y1, x2, y2), outline=outline, width=1)


def _wall_plane_point(spec: Mapping[str, Any], u: float, v: float, *, normal_offset: float = 0.03) -> Tuple[float, float, float]:
    horizontal_axis, vertical_axis, normal_axis = _wall_axes(str(spec["wall"]))
    center = tuple(float(value) for value in spec["world_xyz"])
    return _add_vec(
        center,
        horizontal_axis,
        vertical_axis,
        normal_axis,
        hw=float(spec["wall_width"]) * float(u),
        hh=float(spec["wall_height"]) * float(v),
        normal_offset=float(normal_offset),
    )


def _wall_physical_point(spec: Mapping[str, Any], h: float, v: float, *, normal_offset: float = 0.03) -> Tuple[float, float, float]:
    horizontal_axis, vertical_axis, normal_axis = _wall_axes(str(spec["wall"]))
    center = tuple(float(value) for value in spec["world_xyz"])
    return _add_vec(
        center,
        horizontal_axis,
        vertical_axis,
        normal_axis,
        hw=float(h),
        hh=float(v),
        normal_offset=float(normal_offset),
    )


def _wall_quad_points(
    spec: Mapping[str, Any],
    u_start: float,
    v_start: float,
    u_end: float,
    v_end: float,
    *,
    normal_offset: float = 0.03,
) -> List[Tuple[float, float, float]]:
    return [
        _wall_plane_point(spec, u_start, v_start, normal_offset=normal_offset),
        _wall_plane_point(spec, u_end, v_start, normal_offset=normal_offset),
        _wall_plane_point(spec, u_end, v_end, normal_offset=normal_offset),
        _wall_plane_point(spec, u_start, v_end, normal_offset=normal_offset),
    ]


def _face_camera_distance_sq(face: Sequence[Sequence[float]], camera) -> float:
    center = tuple(sum(float(point[index]) for point in face) / float(len(face)) for index in range(3))
    return sum((float(center[index]) - float(camera.camera_position[index])) ** 2 for index in range(3))


__all__ = [
    "_wall_axes",
    "_add_vec",
    "_wall_rect_points",
    "_project_points",
    "_points_bbox",
    "_projected_polygon_bbox",
    "_draw_poly",
    "_draw_poly_fill_only",
    "_inset_bbox",
    "_draw_screen_scenery",
    "_wall_plane_point",
    "_wall_physical_point",
    "_wall_quad_points",
    "_face_camera_distance_sq",
]
