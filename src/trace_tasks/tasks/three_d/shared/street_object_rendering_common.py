"""Reusable street-object rendering constants and helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .camera_projection import (
    project_screen as _project_screen,
    project_xy as _project_xy,
)
from .color_variation import resolve_three_d_object_fill_rgb
from .object_resources import (
    BUILDING_STYLE_BASE_COLORS,
    BUILDING_STYLE_DISPLAY_NAMES,
    STREET_OBJECT_BASE_DIMENSIONS,
    STREET_OBJECT_COLORS,
    STREET_OBJECT_NAMES,
    STREET_RADIAL_OBJECT_TYPES,
    STREET_VEHICLE_OBJECT_TYPES,
)
from .object_scene_rendering import _bbox_union, _draw_line


VEHICLE_OBJECT_TYPES = set(STREET_VEHICLE_OBJECT_TYPES)
PEDESTRIAN_OBJECT_TYPES = {"pedestrian", "male_pedestrian", "female_pedestrian"}
STREET_BUILDING_CONTEXT_OBJECT_TYPES = {"building", "store", "office_building"}
STREET_FIXED_BUILDING_STYLE_BY_OBJECT_TYPE = {
    "store": "market_shop",
    "office_building": "office_glass",
}
STREET_FULL_BLEED_FALLBACK_EXTENT_MULTIPLIER = 3.2


def _street_object_name(object_type: str) -> str:
    return str(STREET_OBJECT_NAMES.get(str(object_type), str(object_type).replace("_", " ")))


def _street_object_fill_rgb(spec: Mapping[str, Any]) -> Tuple[int, int, int]:
    object_type = str(spec.get("object_type", ""))
    if object_type in STREET_BUILDING_CONTEXT_OBJECT_TYPES:
        base = BUILDING_STYLE_BASE_COLORS.get(
            str(spec.get("building_style", "")),
            STREET_OBJECT_COLORS.get(object_type, (116, 126, 139)),
        )
    else:
        base = STREET_OBJECT_COLORS.get(object_type, (116, 126, 139))
    if bool(spec.get("is_answer_candidate", False)):
        strength = 0.18
        salt = "street.candidate"
    elif object_type in {*STREET_BUILDING_CONTEXT_OBJECT_TYPES, "bench", "street_sign"}:
        strength = 0.28
        salt = "street.context.strong"
    else:
        strength = 0.16
        salt = "street.context"
    return resolve_three_d_object_fill_rgb(
        spec,
        base_rgb=base,
        salt=salt,
        variation_strength=float(strength),
    )


def _base_street_object_dimensions(object_type: str) -> Tuple[float, float, float]:
    return tuple(float(value) for value in STREET_OBJECT_BASE_DIMENSIONS.get(str(object_type), (0.6, 0.44, 0.42)))


def _fixed_building_style_for_street_object(object_type: str) -> str | None:
    return STREET_FIXED_BUILDING_STYLE_BY_OBJECT_TYPE.get(str(object_type))


def _apply_street_building_style(spec: Mapping[str, Any], *, style: str | None = None) -> Dict[str, Any]:
    updated = dict(spec)
    object_type = str(updated.get("object_type", ""))
    if object_type not in STREET_BUILDING_CONTEXT_OBJECT_TYPES:
        return updated
    resolved_style = str(style or _fixed_building_style_for_street_object(object_type) or updated.get("building_style") or "concrete_midrise")
    updated.update(
        {
            "building_style": str(resolved_style),
            "building_style_name": str(BUILDING_STYLE_DISPLAY_NAMES.get(str(resolved_style), _street_object_name(object_type))),
        }
    )
    return updated


def _dimensions_for_orientation(
    object_type: str,
    *,
    orientation_axis: str,
    scale: float,
) -> Tuple[float, float, float]:
    length, width, height = _base_street_object_dimensions(str(object_type))
    if str(object_type) in STREET_RADIAL_OBJECT_TYPES:
        return (round(float(width * scale), 4), round(float(width * scale), 4), round(float(height * scale), 4))
    if str(orientation_axis) == "y":
        return (round(float(width * scale), 4), round(float(length * scale), 4), round(float(height * scale), 4))
    return (round(float(length * scale), 4), round(float(width * scale), 4), round(float(height * scale), 4))


def _orientation_axis_for_xy(xy: Sequence[float]) -> str:
    return "x" if abs(float(xy[0])) >= abs(float(xy[1])) else "y"


def _missing_arm_for_layout(intersection_layout: str) -> str | None:
    layout = str(intersection_layout)
    if layout.startswith("t_missing_"):
        return layout.removeprefix("t_missing_")
    return None


def _arm_is_present(intersection_layout: str, arm: str) -> bool:
    return _missing_arm_for_layout(str(intersection_layout)) != str(arm)


def _draw_shadow(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> None:
    return None


def _screen_points_bbox(points: Sequence[Sequence[float]], *, pad_px: float = 0.0) -> List[float]:
    return [
        round(float(min(point[0] for point in points) - float(pad_px)), 3),
        round(float(min(point[1] for point in points) - float(pad_px)), 3),
        round(float(max(point[0] for point in points) + float(pad_px)), 3),
        round(float(max(point[1] for point in points) + float(pad_px)), 3),
    ]


def _draw_projected_limb(
    draw: ImageDraw.ImageDraw,
    start_world: Sequence[float],
    end_world: Sequence[float],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
    width_px: int,
) -> List[float]:
    start = _project_xy(start_world, camera, frame)
    end = _project_xy(end_world, camera, frame)
    width = max(2, int(width_px))
    radius = float(width) * 0.55
    draw.line([start, end], fill=fill, width=width)
    draw.ellipse(
        (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
        fill=fill,
    )
    draw.ellipse(
        (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
        fill=fill,
    )
    return [
        round(float(min(start[0], end[0]) - radius), 3),
        round(float(min(start[1], end[1]) - radius), 3),
        round(float(max(start[0], end[0]) + radius), 3),
        round(float(max(start[1], end[1]) + radius), 3),
    ]


def _screen_line_bbox(p1: Sequence[float], p2: Sequence[float], *, pad_px: float) -> List[float]:
    return [
        round(float(min(float(p1[0]), float(p2[0])) - pad_px), 3),
        round(float(min(float(p1[1]), float(p2[1])) - pad_px), 3),
        round(float(max(float(p1[0]), float(p2[0])) + pad_px), 3),
        round(float(max(float(p1[1]), float(p2[1])) + pad_px), 3),
    ]


def _screen_rect_bbox(center: Sequence[float], *, width_px: float, height_px: float) -> List[float]:
    half_w = float(width_px) * 0.5
    half_h = float(height_px) * 0.5
    return [
        round(float(center[0]) - half_w, 3),
        round(float(center[1]) - half_h, 3),
        round(float(center[0]) + half_w, 3),
        round(float(center[1]) + half_h, 3),
    ]


def _draw_screen_pole(
    draw: ImageDraw.ImageDraw,
    base: Sequence[float],
    top: Sequence[float],
    *,
    fill: Tuple[int, int, int],
    width_px: int,
) -> List[float]:
    _draw_line(draw, base, top, fill=(23, 27, 31), width=max(1, int(width_px) + 2))
    _draw_line(draw, base, top, fill=fill, width=max(1, int(width_px)))
    radius = max(2.0, float(width_px) * 0.78)
    foot_bbox = [
        float(base[0]) - radius * 1.55,
        float(base[1]) - radius * 0.45,
        float(base[0]) + radius * 1.55,
        float(base[1]) + radius * 0.45,
    ]
    draw.ellipse(foot_bbox, fill=(52, 56, 60), outline=(23, 27, 31), width=1)
    return _bbox_union(_screen_line_bbox(base, top, pad_px=float(width_px) + 1.0), foot_bbox)


def _stable_palette_index(value: str, modulo: int) -> int:
    if int(modulo) <= 0:
        return 0
    return sum(ord(char) for char in str(value)) % int(modulo)


def _upright_screen_basis(spec: Mapping[str, Any], camera, frame) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
    x, y, _base_z = (float(value) for value in spec["base_xyz"])
    height = float(spec["dimensions_xyz"][2])
    base = _project_xy((x, y, 0.0), camera, frame)
    top = _project_xy((x, y, height), camera, frame)
    up = (float(top[0]) - float(base[0]), float(top[1]) - float(base[1]))
    height_px = max(1.0, math.hypot(up[0], up[1]))
    up_unit = (up[0] / height_px, up[1] / height_px)
    side_unit = (-up_unit[1], up_unit[0])
    return (float(base[0]), float(base[1])), up_unit, side_unit, height_px


__all__ = [
    "VEHICLE_OBJECT_TYPES",
    "PEDESTRIAN_OBJECT_TYPES",
    "STREET_BUILDING_CONTEXT_OBJECT_TYPES",
    "STREET_FIXED_BUILDING_STYLE_BY_OBJECT_TYPE",
    "STREET_FULL_BLEED_FALLBACK_EXTENT_MULTIPLIER",
    "_street_object_name",
    "_street_object_fill_rgb",
    "_base_street_object_dimensions",
    "_fixed_building_style_for_street_object",
    "_apply_street_building_style",
    "_dimensions_for_orientation",
    "_orientation_axis_for_xy",
    "_missing_arm_for_layout",
    "_arm_is_present",
    "_draw_shadow",
    "_screen_points_bbox",
    "_draw_projected_limb",
    "_screen_line_bbox",
    "_screen_rect_bbox",
    "_draw_screen_pole",
    "_stable_palette_index",
    "_upright_screen_basis",
]
