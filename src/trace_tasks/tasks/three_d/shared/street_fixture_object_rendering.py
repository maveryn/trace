"""Street fixture and signage rendering helpers."""

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

def _draw_fire_hydrant_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    x, y, _z = (float(value) for value in spec["base_xyz"])
    base = _project_xy((x, y, 0.0), camera, frame)
    top = _project_xy((x, y, height), camera, frame)
    up = (float(top[0]) - float(base[0]), float(top[1]) - float(base[1]))
    height_px = max(1.0, math.hypot(up[0], up[1]))
    up_unit = (up[0] / height_px, up[1] / height_px)
    side_unit = (-up_unit[1], up_unit[0])
    hydrant_h = max(48.0, min(86.0, height_px * 1.05))
    body_w = max(7, min(13, int(round(hydrant_h * 0.16))))

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            float(base[0]) + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            float(base[1]) + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    outline = (72, 38, 34)
    bboxes: List[List[float]] = []
    base_center = p(0.0, hydrant_h * 0.10)
    base_bbox = [
        base_center[0] - body_w * 1.45,
        base_center[1] - hydrant_h * 0.045,
        base_center[0] + body_w * 1.45,
        base_center[1] + hydrant_h * 0.045,
    ]
    draw.ellipse(tuple(base_bbox), fill=_shade(fill, 0.72), outline=outline, width=1)
    bboxes.append(base_bbox)

    body_bottom = p(0.0, hydrant_h * 0.14)
    body_top = p(0.0, hydrant_h * 0.75)
    _draw_line(draw, body_bottom, body_top, fill=outline, width=body_w + 4)
    _draw_line(draw, body_bottom, body_top, fill=fill, width=body_w)
    bboxes.append(_screen_line_bbox(body_bottom, body_top, pad_px=float(body_w + 3)))

    center = p(0.0, hydrant_h * 0.46)
    radius = max(3.0, min(6.0, hydrant_h * 0.055))
    for lateral in (-hydrant_h * 0.23, hydrant_h * 0.23):
        nozzle = p(lateral, hydrant_h * 0.46)
        _draw_line(draw, center, nozzle, fill=_shade(fill, 0.82), width=max(3, body_w // 2))
        nozzle_bbox = [nozzle[0] - radius, nozzle[1] - radius, nozzle[0] + radius, nozzle[1] + radius]
        draw.ellipse(tuple(nozzle_bbox), fill=_tint(fill, 0.18), outline=outline, width=1)
        bboxes.extend([_screen_line_bbox(center, nozzle, pad_px=3.0), nozzle_bbox])

    cap_center = p(0.0, hydrant_h * 0.81)
    cap_bbox = [
        cap_center[0] - body_w * 0.95,
        cap_center[1] - hydrant_h * 0.055,
        cap_center[0] + body_w * 0.95,
        cap_center[1] + hydrant_h * 0.055,
    ]
    draw.ellipse(tuple(cap_bbox), fill=_tint(fill, 0.12), outline=outline, width=1)
    bboxes.append(cap_bbox)

    stripe_left = p(-body_w * 0.70, hydrant_h * 0.57)
    stripe_right = p(body_w * 0.70, hydrant_h * 0.57)
    _draw_line(draw, stripe_left, stripe_right, fill=(244, 218, 132), width=3)
    bboxes.append(_screen_line_bbox(stripe_left, stripe_right, pad_px=3.0))
    return _bbox_union(*bboxes)

def _draw_trash_bin_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.05), dimensions_xyz=(width * 0.88, depth * 0.88, height * 0.76))
    lid = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.80), dimensions_xyz=(width, depth, height * 0.12))
    bbox = _bbox_union(
        _draw_cylinder_object(draw, body, camera=camera, frame=frame, fill=fill),
        _draw_cylinder_object(draw, lid, camera=camera, frame=frame, fill=_shade(fill, 0.70)),
    )
    center = _project_xy(spec["world_xyz"], camera, frame)
    handle_w = max(14.0, min(24.0, 17.0 * (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.35))
    handle_bbox = [center[0] - handle_w * 0.5, center[1] - handle_w * 0.18, center[0] + handle_w * 0.5, center[1] + handle_w * 0.18]
    draw.rectangle(handle_bbox, fill=_shade(fill, 0.48), outline=(31, 39, 35), width=1)
    body_bbox = _object_screen_bbox(body, camera, frame, pad_px=0.0)
    x0, y0, x1, y1 = (float(value) for value in body_bbox)
    slat_bboxes: List[List[float]] = []
    for frac in (0.34, 0.50, 0.66):
        x_line = x0 + (x1 - x0) * frac
        _draw_line(draw, (x_line, y0 + (y1 - y0) * 0.20), (x_line, y1 - (y1 - y0) * 0.12), fill=_shade(fill, 0.54), width=2)
        slat_bboxes.append([x_line - 2.0, y0 + (y1 - y0) * 0.20, x_line + 2.0, y1 - (y1 - y0) * 0.12])
    return _bbox_union(bbox, handle_bbox, *slat_bboxes)

def _draw_mailbox_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.18), dimensions_xyz=(width, depth, height * 0.50))
    roof_dims = (width, depth * 0.90, height * 0.22) if axis == "x" else (width * 0.90, depth, height * 0.22)
    roof = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.62), dimensions_xyz=roof_dims)
    post = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.16, depth * 0.16, height * 0.38))
    bbox = _bbox_union(
        _draw_box_object(draw, post, camera=camera, frame=frame, fill=(76, 70, 62)),
        _draw_box_object(draw, body, camera=camera, frame=frame, fill=fill),
        _draw_box_object(draw, roof, camera=camera, frame=frame, fill=_tint(fill, 0.08)),
    )
    x, y, _z = (float(value) for value in spec["base_xyz"])
    flag_world = (x + width * 0.56, y, height * 0.62) if axis == "x" else (x, y + depth * 0.56, height * 0.62)
    flag_center = _project_xy(flag_world, camera, frame)
    scale = max(0.75, min(1.25, (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.35))
    flag_bbox = _screen_rect_bbox(flag_center, width_px=18.0 * scale, height_px=11.0 * scale)
    draw.rectangle(flag_bbox, fill=(202, 55, 55), outline=(80, 29, 29), width=1)
    return _bbox_union(bbox, flag_bbox)

def _draw_construction_barrier_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    axis = str(spec.get("orientation_axis", "x"))
    if axis == "x":
        rail_dims = (width, depth * 0.34, height * 0.14)
        leg_dims = (width * 0.10, depth * 0.44, height * 0.58)
        leg_offsets = [(-width * 0.36, 0.0, height * 0.08), (width * 0.36, 0.0, height * 0.08)]
    else:
        rail_dims = (width * 0.34, depth, height * 0.14)
        leg_dims = (width * 0.44, depth * 0.10, height * 0.58)
        leg_offsets = [(0.0, -depth * 0.36, height * 0.08), (0.0, depth * 0.36, height * 0.08)]
    rail_parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.28), dimensions_xyz=rail_dims),
        _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.56), dimensions_xyz=rail_dims),
    ]
    leg_parts = [_sub_box_spec(spec, offset_xyz=offset, dimensions_xyz=leg_dims) for offset in leg_offsets]
    bboxes: List[List[float]] = []
    for part in leg_parts:
        leg_bbox = _draw_box_object(draw, part, camera=camera, frame=frame, fill=_shade(fill, 0.60))
        draw.rectangle(tuple(leg_bbox), outline=(88, 44, 24), width=1)
        bboxes.append(leg_bbox)
    rail_bboxes: List[List[float]] = []
    for part in rail_parts:
        rail_bbox = _draw_box_object(draw, part, camera=camera, frame=frame, fill=(226, 105, 42))
        draw.rectangle(tuple(rail_bbox), outline=(94, 47, 25), width=2)
        rail_bboxes.append(rail_bbox)
        bboxes.append(rail_bbox)
    stripe_color = (246, 242, 216)
    stripe_outline = (156, 84, 42)
    for rail_bbox in rail_bboxes:
        x0, y0, x1, y1 = (float(value) for value in rail_bbox)
        rail_w = max(1.0, x1 - x0)
        rail_h = max(1.0, y1 - y0)
        if rail_w >= rail_h:
            for start_frac in (0.10, 0.36, 0.62):
                rect = [
                    x0 + rail_w * start_frac,
                    y0 + rail_h * 0.20,
                    x0 + rail_w * (start_frac + 0.16),
                    y1 - rail_h * 0.20,
                ]
                draw.rectangle(rect, fill=stripe_color, outline=stripe_outline, width=1)
                _draw_line(draw, (rect[0], rect[3]), (rect[2], rect[1]), fill=(156, 84, 42), width=1)
                bboxes.append(rect)
        else:
            for start_frac in (0.10, 0.36, 0.62):
                rect = [
                    x0 + rail_w * 0.20,
                    y0 + rail_h * start_frac,
                    x1 - rail_w * 0.20,
                    y0 + rail_h * (start_frac + 0.16),
                ]
                draw.rectangle(rect, fill=stripe_color, outline=stripe_outline, width=1)
                _draw_line(draw, (rect[0], rect[3]), (rect[2], rect[1]), fill=(156, 84, 42), width=1)
                bboxes.append(rect)
    return _bbox_union(*bboxes)

def _draw_road_barrel_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height))
    bbox = _draw_cylinder_object(draw, body, camera=camera, frame=frame, fill=fill)
    center = _project_xy(spec["world_xyz"], camera, frame)
    scale = max(0.72, min(1.22, (8.0 / max(2.2, float(spec["camera_distance"]))) ** 0.38))
    band_w = max(18.0, min(34.0, 24.0 * scale))
    band_h = max(5.0, min(9.0, 6.5 * scale))
    bands: List[List[float]] = []
    for offset in (-0.22, 0.18):
        band_center = (float(center[0]), float(center[1]) + offset * 42.0 * scale)
        band_bbox = _screen_rect_bbox(band_center, width_px=band_w, height_px=band_h)
        draw.rectangle(band_bbox, fill=(246, 238, 211), outline=(145, 83, 45), width=1)
        bands.append(band_bbox)
    return _bbox_union(bbox, *bands)

def _draw_traffic_cone_object(
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
    height_px = max(1.0, math.hypot(up[0], up[1]))
    up_unit = (up[0] / height_px, up[1] / height_px)
    side_unit = (-up_unit[1], up_unit[0])
    cone_h = max(34.0, min(70.0, height_px * 1.12))
    base_center = (float(base[0]), float(base[1]))

    def p(lateral: float, upward: float) -> Tuple[float, float]:
        return (
            base_center[0] + side_unit[0] * float(lateral) + up_unit[0] * float(upward),
            base_center[1] + side_unit[1] * float(lateral) + up_unit[1] * float(upward),
        )

    outline = (93, 50, 30)
    base_w = cone_h * 0.62
    foot = [p(-base_w * 0.58, cone_h * 0.03), p(base_w * 0.58, cone_h * 0.03), p(base_w * 0.50, -cone_h * 0.05), p(-base_w * 0.50, -cone_h * 0.05)]
    draw.polygon(foot, fill=_shade(fill, 0.62), outline=outline)
    cone_points = [p(-base_w * 0.38, cone_h * 0.12), p(0.0, cone_h * 0.94), p(base_w * 0.38, cone_h * 0.12)]
    draw.polygon(cone_points, fill=fill, outline=outline)
    bboxes: List[List[float]] = [_screen_points_bbox(foot), _screen_points_bbox(cone_points)]
    for lower, upper, scale in ((0.28, 0.37, 0.56), (0.55, 0.63, 0.30)):
        band = [p(-base_w * scale, cone_h * lower), p(base_w * scale, cone_h * lower), p(base_w * scale * 0.78, cone_h * upper), p(-base_w * scale * 0.78, cone_h * upper)]
        draw.polygon(band, fill=(246, 238, 211), outline=(150, 86, 43))
        bboxes.append(_screen_points_bbox(band))
    return _bbox_union(*bboxes)

def _draw_traffic_light_context_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _base_z = (float(value) for value in spec["base_xyz"])
    _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    distance = max(2.2, float(spec["camera_distance"]))
    scale = max(0.78, min(1.22, (8.0 / distance) ** 0.36))
    base = _project_xy((x, y, 0.0), camera, frame)
    top = _project_xy((x, y, height * 0.98), camera, frame)
    head_center = _project_xy((x, y, height * 0.86), camera, frame)
    pole_bbox = _draw_screen_pole(
        draw,
        base,
        top,
        fill=(78, 82, 86),
        width_px=max(3, int(round(4.5 * scale))),
    )
    box_w = max(15.0, min(24.0, 18.5 * scale))
    box_h = max(34.0, min(48.0, 39.0 * scale))
    signal_bbox = _screen_rect_bbox(head_center, width_px=box_w, height_px=box_h)
    draw.rectangle(signal_bbox, fill=_shade(fill, 0.72), outline=(18, 21, 25), width=2)
    lens_radius = max(3.3, min(5.8, box_w * 0.26))
    lens_x = float(head_center[0])
    lens_gap = box_h * 0.25
    for offset, color in ((-lens_gap, (199, 45, 42)), (0.0, (235, 185, 48)), (lens_gap, (62, 159, 87))):
        cy = float(head_center[1]) + float(offset)
        draw.ellipse(
            (
                lens_x - lens_radius,
                cy - lens_radius,
                lens_x + lens_radius,
                cy + lens_radius,
            ),
            fill=color,
            outline=(18, 21, 25),
            width=1,
        )
    cap_y = float(signal_bbox[1]) - max(1.5, 2.0 * scale)
    cap_bbox = [
        float(signal_bbox[0]) - 1.0,
        cap_y,
        float(signal_bbox[2]) + 1.0,
        cap_y + max(3.0, 4.0 * scale),
    ]
    draw.rectangle(cap_bbox, fill=_shade(fill, 0.54), outline=(18, 21, 25), width=1)
    return _bbox_union(pole_bbox, signal_bbox, cap_bbox)

def _draw_street_sign_context_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    x, y, _base_z = (float(value) for value in spec["base_xyz"])
    _width, _depth, height = (float(value) for value in spec["dimensions_xyz"])
    distance = max(2.2, float(spec["camera_distance"]))
    scale = max(0.78, min(1.20, (8.0 / distance) ** 0.36))
    base = _project_xy((x, y, 0.0), camera, frame)
    top = _project_xy((x, y, height * 0.96), camera, frame)
    sign_center = _project_xy((x, y, height * 0.79), camera, frame)
    pole_bbox = _draw_screen_pole(
        draw,
        base,
        top,
        fill=(79, 83, 87),
        width_px=max(3, int(round(4.0 * scale))),
    )
    panel_w = max(28.0, min(46.0, 35.0 * scale))
    panel_h = max(16.0, min(23.0, 18.5 * scale))
    panel_bbox = _screen_rect_bbox(sign_center, width_px=panel_w, height_px=panel_h)
    draw.rectangle(panel_bbox, fill=fill, outline=(241, 246, 241), width=2)
    line_y = float(sign_center[1])
    draw.line(
        (
            float(panel_bbox[0]) + panel_w * 0.20,
            line_y,
            float(panel_bbox[2]) - panel_w * 0.20,
            line_y,
        ),
        fill=(237, 244, 237),
        width=max(1, int(round(1.6 * scale))),
    )
    return _bbox_union(pole_bbox, panel_bbox)


__all__ = [
    '_draw_fire_hydrant_object',
    '_draw_trash_bin_object',
    '_draw_mailbox_object',
    '_draw_construction_barrier_object',
    '_draw_road_barrel_object',
    '_draw_traffic_cone_object',
    '_draw_traffic_light_context_object',
    '_draw_street_sign_context_object',
]
