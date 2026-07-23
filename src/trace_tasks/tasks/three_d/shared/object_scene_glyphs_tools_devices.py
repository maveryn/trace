"""Tool, device, and toy glyphs for shared three_d object scenes."""

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
    _project_face,
    _project_local_xy_point,
    _project_local_xy_rect,
    _radius_px_for_object,
    _shade,
    _star_footprint_points,
    _sub_box_spec,
    _tint,
    _upright_profile_world_points,
    _upright_screen_points,
)


def _draw_pencil_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    shape_type = str(spec.get("shape_type", "pencil"))
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    if shape_type == "highlighter":
        body = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.04, 0.0), dimensions_xyz=(width * 0.64, depth * 0.70, height * 0.68))
        cap = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.48, 0.0), dimensions_xyz=(width * 0.68, depth * 0.18, height * 0.72))
        chisel = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.40, 0.0), dimensions_xyz=(width * 0.50, depth * 0.16, height * 0.44))
        body_bbox = _draw_box_object(draw, body, camera=camera, frame=frame, fill=_tint(fill, 0.12))
        cap_bbox = _draw_box_object(draw, cap, camera=camera, frame=frame, fill=(45, 54, 67))
        chisel_bbox = _draw_box_object(draw, chisel, camera=camera, frame=frame, fill=_shade(fill, 0.72))
        x0, y0, x1, y1 = (float(value) for value in body_bbox)
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        bboxes = [body_bbox, cap_bbox, chisel_bbox]
        bboxes.append(_draw_detail_line(draw, [(x0 + w * 0.18, y0 + h * 0.23), (x1 - w * 0.14, y0 + h * 0.17)], fill=(250, 254, 242), width=2))
        for offset in (0.18, 0.82):
            bboxes.append(_draw_detail_line(draw, [(x0 + w * 0.10, y0 + h * offset), (x1 - w * 0.10, y0 + h * (offset - 0.04))], fill=_shade(fill, 0.62), width=1))
        return _bbox_union(*bboxes)
    if shape_type == "pen":
        center, direction, normal, length_px = _diagonal_ground_axis_basis(
            spec,
            camera,
            frame,
            center_height_frac=0.58,
            length_scale=0.84,
            min_length_px=52.0,
            max_length_px=92.0,
        )
        radius = max(3.8, min(6.0, length_px * 0.060))

        def point_at(axis_frac: float) -> Tuple[float, float]:
            return (
                center[0] + direction[0] * length_px * float(axis_frac),
                center[1] + direction[1] * length_px * float(axis_frac),
            )

        def strip_between(a: Sequence[float], b: Sequence[float], ra: float, rb: float | None = None) -> List[Tuple[float, float]]:
            rb = float(ra if rb is None else rb)
            return [
                (float(a[0]) + normal[0] * float(ra), float(a[1]) + normal[1] * float(ra)),
                (float(b[0]) + normal[0] * rb, float(b[1]) + normal[1] * rb),
                (float(b[0]) - normal[0] * rb, float(b[1]) - normal[1] * rb),
                (float(a[0]) - normal[0] * float(ra), float(a[1]) - normal[1] * float(ra)),
            ]

        bboxes: List[List[float]] = []
        rear = point_at(-0.50)
        cap_end = point_at(-0.34)
        rear_plug_end = point_at(-0.44)
        body_end = point_at(0.18)
        grip_end = point_at(0.34)
        metal_end = point_at(0.46)
        nib = point_at(0.52)
        body_poly = strip_between(cap_end, body_end, radius)
        cap_poly = strip_between(rear, cap_end, radius * 1.05)
        rear_plug_poly = strip_between(rear, rear_plug_end, radius * 1.08)
        grip_poly = strip_between(body_end, grip_end, radius * 0.94, radius * 0.72)
        metal_poly = strip_between(grip_end, metal_end, radius * 0.70, radius * 0.38)
        nib_poly = strip_between(metal_end, nib, radius * 0.42, radius * 0.12)
        for poly, color, outline in (
            (body_poly, _tint(fill, 0.18), (35, 43, 52)),
            (cap_poly, (64, 73, 85), (28, 34, 42)),
            (rear_plug_poly, (28, 33, 40), (18, 22, 27)),
            (grip_poly, (42, 48, 56), (22, 26, 31)),
            (metal_poly, (184, 191, 198), (74, 83, 92)),
            (nib_poly, (42, 47, 53), (22, 26, 31)),
        ):
            draw.polygon(poly, fill=color, outline=outline)
            bboxes.append(_bbox_from_screen_points(poly))
        for frac, color in ((-0.35, (35, 43, 52)), (0.20, (85, 92, 100)), (0.27, (85, 92, 100)), (0.33, (85, 92, 100))):
            p = point_at(frac)
            line = [
                (p[0] + normal[0] * radius * 0.98, p[1] + normal[1] * radius * 0.98),
                (p[0] - normal[0] * radius * 0.98, p[1] - normal[1] * radius * 0.98),
            ]
            draw.line(line, fill=color, width=1)
            bboxes.append(_padded_screen_line_bbox(line, pad_px=1.0))
        clip_start = point_at(-0.31)
        clip_mid = point_at(-0.08)
        clip_end = point_at(0.12)
        clip_points = [
            (clip_start[0] + normal[0] * radius * 0.96, clip_start[1] + normal[1] * radius * 0.96),
            (clip_mid[0] + normal[0] * radius * 1.18, clip_mid[1] + normal[1] * radius * 1.18),
            (clip_end[0] + normal[0] * radius * 0.72, clip_end[1] + normal[1] * radius * 0.72),
        ]
        draw.line(clip_points, fill=(238, 243, 246), width=max(3, int(round(radius * 0.58))), joint="curve")
        draw.line(clip_points, fill=(70, 80, 90), width=1)
        bboxes.append(_padded_screen_line_bbox(clip_points, pad_px=max(2.0, radius * 0.55)))
        shine_start = point_at(-0.18)
        shine_end = point_at(0.14)
        shine = [
            (shine_start[0] - normal[0] * radius * 0.42, shine_start[1] - normal[1] * radius * 0.42),
            (shine_end[0] - normal[0] * radius * 0.42, shine_end[1] - normal[1] * radius * 0.42),
        ]
        draw.line(shine, fill=(247, 250, 251), width=1)
        bboxes.append(_padded_screen_line_bbox(shine, pad_px=1.0))
        nib_radius = max(1.3, radius * 0.20)
        draw.ellipse((nib[0] - nib_radius, nib[1] - nib_radius, nib[0] + nib_radius, nib[1] + nib_radius), fill=(18, 22, 26))
        bboxes.append([nib[0] - nib_radius, nib[1] - nib_radius, nib[0] + nib_radius, nib[1] + nib_radius])
        return _bbox_union(*bboxes)
    if shape_type == "marker":
        body_scale = 0.44 if shape_type == "marker" else 0.38
        body = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.04, 0.0), dimensions_xyz=(width * body_scale, depth * 0.76, height * 0.44))
        cap = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.48, 0.0), dimensions_xyz=(width * 0.48, depth * 0.16, height * 0.46))
        tip = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.43, 0.0), dimensions_xyz=(width * 0.16, depth * 0.12, height * 0.26))
        clip = _sub_box_spec(spec, offset_xyz=(width * 0.24, -depth * 0.26, height * 0.08), dimensions_xyz=(width * 0.06, depth * 0.34, height * 0.12))
        body_bbox = _draw_box_object(draw, body, camera=camera, frame=frame, fill=_tint(fill, 0.26))
        cap_bbox = _draw_box_object(draw, cap, camera=camera, frame=frame, fill=_shade(_tint(fill, 0.10), 0.68))
        tip_bbox = _draw_cone_object(draw, tip, camera=camera, frame=frame, fill=(182, 188, 194))
        clip_bbox = _draw_box_object(draw, clip, camera=camera, frame=frame, fill=(224, 229, 232))
        x0, y0, x1, y1 = (float(value) for value in body_bbox)
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        bboxes = [body_bbox, cap_bbox, tip_bbox, clip_bbox]
        bboxes.append(_draw_detail_line(draw, [(x0 + w * 0.22, y0 + h * 0.18), (x1 - w * 0.18, y0 + h * 0.13)], fill=(245, 249, 250), width=2))
        bboxes.append(_draw_detail_line(draw, [(x0 + w * 0.50, y0 + h * 0.12), (x0 + w * 0.50, y1 - h * 0.12)], fill=_shade(fill, 0.62), width=1))
        cx0, cy0, cx1, cy1 = (float(value) for value in clip_bbox)
        bboxes.append(_draw_detail_line(draw, [(cx0 + (cx1 - cx0) * 0.50, cy0 + (cy1 - cy0) * 0.10), (cx0 + (cx1 - cx0) * 0.50, cy1 - (cy1 - cy0) * 0.10)], fill=(76, 86, 96), width=1))
        tx0, ty0, tx1, ty1 = (float(value) for value in tip_bbox)
        nib = [
            (tx0 + (tx1 - tx0) * 0.42, ty1 - (ty1 - ty0) * 0.18),
            (tx1 - (tx1 - tx0) * 0.42, ty1 - (ty1 - ty0) * 0.18),
            ((tx0 + tx1) * 0.5, ty1 + (ty1 - ty0) * 0.12),
        ]
        draw.polygon(nib, fill=(35, 42, 49), outline=(22, 26, 31))
        bboxes.append(_bbox_from_screen_points(nib))
        if shape_type == "marker":
            color_band = [x0 + w * 0.26, y0 + h * 0.62, x1 - w * 0.26, y0 + h * 0.76]
            draw.rounded_rectangle(color_band, radius=max(2, int(w * 0.04)), fill=_tint(fill, 0.18), outline=_shade(fill, 0.52), width=1)
            broad_tip = [x0 + w * 0.36, y1 - h * 0.18, x1 - w * 0.36, y1 - h * 0.04]
            draw.polygon([(broad_tip[0], broad_tip[1]), (broad_tip[2], broad_tip[1]), ((broad_tip[0] + broad_tip[2]) * 0.5, broad_tip[3])], fill=(34, 40, 48))
            bboxes.extend([color_band, broad_tip])
        return _bbox_union(*bboxes)
    center, direction, normal, length_px = _diagonal_ground_axis_basis(
        spec,
        camera,
        frame,
        center_height_frac=0.56,
        length_scale=0.86,
        min_length_px=54.0,
        max_length_px=94.0,
    )
    radius = max(6.6, min(9.5, length_px * 0.085))

    def point_at(axis_frac: float, *, z_frac: float = 0.0) -> Tuple[float, float]:
        return (
            center[0] + direction[0] * length_px * float(axis_frac) - normal[0] * length_px * 0.012 * float(z_frac),
            center[1] + direction[1] * length_px * float(axis_frac) - normal[1] * length_px * 0.012 * float(z_frac),
        )

    def strip_between(a: Sequence[float], b: Sequence[float], ra: float, rb: float | None = None) -> List[Tuple[float, float]]:
        rb = float(ra if rb is None else rb)
        return [
            (float(a[0]) + normal[0] * float(ra), float(a[1]) + normal[1] * float(ra)),
            (float(b[0]) + normal[0] * rb, float(b[1]) + normal[1] * rb),
            (float(b[0]) - normal[0] * rb, float(b[1]) - normal[1] * rb),
            (float(a[0]) - normal[0] * float(ra), float(a[1]) - normal[1] * float(ra)),
        ]

    bboxes: List[List[float]] = []
    eraser_rear = point_at(-0.52)
    eraser_front = point_at(-0.42)
    ferrule_front = point_at(-0.32)
    body_front = point_at(0.32)
    wood_front = point_at(0.48)
    graphite_tip = point_at(0.55)
    parts = [
        (strip_between(eraser_rear, eraser_front, radius * 0.95), (214, 108, 126), (95, 58, 66)),
        (strip_between(eraser_front, ferrule_front, radius * 1.02), (180, 187, 194), (70, 78, 86)),
        (strip_between(ferrule_front, body_front, radius), _tint(fill, 0.16), _shade(fill, 0.42)),
        (strip_between(body_front, wood_front, radius * 0.98, radius * 0.34), (188, 139, 78), (91, 61, 35)),
        (strip_between(wood_front, graphite_tip, radius * 0.34, radius * 0.10), (34, 38, 43), (20, 24, 28)),
    ]
    for poly, color, outline in parts:
        draw.polygon(poly, fill=color, outline=outline)
        bboxes.append(_bbox_from_screen_points(poly))
    for frac, color in ((-0.40, (82, 91, 99)), (-0.36, (226, 230, 232))):
        p = point_at(frac)
        line = [
            (p[0] + normal[0] * radius * 1.08, p[1] + normal[1] * radius * 1.08),
            (p[0] - normal[0] * radius * 1.08, p[1] - normal[1] * radius * 1.08),
        ]
        draw.line(line, fill=color, width=1)
        bboxes.append(_padded_screen_line_bbox(line, pad_px=1.0))
    for offset in (-0.45, 0.0, 0.45):
        start = point_at(-0.26 + offset * 0.03, z_frac=0.02)
        end = point_at(0.24 + offset * 0.03, z_frac=0.02)
        line = [
            (start[0] + normal[0] * radius * offset, start[1] + normal[1] * radius * offset),
            (end[0] + normal[0] * radius * offset, end[1] + normal[1] * radius * offset),
        ]
        draw.line(line, fill=_shade(fill, 0.58), width=1)
        bboxes.append(_padded_screen_line_bbox(line, pad_px=1.0))
    tip_radius = max(1.1, radius * 0.18)
    draw.ellipse((graphite_tip[0] - tip_radius, graphite_tip[1] - tip_radius, graphite_tip[0] + tip_radius, graphite_tip[1] + tip_radius), fill=(22, 25, 29))
    bboxes.append([graphite_tip[0] - tip_radius, graphite_tip[1] - tip_radius, graphite_tip[0] + tip_radius, graphite_tip[1] + tip_radius])
    return _bbox_union(*bboxes)


def _draw_thumb_pin_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    head = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.22, height * 0.04), dimensions_xyz=(width * 0.88, depth * 0.34, height * 0.46))
    shaft = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.16, height * 0.06), dimensions_xyz=(width * 0.18, depth * 0.48, height * 0.22))
    point = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.44, height * 0.06), dimensions_xyz=(width * 0.22, depth * 0.18, height * 0.22))
    head_fill = _tint(fill, 0.10)
    return _bbox_union(
        _draw_cylinder_object(draw, head, camera=camera, frame=frame, fill=head_fill),
        _draw_box_object(draw, shaft, camera=camera, frame=frame, fill=(171, 178, 184)),
        _draw_cone_object(draw, point, camera=camera, frame=frame, fill=(120, 128, 136)),
    )


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def _draw_small_ellipse(draw: ImageDraw.ImageDraw, center: Sequence[float], rx: float, ry: float, *, fill: Tuple[int, int, int], outline: Tuple[int, int, int] | None = None, width: int = 1) -> List[float]:
    cx, cy = float(center[0]), float(center[1])
    bbox = [cx - float(rx), cy - float(ry), cx + float(rx), cy + float(ry)]
    draw.ellipse(bbox, fill=fill, outline=outline, width=int(width))
    return bbox


def _padded_screen_line_bbox(points: Sequence[Sequence[float]], *, pad_px: float = 1.0) -> List[float]:
    bbox = _bbox_from_screen_points(points)
    return [
        round(float(bbox[0]) - float(pad_px), 3),
        round(float(bbox[1]) - float(pad_px), 3),
        round(float(bbox[2]) + float(pad_px), 3),
        round(float(bbox[3]) + float(pad_px), 3),
    ]


def _draw_detail_line(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    fill: Tuple[int, int, int],
    width: int = 1,
    joint: str | None = None,
) -> List[float]:
    draw.line(points, fill=fill, width=int(width), joint=joint)
    return _padded_screen_line_bbox(points, pad_px=max(1.0, float(width) * 0.65))


def _draw_flat_rect_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    shape_type = str(spec.get("shape_type", "card"))
    if shape_type == "card":
        face_fill = _tint(fill, 0.06)
        edge = _shade(fill, 0.46)
        mark = (248, 250, 245)
        body = _project_local_xy_rect(spec, camera, frame, u0=0.06, v0=0.06, u1=0.94, v1=0.94, z_frac=1.0)
        draw.polygon(body, fill=face_fill, outline=edge)
        _draw_polyline(draw, body, fill=edge, width=2)
        bboxes: List[List[float]] = [_bbox_from_screen_points(body)]
        inset = _project_local_xy_rect(spec, camera, frame, u0=0.14, v0=0.14, u1=0.86, v1=0.86, z_frac=1.06)
        _draw_polyline(draw, inset, fill=mark, width=1)
        bboxes.append(_bbox_from_screen_points(inset))

        body_bbox = _bbox_from_screen_points(body)
        pip_radius = max(1.6, min(float(body_bbox[2]) - float(body_bbox[0]), float(body_bbox[3]) - float(body_bbox[1])) * 0.045)
        for u, v, scale in ((0.26, 0.24, 0.74), (0.74, 0.76, 0.74), (0.50, 0.50, 1.0)):
            center = _project_local_xy_point(spec, camera, frame, u=u, v=v, z_frac=1.08)
            bboxes.append(_draw_small_ellipse(draw, center, pip_radius * scale, pip_radius * scale, fill=mark, outline=None, width=1))
        for u, v in ((0.26, 0.76), (0.74, 0.24)):
            diamond = [
                _project_local_xy_point(spec, camera, frame, u=u, v=v - 0.035, z_frac=1.08),
                _project_local_xy_point(spec, camera, frame, u=u + 0.035, v=v, z_frac=1.08),
                _project_local_xy_point(spec, camera, frame, u=u, v=v + 0.035, z_frac=1.08),
                _project_local_xy_point(spec, camera, frame, u=u - 0.035, v=v, z_frac=1.08),
            ]
            draw.polygon(diamond, fill=mark)
            bboxes.append(_bbox_from_screen_points(diamond))
        return _bbox_union(*bboxes)

    if shape_type == "bookmark":
        fabric_fill = _tint(fill, 0.18)
        fabric_edge = _shade(fill, 0.50)
        stitch = (248, 250, 245) if sum(fill) < 420 else (78, 84, 92)
        body = _project_local_xy_rect(spec, camera, frame, u0=0.08, v0=0.05, u1=0.92, v1=0.96, z_frac=1.0)
        draw.polygon(body, fill=fabric_fill, outline=fabric_edge)
        _draw_polyline(draw, body, fill=fabric_edge, width=2)
        bboxes: List[List[float]] = [_bbox_from_screen_points(body)]
        inset = _project_local_xy_rect(spec, camera, frame, u0=0.15, v0=0.12, u1=0.85, v1=0.90, z_frac=1.06)
        _draw_polyline(draw, inset, fill=stitch, width=1)
        bboxes.append(_bbox_from_screen_points(inset))
        for u in (0.38, 0.50, 0.62):
            lace = [
                _project_local_xy_point(spec, camera, frame, u=u, v=0.16, z_frac=1.07),
                _project_local_xy_point(spec, camera, frame, u=u, v=0.84, z_frac=1.07),
            ]
            bboxes.append(_draw_detail_line(draw, lace, fill=_shade(stitch, 0.86), width=1))
        for v0, v1 in ((0.20, 0.30), (0.42, 0.52), (0.64, 0.74)):
            seam = [
                _project_local_xy_point(spec, camera, frame, u=0.24, v=v0, z_frac=1.07),
                _project_local_xy_point(spec, camera, frame, u=0.76, v=v1, z_frac=1.07),
            ]
            bboxes.append(_draw_detail_line(draw, seam, fill=_shade(stitch, 0.82), width=1))
        hole = _project_local_xy_point(spec, camera, frame, u=0.50, v=0.09, z_frac=1.08)
        body_bbox = _bbox_from_screen_points(body)
        hole_r = max(1.4, min(float(body_bbox[2]) - float(body_bbox[0]), float(body_bbox[3]) - float(body_bbox[1])) * 0.035)
        bboxes.append(_draw_small_ellipse(draw, hole, hole_r, hole_r, fill=(238, 242, 240), outline=fabric_edge, width=1))
        for u in (0.47, 0.53):
            marker = [
                _project_local_xy_point(spec, camera, frame, u=u, v=0.06, z_frac=1.08),
                _project_local_xy_point(spec, camera, frame, u=u + (0.02 if u < 0.5 else -0.02), v=-0.08, z_frac=1.08),
            ]
            bboxes.append(_draw_detail_line(draw, marker, fill=fabric_edge, width=1))
        return _bbox_union(*bboxes)

    base_fill = {
        "card": (238, 240, 232),
        "sachet": (221, 226, 216),
        "packet": _tint(fill, 0.12),
        "small_box": _tint(fill, 0.06),
        "towel": _tint(fill, 0.20),
    }.get(shape_type, _tint(fill, 0.08))
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=base_fill)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    detail = (47, 56, 66)
    bboxes: List[List[float]] = [bbox]
    if shape_type in {"sachet", "packet"}:
        for offset in (0.13, 0.87):
            line = [(x0 + w * 0.10, y0 + h * offset), (x1 - w * 0.10, y0 + h * offset)]
            draw.line(line, fill=detail, width=2)
            bboxes.append(_padded_screen_line_bbox(line, pad_px=1.0))
        crease = [(x0 + w * 0.22, y0 + h * 0.24), (x1 - w * 0.20, y1 - h * 0.22)]
        draw.line(crease, fill=_shade(base_fill, 0.70), width=1)
        bboxes.append(_padded_screen_line_bbox(crease, pad_px=1.0))
        label = [x0 + w * 0.25, y0 + h * 0.38, x1 - w * 0.25, y0 + h * 0.62]
        draw.rounded_rectangle(label, radius=max(2, int(min(w, h) * 0.05)), fill=(245, 240, 205), outline=_shade(base_fill, 0.58), width=1)
        bboxes.append(label)
        for px in (0.17, 0.83):
            crimp = [(x0 + w * px, y0 + h * 0.10), (x0 + w * px, y1 - h * 0.10)]
            bboxes.append(_draw_detail_line(draw, crimp, fill=_shade(base_fill, 0.58), width=1))
    elif shape_type == "small_box":
        box_detail = _shade(base_fill, 0.54)
        box_soft_detail = _shade(base_fill, 0.68)
        seam1 = [(x0 + w * 0.18, y0 + h * 0.34), (x1 - w * 0.12, y0 + h * 0.34)]
        seam2 = [(x0 + w * 0.50, y0 + h * 0.12), (x0 + w * 0.50, y1 - h * 0.12)]
        draw.line(seam1, fill=box_detail, width=2)
        draw.line(seam2, fill=box_soft_detail, width=1)
        bboxes.extend([_padded_screen_line_bbox(seam1, pad_px=1.0), _padded_screen_line_bbox(seam2, pad_px=1.0)])
        lid_line = [(x0 + w * 0.28, y0 + h * 0.18), (x1 - w * 0.18, y0 + h * 0.20)]
        bboxes.append(_draw_detail_line(draw, lid_line, fill=box_soft_detail, width=1))
    elif shape_type == "towel":
        for offset in (0.30, 0.52, 0.74):
            fold = [(x0 + w * 0.10, y0 + h * offset), (x1 - w * 0.10, y0 + h * (offset - 0.04))]
            draw.line(fold, fill=_shade(base_fill, 0.70), width=2)
            bboxes.append(_padded_screen_line_bbox(fold, pad_px=1.0))
        hem = [x0 + w * 0.10, y1 - h * 0.18, x1 - w * 0.10, y1 - h * 0.10]
        draw.rectangle(hem, outline=_shade(base_fill, 0.62), width=1)
        bboxes.append(hem)
        for px in (0.18, 0.30, 0.42, 0.54, 0.66, 0.78):
            fringe = [(x0 + w * px, y1 - h * 0.12), (x0 + w * (px - 0.02), y1 - h * 0.02)]
            bboxes.append(_draw_detail_line(draw, fringe, fill=_shade(base_fill, 0.58), width=1))
    else:
        inset = [x0 + w * 0.12, y0 + h * 0.13, x1 - w * 0.12, y1 - h * 0.13]
        draw.rectangle(inset, outline=detail, width=2)
        bboxes.append(inset)
        for cx, cy in ((x0 + w * 0.25, y0 + h * 0.28), (x1 - w * 0.25, y1 - h * 0.28)):
            mark = [cx - w * 0.035, cy - h * 0.035, cx + w * 0.035, cy + h * 0.035]
            draw.ellipse(mark, fill=(180, 43, 54))
            bboxes.append(mark)
    return _bbox_union(*bboxes)


def _draw_ticket_tag_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    shape_type = str(spec.get("shape_type", "ticket"))
    if shape_type == "ticket":
        face_fill = _tint(fill, 0.24)
        edge = (58, 66, 74)
        accent = _shade(fill, 0.56)
        if sum(accent) > 530:
            accent = (133, 92, 42)
        body = _project_local_xy_rect(spec, camera, frame, u0=0.06, v0=0.06, u1=0.94, v1=0.94, z_frac=1.0)
        draw.polygon(body, fill=face_fill, outline=edge)
        _draw_polyline(draw, body, fill=edge, width=2)
        bboxes: List[List[float]] = [_bbox_from_screen_points(body)]

        inset = _project_local_xy_rect(spec, camera, frame, u0=0.14, v0=0.14, u1=0.86, v1=0.86, z_frac=1.06)
        _draw_polyline(draw, inset, fill=accent, width=1)
        bboxes.append(_bbox_from_screen_points(inset))

        for start_v, end_v in ((0.18, 0.28), (0.36, 0.46), (0.54, 0.64), (0.72, 0.82)):
            segment = [
                _project_local_xy_point(spec, camera, frame, u=0.50, v=start_v, z_frac=1.08),
                _project_local_xy_point(spec, camera, frame, u=0.50, v=end_v, z_frac=1.08),
            ]
            bboxes.append(_draw_detail_line(draw, segment, fill=accent, width=1))
        for u, v in ((0.24, 0.24), (0.76, 0.24), (0.24, 0.76), (0.76, 0.76)):
            center = _project_local_xy_point(spec, camera, frame, u=u, v=v, z_frac=1.08)
            body_bbox = _bbox_from_screen_points(body)
            radius = max(1.5, min(float(body_bbox[2]) - float(body_bbox[0]), float(body_bbox[3]) - float(body_bbox[1])) * 0.030)
            bboxes.append(_draw_small_ellipse(draw, center, radius, radius, fill=accent, outline=None, width=1))
        for v in (0.28, 0.50, 0.72):
            notch = [
                _project_local_xy_point(spec, camera, frame, u=0.08, v=v, z_frac=1.09),
                _project_local_xy_point(spec, camera, frame, u=0.16, v=v, z_frac=1.09),
            ]
            bboxes.append(_draw_detail_line(draw, notch, fill=(238, 242, 240), width=2))
            notch = [
                _project_local_xy_point(spec, camera, frame, u=0.84, v=v, z_frac=1.09),
                _project_local_xy_point(spec, camera, frame, u=0.92, v=v, z_frac=1.09),
            ]
            bboxes.append(_draw_detail_line(draw, notch, fill=(238, 242, 240), width=2))
        return _bbox_union(*bboxes)

    base_fill = _tint(fill, 0.18)
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=base_fill)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    detail = (58, 66, 74)
    bboxes: List[List[float]] = [bbox]
    hole = [x0 + w * 0.38, y0 + h * 0.08, x0 + w * 0.62, y0 + h * 0.24]
    draw.ellipse(hole, fill=(238, 242, 240), outline=detail, width=2)
    string = [(x0 + w * 0.50, y0 + h * 0.16), (x0 + w * 0.76, y0 - h * 0.12)]
    draw.line(string, fill=(72, 62, 45), width=2)
    slash = [(x0 + w * 0.18, y1 - h * 0.22), (x1 - w * 0.16, y0 + h * 0.44)]
    draw.line(slash, fill=_shade(base_fill, 0.70), width=1)
    bboxes.extend([hole, _padded_screen_line_bbox(string, pad_px=1.0), _padded_screen_line_bbox(slash, pad_px=1.0)])
    return _bbox_union(*bboxes)


def _draw_puzzle_piece_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    footprint = [
        (-0.82, -0.82),
        (-0.28, -0.82),
        (-0.28, -1.10),
        (0.28, -1.10),
        (0.28, -0.82),
        (0.82, -0.82),
        (0.82, -0.28),
        (1.10, -0.28),
        (1.10, 0.28),
        (0.82, 0.28),
        (0.82, 0.82),
        (0.28, 0.82),
        (0.28, 0.54),
        (-0.28, 0.54),
        (-0.28, 0.82),
        (-0.82, 0.82),
        (-0.82, 0.28),
        (-0.56, 0.28),
        (-0.56, -0.28),
        (-0.82, -0.28),
    ]
    bbox = _draw_footprint_prism_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.10), footprint_xy=footprint)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    socket = [x0 + w * 0.08, y0 + h * 0.38, x0 + w * 0.30, y0 + h * 0.62]
    top_tab = [x0 + w * 0.38, y0 + h * 0.04, x0 + w * 0.62, y0 + h * 0.28]
    side_tab = [x1 - w * 0.28, y0 + h * 0.38, x1 - w * 0.04, y0 + h * 0.62]
    draw.arc(socket, start=90, end=270, fill=(35, 43, 54), width=2)
    draw.arc(top_tab, start=180, end=360, fill=(35, 43, 54), width=2)
    draw.arc(side_tab, start=270, end=90, fill=(35, 43, 54), width=2)
    inner = [x0 + w * 0.20, y0 + h * 0.20, x1 - w * 0.20, y1 - h * 0.20]
    draw.line([(inner[0], inner[1]), (inner[2], inner[3])], fill=_shade(fill, 0.62), width=1)
    return _bbox_union(bbox, socket, top_tab, side_tab, inner)


def _draw_candy_disc_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18))
    cx, cy = _bbox_center(bbox)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    rim = [x0 + w * 0.12, y0 + h * 0.18, x1 - w * 0.12, y1 - h * 0.18]
    inner = [x0 + w * 0.27, y0 + h * 0.32, x1 - w * 0.27, y1 - h * 0.32]
    highlight = [cx - (x1 - x0) * 0.18, cy - (y1 - y0) * 0.22, cx + (x1 - x0) * 0.08, cy - (y1 - y0) * 0.04]
    draw.ellipse(rim, outline=_shade(fill, 0.58), width=2)
    draw.ellipse(inner, outline=(245, 249, 247), width=2)
    draw.ellipse(highlight, fill=(250, 250, 244))
    return _bbox_union(bbox, rim, inner, highlight)


def _draw_cd_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=(214, 221, 224))
    cx, cy = _bbox_center(bbox)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    ring1 = [x0 + w * 0.18, y0 + h * 0.24, x1 - w * 0.18, y1 - h * 0.24]
    ring2 = [x0 + w * 0.28, y0 + h * 0.34, x1 - w * 0.28, y1 - h * 0.34]
    hole = [cx - w * 0.08, cy - h * 0.08, cx + w * 0.08, cy + h * 0.08]
    draw.ellipse(ring1, outline=(125, 145, 160), width=2)
    draw.ellipse(ring2, outline=(221, 174, 88), width=2)
    shine = [(x0 + w * 0.22, cy), (cx, y0 + h * 0.18), (x1 - w * 0.18, cy - h * 0.08)]
    draw.line(shine, fill=(129, 197, 211), width=2)
    draw.ellipse(hole, fill=floor_rgb, outline=(42, 50, 58), width=2)
    return _bbox_union(bbox, ring1, ring2, hole, _bbox_from_screen_points(shine))


def _draw_berry_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_sphere_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.06))
    cx, cy = _bbox_center(bbox)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    top = [cx - (x1 - x0) * 0.10, y0 + (y1 - y0) * 0.12, cx + (x1 - x0) * 0.10, y0 + (y1 - y0) * 0.26]
    draw.ellipse(top, fill=(59, 96, 56), outline=(34, 70, 38), width=1)
    return _bbox_union(bbox, top)


def _draw_marble_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_sphere_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.12))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    swirl = [
        (x0 + w * 0.22, y0 + h * 0.56),
        (x0 + w * 0.42, y0 + h * 0.34),
        (x0 + w * 0.62, y0 + h * 0.44),
        (x0 + w * 0.78, y0 + h * 0.25),
    ]
    draw.line(swirl, fill=(246, 248, 244), width=max(2, int(min(w, h) * 0.055)))
    draw.line(swirl, fill=_shade(fill, 0.68), width=1)
    return _bbox_union(bbox, _padded_screen_line_bbox(swirl, pad_px=2.0))


def _draw_bead_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.12))
    cx, cy = _bbox_center(bbox)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    hole = [cx - w * 0.14, cy - h * 0.11, cx + w * 0.14, cy + h * 0.11]
    rim = [cx - w * 0.23, cy - h * 0.18, cx + w * 0.23, cy + h * 0.18]
    draw.ellipse(rim, fill=_tint(fill, 0.28), outline=_shade(fill, 0.70), width=1)
    draw.ellipse(hole, fill=(42, 49, 58), outline=(238, 241, 240), width=1)
    return _bbox_union(bbox, rim, hole)


def _draw_dot_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    shine = [x0 + w * 0.24, y0 + h * 0.24, x0 + w * 0.48, y0 + h * 0.42]
    draw.ellipse(shine, fill=(255, 255, 255))
    return _bbox_union(bbox, shine)


def _draw_button_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.12))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    radius = max(1.35, min(x1 - x0, y1 - y0) * 0.036)
    holes = [
        _draw_small_ellipse(
            draw,
            _project_local_xy_point(spec, camera, frame, u=u, v=v, z_frac=1.08),
            radius,
            radius * 0.82,
            fill=(45, 52, 61),
        )
        for u, v in ((0.40, 0.40), (0.60, 0.40), (0.40, 0.60), (0.60, 0.60))
    ]
    return _bbox_union(bbox, *holes)


def _draw_plate_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    plate_fill = _tint(fill, 0.12)
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=plate_fill)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    rim = [x0 + (x1 - x0) * 0.16, y0 + (y1 - y0) * 0.22, x1 - (x1 - x0) * 0.16, y1 - (y1 - y0) * 0.22]
    draw.ellipse(rim, outline=_shade(plate_fill, 0.58), width=2)
    return _bbox_union(bbox, rim)


def _draw_bowl_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.10))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    inner = [x0 + (x1 - x0) * 0.14, y0 + (y1 - y0) * 0.16, x1 - (x1 - x0) * 0.14, y0 + (y1 - y0) * 0.52]
    draw.ellipse(inner, fill=_shade(fill, 0.55), outline=(36, 44, 54), width=2)
    return _bbox_union(bbox, inner)


def _draw_screw_object(
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
        center_height_frac=0.58,
        length_scale=0.76,
        min_length_px=58.0,
        max_length_px=96.0,
    )
    shaft_width = max(4, int(round(length_px * 0.064)))

    def point_at(axis_frac: float) -> Tuple[float, float]:
        return (
            center[0] + direction[0] * length_px * float(axis_frac),
            center[1] + direction[1] * length_px * float(axis_frac),
        )

    head_center = point_at(-0.44)
    shaft_start = point_at(-0.46)
    shaft_end = point_at(0.40)
    tip = point_at(0.50)
    shaft_points = [shaft_start, shaft_end]
    draw.line(shaft_points, fill=(63, 72, 82), width=shaft_width + 2)
    draw.line(shaft_points, fill=(166, 174, 181), width=shaft_width)
    head_rx = max(3.2, float(shaft_width) * 1.25)
    head_ry = max(2.2, float(shaft_width) * 0.78)
    head_bbox = [head_center[0] - head_rx, head_center[1] - head_ry, head_center[0] + head_rx, head_center[1] + head_ry]
    draw.ellipse(head_bbox, fill=(194, 200, 205), outline=(55, 64, 74), width=1)
    slot = [
        (head_center[0] - direction[0] * head_rx * 0.54, head_center[1] - direction[1] * head_rx * 0.54),
        (head_center[0] + direction[0] * head_rx * 0.54, head_center[1] + direction[1] * head_rx * 0.54),
    ]
    draw.line(slot, fill=(43, 50, 59), width=1)
    tip_poly = [
        (shaft_end[0] + normal[0] * shaft_width * 0.62, shaft_end[1] + normal[1] * shaft_width * 0.62),
        (shaft_end[0] - normal[0] * shaft_width * 0.62, shaft_end[1] - normal[1] * shaft_width * 0.62),
        (tip[0], tip[1]),
    ]
    draw.polygon(tip_poly, fill=(118, 126, 134), outline=(54, 63, 72))
    thread_boxes: List[List[float]] = []
    for offset in (0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.84):
        cx = shaft_start[0] * (1.0 - offset) + shaft_end[0] * offset
        cy = shaft_start[1] * (1.0 - offset) + shaft_end[1] * offset
        line = [
            (cx - normal[0] * shaft_width * 0.72 - direction[0] * shaft_width * 0.45, cy - normal[1] * shaft_width * 0.72 - direction[1] * shaft_width * 0.45),
            (cx + normal[0] * shaft_width * 0.72 + direction[0] * shaft_width * 0.45, cy + normal[1] * shaft_width * 0.72 + direction[1] * shaft_width * 0.45),
        ]
        draw.line(line, fill=(95, 103, 112), width=1)
        thread_boxes.append(_padded_screen_line_bbox(line, pad_px=1.0))
    return _bbox_union(
        _padded_screen_line_bbox(shaft_points, pad_px=float(shaft_width + 2)),
        head_bbox,
        _bbox_from_screen_points(tip_poly),
        _padded_screen_line_bbox(slot, pad_px=1.0),
        *thread_boxes,
    )


def _draw_bolt_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    metal = (168, 176, 183)
    shaft = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.10, height * 0.03), dimensions_xyz=(width * 0.30, depth * 0.70, height * 0.40))
    head = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.40, height * 0.01), dimensions_xyz=(width * 0.86, depth * 0.22, height * 0.58))
    shaft_bbox = _draw_box_object(draw, shaft, camera=camera, frame=frame, fill=metal)
    head_bbox = _draw_footprint_prism_object(draw, head, camera=camera, frame=frame, fill=(185, 192, 198), footprint_xy=_hexagon_footprint_points())
    sx0, sy0, sx1, sy1 = (float(value) for value in shaft_bbox)
    thread_boxes: List[List[float]] = []
    for offset in (0.22, 0.38, 0.54, 0.70, 0.84):
        line = [
            (sx0 + (sx1 - sx0) * 0.18, sy0 + (sy1 - sy0) * offset),
            (sx1 - (sx1 - sx0) * 0.12, sy0 + (sy1 - sy0) * (offset - 0.06)),
        ]
        draw.line(line, fill=(88, 98, 108), width=1)
        thread_boxes.append(_padded_screen_line_bbox(line, pad_px=1.0))
    return _bbox_union(shaft_bbox, head_bbox, *thread_boxes)


def _draw_hex_nut_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_footprint_prism_object(draw, spec, camera=camera, frame=frame, fill=(172, 178, 184), footprint_xy=_hexagon_footprint_points())
    cx, cy = _bbox_center(bbox)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    rx = max(4.0, (x1 - x0) * 0.18)
    ry = max(3.0, (y1 - y0) * 0.16)
    hole = [(cx + math.cos(i * math.pi / 3.0) * rx, cy + math.sin(i * math.pi / 3.0) * ry) for i in range(6)]
    draw.polygon(hole, fill=floor_rgb, outline=(38, 45, 54))
    bevel = [cx - rx * 1.55, cy - ry * 1.45, cx + rx * 1.55, cy + ry * 1.45]
    draw.ellipse(bevel, outline=(226, 232, 235), width=1)
    facet_boxes: List[List[float]] = []
    for angle in (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0):
        line = [(cx + math.cos(angle) * rx * 1.55, cy + math.sin(angle) * ry * 1.45), (cx - math.cos(angle) * rx * 1.55, cy - math.sin(angle) * ry * 1.45)]
        facet_boxes.append(_draw_detail_line(draw, line, fill=(111, 121, 130), width=1))
    return _bbox_union(bbox, _bbox_from_screen_points(hole), bevel, *facet_boxes)


def _draw_washer_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_torus_object(draw, spec, camera=camera, frame=frame, fill=(184, 190, 196), floor_rgb=floor_rgb)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    outer_highlight = [x0 + w * 0.10, y0 + h * 0.16, x1 - w * 0.10, y1 - h * 0.16]
    inner_shadow = [x0 + w * 0.32, y0 + h * 0.34, x1 - w * 0.32, y1 - h * 0.34]
    draw.ellipse(outer_highlight, outline=(236, 241, 243), width=2)
    draw.ellipse(inner_shadow, outline=(82, 92, 102), width=2)
    shine = [(x0 + w * 0.24, y0 + h * 0.28), (x1 - w * 0.20, y0 + h * 0.22)]
    return _bbox_union(bbox, outer_highlight, inner_shadow, _draw_detail_line(draw, shine, fill=(246, 250, 250), width=2))


def _draw_paper_clip_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=(239, 243, 244))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    outline = (73, 87, 98)
    outer = [x0 + w * 0.17, y0 + h * 0.08, x1 - w * 0.20, y1 - h * 0.08]
    inner = [x0 + w * 0.34, y0 + h * 0.22, x1 - w * 0.36, y1 - h * 0.24]
    tail = [(x0 + w * 0.50, y0 + h * 0.22), (x0 + w * 0.50, y1 - h * 0.18), (x1 - w * 0.20, y1 - h * 0.18)]
    draw.rounded_rectangle(outer, radius=max(4, int(w * 0.16)), outline=outline, width=4)
    draw.rounded_rectangle(inner, radius=max(3, int(w * 0.10)), outline=outline, width=3)
    tail_bbox = _draw_detail_line(draw, tail, fill=outline, width=3, joint="curve")
    shine = _draw_detail_line(draw, [(outer[0] + w * 0.10, outer[1] + h * 0.12), (outer[2] - w * 0.18, outer[1] + h * 0.05)], fill=(230, 237, 240), width=1)
    return _bbox_union(bbox, outer, inner, tail_bbox, shine)


def _draw_u_bolt_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    metal = (176, 184, 190)
    parts = [
        _sub_box_spec(spec, offset_xyz=(-width * 0.24, depth * 0.12, 0.0), dimensions_xyz=(width * 0.14, depth * 0.58, height * 0.44)),
        _sub_box_spec(spec, offset_xyz=(width * 0.24, depth * 0.12, 0.0), dimensions_xyz=(width * 0.14, depth * 0.58, height * 0.44)),
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.27, 0.0), dimensions_xyz=(width * 0.62, depth * 0.16, height * 0.44)),
    ]
    bboxes = [_draw_box_object(draw, part, camera=camera, frame=frame, fill=metal) for part in parts]
    for offset_x in (-0.24, 0.24):
        nut = _sub_box_spec(spec, offset_xyz=(width * offset_x, depth * 0.43, 0.0), dimensions_xyz=(width * 0.26, depth * 0.16, height * 0.52))
        bboxes.append(_draw_footprint_prism_object(draw, nut, camera=camera, frame=frame, fill=(151, 158, 165), footprint_xy=_hexagon_footprint_points()))
    return _bbox_union(*bboxes)


def _draw_nail_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    metal = (166, 173, 180)
    shaft = _sub_box_spec(spec, offset_xyz=(0.0, 0.03 * depth, 0.0), dimensions_xyz=(width * 0.34, depth * 0.78, height * 0.52))
    head = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.42, 0.0), dimensions_xyz=(width * 0.90, depth * 0.16, height * 0.64))
    tip = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.47, 0.0), dimensions_xyz=(width * 0.48, depth * 0.14, height * 0.48))
    shaft_bbox = _draw_box_object(draw, shaft, camera=camera, frame=frame, fill=metal)
    head_bbox = _draw_cylinder_object(draw, head, camera=camera, frame=frame, fill=(190, 195, 200))
    tip_bbox = _draw_cone_object(draw, tip, camera=camera, frame=frame, fill=(139, 145, 152))
    sx0, sy0, sx1, sy1 = (float(value) for value in shaft_bbox)
    hx0, hy0, hx1, hy1 = (float(value) for value in head_bbox)
    shine = _draw_detail_line(draw, [(sx0 + (sx1 - sx0) * 0.34, sy0 + (sy1 - sy0) * 0.10), (sx1 - (sx1 - sx0) * 0.20, sy1 - (sy1 - sy0) * 0.16)], fill=(229, 235, 238), width=2)
    head_rim = [hx0 + (hx1 - hx0) * 0.10, hy0 + (hy1 - hy0) * 0.18, hx1 - (hx1 - hx0) * 0.10, hy1 - (hy1 - hy0) * 0.18]
    draw.ellipse(head_rim, outline=(101, 111, 121), width=1)
    return _bbox_union(shaft_bbox, head_bbox, tip_bbox, shine, head_rim)


def _draw_rod_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=(165, 174, 181))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    highlight = [(x0 + (x1 - x0) * 0.20, y0 + (y1 - y0) * 0.28), (x1 - (x1 - x0) * 0.16, y0 + (y1 - y0) * 0.20)]
    draw.line(highlight, fill=(225, 231, 234), width=2)
    return _bbox_union(bbox, _padded_screen_line_bbox(highlight, pad_px=1.0))


def _draw_stick_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    center, direction, normal, length_px = _diagonal_ground_axis_basis(
        spec,
        camera,
        frame,
        center_height_frac=0.62,
        length_scale=0.86,
        min_length_px=62.0,
        max_length_px=108.0,
    )

    def point_at(axis_frac: float, offset_px: float = 0.0) -> Tuple[float, float]:
        return (
            center[0] + direction[0] * length_px * float(axis_frac) + normal[0] * float(offset_px),
            center[1] + direction[1] * length_px * float(axis_frac) + normal[1] * float(offset_px),
        )

    points = [
        point_at(-0.52, -2.0),
        point_at(-0.18, 3.0),
        point_at(0.14, -2.5),
        point_at(0.52, 2.0),
    ]
    main_width = max(5, int(round(length_px * 0.070)))
    draw.line(points, fill=(71, 43, 23), width=main_width + 2, joint="curve")
    draw.line(points, fill=(132, 84, 43), width=main_width, joint="curve")
    bboxes: List[List[float]] = [_padded_screen_line_bbox(points, pad_px=float(main_width + 2))]
    branch = [point_at(0.06, -1.0), point_at(0.27, 18.0)]
    draw.line(branch, fill=(83, 50, 26), width=max(3, main_width - 2), joint="curve")
    bboxes.append(_padded_screen_line_bbox(branch, pad_px=float(main_width)))
    for offset in (0.22, 0.48, 0.72):
        index = min(len(points) - 2, int(offset * (len(points) - 1)))
        p0 = points[index]
        p1 = points[index + 1]
        mark = [
            (p0[0] * (1.0 - offset) + p1[0] * offset - 3.0, p0[1] * (1.0 - offset) + p1[1] * offset - 2.0),
            (p0[0] * (1.0 - offset) + p1[0] * offset + 4.0, p0[1] * (1.0 - offset) + p1[1] * offset + 1.0),
        ]
        bboxes.append(_draw_detail_line(draw, mark, fill=(53, 31, 18), width=2))
    return _bbox_union(*bboxes)


def _draw_tube_object(
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
        center_height_frac=0.56,
        length_scale=0.80,
        min_length_px=58.0,
        max_length_px=100.0,
    )

    def point_at(axis_frac: float) -> Tuple[float, float]:
        return (
            center[0] + direction[0] * length_px * float(axis_frac),
            center[1] + direction[1] * length_px * float(axis_frac),
        )

    body_width = max(12, int(round(length_px * 0.150)))
    start = point_at(-0.50)
    end = point_at(0.50)
    line = [start, end]
    draw.line(line, fill=(73, 82, 92), width=body_width + 4)
    draw.line(line, fill=(170, 181, 187), width=body_width)
    shine = [
        (start[0] + normal[0] * body_width * 0.20, start[1] + normal[1] * body_width * 0.20),
        (end[0] + normal[0] * body_width * 0.20, end[1] + normal[1] * body_width * 0.20),
    ]
    draw.line(shine, fill=(224, 230, 232), width=2)
    bboxes = [
        _padded_screen_line_bbox(line, pad_px=float(body_width + 2)),
        _padded_screen_line_bbox(shine, pad_px=1.0),
    ]
    end_radius = max(5.0, float(body_width) * 0.52)
    for end_x, end_y in (start, end):
        outer = [end_x - end_radius, end_y - end_radius, end_x + end_radius, end_y + end_radius]
        inner_radius = end_radius * 0.44
        inner = [end_x - inner_radius, end_y - inner_radius, end_x + inner_radius, end_y + inner_radius]
        draw.ellipse(outer, fill=(154, 165, 173), outline=(53, 62, 72), width=1)
        draw.ellipse(inner, fill=(58, 68, 78), outline=(224, 230, 232), width=1)
        bboxes.extend([outer, inner])
    return _bbox_union(*bboxes)


def _draw_clip_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.03 * depth, 0.0), dimensions_xyz=(width * 0.74, depth * 0.56, height * 0.64))
    hinge = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.34, 0.0), dimensions_xyz=(width * 0.78, depth * 0.14, height * 0.56))
    body_bbox = _draw_wedge_object(draw, body, camera=camera, frame=frame, fill=_shade(fill, 0.76))
    hinge_bbox = _draw_box_object(draw, hinge, camera=camera, frame=frame, fill=(183, 190, 196))
    x0, y0, x1, y1 = (float(value) for value in body_bbox)
    body_w = max(1.0, x1 - x0)
    body_h = max(1.0, y1 - y0)
    lip = [x0 + body_w * 0.16, y1 - body_h * 0.20, x1 - body_w * 0.16, y1 - body_h * 0.10]
    draw.rectangle(lip, fill=(55, 63, 73), outline=(24, 29, 35), width=1)
    arms = [
        [(x0 + (x1 - x0) * 0.16, y0 + (y1 - y0) * 0.18), (x0 + (x1 - x0) * 0.32, y1 - (y1 - y0) * 0.08)],
        [(x1 - (x1 - x0) * 0.16, y0 + (y1 - y0) * 0.18), (x1 - (x1 - x0) * 0.32, y1 - (y1 - y0) * 0.08)],
    ]
    arm_bboxes = []
    for arm in arms:
        draw.line(arm, fill=(224, 230, 233), width=2)
        arm_bboxes.append(_padded_screen_line_bbox(arm, pad_px=1.0))
    loops = [
        [x0 + body_w * 0.08, y0 + body_h * 0.03, x0 + body_w * 0.34, y0 + body_h * 0.22],
        [x1 - body_w * 0.34, y0 + body_h * 0.03, x1 - body_w * 0.08, y0 + body_h * 0.22],
    ]
    for loop in loops:
        draw.arc(loop, start=190, end=355, fill=(225, 231, 234), width=2)
    return _bbox_union(body_bbox, hinge_bbox, lip, *loops, *arm_bboxes)


def _draw_socket_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18))
    plate = _project_local_xy_rect(spec, camera, frame, u0=0.14, v0=0.16, u1=0.86, v1=0.84, z_frac=1.06)
    _draw_polyline(draw, plate, fill=(56, 64, 73), width=1)
    bboxes: List[List[float]] = [bbox, _bbox_from_screen_points(plate)]
    for u in (0.40, 0.60):
        slot = _project_local_xy_rect(spec, camera, frame, u0=u - 0.045, v0=0.34, u1=u + 0.045, v1=0.66, z_frac=1.08)
        draw.polygon(slot, fill=(38, 45, 54), outline=(24, 30, 36))
        bboxes.append(_bbox_from_screen_points(slot))
    for v in (0.25, 0.75):
        screw_center = _project_local_xy_point(spec, camera, frame, u=0.50, v=v, z_frac=1.09)
        bboxes.append(_draw_small_ellipse(draw, screw_center, 2.6, 2.0, fill=(176, 182, 186), outline=(72, 80, 88), width=1))
    return _bbox_union(*bboxes)


def _draw_magnet_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    parts = [
        (_sub_box_spec(spec, offset_xyz=(-width * 0.25, 0.06 * depth, 0.0), dimensions_xyz=(width * 0.22, depth * 0.62, height * 0.72)), (201, 54, 55)),
        (_sub_box_spec(spec, offset_xyz=(width * 0.25, 0.06 * depth, 0.0), dimensions_xyz=(width * 0.22, depth * 0.62, height * 0.72)), (60, 112, 201)),
        (_sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.29, 0.0), dimensions_xyz=(width * 0.72, depth * 0.18, height * 0.72)), (122, 132, 142)),
    ]
    bboxes = [_draw_box_object(draw, part, camera=camera, frame=frame, fill=color) for part, color in parts]
    union = _bbox_union(*bboxes)
    x0, y0, x1, y1 = (float(value) for value in union)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    gap = [x0 + w * 0.38, y0 + h * 0.42, x1 - w * 0.38, y1 - h * 0.05]
    draw.rounded_rectangle(gap, radius=max(2, int(w * 0.04)), fill=(236, 240, 238))
    for px, color in ((0.22, (248, 235, 236)), (0.78, (232, 239, 252))):
        pole = [x0 + w * (px - 0.08), y1 - h * 0.24, x0 + w * (px + 0.08), y1 - h * 0.08]
        draw.rectangle(pole, fill=color, outline=(50, 58, 66), width=1)
        bboxes.append(pole)
    return _bbox_union(union, gap, *bboxes)


def _draw_heater_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=(211, 217, 219))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    bboxes = [bbox]
    for offset in (0.24, 0.36, 0.48, 0.60, 0.72):
        vent = [(x0 + w * 0.18, y0 + h * offset), (x1 - w * 0.14, y0 + h * offset)]
        draw.line(vent, fill=(92, 103, 112), width=2)
        bboxes.append(_padded_screen_line_bbox(vent, pad_px=1.0))
    side_handle = [x0 + w * 0.08, y0 + h * 0.22, x0 + w * 0.16, y0 + h * 0.62]
    draw.rounded_rectangle(side_handle, radius=max(2, int(w * 0.025)), fill=(166, 176, 181), outline=(79, 91, 99), width=1)
    bboxes.append(side_handle)
    dial = [x1 - w * 0.22, y0 + h * 0.12, x1 - w * 0.10, y0 + h * 0.25]
    draw.ellipse(dial, fill=(120, 129, 137), outline=(42, 49, 58), width=1)
    bboxes.append(dial)
    for fx in (0.24, 0.76):
        foot = [x0 + w * (fx - 0.08), y1 - h * 0.10, x0 + w * (fx + 0.08), y1 + h * 0.02]
        draw.rounded_rectangle(foot, radius=max(2, int(w * 0.035)), fill=(92, 103, 112), outline=(42, 49, 58), width=1)
        bboxes.append(foot)
    return _bbox_union(*bboxes)


def _draw_flower_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    center = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.68)])[0]
    radius = max(4.0, float(frame.scale) * width * 0.028)
    petal_radius = max(5.0, radius * 1.35)
    stem = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.76), (0.0, 0.56)])
    bboxes = [_draw_detail_line(draw, stem, fill=(54, 128, 66), width=3)]
    leaves = [
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.02, -0.20), (-0.42, 0.02), (-0.08, 0.16)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.02, 0.08), (0.40, 0.30), (0.08, 0.42)]),
    ]
    for leaf in leaves:
        draw.polygon(leaf, fill=(61, 132, 70), outline=(36, 90, 48))
        bboxes.append(_bbox_from_screen_points(leaf))
    petal_fill = _tint(fill, 0.10)
    for angle in (0.0, math.pi * 0.4, math.pi * 0.8, math.pi * 1.2, math.pi * 1.6):
        cx = float(center[0]) + math.cos(angle) * radius * 1.45
        cy = float(center[1]) + math.sin(angle) * radius * 1.15
        bboxes.append(_draw_small_ellipse(draw, (cx, cy), petal_radius, petal_radius * 0.72, fill=petal_fill, outline=(104, 48, 76), width=1))
    bboxes.append(_draw_small_ellipse(draw, center, radius * 0.95, radius * 0.82, fill=petal_fill, outline=_shade(fill, 0.42), width=1))
    return _bbox_union(*bboxes)


def _draw_plant_pot_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    pot_profile = [(-0.72, -0.90), (0.72, -0.90), (0.54, 0.08), (0.38, 0.20), (-0.38, 0.20), (-0.54, 0.08)]
    pot = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(176, 91, 57), profile_xz=pot_profile, inset_scale=0.70)
    bboxes = [pot]
    leaves = [
        [(-0.06, 0.10), (-0.56, 0.58), (-0.18, 0.88)],
        [(0.04, 0.14), (0.52, 0.64), (0.14, 0.92)],
        [(-0.02, 0.12), (-0.08, 0.98), (0.20, 0.56)],
    ]
    for leaf in leaves:
        points = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=leaf)
        draw.polygon(points, fill=(72, 142, 80), outline=(38, 91, 50))
        bboxes.append(_bbox_from_screen_points(points))
    return _bbox_union(*bboxes)


def _draw_glass_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=(190, 220, 228))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    inner = [x0 + w * 0.16, y0 + h * 0.12, x1 - w * 0.16, y0 + h * 0.34]
    liquid = [x0 + w * 0.20, y0 + h * 0.52, x1 - w * 0.20, y0 + h * 0.74]
    shine = [(x0 + w * 0.30, y0 + h * 0.22), (x0 + w * 0.20, y1 - h * 0.18)]
    draw.ellipse(inner, fill=(218, 239, 244), outline=(88, 126, 145), width=2)
    draw.ellipse(liquid, fill=(143, 202, 219), outline=(77, 136, 156), width=1)
    draw.line([(x0 + w * 0.12, y1 - h * 0.16), (x1 - w * 0.12, y1 - h * 0.16)], fill=(103, 139, 151), width=2)
    draw.line(shine, fill=(246, 252, 252), width=2)
    return _bbox_union(bbox, inner, liquid, _padded_screen_line_bbox(shine, pad_px=1.0))


def _draw_jar_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.88, depth * 0.88, height * 0.76))
    lid = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.72), dimensions_xyz=(width * 0.72, depth * 0.72, height * 0.16))
    body_bbox = _draw_cylinder_object(draw, body, camera=camera, frame=frame, fill=(199, 224, 226))
    lid_bbox = _draw_cylinder_object(draw, lid, camera=camera, frame=frame, fill=(142, 151, 156))
    x0, y0, x1, y1 = (float(value) for value in body_bbox)
    label = [x0 + (x1 - x0) * 0.20, y0 + (y1 - y0) * 0.45, x1 - (x1 - x0) * 0.20, y0 + (y1 - y0) * 0.66]
    draw.rectangle(label, fill=(237, 228, 185), outline=(92, 102, 108), width=1)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    shine = _draw_detail_line(draw, [(x0 + w * 0.28, y0 + h * 0.20), (x0 + w * 0.20, y1 - h * 0.20)], fill=(244, 252, 252), width=2)
    for offset in (0.20, 0.28):
        line = [(x0 + w * 0.28, y0 + h * offset), (x1 - w * 0.28, y0 + h * (offset - 0.02))]
        draw.line(line, fill=(103, 112, 118), width=1)
    return _bbox_union(body_bbox, lid_bbox, label, shine)


def _draw_can_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.04))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    label = [x0 + w * 0.10, y0 + h * 0.34, x1 - w * 0.10, y0 + h * 0.70]
    pull_tab = [x0 + w * 0.38, y0 + h * 0.13, x0 + w * 0.62, y0 + h * 0.26]
    top_rim = [x0 + w * 0.09, y0 + h * 0.08, x1 - w * 0.09, y0 + h * 0.30]
    bottom_rim = [x0 + w * 0.10, y1 - h * 0.22, x1 - w * 0.10, y1 - h * 0.06]
    top_fill = [x0 + w * 0.12, y0 + h * 0.10, x1 - w * 0.12, y0 + h * 0.28]
    draw.ellipse(top_fill, fill=(205, 211, 214), outline=(70, 80, 89), width=1)
    draw.ellipse(top_rim, outline=(92, 101, 108), width=2)
    draw.ellipse(bottom_rim, outline=(92, 101, 108), width=1)
    draw.rectangle(label, fill=_shade(fill, 0.88), outline=(42, 50, 58), width=1)
    stripe = [label[0] + w * 0.08, label[1] + h * 0.04, label[0] + w * 0.20, label[3] - h * 0.04]
    shine = [(x0 + w * 0.26, y0 + h * 0.18), (x0 + w * 0.20, y1 - h * 0.16)]
    draw.rectangle(stripe, fill=(246, 246, 232))
    draw.line(shine, fill=(244, 249, 249), width=2)
    draw.ellipse(pull_tab, outline=(82, 91, 99), width=2)
    return _bbox_union(bbox, label, stripe, pull_tab, top_fill, top_rim, bottom_rim, _padded_screen_line_bbox(shine, pad_px=1.0))


def _draw_lid_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.10))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    rim = [x0 + (x1 - x0) * 0.12, y0 + (y1 - y0) * 0.20, x1 - (x1 - x0) * 0.12, y1 - (y1 - y0) * 0.20]
    draw.ellipse(rim, outline=(70, 80, 90), width=2)
    return _bbox_union(bbox, rim)


def _draw_pillow_cushion_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    shape_type = str(spec.get("shape_type", "pillow"))
    body = _project_local_xy_rect(spec, camera, frame, u0=0.04, v0=0.06, u1=0.96, v1=0.94, z_frac=1.0)
    seam = _project_local_xy_rect(spec, camera, frame, u0=0.13, v0=0.17, u1=0.87, v1=0.83, z_frac=1.02)
    draw.polygon(body, fill=_tint(fill, 0.32), outline=_shade(fill, 0.64))
    _draw_polyline(draw, body, fill=_shade(fill, 0.62), width=2)
    _draw_polyline(draw, seam, fill=_shade(fill, 0.72), width=1)
    bboxes: List[List[float]] = [_bbox_from_screen_points(body), _bbox_from_screen_points(seam)]
    if shape_type == "cushion":
        center = _project_local_xy_point(spec, camera, frame, u=0.50, v=0.50, z_frac=1.04)
        body_bbox = _bbox_from_screen_points(body)
        x0, y0, x1, y1 = (float(value) for value in body_bbox)
        button = [
            float(center[0]) - max(2.0, (x1 - x0) * 0.045),
            float(center[1]) - max(2.0, (y1 - y0) * 0.045),
            float(center[0]) + max(2.0, (x1 - x0) * 0.045),
            float(center[1]) + max(2.0, (y1 - y0) * 0.045),
        ]
        draw.ellipse(button, fill=_shade(fill, 0.68), outline=(45, 52, 61), width=1)
        bboxes.append(button)
    else:
        fold = [
            _project_local_xy_point(spec, camera, frame, u=0.20, v=0.50, z_frac=1.04),
            _project_local_xy_point(spec, camera, frame, u=0.80, v=0.46, z_frac=1.04),
        ]
        draw.line(fold, fill=_shade(fill, 0.70), width=2)
        bboxes.append(_padded_screen_line_bbox(fold, pad_px=1.0))
        body_bbox = _bbox_from_screen_points(body)
        x0, y0, x1, y1 = (float(value) for value in body_bbox)
        rx = max(1.5, (x1 - x0) * 0.025)
        ry = max(1.5, (y1 - y0) * 0.025)
        for cx, cy in (
            _project_local_xy_point(spec, camera, frame, u=0.13, v=0.16, z_frac=1.04),
            _project_local_xy_point(spec, camera, frame, u=0.87, v=0.16, z_frac=1.04),
            _project_local_xy_point(spec, camera, frame, u=0.13, v=0.84, z_frac=1.04),
            _project_local_xy_point(spec, camera, frame, u=0.87, v=0.84, z_frac=1.04),
        ):
            pinch = [cx - rx, cy - ry, cx + rx, cy + ry]
            draw.ellipse(pinch, fill=_shade(fill, 0.62))
            bboxes.append(pinch)
    return _bbox_union(*bboxes)


def _draw_stool_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    leg_h = height * 0.66
    seat_h = height * 0.20
    leg_w = min(width, depth) * 0.12
    parts = [
        _sub_box_spec(spec, offset_xyz=(sx * width * 0.30, sy * depth * 0.28, 0.0), dimensions_xyz=(leg_w, leg_w, leg_h))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    ]
    bboxes = [_draw_box_object(draw, part, camera=camera, frame=frame, fill=_shade(fill, 0.74)) for part in parts]
    brace_specs = [
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.30, height * 0.28), dimensions_xyz=(width * 0.68, leg_w * 0.55, height * 0.08)),
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.30, height * 0.28), dimensions_xyz=(width * 0.68, leg_w * 0.55, height * 0.08)),
    ]
    bboxes.extend(_draw_box_object(draw, brace, camera=camera, frame=frame, fill=_shade(fill, 0.82)) for brace in brace_specs)
    seat = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, leg_h), dimensions_xyz=(width, depth, seat_h))
    seat_bbox = _draw_cylinder_object(draw, seat, camera=camera, frame=frame, fill=_tint(fill, 0.12))
    bboxes.append(seat_bbox)
    x0, y0, x1, y1 = (float(value) for value in seat_bbox)
    rim = [x0 + (x1 - x0) * 0.12, y0 + (y1 - y0) * 0.18, x1 - (x1 - x0) * 0.12, y1 - (y1 - y0) * 0.22]
    draw.ellipse(rim, outline=_shade(fill, 0.62), width=2)
    bboxes.append(rim)
    return _bbox_union(*bboxes)


def _draw_drawer_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.10))
    x0, y0, x1, y1 = (float(value) for value in body)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    front = [x0 + w * 0.09, y0 + h * 0.24, x1 - w * 0.09, y1 - h * 0.13]
    draw.rectangle(front, fill=_tint(fill, 0.20), outline=_shade(fill, 0.55), width=2)
    panel = [front[0] + w * 0.10, front[1] + h * 0.10, front[2] - w * 0.10, front[3] - h * 0.12]
    draw.rectangle(panel, outline=_shade(fill, 0.68), width=1)
    handle = [x0 + w * 0.38, y0 + h * 0.52, x0 + w * 0.62, y0 + h * 0.63]
    draw.rounded_rectangle(handle, radius=max(2, int(h * 0.04)), fill=(72, 61, 48), outline=(36, 31, 26), width=1)
    return _bbox_union(body, front, panel, handle)


def _draw_cap_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base_bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18))
    bx0, by0, bx1, by1 = (float(value) for value in base_bbox)
    bw = max(1.0, bx1 - bx0)
    bh = max(1.0, by1 - by0)
    crown = _upright_screen_points(
        spec,
        camera=camera,
        frame=frame,
        profile_xz=[
            (-0.72, -0.18),
            (-0.56, 0.38),
            (-0.28, 0.76),
            (0.08, 0.88),
            (0.46, 0.66),
            (0.70, 0.24),
            (0.66, -0.18),
        ],
    )
    bill = [
        (bx0 + bw * 0.50, by0 + bh * 0.50),
        (bx1 + bw * 0.32, by0 + bh * 0.42),
        (bx1 + bw * 0.24, by0 + bh * 0.72),
        (bx0 + bw * 0.46, by0 + bh * 0.68),
    ]
    draw.polygon(crown, fill=_tint(fill, 0.16), outline=_shade(fill, 0.54))
    draw.polygon(bill, fill=_shade(fill, 0.82), outline=_shade(fill, 0.48))
    dome = [bx0 + bw * 0.08, by0 - max(3.0, bh * 0.75), bx1 - bw * 0.10, by1 - bh * 0.10]
    draw.arc(dome, start=180, end=360, fill=_shade(fill, 0.45), width=2)
    bboxes = [base_bbox, _bbox_from_screen_points(crown), _bbox_from_screen_points(bill), dome]
    seam = [(bx0 + bw * 0.50, by0 - max(2.0, bh * 0.55)), (bx0 + bw * 0.50, by1 - bh * 0.12)]
    bboxes.append(_draw_detail_line(draw, seam, fill=_shade(fill, 0.58), width=2))
    for px in (0.34, 0.66):
        panel = [(bx0 + bw * px, by0 - max(2.0, bh * 0.42)), (bx0 + bw * (0.50 + (px - 0.50) * 0.35), by1 - bh * 0.14)]
        bboxes.append(_draw_detail_line(draw, panel, fill=_shade(fill, 0.64), width=1))
    button_center = (bx0 + bw * 0.50, by0 - max(2.0, bh * 0.58))
    bboxes.append(_draw_small_ellipse(draw, button_center, 3.0, 2.3, fill=_shade(fill, 0.72), outline=_shade(fill, 0.44)))
    return _bbox_union(*bboxes)


def _draw_bucket_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.54, -0.92), (0.54, -0.92), (0.86, 0.60), (0.68, 0.88), (-0.68, 0.88), (-0.86, 0.60)]
    bucket_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.12), profile_xz=profile, inset_scale=0.58)
    x0, y0, x1, y1 = (float(value) for value in bucket_bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    opening = [x0 + w * 0.09, y0 + h * 0.08, x1 - w * 0.09, y0 + h * 0.36]
    inner = [x0 + w * 0.18, y0 + h * 0.15, x1 - w * 0.18, y0 + h * 0.31]
    draw.ellipse(opening, fill=_shade(fill, 0.70), outline=(41, 50, 60), width=3)
    draw.ellipse(inner, fill=_shade(fill, 0.50), outline=(36, 44, 54), width=1)
    handle = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.82, 0.36), (-0.48, 1.10), (0.0, 1.24), (0.48, 1.10), (0.82, 0.36)])
    draw.line(handle, fill=(82, 90, 98), width=4, joint="curve")
    lugs = [
        _draw_small_ellipse(draw, (x0 + w * 0.14, y0 + h * 0.44), max(2.0, w * 0.035), max(2.0, h * 0.035), fill=(78, 88, 98), outline=(36, 42, 50)),
        _draw_small_ellipse(draw, (x1 - w * 0.14, y0 + h * 0.44), max(2.0, w * 0.035), max(2.0, h * 0.035), fill=(78, 88, 98), outline=(36, 42, 50)),
    ]
    return _bbox_union(bucket_bbox, opening, inner, _padded_screen_line_bbox(handle, pad_px=4.0), *lugs)


def _draw_tray_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    base_bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.10))
    lip = _project_local_xy_rect(spec, camera, frame, u0=0.04, v0=0.08, u1=0.96, v1=0.92, z_frac=1.04)
    inner = _project_local_xy_rect(spec, camera, frame, u0=0.14, v0=0.24, u1=0.86, v1=0.78, z_frac=1.06)
    draw.polygon(lip, fill=_tint(fill, 0.16), outline=_shade(fill, 0.48))
    _draw_polyline(draw, lip, fill=_shade(fill, 0.48), width=3)
    draw.polygon(inner, fill=_shade(fill, 0.80), outline=_shade(fill, 0.52))
    _draw_polyline(draw, inner, fill=_shade(fill, 0.52), width=1)
    handles = []
    for px in (0.08, 0.92):
        handle = _project_local_xy_rect(spec, camera, frame, u0=px - 0.055, v0=0.42, u1=px + 0.055, v1=0.62, z_frac=1.08)
        draw.polygon(handle, fill=(236, 240, 238), outline=_shade(fill, 0.50))
        _draw_polyline(draw, handle, fill=_shade(fill, 0.50), width=2)
        handles.append(_bbox_from_screen_points(handle))
    return _bbox_union(base_bbox, _bbox_from_screen_points(lip), _bbox_from_screen_points(inner), *handles)


def _draw_coaster_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_cylinder_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    rim = [x0 + w * 0.12, y0 + h * 0.20, x1 - w * 0.12, y1 - h * 0.20]
    inner = [x0 + w * 0.28, y0 + h * 0.34, x1 - w * 0.28, y1 - h * 0.34]
    draw.ellipse(rim, outline=_shade(fill, 0.58), width=2)
    draw.ellipse(inner, outline=_shade(fill, 0.70), width=1)
    return _bbox_union(bbox, rim, inner)


def _draw_rose_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    stem = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.76), (0.0, 0.60)])
    bboxes = [_draw_detail_line(draw, stem, fill=(54, 124, 66), width=3)]
    leaf_a = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.02, 0.02), (-0.46, 0.22), (-0.10, 0.34)])
    leaf_b = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.02, 0.18), (0.42, 0.38), (0.10, 0.48)])
    for leaf in (leaf_a, leaf_b):
        draw.polygon(leaf, fill=(58, 128, 67), outline=(34, 84, 46))
        bboxes.append(_bbox_from_screen_points(leaf))
    center = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.78)])[0]
    radius = max(4.0, float(frame.scale) * width * 0.032)
    for layer, color, scale in ((0, (198, 42, 68), 1.85), (1, (222, 65, 94), 1.35), (2, (151, 32, 56), 0.82)):
        count = 6 if layer == 0 else 5
        for index in range(count):
            angle = (math.tau * float(index) / float(count)) + (0.34 if layer else 0.0)
            cx = float(center[0]) + math.cos(angle) * radius * scale * 0.58
            cy = float(center[1]) + math.sin(angle) * radius * scale * 0.44
            bboxes.append(_draw_small_ellipse(draw, (cx, cy), radius * scale * 0.62, radius * scale * 0.42, fill=color, outline=(105, 28, 43), width=1))
    bboxes.append(_draw_small_ellipse(draw, center, radius * 0.58, radius * 0.48, fill=(107, 25, 42), outline=(70, 20, 31), width=1))
    return _bbox_union(*bboxes)


def _draw_banana_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    crescent = _upright_screen_points(
        spec,
        camera=camera,
        frame=frame,
        profile_xz=[
            (-0.74, -0.26),
            (-0.50, 0.32),
            (-0.08, 0.62),
            (0.52, 0.42),
            (0.84, -0.10),
            (0.66, -0.36),
            (0.12, -0.20),
            (-0.42, -0.48),
        ],
    )
    draw.polygon(crescent, fill=(232, 197, 54), outline=(124, 91, 32))
    inner = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.42, -0.16), (-0.04, 0.16), (0.44, 0.06)])
    draw.line(inner, fill=(255, 229, 95), width=2)
    stem = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.78, -0.18), (-0.88, -0.26)])
    draw.line(stem, fill=(91, 58, 30), width=3)
    return _bbox_union(_bbox_from_screen_points(crescent), _padded_screen_line_bbox(inner, pad_px=2.0), _padded_screen_line_bbox(stem, pad_px=3.0))


def _draw_tomato_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_sphere_object(draw, spec, camera=camera, frame=frame, fill=(203, 54, 49))
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    cx, cy = _bbox_center(bbox)
    calyx_points = []
    for angle in (math.pi * 0.05, math.pi * 0.45, math.pi * 0.85, math.pi * 1.25, math.pi * 1.65):
        p1 = (cx + math.cos(angle) * w * 0.07, y0 + h * 0.20 + math.sin(angle) * h * 0.04)
        p2 = (cx + math.cos(angle) * w * 0.24, y0 + h * 0.13 + math.sin(angle) * h * 0.10)
        draw.line([p1, p2], fill=(50, 126, 54), width=3)
        calyx_points.extend([p1, p2])
    highlight = [x0 + w * 0.24, y0 + h * 0.30, x0 + w * 0.42, y0 + h * 0.46]
    draw.ellipse(highlight, fill=(235, 104, 84))
    return _bbox_union(bbox, highlight, _bbox_from_screen_points(calyx_points))


def _draw_peanut_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    lobe_a = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.22, 0.0), dimensions_xyz=(width * 0.86, depth * 0.52, height * 0.92))
    lobe_b = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.22, 0.0), dimensions_xyz=(width * 0.78, depth * 0.52, height * 0.84))
    bboxes = [
        _draw_upright_profile_object(draw, lobe_a, camera=camera, frame=frame, fill=(196, 145, 83), profile_xz=_oval_profile_points(28, z_scale=0.78), inset_scale=0.62),
        _draw_upright_profile_object(draw, lobe_b, camera=camera, frame=frame, fill=(210, 160, 93), profile_xz=_oval_profile_points(28, z_scale=0.76), inset_scale=0.62),
    ]
    texture_points: List[Tuple[float, float]] = []
    x0, y0, x1, y1 = (float(value) for value in _bbox_union(*bboxes))
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    for offset in (0.25, 0.48, 0.70):
        line = [(x0 + w * 0.24, y0 + h * offset), (x1 - w * 0.20, y0 + h * (offset - 0.06))]
        draw.line(line, fill=(137, 95, 54), width=1)
        texture_points.extend(line)
    dots: List[List[float]] = []
    for px, py in ((0.34, 0.30), (0.58, 0.40), (0.42, 0.62), (0.68, 0.68)):
        dots.append(_draw_small_ellipse(draw, (x0 + w * px, y0 + h * py), max(1.2, w * 0.025), max(1.0, h * 0.018), fill=(150, 104, 59)))
    waist = _draw_detail_line(draw, [(x0 + w * 0.20, y0 + h * 0.50), (x1 - w * 0.22, y0 + h * 0.47)], fill=(121, 83, 48), width=1)
    return _bbox_union(*bboxes, _bbox_from_screen_points(texture_points), waist, *dots)


def _draw_coffee_bean_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(104, 62, 36), profile_xz=_oval_profile_points(34, z_scale=0.72), inset_scale=0.66)
    groove = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.08, -0.70), (0.06, -0.34), (-0.03, 0.05), (0.10, 0.42), (0.00, 0.72)])
    draw.line(groove, fill=(52, 31, 21), width=3, joint="curve")
    draw.line(groove, fill=(155, 105, 66), width=1, joint="curve")
    return _bbox_union(bbox, _padded_screen_line_bbox(groove, pad_px=3.0))


def _draw_hook_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    plate = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.30, height * 0.02), dimensions_xyz=(width * 0.52, depth * 0.18, height * 0.62))
    plate_bbox = _draw_box_object(draw, plate, camera=camera, frame=frame, fill=_shade(fill, 0.78))
    hook = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.10, 0.76), (0.10, 0.18), (0.00, -0.20), (-0.26, -0.30), (-0.36, -0.08), (-0.20, 0.10)])
    draw.line(hook, fill=_tint(fill, 0.14), width=6, joint="curve")
    draw.line(hook, fill=_shade(fill, 0.52), width=2, joint="curve")
    x0, y0, x1, y1 = (float(value) for value in plate_bbox)
    w = max(1.0, x1 - x0)
    holes = [
        _draw_small_ellipse(draw, (x0 + w * 0.50, y0 + (y1 - y0) * 0.28), max(1.8, w * 0.06), max(1.6, w * 0.05), fill=(45, 52, 60)),
        _draw_small_ellipse(draw, (x0 + w * 0.50, y0 + (y1 - y0) * 0.72), max(1.8, w * 0.06), max(1.6, w * 0.05), fill=(45, 52, 60)),
    ]
    return _bbox_union(plate_bbox, _padded_screen_line_bbox(hook, pad_px=6.0), *holes)


def _draw_bracket_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    metal = _tint(fill, 0.10)
    vertical = _sub_box_spec(spec, offset_xyz=(-width * 0.24, 0.0, height * 0.08), dimensions_xyz=(width * 0.18, depth * 0.70, height * 0.70))
    foot = _sub_box_spec(spec, offset_xyz=(width * 0.08, 0.0, 0.0), dimensions_xyz=(width * 0.66, depth * 0.70, height * 0.18))
    brace = _sub_box_spec(spec, offset_xyz=(-width * 0.02, 0.0, height * 0.20), dimensions_xyz=(width * 0.50, depth * 0.18, height * 0.16))
    bboxes = [
        _draw_box_object(draw, vertical, camera=camera, frame=frame, fill=metal),
        _draw_box_object(draw, foot, camera=camera, frame=frame, fill=_shade(metal, 0.86)),
        _draw_wedge_object(draw, brace, camera=camera, frame=frame, fill=_shade(metal, 0.74)),
    ]
    hole_bboxes: List[List[float]] = []
    for bbox in bboxes[:2]:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        for cx, cy in ((x0 + w * 0.40, y0 + h * 0.36), (x0 + w * 0.62, y0 + h * 0.66)):
            hole_bboxes.append(_draw_small_ellipse(draw, (cx, cy), max(1.8, w * 0.055), max(1.6, h * 0.055), fill=(46, 54, 62)))
    return _bbox_union(*bboxes, *hole_bboxes)


def _draw_battery_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.84, depth * 0.78, height * 0.78))
    nub = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.46, height * 0.02), dimensions_xyz=(width * 0.42, depth * 0.12, height * 0.46))
    body_bbox = _draw_box_object(draw, body, camera=camera, frame=frame, fill=_tint(fill, 0.12))
    nub_bbox = _draw_cylinder_object(draw, nub, camera=camera, frame=frame, fill=(190, 194, 198))
    x0, y0, x1, y1 = (float(value) for value in body_bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    terminal = [x0 + w * 0.12, y0 + h * 0.24, x0 + w * 0.28, y0 + h * 0.40]
    draw.rectangle(terminal, fill=(230, 233, 226), outline=(50, 57, 64), width=1)
    plus_v = [(terminal[0] + w * 0.08, terminal[1] + h * 0.03), (terminal[0] + w * 0.08, terminal[3] - h * 0.03)]
    plus_h = [(terminal[0] + w * 0.03, terminal[1] + h * 0.08), (terminal[2] - w * 0.03, terminal[1] + h * 0.08)]
    draw.line(plus_v, fill=(42, 49, 56), width=1)
    draw.line(plus_h, fill=(42, 49, 56), width=1)
    band = [x1 - w * 0.26, y0 + h * 0.10, x1 - w * 0.13, y1 - h * 0.10]
    draw.rectangle(band, fill=(49, 57, 66))
    return _bbox_union(body_bbox, nub_bbox, terminal, band, _padded_screen_line_bbox(plus_v), _padded_screen_line_bbox(plus_h))


def _draw_tape_roll_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
    floor_rgb: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_torus_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18), floor_rgb=floor_rgb)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    core = [x0 + w * 0.32, y0 + h * 0.33, x1 - w * 0.32, y1 - h * 0.33]
    draw.ellipse(core, fill=(184, 137, 87), outline=(82, 58, 38), width=1)
    edge = [x0 + w * 0.10, y0 + h * 0.20, x1 - w * 0.10, y1 - h * 0.20]
    draw.ellipse(edge, outline=_shade(fill, 0.58), width=2)
    tab = [(x1 - w * 0.20, y0 + h * 0.42), (x1 + w * 0.16, y0 + h * 0.32), (x1 + w * 0.20, y0 + h * 0.48), (x1 - w * 0.12, y0 + h * 0.58)]
    draw.polygon(tab, fill=_tint(fill, 0.35), outline=_shade(fill, 0.62))
    return _bbox_union(bbox, core, edge, _bbox_from_screen_points(tab))


def _draw_bag_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.62, -0.92), (0.58, -0.92), (0.70, -0.22), (0.54, 0.62), (0.32, 0.82), (-0.32, 0.82), (-0.54, 0.62), (-0.70, -0.22)]
    body = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.18), profile_xz=profile, inset_scale=0.62)
    handle_a = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.34, 0.58), (-0.24, 1.02), (-0.04, 1.10), (0.04, 0.78)])
    handle_b = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.34, 0.58), (0.24, 1.02), (0.04, 1.10), (-0.04, 0.78)])
    for handle in (handle_a, handle_b):
        draw.line(handle, fill=_shade(fill, 0.58), width=4, joint="curve")
    x0, y0, x1, y1 = (float(value) for value in body)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    crease = [(x0 + w * 0.24, y0 + h * 0.24), (x0 + w * 0.18, y1 - h * 0.20)]
    draw.line(crease, fill=_shade(fill, 0.65), width=2)
    return _bbox_union(body, _padded_screen_line_bbox(handle_a, pad_px=4.0), _padded_screen_line_bbox(handle_b, pad_px=4.0), _padded_screen_line_bbox(crease, pad_px=2.0))


def _draw_chess_piece_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    base = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.88, depth * 0.88, height * 0.18))
    foot = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.13), dimensions_xyz=(width * 0.72, depth * 0.72, height * 0.16))
    tower = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.30), dimensions_xyz=(width * 0.48, depth * 0.48, height * 0.38))
    piece_fill = _tint(fill, 0.16)
    bboxes: List[List[float]] = [
        _draw_cylinder_object(draw, base, camera=camera, frame=frame, fill=_shade(piece_fill, 0.82)),
        _draw_cylinder_object(draw, foot, camera=camera, frame=frame, fill=_shade(piece_fill, 0.90)),
        _draw_cylinder_object(draw, tower, camera=camera, frame=frame, fill=piece_fill),
    ]
    tx0, ty0, tx1, ty1 = (float(value) for value in bboxes[-1])
    tw = max(1.0, tx1 - tx0)
    th = max(1.0, ty1 - ty0)
    top = [tx0 - tw * 0.08, ty0 - th * 0.36, tx1 + tw * 0.08, ty0 + th * 0.16]
    draw.rectangle(top, fill=piece_fill, outline=(34, 39, 48), width=2)
    battlements: List[List[float]] = []
    for px in (0.18, 0.50, 0.82):
        block = [
            top[0] + (top[2] - top[0]) * (px - 0.10),
            top[1] - th * 0.22,
            top[0] + (top[2] - top[0]) * (px + 0.10),
            top[1] + th * 0.06,
        ]
        draw.rectangle(block, fill=piece_fill, outline=(34, 39, 48), width=1)
        battlements.append(block)
    bboxes.append(top)
    bboxes.extend(battlements)
    x0, y0, x1, y1 = (float(value) for value in _bbox_union(*bboxes))
    bboxes.append(_draw_detail_line(draw, [(x0 + (x1 - x0) * 0.22, y1 - (y1 - y0) * 0.25), (x1 - (x1 - x0) * 0.22, y1 - (y1 - y0) * 0.27)], fill=(248, 248, 238), width=2))
    return _bbox_union(*bboxes)


def _draw_hanger_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    wood = _shade(fill, 0.86)
    hook = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.08, 0.42), (-0.02, 0.72), (0.20, 0.90), (0.34, 0.70), (0.18, 0.58)])
    shoulders = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.92, 0.18), (0.0, 0.50), (0.92, 0.18)])
    lower = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.66, 0.16), (0.0, -0.44), (0.66, 0.16)])
    crossbar = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.58, -0.08), (0.58, -0.08)])
    bboxes = []
    for points, color, width_px in ((hook, (112, 118, 124), 5), (shoulders, wood, 8), (lower, wood, 7), (crossbar, _tint(wood, 0.10), 5)):
        draw.line(points, fill=color, width=int(width_px), joint="curve")
        bboxes.append(_padded_screen_line_bbox(points, pad_px=float(width_px)))
    return _bbox_union(*bboxes)


def _draw_light_bulb_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    glass_profile = [
        (-0.30, -0.56),
        (-0.54, -0.22),
        (-0.66, 0.24),
        (-0.48, 0.68),
        (-0.18, 0.92),
        (0.18, 0.92),
        (0.48, 0.68),
        (0.66, 0.24),
        (0.54, -0.22),
        (0.30, -0.56),
    ]
    glass_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(231, 236, 211), profile_xz=glass_profile, inset_scale=0.0)
    base = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width * 0.50, depth * 0.50, height * 0.24))
    base_bbox = _draw_cylinder_object(draw, base, camera=camera, frame=frame, fill=(157, 164, 171))
    x0, y0, x1, y1 = (float(value) for value in base_bbox)
    ribs: List[List[float]] = []
    for offset in (0.34, 0.50, 0.66):
        rib = [(x0 + (x1 - x0) * 0.18, y0 + (y1 - y0) * offset), (x1 - (x1 - x0) * 0.18, y0 + (y1 - y0) * (offset - 0.06))]
        draw.line(rib, fill=(79, 88, 96), width=1)
        ribs.append(_padded_screen_line_bbox(rib, pad_px=1.0))
    filament = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.20, -0.12), (-0.06, 0.04), (0.06, -0.12), (0.20, 0.04)])
    draw.line(filament, fill=(206, 148, 48), width=2)
    ray_boxes: List[List[float]] = []
    for start, end in (((-0.62, 0.78), (-0.84, 1.02)), ((0.62, 0.78), (0.84, 1.02)), ((0.0, 0.94), (0.0, 1.20))):
        ray = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[start, end])
        ray_boxes.append(_draw_detail_line(draw, ray, fill=(236, 201, 82), width=2))
    return _bbox_union(glass_bbox, base_bbox, _padded_screen_line_bbox(filament, pad_px=1.0), *ribs, *ray_boxes)


def _draw_egg_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.36, -0.92),
        (-0.62, -0.64),
        (-0.68, -0.18),
        (-0.54, 0.30),
        (-0.28, 0.70),
        (0.0, 0.94),
        (0.28, 0.70),
        (0.54, 0.30),
        (0.68, -0.18),
        (0.62, -0.64),
        (0.36, -0.92),
        (0.0, -1.00),
    ]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(236, 229, 202), profile_xz=profile, inset_scale=0.76)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    highlight = [x0 + w * 0.30, y0 + h * 0.22, x0 + w * 0.48, y0 + h * 0.40]
    lower_shade = [x0 + w * 0.22, y0 + h * 0.66, x1 - w * 0.18, y1 - h * 0.08]
    draw.ellipse(lower_shade, outline=(203, 191, 159), width=1)
    draw.ellipse(highlight, fill=(252, 249, 236))
    return _bbox_union(bbox, highlight, lower_shade)


def _draw_chili_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body_profile = [
        (-0.34, 0.72),
        (0.20, 0.78),
        (0.76, 0.44),
        (0.98, -0.10),
        (0.54, -0.72),
        (-0.10, -0.90),
        (-0.54, -0.54),
        (-0.42, -0.02),
    ]
    body_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(199, 47, 42), profile_xz=body_profile, inset_scale=0.68)
    stem = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.24, 0.70), (-0.42, 0.96), (-0.18, 1.04), (-0.04, 0.78)])
    draw.polygon(stem, fill=(58, 132, 62), outline=(31, 86, 39))
    shine = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.02, 0.46), (0.36, 0.20), (0.36, -0.28)])
    draw.line(shine, fill=(241, 112, 91), width=2, joint="curve")
    x0, y0, x1, y1 = (float(value) for value in body_bbox)
    cx, cy = _bbox_center(body_bbox)
    dw = max(16.0, x1 - x0)
    dh = max(8.5, y1 - y0)
    overlay = [
        (cx - dw * 0.46, cy - dh * 0.04),
        (cx - dw * 0.18, cy - dh * 0.42),
        (cx + dw * 0.34, cy - dh * 0.24),
        (cx + dw * 0.48, cy + dh * 0.12),
        (cx + dw * 0.20, cy + dh * 0.42),
        (cx - dw * 0.34, cy + dh * 0.30),
    ]
    draw.polygon(overlay, fill=(199, 47, 42), outline=(112, 30, 28))
    draw.line([(cx - dw * 0.14, cy - dh * 0.20), (cx + dw * 0.32, cy - dh * 0.02)], fill=(241, 112, 91), width=2)
    screen_stem = [(cx - dw * 0.50, cy - dh * 0.04), (cx - dw * 0.62, cy - dh * 0.36), (cx - dw * 0.42, cy - dh * 0.30)]
    draw.polygon(screen_stem, fill=(58, 132, 62), outline=(31, 86, 39))
    return _bbox_union(
        body_bbox,
        _bbox_from_screen_points(stem),
        _padded_screen_line_bbox(shine, pad_px=1.0),
        _bbox_from_screen_points(overlay),
        _bbox_from_screen_points(screen_stem),
    )


def _draw_fork_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    metal = (183, 192, 199)
    parts = [
        _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.20, 0.0), dimensions_xyz=(width * 0.16, depth * 0.60, height * 0.42)),
        _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.17, 0.0), dimensions_xyz=(width * 0.62, depth * 0.13, height * 0.36)),
    ]
    for offset_x in (-0.30, -0.10, 0.10, 0.30):
        parts.append(_sub_box_spec(spec, offset_xyz=(width * offset_x, depth * 0.39, 0.0), dimensions_xyz=(width * 0.12, depth * 0.32, height * 0.34)))
    bboxes = [_draw_box_object(draw, part, camera=camera, frame=frame, fill=metal) for part in parts]
    union = _bbox_union(*bboxes)
    x0, y0, x1, y1 = (float(value) for value in union)
    bboxes.append(_draw_detail_line(draw, [(x0 + (x1 - x0) * 0.50, y0 + (y1 - y0) * 0.20), (x0 + (x1 - x0) * 0.50, y1 - (y1 - y0) * 0.18)], fill=(237, 242, 244), width=1))
    return _bbox_union(*bboxes)


def _draw_knife_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    handle = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.28, 0.0), dimensions_xyz=(width * 0.30, depth * 0.46, height * 0.50))
    blade = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.22, 0.0), dimensions_xyz=(width * 0.36, depth * 0.58, height * 0.42))
    handle_bbox = _draw_box_object(draw, handle, camera=camera, frame=frame, fill=(89, 67, 47))
    blade_bbox = _draw_wedge_object(draw, blade, camera=camera, frame=frame, fill=(194, 203, 209))
    x0, y0, x1, y1 = (float(value) for value in blade_bbox)
    edge = [(x0 + (x1 - x0) * 0.60, y0 + (y1 - y0) * 0.12), (x1 - (x1 - x0) * 0.12, y1 - (y1 - y0) * 0.16)]
    draw.line(edge, fill=(76, 86, 94), width=2)
    hx0, hy0, hx1, hy1 = (float(value) for value in handle_bbox)
    rivets = []
    for cy in (hy0 + (hy1 - hy0) * 0.32, hy0 + (hy1 - hy0) * 0.66):
        rivet = [hx0 + (hx1 - hx0) * 0.42, cy - (hy1 - hy0) * 0.035, hx0 + (hx1 - hx0) * 0.58, cy + (hy1 - hy0) * 0.035]
        draw.ellipse(rivet, fill=(208, 196, 162), outline=(61, 47, 32), width=1)
        rivets.append(rivet)
    spine = _draw_detail_line(draw, [(x0 + (x1 - x0) * 0.22, y0 + (y1 - y0) * 0.22), (x1 - (x1 - x0) * 0.18, y0 + (y1 - y0) * 0.32)], fill=(239, 244, 246), width=2)
    return _bbox_union(handle_bbox, blade_bbox, _padded_screen_line_bbox(edge, pad_px=1.0), spine, *rivets)


def _draw_spoon_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    handle = _sub_box_spec(spec, offset_xyz=(0.0, -depth * 0.24, 0.0), dimensions_xyz=(width * 0.13, depth * 0.58, height * 0.42))
    bowl = _sub_box_spec(spec, offset_xyz=(0.0, depth * 0.30, 0.0), dimensions_xyz=(width * 1.08, depth * 0.48, height * 0.76))
    handle_bbox = _draw_box_object(draw, handle, camera=camera, frame=frame, fill=(158, 168, 178))
    bowl_bbox = _draw_upright_profile_object(draw, bowl, camera=camera, frame=frame, fill=(194, 204, 212), profile_xz=_oval_profile_points(32, z_scale=0.78), inset_scale=0.38)
    x0, y0, x1, y1 = (float(value) for value in bowl_bbox)
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    inner = [x0 + w * 0.18, y0 + h * 0.20, x1 - w * 0.18, y1 - h * 0.16]
    shine = [(x0 + w * 0.28, y0 + h * 0.34), (x1 - w * 0.24, y0 + h * 0.24)]
    draw.ellipse(inner, fill=(174, 185, 194), outline=(91, 103, 113), width=2)
    draw.line(shine, fill=(244, 248, 249), width=2)
    return _bbox_union(handle_bbox, bowl_bbox, inner, _padded_screen_line_bbox(shine, pad_px=1.0))


def _draw_calculator_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(67, 78, 92), profile_xz=[(-0.72, -0.98), (0.72, -0.98), (0.72, 0.98), (-0.72, 0.98)], inset_scale=0.0)
    display = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.50, 0.54), (0.50, 0.54), (0.50, 0.82), (-0.50, 0.82)])
    draw.polygon(display, fill=(164, 188, 169))
    _draw_polyline(draw, display, fill=(26, 35, 38), width=1)
    bboxes = [bbox, _bbox_from_screen_points(display)]
    for px in (-0.45, -0.15, 0.15, 0.45):
        for pz in (-0.66, -0.40, -0.14, 0.12, 0.36):
            center = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px, pz)])[0]
            radius = max(2.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.017)
            button = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
            draw.rectangle(button, fill=(214, 218, 220), outline=(31, 36, 42), width=1)
            bboxes.append(button)
    return _bbox_union(*bboxes)


def _draw_dice_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.12))
    bboxes = [bbox]

    vertices = _object_vertices(spec)
    sx = 1 if camera.camera_position[0] >= float(spec["world_xyz"][0]) else -1
    sy = 1 if camera.camera_position[1] >= float(spec["world_xyz"][1]) else -1
    face_quads = [
        _project_face([vertices[f"{-sx}{-sy}1"], vertices[f"{sx}{-sy}1"], vertices[f"{sx}{sy}1"], vertices[f"{-sx}{sy}1"]], camera, frame),
        _project_face([vertices[f"{sx}{-sy}0"], vertices[f"{sx}{sy}0"], vertices[f"{sx}{sy}1"], vertices[f"{sx}{-sy}1"]], camera, frame),
        _project_face([vertices[f"{-sx}{sy}0"], vertices[f"{sx}{sy}0"], vertices[f"{sx}{sy}1"], vertices[f"{-sx}{sy}1"]], camera, frame),
    ]

    def point_on_quad(quad: Sequence[Sequence[float]], u: float, v: float) -> Tuple[float, float]:
        top_x = float(quad[0][0]) * (1.0 - u) + float(quad[1][0]) * u
        top_y = float(quad[0][1]) * (1.0 - u) + float(quad[1][1]) * u
        bottom_x = float(quad[3][0]) * (1.0 - u) + float(quad[2][0]) * u
        bottom_y = float(quad[3][1]) * (1.0 - u) + float(quad[2][1]) * u
        return (top_x * (1.0 - v) + bottom_x * v, top_y * (1.0 - v) + bottom_y * v)

    def add_pip(quad: Sequence[Sequence[float]], u: float, v: float) -> None:
        fx0, fy0, fx1, fy1 = (float(value) for value in _bbox_from_screen_points(quad))
        radius = max(1.8, min(4.8, min(fx1 - fx0, fy1 - fy0) * 0.085))
        bboxes.append(_draw_small_ellipse(draw, point_on_quad(quad, u, v), radius, radius * 0.90, fill=(38, 44, 52)))

    add_pip(face_quads[0], 0.50, 0.50)
    for u, v in ((0.36, 0.35), (0.64, 0.65)):
        add_pip(face_quads[1], u, v)
    for u, v in ((0.32, 0.28), (0.50, 0.50), (0.68, 0.72)):
        add_pip(face_quads[2], u, v)
    return _bbox_union(*bboxes)


def _draw_kite_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    diamond = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.98), (0.78, 0.12), (0.0, -0.98), (-0.78, 0.12)])
    kite_fill = _tint(fill, 0.06)
    draw.polygon(diamond, fill=kite_fill)
    _draw_polyline(draw, diamond, fill=_shade(kite_fill, 0.45), width=2)
    cross_a = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.98), (0.0, -0.98)])
    cross_b = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.78, 0.12), (0.78, 0.12)])
    tail = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.98), (0.16, -1.22), (-0.10, -1.46), (0.12, -1.70)])
    rib_fill = (248, 250, 245) if sum(kite_fill) < 430 else _shade(kite_fill, 0.48)
    draw.line(cross_a, fill=rib_fill, width=2)
    draw.line(cross_b, fill=rib_fill, width=2)
    draw.line(tail, fill=(56, 65, 74), width=2)
    return _bbox_union(_bbox_from_screen_points(diamond), _bbox_from_screen_points(cross_a), _bbox_from_screen_points(cross_b), _bbox_from_screen_points(tail))


def _draw_cactus_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.18, -1.0),
        (0.18, -1.0),
        (0.18, 0.70),
        (0.30, 0.92),
        (0.12, 1.0),
        (-0.12, 1.0),
        (-0.30, 0.92),
        (-0.18, 0.70),
    ]
    trunk_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(72, 151, 91), profile_xz=profile, inset_scale=0.72)
    bboxes = [trunk_bbox]
    arms = [
        [(-0.18, 0.12), (-0.48, 0.12), (-0.48, 0.52), (-0.68, 0.52), (-0.68, -0.08), (-0.18, -0.08)],
        [(0.18, 0.26), (0.50, 0.26), (0.50, 0.66), (0.70, 0.66), (0.70, 0.06), (0.18, 0.06)],
    ]
    for arm in arms:
        points = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=arm)
        draw.polygon(points, fill=(62, 135, 82))
        _draw_polyline(draw, points, fill=(34, 94, 56), width=2)
        bboxes.append(_bbox_from_screen_points(points))
    for px in (-0.08, 0.0, 0.08):
        stripe = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px, -0.74), (px * 0.6, 0.72)])
        draw.line(stripe, fill=(116, 188, 127), width=1)
        bboxes.append(_bbox_from_screen_points(stripe))
    return _bbox_union(*bboxes)


__all__ = [
    "_draw_pencil_object",
    "_draw_thumb_pin_object",
    "_draw_flat_rect_object",
    "_draw_ticket_tag_object",
    "_draw_puzzle_piece_object",
    "_draw_candy_disc_object",
    "_draw_cd_object",
    "_draw_berry_object",
    "_draw_marble_object",
    "_draw_bead_object",
    "_draw_dot_object",
    "_draw_button_object",
    "_draw_plate_object",
    "_draw_bowl_object",
    "_draw_screw_object",
    "_draw_bolt_object",
    "_draw_hex_nut_object",
    "_draw_washer_object",
    "_draw_paper_clip_object",
    "_draw_u_bolt_object",
    "_draw_nail_object",
    "_draw_rod_object",
    "_draw_stick_object",
    "_draw_tube_object",
    "_draw_clip_object",
    "_draw_socket_object",
    "_draw_magnet_object",
    "_draw_heater_object",
    "_draw_flower_object",
    "_draw_plant_pot_object",
    "_draw_glass_object",
    "_draw_jar_object",
    "_draw_can_object",
    "_draw_lid_object",
    "_draw_pillow_cushion_object",
    "_draw_stool_object",
    "_draw_drawer_object",
    "_draw_cap_object",
    "_draw_bucket_object",
    "_draw_tray_object",
    "_draw_coaster_object",
    "_draw_rose_object",
    "_draw_banana_object",
    "_draw_tomato_object",
    "_draw_peanut_object",
    "_draw_coffee_bean_object",
    "_draw_hook_object",
    "_draw_bracket_object",
    "_draw_battery_object",
    "_draw_tape_roll_object",
    "_draw_bag_object",
    "_draw_chess_piece_object",
    "_draw_hanger_object",
    "_draw_light_bulb_object",
    "_draw_egg_object",
    "_draw_chili_object",
    "_draw_fork_object",
    "_draw_knife_object",
    "_draw_spoon_object",
    "_draw_calculator_object",
    "_draw_dice_object",
    "_draw_kite_object",
    "_draw_cactus_object",
]
