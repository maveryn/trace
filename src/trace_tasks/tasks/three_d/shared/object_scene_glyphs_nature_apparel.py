"""Nature, clothing, and container glyphs for shared three_d object scenes."""

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


def _draw_apple_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (0.0, 0.72),
        (0.22, 0.88),
        (0.58, 0.70),
        (0.78, 0.28),
        (0.68, -0.42),
        (0.34, -0.88),
        (0.0, -0.96),
        (-0.34, -0.88),
        (-0.68, -0.42),
        (-0.78, 0.28),
        (-0.58, 0.70),
        (-0.22, 0.88),
    ]
    body_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(204, 58, 50), profile_xz=profile, inset_scale=0.72)
    notch = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.18, 0.72), (0.0, 0.60), (0.18, 0.72)])
    stem = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.02, 0.68), (0.16, 1.02)])
    leaf = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.14, 0.86), (0.58, 0.96), (0.28, 0.66)])
    draw.line(notch, fill=(136, 38, 38), width=2)
    draw.line(stem, fill=(84, 57, 32), width=3)
    draw.polygon(leaf, fill=(73, 143, 70))
    _draw_polyline(draw, leaf, fill=(34, 88, 42), width=1)
    return _bbox_union(body_bbox, _bbox_from_screen_points(notch), _bbox_from_screen_points(stem), _bbox_from_screen_points(leaf))


def _draw_carrot_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.58, 0.54), (0.58, 0.54), (0.18, -0.96), (0.0, -1.10), (-0.18, -0.96)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(226, 118, 38), profile_xz=profile, inset_scale=0.0)
    leaves = [
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.00, 0.50), (-0.42, 1.02)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.00, 0.54), (0.00, 1.12)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.00, 0.50), (0.42, 1.02)]),
    ]
    bboxes = [bbox]
    for leaf in leaves:
        draw.line(leaf, fill=(66, 140, 70), width=4)
        bboxes.append(_bbox_from_screen_points(leaf))
    return _bbox_union(*bboxes)


def _draw_pear_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.40, 0.92), (0.40, 0.92), (0.64, 0.42), (0.82, -0.22), (0.48, -0.92), (0.0, -1.04), (-0.48, -0.92), (-0.82, -0.22), (-0.64, 0.42)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(168, 181, 75), profile_xz=profile, inset_scale=0.0)
    stem = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.00, 0.82), (0.12, 1.10)])
    draw.line(stem, fill=(87, 58, 34), width=3)
    return _bbox_union(bbox, _bbox_from_screen_points(stem))


def _draw_fish_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    body_profile = [
        (-0.66, 0.00),
        (-0.44, 0.38),
        (0.04, 0.58),
        (0.58, 0.42),
        (0.96, 0.00),
        (0.58, -0.42),
        (0.04, -0.58),
        (-0.44, -0.38),
    ]
    tail = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.62, 0.00), (-1.12, 0.62), (-0.96, 0.00), (-1.12, -0.62)])
    dorsal = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.14, 0.48), (0.20, 0.92), (0.44, 0.42)])
    ventral = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.02, -0.48), (0.28, -0.84), (0.50, -0.40)])
    draw.polygon(tail, fill=(50, 120, 162))
    _draw_polyline(draw, tail, fill=(25, 62, 82), width=2)
    draw.polygon(dorsal, fill=(55, 132, 170))
    draw.polygon(ventral, fill=(55, 132, 170))
    body_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(70, 154, 190), profile_xz=body_profile, inset_scale=0.0)
    pectoral = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.14, -0.04), (0.46, -0.34), (0.40, 0.02)])
    draw.polygon(pectoral, fill=(44, 116, 158))
    _draw_polyline(draw, pectoral, fill=(25, 62, 82), width=1)
    gill = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.50, 0.26), (0.42, -0.24)])
    mouth = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.80, -0.04), (0.96, 0.02)])
    draw.line(gill, fill=(25, 62, 82), width=2)
    draw.line(mouth, fill=(25, 62, 82), width=2)
    eye = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.64, 0.20)])[0]
    radius = max(2.0, min(5.0, float(frame.scale) * float(spec["dimensions_xyz"][0]) * 0.013))
    draw.ellipse((eye[0] - radius * 1.35, eye[1] - radius * 1.35, eye[0] + radius * 1.35, eye[1] + radius * 1.35), fill=(245, 248, 245), outline=(25, 62, 82), width=1)
    draw.ellipse((eye[0] - radius * 0.58, eye[1] - radius * 0.58, eye[0] + radius * 0.58, eye[1] + radius * 0.58), fill=(18, 24, 30))
    return _bbox_union(
        body_bbox,
        _bbox_from_screen_points(tail),
        _bbox_from_screen_points(dorsal),
        _bbox_from_screen_points(ventral),
        _bbox_from_screen_points(pectoral),
        _bbox_from_screen_points(gill),
        _bbox_from_screen_points(mouth),
        [eye[0] - radius * 1.35, eye[1] - radius * 1.35, eye[0] + radius * 1.35, eye[1] + radius * 1.35],
    )


def _draw_leaf_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.94, 0.00), (-0.54, 0.54), (0.00, 0.86), (0.54, 0.54), (0.94, 0.00), (0.54, -0.54), (0.00, -0.86), (-0.54, -0.54)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(82, 153, 75), profile_xz=profile, inset_scale=0.0)
    vein = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.82, 0.00), (0.82, 0.00)])
    draw.line(vein, fill=(44, 98, 52), width=2)
    bboxes = [bbox, _bbox_from_screen_points(vein)]
    for px in (-0.38, 0.05, 0.42):
        for pz in (-0.32, 0.32):
            detail = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(px, 0.0), (px + 0.20, pz)])
            draw.line(detail, fill=(55, 114, 60), width=1)
            bboxes.append(_bbox_from_screen_points(detail))
    return _bbox_union(*bboxes)


def _draw_feather_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.10, -0.96), (-0.56, -0.36), (-0.64, 0.30), (-0.26, 0.88), (0.12, 1.00), (0.52, 0.44), (0.46, -0.24)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(173, 183, 155), profile_xz=profile, inset_scale=0.0)
    shaft = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.02, -0.98), (0.10, 0.90)])
    draw.line(shaft, fill=(78, 72, 56), width=2)
    bboxes = [bbox, _bbox_from_screen_points(shaft)]
    for pz in (-0.48, -0.18, 0.12, 0.42):
        left = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.02, pz), (-0.38, pz + 0.15)])
        right = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.04, pz + 0.08), (0.34, pz + 0.24)])
        draw.line(left, fill=(103, 108, 89), width=1)
        draw.line(right, fill=(103, 108, 89), width=1)
        bboxes.extend([_bbox_from_screen_points(left), _bbox_from_screen_points(right)])
    return _bbox_union(*bboxes)


def _draw_shoe_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.94, -0.72),
        (-0.52, -0.82),
        (0.44, -0.82),
        (0.94, -0.60),
        (0.84, -0.34),
        (0.42, -0.14),
        (0.08, 0.10),
        (-0.36, 0.06),
        (-0.70, -0.22),
    ]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(72, 104, 151), profile_xz=profile, inset_scale=0.0)
    sole = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.74, -0.72), (0.58, -0.72), (0.88, -0.58)])
    tongue = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.18, -0.20), (0.18, -0.08), (0.02, 0.08)])
    laces = [
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.26, -0.24), (0.18, -0.34)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.18, -0.08), (0.28, -0.20)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.08, 0.04), (0.36, -0.08)]),
    ]
    heel = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.74, -0.22), (-0.62, -0.62)])
    draw.line(sole, fill=(37, 40, 44), width=3)
    draw.polygon(tongue, fill=(51, 77, 120))
    _draw_polyline(draw, tongue, fill=(30, 43, 66), width=1)
    for lace in laces:
        draw.line(lace, fill=(238, 240, 235), width=2)
    draw.line(heel, fill=(40, 49, 66), width=2)
    return _bbox_union(bbox, _bbox_from_screen_points(sole), _bbox_from_screen_points(tongue), _bbox_from_screen_points(heel), *[_bbox_from_screen_points(lace) for lace in laces])


def _draw_glove_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.62, -0.92),
        (0.34, -0.92),
        (0.52, -0.48),
        (0.50, -0.18),
        (0.88, 0.02),
        (0.78, 0.34),
        (0.48, 0.26),
        (0.44, 0.90),
        (0.28, 1.04),
        (0.12, 0.44),
        (0.04, 1.08),
        (-0.12, 1.08),
        (-0.16, 0.42),
        (-0.34, 1.00),
        (-0.50, 0.92),
        (-0.38, 0.34),
        (-0.58, 0.76),
        (-0.76, 0.62),
        (-0.58, -0.04),
        (-0.76, -0.36),
    ]
    glove_fill = _tint(fill, 0.06)
    seam_fill = _shade(glove_fill, 0.55)
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=glove_fill, profile_xz=profile, inset_scale=0.0)
    palm = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.48, -0.36), (0.34, -0.36)])
    finger_seams = [
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.22, 0.42), (0.28, 0.82)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.06, 0.40), (-0.06, 0.88)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.28, 0.36), (-0.36, 0.76)]),
        _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.44, -0.02), (0.68, 0.18)]),
    ]
    draw.line(palm, fill=seam_fill, width=2)
    for seam in finger_seams:
        draw.line(seam, fill=seam_fill, width=2)
    return _bbox_union(bbox, _bbox_from_screen_points(palm), *[_bbox_from_screen_points(seam) for seam in finger_seams])


def _draw_hat_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    brim = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, 0.0), dimensions_xyz=(width, depth, height * 0.14))
    crown_spec = _sub_box_spec(spec, offset_xyz=(0.0, 0.0, height * 0.08), dimensions_xyz=(width * 0.66, depth * 0.70, height * 0.76))
    brim_bbox = _draw_cylinder_object(draw, brim, camera=camera, frame=frame, fill=_shade(fill, 0.62))
    crown_profile = [
        (-0.76, -0.88),
        (0.76, -0.88),
        (0.66, 0.24),
        (0.42, 0.72),
        (0.00, 0.92),
        (-0.42, 0.72),
        (-0.66, 0.24),
    ]
    crown = _draw_upright_profile_object(draw, crown_spec, camera=camera, frame=frame, fill=_tint(fill, 0.08), profile_xz=crown_profile, inset_scale=0.0)
    band = _upright_screen_points(crown_spec, camera=camera, frame=frame, profile_xz=[(-0.62, -0.36), (0.62, -0.36)])
    brim_curve = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.84, -0.70), (-0.36, -0.56), (0.36, -0.56), (0.84, -0.70)])
    draw.line(band, fill=_shade(fill, 0.36), width=4)
    draw.line(brim_curve, fill=_shade(fill, 0.40), width=2)
    return _bbox_union(brim_bbox, crown, _bbox_from_screen_points(band), _bbox_from_screen_points(brim_curve))


def _draw_helmet_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.88, -0.12),
        (-0.74, 0.46),
        (-0.36, 0.86),
        (0.00, 0.98),
        (0.42, 0.84),
        (0.76, 0.46),
        (0.90, -0.14),
        (0.68, -0.74),
        (0.24, -0.58),
        (0.00, -0.84),
        (-0.26, -0.58),
        (-0.70, -0.74),
    ]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=_tint(fill, 0.06), profile_xz=profile, inset_scale=0.0)
    visor = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.58, 0.20), (0.58, 0.20), (0.46, -0.18), (0.12, -0.32), (-0.46, -0.18)])
    chin = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.32, -0.30), (0.00, -0.56), (0.32, -0.30), (0.20, -0.74), (-0.20, -0.74)])
    ridge = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.86), (0.0, 0.28)])
    cheek_left = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.68, 0.12), (-0.46, -0.20), (-0.48, -0.52)])
    cheek_right = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.68, 0.12), (0.46, -0.20), (0.48, -0.52)])
    draw.polygon(visor, fill=(74, 190, 218))
    _draw_polyline(draw, visor, fill=(15, 28, 42), width=2)
    draw.polygon(chin, fill=_shade(fill, 0.46))
    _draw_polyline(draw, chin, fill=_shade(fill, 0.30), width=1)
    draw.line(ridge, fill=_tint(fill, 0.36), width=3)
    draw.line(cheek_left, fill=_shade(fill, 0.40), width=3)
    draw.line(cheek_right, fill=_shade(fill, 0.40), width=3)
    return _bbox_union(
        bbox,
        _bbox_from_screen_points(visor),
        _bbox_from_screen_points(chin),
        _bbox_from_screen_points(ridge),
        _bbox_from_screen_points(cheek_left),
        _bbox_from_screen_points(cheek_right),
    )


def _draw_cup_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    body = _sub_box_spec(spec, offset_xyz=(-width * 0.06, 0.0, 0.0), dimensions_xyz=(width * 0.76, depth * 0.82, height * 0.86))
    body_bbox = _draw_cylinder_object(draw, body, camera=camera, frame=frame, fill=_tint(fill, 0.10))
    rim_center = _project_xy((float(spec["world_xyz"][0]) - width * 0.06, float(spec["world_xyz"][1]), float(spec["world_xyz"][2]) + height * 0.34), camera, frame)
    rim_radius = max(5.0, _radius_px_for_object(body, camera, frame) * 0.58)
    rim_h = max(4.0, rim_radius * 0.28)
    draw.ellipse(
        (rim_center[0] - rim_radius, rim_center[1] - rim_h, rim_center[0] + rim_radius, rim_center[1] + rim_h),
        outline=(35, 42, 50),
        width=2,
    )
    x0, y0, x1, y1 = (float(value) for value in body_bbox)
    body_w = max(1.0, x1 - x0)
    body_h = max(1.0, y1 - y0)
    top_body = (x1 - body_w * 0.12, y0 + body_h * 0.35)
    bottom_body = (x1 - body_w * 0.12, y0 + body_h * 0.64)
    outer_top = (x1 + body_w * 0.28, y0 + body_h * 0.38)
    outer_mid = (x1 + body_w * 0.42, y0 + body_h * 0.50)
    outer_bottom = (x1 + body_w * 0.28, y0 + body_h * 0.62)
    handle_path = [top_body, outer_top, outer_mid, outer_bottom, bottom_body]
    draw.line(handle_path, fill=(35, 42, 50), width=8, joint="curve")
    draw.line(handle_path, fill=_tint(fill, 0.10), width=5, joint="curve")
    attachment_boxes = []
    for cx, cy in (top_body, bottom_body):
        pad = [cx - body_w * 0.075, cy - body_h * 0.060, cx + body_w * 0.105, cy + body_h * 0.060]
        draw.ellipse(pad, fill=_tint(fill, 0.10), outline=(35, 42, 50), width=1)
        attachment_boxes.append(pad)
    connector_boxes = [
        _bbox_from_screen_points(handle_path),
    ]
    return _bbox_union(body_bbox, *connector_boxes, *attachment_boxes)


def _draw_bottle_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [
        (-0.34, 0.96),
        (0.34, 0.96),
        (0.34, 0.62),
        (0.56, 0.40),
        (0.62, -0.66),
        (0.42, -0.96),
        (-0.42, -0.96),
        (-0.62, -0.66),
        (-0.56, 0.40),
        (-0.34, 0.62),
    ]
    body_bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(82, 151, 134), profile_xz=profile, inset_scale=0.72)
    label = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.40, -0.38), (0.40, -0.38), (0.40, 0.04), (-0.40, 0.04)])
    cap = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.36, 0.86), (0.36, 0.86), (0.36, 1.02), (-0.36, 1.02)])
    shoulder = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.54, 0.38), (-0.30, 0.56), (0.30, 0.56), (0.54, 0.38)])
    draw.polygon(cap, fill=(58, 67, 74))
    _draw_polyline(draw, cap, fill=(32, 38, 44), width=1)
    draw.polygon(label, fill=(230, 235, 210))
    _draw_polyline(draw, label, fill=(83, 96, 92), width=1)
    draw.line(shoulder, fill=(48, 105, 96), width=1)
    return _bbox_union(body_bbox, _bbox_from_screen_points(label), _bbox_from_screen_points(cap), _bbox_from_screen_points(shoulder))


def _draw_vase_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    profile = [(-0.30, 0.92), (0.30, 0.92), (0.44, 0.38), (0.72, -0.22), (0.46, -0.92), (-0.46, -0.92), (-0.72, -0.22), (-0.44, 0.38)]
    bbox = _draw_upright_profile_object(draw, spec, camera=camera, frame=frame, fill=(120, 157, 174), profile_xz=profile, inset_scale=0.72)
    return list(bbox)


def _draw_umbrella_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    canopy = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(-0.96, 0.20), (-0.62, 0.68), (0.00, 0.86), (0.62, 0.68), (0.96, 0.20), (0.58, 0.06), (0.20, 0.18), (-0.20, 0.06), (-0.58, 0.18)])
    draw.polygon(canopy, fill=_tint(fill, 0.08))
    _draw_polyline(draw, canopy, fill=_shade(fill, 0.42), width=2)
    shaft = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, 0.14), (0.0, -0.88)])
    hook = _upright_screen_points(spec, camera=camera, frame=frame, profile_xz=[(0.0, -0.88), (0.20, -0.98), (0.34, -0.78)])
    draw.line(shaft, fill=(54, 61, 68), width=3)
    draw.line(hook, fill=(54, 61, 68), width=3, joint="curve")
    return _bbox_union(_bbox_from_screen_points(canopy), _bbox_from_screen_points(shaft), _bbox_from_screen_points(hook))


__all__ = [
    "_draw_apple_object",
    "_draw_carrot_object",
    "_draw_pear_object",
    "_draw_fish_object",
    "_draw_leaf_object",
    "_draw_feather_object",
    "_draw_shoe_object",
    "_draw_glove_object",
    "_draw_hat_object",
    "_draw_helmet_object",
    "_draw_cup_object",
    "_draw_bottle_object",
    "_draw_vase_object",
    "_draw_umbrella_object",
]
