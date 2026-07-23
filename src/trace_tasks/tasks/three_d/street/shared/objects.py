"""Building and storefront rendering helpers for street-intersection scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ...shared.camera_projection import (
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    polygon_axis_line_segment as _polygon_axis_line_segment,
    project_screen as _project_screen,
    project_xy as _project_xy,
    screen_to_floor_xy as _screen_to_floor_xy,
)
from ...shared.color_variation import resolve_three_d_object_fill_rgb
from ...shared.object_resources import (
    BUILDING_STYLE_BASE_COLORS,
    BUILDING_STYLE_DIMENSION_FACTORS,
    BUILDING_STYLE_DISPLAY_NAMES,
    STREET_OBJECT_BASE_DIMENSIONS,
    STREET_OBJECT_COLORS,
    STREET_OBJECT_NAMES,
    STREET_RADIAL_OBJECT_TYPES,
    STREET_VEHICLE_OBJECT_TYPES,
)
from ...shared.object_scene_rendering import (
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
from ...shared.object_scene import _object_screen_bbox

from ...shared.street_object_rendering_common import *  # noqa: F403

def _draw_building_face_rect(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    face_axis: str,
    face_side: int,
    span0: float,
    span1: float,
    z0: float,
    z1: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int] | None = None,
    width: int = 1,
) -> List[float]:
    """Project one normalized facade rectangle onto a visible building face."""

    x, y, base_z = (float(value) for value in spec["base_xyz"])
    building_w, building_d, building_h = (float(value) for value in spec["dimensions_xyz"])
    span0 = max(0.0, min(1.0, float(span0)))
    span1 = max(0.0, min(1.0, float(span1)))
    if span1 <= span0:
        span1 = min(1.0, span0 + 0.02)
    z0 = max(0.0, min(1.04, float(z0)))
    z1 = max(0.0, min(1.04, float(z1)))
    if z1 <= z0:
        z1 = min(1.04, z0 + 0.02)
    if str(face_axis) == "x":
        fixed_x = x + int(face_side) * building_w * 0.501
        y0 = y - building_d * 0.5 + building_d * span0
        y1 = y - building_d * 0.5 + building_d * span1
        points = [
            (fixed_x, y0, base_z + building_h * z0),
            (fixed_x, y1, base_z + building_h * z0),
            (fixed_x, y1, base_z + building_h * z1),
            (fixed_x, y0, base_z + building_h * z1),
        ]
    else:
        fixed_y = y + int(face_side) * building_d * 0.501
        x0 = x - building_w * 0.5 + building_w * span0
        x1 = x - building_w * 0.5 + building_w * span1
        points = [
            (x0, fixed_y, base_z + building_h * z0),
            (x1, fixed_y, base_z + building_h * z0),
            (x1, fixed_y, base_z + building_h * z1),
            (x0, fixed_y, base_z + building_h * z1),
        ]
    projected = [_project_xy(point, camera, frame) for point in points]
    draw.polygon(projected, fill=fill)
    if outline is not None:
        draw.line(projected + [projected[0]], fill=outline, width=max(1, int(width)))
    return _bbox_union(*[[point[0], point[1], point[0], point[1]] for point in projected])

def _draw_building_window_grid(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    face_axis: str,
    face_side: int,
    cols: int,
    rows: int,
    z_min: float,
    z_max: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    span_margin: float = 0.12,
) -> List[List[float]]:
    """Draw repeated facade windows and return their projected bboxes."""

    cols = max(1, int(cols))
    rows = max(1, int(rows))
    span_width = max(0.12, 1.0 - 2.0 * float(span_margin))
    cell_w = span_width / float(cols)
    cell_h = max(0.05, (float(z_max) - float(z_min)) / float(rows))
    bboxes: List[List[float]] = []
    for col in range(cols):
        span0 = float(span_margin) + col * cell_w + cell_w * 0.17
        span1 = float(span_margin) + (col + 1) * cell_w - cell_w * 0.17
        for row in range(rows):
            row_z0 = float(z_min) + row * cell_h + cell_h * 0.20
            row_z1 = float(z_min) + (row + 1) * cell_h - cell_h * 0.18
            if span1 - span0 < 0.025 or row_z1 - row_z0 < 0.022:
                continue
            bboxes.append(
                _draw_building_face_rect(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=str(face_axis),
                    face_side=int(face_side),
                    span0=span0,
                    span1=span1,
                    z0=row_z0,
                    z1=row_z1,
                    fill=fill,
                    outline=outline,
                    width=1,
                )
            )
    return list(bboxes)

def _draw_building_vertical_glass(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    face_axis: str,
    face_side: int,
    cols: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> List[List[float]]:
    """Draw tower-like vertical glass panels on one facade."""

    bboxes: List[List[float]] = []
    cols = max(2, int(cols))
    for col in range(cols):
        cell_w = 0.80 / float(cols)
        span0 = 0.10 + col * cell_w + cell_w * 0.08
        span1 = 0.10 + (col + 1) * cell_w - cell_w * 0.08
        panel_fill = _tint(fill, 0.08) if col % 2 == 0 else _shade(fill, 0.92)
        bboxes.append(
            _draw_building_face_rect(
                draw,
                spec,
                camera=camera,
                frame=frame,
                face_axis=str(face_axis),
                face_side=int(face_side),
                span0=span0,
                span1=span1,
                z0=0.15,
                z1=0.91,
                fill=panel_fill,
                outline=outline,
                width=1,
            )
        )
    for row in range(1, 5):
        z = 0.15 + row * 0.152
        bboxes.append(
            _draw_building_face_rect(
                draw,
                spec,
                camera=camera,
                frame=frame,
                face_axis=str(face_axis),
                face_side=int(face_side),
                span0=0.11,
                span1=0.89,
                z0=z,
                z1=z + 0.010,
                fill=outline,
            )
        )
    return list(bboxes)

def _draw_building_horizontal_bands(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    face_axis: str,
    face_side: int,
    rows: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> List[List[float]]:
    bboxes: List[List[float]] = []
    rows = max(2, int(rows))
    for row in range(rows):
        cell_h = 0.70 / float(rows)
        z0 = 0.18 + row * cell_h + cell_h * 0.22
        z1 = 0.18 + (row + 1) * cell_h - cell_h * 0.22
        band_fill = _tint(fill, 0.10) if row % 2 == 0 else _shade(fill, 0.94)
        bboxes.append(
            _draw_building_face_rect(
                draw,
                spec,
                camera=camera,
                frame=frame,
                face_axis=str(face_axis),
                face_side=int(face_side),
                span0=0.11,
                span1=0.89,
                z0=z0,
                z1=z1,
                fill=band_fill,
                outline=outline,
                width=1,
            )
        )
    return list(bboxes)

def _draw_retail_front(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    face_axis: str,
    face_side: int,
    fill: Tuple[int, int, int],
) -> List[List[float]]:
    """Draw storefront awning, window, and doorway details on a facade."""

    awning_palette = [(176, 59, 62), (63, 127, 112), (201, 159, 68), (70, 95, 145)]
    accent = awning_palette[_stable_palette_index(str(spec.get("object_id", "")), len(awning_palette))]
    outline = (82, 64, 48)
    bboxes = [
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.12,
            span1=0.88,
            z0=0.48,
            z1=0.60,
            fill=accent,
            outline=outline,
            width=1,
        ),
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.16,
            span1=0.46,
            z0=0.16,
            z1=0.43,
            fill=(156, 190, 205),
            outline=outline,
            width=1,
        ),
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.54,
            span1=0.82,
            z0=0.15,
            z1=0.45,
            fill=_tint(fill, 0.20),
            outline=outline,
            width=1,
        ),
    ]
    return bboxes

def _draw_shopfront(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    face_axis: str,
    face_side: int,
    style: str,
    fill: Tuple[int, int, int],
) -> List[List[float]]:
    """Draw shop-specific facade bands, windows, door, and awning stripes."""

    accents = {
        "cafe_shop": (176, 75, 66),
        "market_shop": (72, 135, 86),
        "bookstore_shop": (82, 95, 151),
    }
    accent = accents.get(str(style), (176, 75, 66))
    trim = _shade(accent, 0.64)
    glass = (168, 204, 210)
    cream = (239, 225, 188)
    outline = (72, 62, 52)
    bboxes: List[List[float]] = [
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.08,
            span1=0.92,
            z0=0.52,
            z1=0.66,
            fill=accent,
            outline=outline,
            width=1,
        ),
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.12,
            span1=0.46,
            z0=0.14,
            z1=0.46,
            fill=glass,
            outline=outline,
            width=1,
        ),
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.56,
            span1=0.78,
            z0=0.12,
            z1=0.46,
            fill=_shade(fill, 0.74),
            outline=outline,
            width=1,
        ),
        _draw_building_face_rect(
            draw,
            spec,
            camera=camera,
            frame=frame,
            face_axis=str(face_axis),
            face_side=int(face_side),
            span0=0.10,
            span1=0.90,
            z0=0.46,
            z1=0.53,
            fill=trim,
            outline=outline,
            width=1,
        ),
    ]
    for stripe_index in range(5):
        span0 = 0.11 + float(stripe_index) * 0.156
        span1 = min(0.89, span0 + 0.116)
        bboxes.append(
            _draw_building_face_rect(
                draw,
                spec,
                camera=camera,
                frame=frame,
                face_axis=str(face_axis),
                face_side=int(face_side),
                span0=span0,
                span1=span1,
                z0=0.47,
                z1=0.52,
                fill=accent if stripe_index % 2 == 0 else cream,
            )
        )
    for span0, span1 in ((0.14, 0.34), (0.38, 0.50), (0.80, 0.90)):
        bboxes.append(
            _draw_building_face_rect(
                draw,
                spec,
                camera=camera,
                frame=frame,
                face_axis=str(face_axis),
                face_side=int(face_side),
                span0=span0,
                span1=span1,
                z0=0.69,
                z1=0.83,
                fill=_tint(glass, 0.12),
                outline=outline,
                width=1,
            )
        )
    return list(bboxes)

def _draw_styled_building_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
    fill: Tuple[int, int, int],
) -> List[float]:
    """Draw one building body plus style-specific facade geometry."""

    style = str(spec.get("building_style", "concrete_midrise"))
    bbox = _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    x, y, _z = (float(value) for value in spec["world_xyz"])
    sx = 1 if float(camera.camera_position[0]) >= x else -1
    sy = 1 if float(camera.camera_position[1]) >= y else -1
    feature_bboxes: List[List[float]] = [list(bbox)]

    if style in {"office_glass", "glass_tower", "concrete_midrise", "stucco_walkup"}:
        cap_h = max(0.035, min(0.075, height * 0.055))
        cap_fill = _shade(fill, 0.72) if style != "stucco_walkup" else (116, 80, 66)
        cap = _sub_box_spec(
            spec,
            offset_xyz=(0.0, 0.0, max(0.0, height - cap_h * 0.72)),
            dimensions_xyz=(width * 1.04, depth * 1.04, cap_h),
        )
        feature_bboxes.append(_draw_box_object(draw, cap, camera=camera, frame=frame, fill=cap_fill))

    visible_faces = (("x", sx), ("y", sy))
    for face_axis, face_side in visible_faces:
        face_span = depth if face_axis == "x" else width
        rows = max(2, min(6, int(round(height / 0.27))))
        cols = max(2, min(5, int(round(face_span / 0.24))))
        if style == "office_glass":
            feature_bboxes.extend(
                _draw_building_window_grid(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    cols=max(3, cols),
                    rows=max(3, rows),
                    z_min=0.18,
                    z_max=0.88,
                    fill=(166, 204, 221),
                    outline=(65, 91, 105),
                    span_margin=0.10,
                )
            )
        elif style == "glass_tower":
            feature_bboxes.extend(
                _draw_building_vertical_glass(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    cols=max(4, cols),
                    fill=(126, 180, 214),
                    outline=(58, 87, 111),
                )
            )
        elif style == "apartment_brick":
            feature_bboxes.extend(
                _draw_building_window_grid(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    cols=max(2, cols - 1),
                    rows=max(3, rows),
                    z_min=0.18,
                    z_max=0.86,
                    fill=(235, 218, 169),
                    outline=(94, 64, 52),
                    span_margin=0.15,
                )
            )
            feature_bboxes.extend(
                _draw_building_face_rect(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    span0=0.12,
                    span1=0.88,
                    z0=z,
                    z1=z + 0.014,
                    fill=(105, 58, 49),
                )
                for z in (0.37, 0.59, 0.81)
            )
        elif style == "retail_corner":
            feature_bboxes.extend(
                _draw_retail_front(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    fill=fill,
                )
            )
            feature_bboxes.extend(
                _draw_building_window_grid(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    cols=max(2, cols - 1),
                    rows=2,
                    z_min=0.66,
                    z_max=0.90,
                    fill=(208, 229, 220),
                    outline=(92, 78, 61),
                    span_margin=0.18,
                )
            )
        elif style in {"cafe_shop", "market_shop", "bookstore_shop"}:
            feature_bboxes.extend(
                _draw_shopfront(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    style=str(style),
                    fill=fill,
                )
            )
        elif style == "stucco_walkup":
            feature_bboxes.extend(
                _draw_building_window_grid(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    cols=max(2, cols - 1),
                    rows=max(2, rows - 1),
                    z_min=0.20,
                    z_max=0.80,
                    fill=(96, 132, 146),
                    outline=(119, 83, 63),
                    span_margin=0.17,
                )
            )
        else:
            feature_bboxes.extend(
                _draw_building_horizontal_bands(
                    draw,
                    spec,
                    camera=camera,
                    frame=frame,
                    face_axis=face_axis,
                    face_side=face_side,
                    rows=max(4, rows),
                    fill=(174, 191, 199),
                    outline=(92, 102, 108),
                )
            )
    return _bbox_union(*feature_bboxes)


__all__ = [
    '_draw_building_face_rect',
    '_draw_building_window_grid',
    '_draw_building_vertical_glass',
    '_draw_building_horizontal_bands',
    '_draw_retail_front',
    '_draw_shopfront',
    '_draw_styled_building_object',
]
