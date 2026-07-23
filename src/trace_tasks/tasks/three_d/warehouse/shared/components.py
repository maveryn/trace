"""Warehouse scene-support rendering helpers."""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

from PIL import ImageDraw

from ...shared.object_scene import _CameraSpec, _ProjectionFrame
from ...shared.object_scene_rendering import (
    _bbox_union,
    _draw_box_object,
    _draw_box_parts_object,
    _sub_box_spec,
)
from .state import SHELF_LOAD_COLORS


def _draw_shelf_rack_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera: _CameraSpec,
    frame: _ProjectionFrame,
    fill: Tuple[int, int, int],
) -> List[float]:
    """Draw one shelf rack from frame and load components."""
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    shelf_style = str(spec.get("shelf_style", "open_frame"))
    level_fracs = [float(value) for value in spec.get("shelf_level_fracs", (0.10, 0.48, 0.86))]
    if not level_fracs:
        level_fracs = [0.10, 0.48, 0.86]
    beam_height = float(spec.get("shelf_beam_height", 0.09))
    post_width = float(spec.get("shelf_post_width", 0.08))
    beam_depth_scale = 0.90 if shelf_style == "open_frame" else 1.0
    if shelf_style == "heavy_low":
        beam_depth_scale = 1.08
    parts: List[Mapping[str, Any]] = []
    for level_frac in level_fracs:
        level_z = max(0.02, min(height - beam_height, height * float(level_frac)))
        parts.append(
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, 0.0, level_z),
                dimensions_xyz=(width, depth * beam_depth_scale, beam_height),
            )
        )
    post_height = height * (0.96 if shelf_style != "heavy_low" else 0.82)
    x_offsets = [-width * 0.43, width * 0.43]
    if shelf_style in {"loaded_bins", "heavy_low"}:
        x_offsets = [-width * 0.43, 0.0, width * 0.43]
    y_offsets = [-depth * 0.38, depth * 0.38]
    for offset_x in x_offsets:
        for offset_y in y_offsets:
            parts.append(
                _sub_box_spec(
                    spec,
                    offset_xyz=(offset_x, offset_y, 0.0),
                    dimensions_xyz=(post_width, post_width, post_height),
                )
            )
    if shelf_style in {"mixed_crates", "heavy_low"}:
        parts.append(
            _sub_box_spec(
                spec,
                offset_xyz=(0.0, -depth * 0.43, height * 0.18),
                dimensions_xyz=(width * 0.92, post_width * 0.62, height * 0.48),
            )
        )
    bboxes: List[List[float]] = [_draw_box_parts_object(draw, parts, camera=camera, frame=frame, fill=fill)]
    for load_index, raw_load in enumerate(spec.get("shelf_load_slots", ())):
        if not isinstance(raw_load, Mapping):
            continue
        color_index = int(raw_load.get("color_index", load_index))
        while color_index < 0:
            color_index += len(SHELF_LOAD_COLORS)
        while color_index >= len(SHELF_LOAD_COLORS):
            color_index -= len(SHELF_LOAD_COLORS)
        load_fill = SHELF_LOAD_COLORS[color_index]
        load_part = _sub_box_spec(
            spec,
            offset_xyz=(
                width * float(raw_load.get("x_frac", 0.0)),
                depth * float(raw_load.get("y_frac", 0.0)),
                height * float(raw_load.get("z_frac", 0.12)),
            ),
            dimensions_xyz=(
                width * float(raw_load.get("w_frac", 0.18)),
                depth * float(raw_load.get("d_frac", 0.46)),
                height * float(raw_load.get("h_frac", 0.16)),
            ),
        )
        bboxes.append(_draw_box_object(draw, load_part, camera=camera, frame=frame, fill=load_fill))
    return _bbox_union(*bboxes)


__all__ = ["_draw_shelf_rack_object"]
