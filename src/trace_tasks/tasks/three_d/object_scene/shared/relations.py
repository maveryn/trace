"""Shared relation-scene object helpers for 3D object-scene tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ...shared.object_scene import (
    _bbox_intersection_area,
    _make_object_spec,
    _object_screen_bbox,
    _project_screen,
    _sample_shape_dimensions,
)
from ...shared.object_scene_primitives import _sub_box_spec


def make_sampled_object(
    *,
    rng,
    object_id: str,
    shape_type: str,
    object_role: str,
    xy: Tuple[float, float],
    label: str | None = None,
) -> Dict[str, Any]:
    """Create one sampled object spec while preserving domain object-resource metadata."""
    dimensions_xyz, dimension_scale = _sample_shape_dimensions(str(shape_type), object_role=str(object_role), rng=rng)
    return _make_object_spec(
        object_id=str(object_id),
        shape_type=str(shape_type),
        object_role=str(object_role),
        xy=xy,
        dimensions_xyz=dimensions_xyz,
        dimension_scale=float(dimension_scale),
        label=label,
    )


def set_xy(spec: Mapping[str, Any], xy: Tuple[float, float]) -> Dict[str, Any]:
    """Move an object spec on the floor plane without changing its dimensions or identity."""
    updated = dict(spec)
    height = float(updated["dimensions_xyz"][2])
    updated["base_xyz"] = [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0]
    updated["world_xyz"] = [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(height * 0.5), 4)]
    return updated


def prompt_name(spec: Mapping[str, Any]) -> str:
    """Return the user-facing object name recorded on a sampled object spec."""
    return str(spec.get("prompt_name", spec.get("object_name", spec.get("shape_type", "object"))))


def support_part_specs(reference_spec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return visible support components for under-relation clearance checks."""
    width, depth, height = (float(value) for value in reference_spec["dimensions_xyz"])
    shape_type = str(reference_spec.get("shape_type", ""))
    if shape_type == "arch":
        post_w = width * 0.22
        lintel_h = height * 0.24
        return [
            _sub_box_spec(
                reference_spec,
                offset_xyz=(-width * 0.34, 0.0, 0.0),
                dimensions_xyz=(post_w, depth, height),
            ),
            _sub_box_spec(
                reference_spec,
                offset_xyz=(width * 0.34, 0.0, 0.0),
                dimensions_xyz=(post_w, depth, height),
            ),
            _sub_box_spec(
                reference_spec,
                offset_xyz=(0.0, 0.0, height - lintel_h),
                dimensions_xyz=(width, depth, lintel_h),
            ),
        ]
    if shape_type == "table":
        top_h = height * 0.18
        leg_w = min(width, depth) * 0.16
        leg_h = height - top_h
        parts = [
            _sub_box_spec(
                reference_spec,
                offset_xyz=(sx * width * 0.34, sy * depth * 0.34, 0.0),
                dimensions_xyz=(leg_w, leg_w, leg_h),
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        ]
        parts.append(
            _sub_box_spec(
                reference_spec,
                offset_xyz=(0.0, 0.0, leg_h),
                dimensions_xyz=(width, depth, top_h),
            )
        )
        return parts
    return [dict(reference_spec)]


def bbox_area(bbox: Sequence[float]) -> float:
    """Return the pixel-space area of a bbox."""
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def max_support_part_overlap_fraction(
    candidate_spec: Mapping[str, Any],
    reference_spec: Mapping[str, Any],
    *,
    camera,
    frame,
    pad_px: float = 4.0,
) -> float:
    """Measure how much one candidate bbox collides with visible support parts."""
    candidate_bbox = _object_screen_bbox(candidate_spec, camera, frame, pad_px=float(pad_px))
    candidate_area = max(1.0, bbox_area(candidate_bbox))
    overlaps = [
        _bbox_intersection_area(candidate_bbox, _object_screen_bbox(part, camera, frame, pad_px=float(pad_px)))
        for part in support_part_specs(reference_spec)
    ]
    return float(max(overlaps, default=0.0) / candidate_area)


def under_support_xy(
    reference_spec: Mapping[str, Any],
    *,
    camera,
    rng,
    index: int = 0,
    total: int = 1,
) -> Tuple[float, float]:
    """Place an under-relation object inside the open support footprint, not on the front skin."""
    ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["world_xyz"])
    width, depth, _height = (float(value) for value in reference_spec["dimensions_xyz"])
    to_camera_x = float(camera.camera_position[0]) - float(ref_x)
    to_camera_y = float(camera.camera_position[1]) - float(ref_y)
    length = max(1e-6, math.hypot(to_camera_x, to_camera_y))
    unit_x = float(to_camera_x) / float(length)
    unit_y = float(to_camera_y) / float(length)
    lateral_x = -unit_y
    lateral_y = unit_x
    slot_count = max(1, int(total))
    slot_index = max(0, min(int(index), slot_count - 1))
    if slot_count == 1:
        lateral_slot, away_slot = 0.0, 0.0
    elif slot_count == 2:
        lateral_slot, away_slot = ((-0.90, 0.0), (0.90, 0.0))[slot_index]
    elif slot_count == 3:
        lateral_slot, away_slot = ((-0.90, 0.0), (0.90, 0.0), (0.0, 0.55))[slot_index]
    else:
        lateral_slot, away_slot = ((-0.85, 0.0), (0.85, 0.0), (-0.85, 0.55), (0.85, 0.55))[
            min(slot_index, 3)
        ]
    shape_type = str(reference_spec.get("shape_type", ""))
    if shape_type == "arch":
        away_scale = 0.06
        away_spread = 0.08
        lateral_scale = 0.18
        jitter = 0.018
    else:
        away_scale = 0.11
        away_spread = 0.09
        lateral_scale = 0.26
        jitter = 0.022
    return (
        float(
            ref_x
            - unit_x * width * (away_scale + away_spread * away_slot)
            + lateral_x * width * lateral_scale * lateral_slot
            + rng.uniform(-jitter, jitter)
        ),
        float(
            ref_y
            - unit_y * depth * (away_scale + away_spread * away_slot)
            + lateral_y * depth * lateral_scale * lateral_slot
            + rng.uniform(-jitter, jitter)
        ),
    )


def can_place(candidate: Mapping[str, Any], placed: Sequence[Mapping[str, Any]], *, clearance: float = 0.12) -> bool:
    """Check floor-plane footprint separation before a relation scene accepts a placement."""
    cx, cy, _cz = (float(value) for value in candidate["world_xyz"])
    for item in placed:
        ix, iy, _iz = (float(value) for value in item["world_xyz"])
        min_distance = float(candidate["footprint_radius"]) + float(item["footprint_radius"]) + float(clearance)
        if math.hypot(float(cx - ix), float(cy - iy)) < min_distance:
            return False
    return True


def finalize_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
    """Attach projected screen/camera coordinates to object specs without altering semantic attrs."""
    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = _project_screen(spec["world_xyz"], camera, frame)
        finalized = dict(spec)
        finalized.update(
            {
                "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_specs.append(finalized)
    return list(finalized_specs)


__all__ = [
    "can_place",
    "bbox_area",
    "finalize_specs",
    "make_sampled_object",
    "max_support_part_overlap_fraction",
    "prompt_name",
    "set_xy",
    "support_part_specs",
    "under_support_xy",
]
