"""Spatial primitive helpers for object-scene marked-point tasks."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from ...shared.object_scene import (
    _RenderParams,
    _object_screen_bbox,
    _project_screen,
)


POINT_MARKER_COUNT = 6
MARKER_WORLD_Z = 0.055
MIN_MARKER_SCREEN_SEPARATION_PX = 76.0


def demote_small_object_spec(spec: Mapping[str, Any], *, index: int, prefix: str) -> Dict[str, Any]:
    updated = dict(spec)
    shape_type = str(updated["shape_type"])
    updated["object_id"] = f"{prefix}_object_{int(index):02d}_{shape_type}"
    updated["object_role"] = "small_context"
    updated["is_answer_candidate"] = False
    for key in ("point_id", "point_label", "object_label"):
        updated.pop(key, None)
    return updated


def prompt_names_are_unique(specs: Sequence[Mapping[str, Any]]) -> bool:
    counts = Counter(str(spec["prompt_name"]) for spec in specs)
    return all(int(count) == 1 for count in counts.values())


def finalize_object_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
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
    return finalized_specs


def make_marker_record(*, marker_id: str, world_xyz: Sequence[float], surface_kind: str) -> Dict[str, Any]:
    return {
        "marker_id": str(marker_id),
        "surface_kind": str(surface_kind),
        "attached_object_id": None,
        "world_xyz": [round(float(value), 4) for value in world_xyz],
    }


def marker_screen_record(record: Mapping[str, Any], *, camera, frame, render_params: _RenderParams) -> Dict[str, Any]:
    screen = _project_screen(record["world_xyz"], camera, frame)
    x, y = float(screen[0]), float(screen[1])
    if not (
        58.0 <= x <= float(render_params.canvas_width) - 58.0
        and 58.0 <= y <= float(render_params.canvas_height) - 58.0
    ):
        raise ValueError("marked point projects outside the readable image area")
    finalized = dict(record)
    finalized.update(
        {
            "screen_xy": [round(float(x), 3), round(float(y), 3)],
            "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
            "camera_distance": round(float(screen[7]), 4),
        }
    )
    return finalized


def object_bboxes_by_id(*, specs: Sequence[Mapping[str, Any]], camera, frame) -> Dict[str, List[float]]:
    return {str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=10.0) for spec in specs}


def floor_sample_extent(render_params: _RenderParams) -> float:
    return min(2.82, max(2.1, float(render_params.room_extent) - 0.34))


def clear_of_existing_world_xy(
    *,
    x: float,
    y: float,
    existing_world_xy: Sequence[Sequence[float]],
    min_distance: float = 0.44,
) -> bool:
    return all(math.hypot(float(x) - float(other[0]), float(y) - float(other[1])) >= float(min_distance) for other in existing_world_xy)


def clear_of_objects(
    *,
    x: float,
    y: float,
    objects: Sequence[Mapping[str, Any]],
    clearance: float = 0.16,
) -> bool:
    return all(
        math.hypot(float(x) - float(obj["base_xyz"][0]), float(y) - float(obj["base_xyz"][1]))
        >= float(obj.get("footprint_radius", 0.42)) + float(clearance)
        for obj in objects
    )


def marker_screen_separation_ok(
    markers: Sequence[Mapping[str, Any]],
    *,
    min_px: float = MIN_MARKER_SCREEN_SEPARATION_PX,
) -> bool:
    centers = [(float(item["screen_xy"][0]), float(item["screen_xy"][1])) for item in markers]
    return all(
        math.hypot(float(a[0] - b[0]), float(a[1] - b[1])) >= float(min_px)
        for index, a in enumerate(centers)
        for b in centers[index + 1 :]
    )


def marker_object_overlap_ok(
    *,
    markers: Sequence[Mapping[str, Any]],
    object_bboxes: Sequence[Sequence[float]],
    intersection_area_fn,
    marker_radius: float = 24.0,
    max_intersection_area: float = 1500.0,
) -> bool:
    centers = [(float(item["screen_xy"][0]), float(item["screen_xy"][1])) for item in markers]
    return all(
        float(
            intersection_area_fn(
                [center[0] - marker_radius, center[1] - marker_radius, center[0] + marker_radius, center[1] + marker_radius],
                object_bbox,
            )
        )
        <= float(max_intersection_area)
        for center in centers
        for object_bbox in object_bboxes
    )


__all__ = [
    "MARKER_WORLD_Z",
    "MIN_MARKER_SCREEN_SEPARATION_PX",
    "POINT_MARKER_COUNT",
    "clear_of_existing_world_xy",
    "clear_of_objects",
    "demote_small_object_spec",
    "finalize_object_specs",
    "floor_sample_extent",
    "make_marker_record",
    "marker_object_overlap_ok",
    "marker_screen_record",
    "marker_screen_separation_ok",
    "object_bboxes_by_id",
    "prompt_names_are_unique",
]
