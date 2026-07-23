"""Trace fragment helpers for indoor-room public tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .annotations import (
    container_bbox_map,
    container_interior_bbox_map,
    furniture_bbox_map,
    placement_map,
    surface_bbox_map,
    surface_support_bbox_map,
)
from .state import RenderedIndoorRoomScene


def render_fallback_from_defaults(defaults: Any) -> dict[str, int]:
    """Return render fallback values from a task-local defaults object."""

    return {
        "canvas_width": int(defaults.canvas_width),
        "canvas_height": int(defaults.canvas_height),
        "object_size_min_px": int(defaults.object_size_min_px),
        "object_size_max_px": int(defaults.object_size_max_px),
        "render_scale": int(defaults.render_scale),
    }


def indoor_render_spec(scene: RenderedIndoorRoomScene, *, scene_id: str) -> dict[str, Any]:
    return {
        "canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
        "coord_space": "pixel",
        "scene_id": str(scene_id),
        "style": {
            "theme_id": str(scene.theme_id),
            "style_id": str(scene.style_id),
            "render_scale": int(scene.render_scale),
        },
    }


def indoor_base_render_map(
    scene: RenderedIndoorRoomScene,
    *,
    object_bboxes: Mapping[str, Sequence[float]],
    part_bboxes: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        "image_id": "img0",
        "object_bboxes_px": dict(object_bboxes),
        "part_bboxes_px": dict(part_bboxes),
        "surface_bboxes_px": surface_bbox_map(scene),
        "surface_support_bboxes_px": surface_support_bbox_map(scene),
        "container_bboxes_px": container_bbox_map(scene),
        "container_interior_bboxes_px": container_interior_bbox_map(scene),
        "furniture_bboxes_px": furniture_bbox_map(scene),
        "placements": placement_map(scene),
    }


def object_type_map(serialized_objects: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(obj["object_id"]): str(obj["object_type"]) for obj in serialized_objects}


__all__ = [
    "indoor_base_render_map",
    "indoor_render_spec",
    "object_type_map",
    "render_fallback_from_defaults",
]
