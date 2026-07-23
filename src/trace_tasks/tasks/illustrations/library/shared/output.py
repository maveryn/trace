"""Trace fragment helpers for library public tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .annotations import book_bbox_map, book_point_map, section_bbox_map
from .state import RenderedLibraryScene


def render_fallback_from_defaults(defaults: Any) -> dict[str, int]:
    """Return render fallback values from a task-local defaults object."""

    return {
        "canvas_width": int(defaults.canvas_width),
        "canvas_height": int(defaults.canvas_height),
        "render_scale": int(defaults.render_scale),
    }


def library_render_spec(scene: RenderedLibraryScene, *, scene_id: str) -> dict[str, Any]:
    return {
        "canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
        "coord_space": "pixel",
        "scene_id": str(scene_id),
        "style": {
            "setting_id": str(scene.setting_id),
            "style_id": str(scene.style_id),
            "render_scale": int(scene.render_scale),
            "layout": dict(scene.layout),
        },
    }


def library_base_render_map(
    scene: RenderedLibraryScene,
    *,
    counted_book_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "image_id": "img0",
        "book_bboxes_px": book_bbox_map(scene),
        "book_points_px": book_point_map(scene),
        "section_bboxes_px": section_bbox_map(scene),
        "counted_book_ids": [str(book_id) for book_id in counted_book_ids],
    }


def library_scene_relations(
    *,
    prompt_query_key: str,
    section_key: str,
    color_name: str | None = None,
    orientation: str | None = None,
) -> dict[str, Any]:
    return {
        "prompt_query_key": str(prompt_query_key),
        "target_section_key": str(section_key),
        "target_color_name": color_name,
        "target_orientation": orientation,
    }


__all__ = [
    "library_base_render_map",
    "library_render_spec",
    "library_scene_relations",
    "render_fallback_from_defaults",
]
