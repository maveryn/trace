"""Annotation helpers for the spring physics scene."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .state import RenderedSpringScene

SPRING_EXTENSION_MARKER_ANNOTATION_MIN_HEIGHT_PX = 32.0


def expanded_spring_extension_marker_bbox(
    bbox: Sequence[float],
    *,
    canvas_width: int,
    canvas_height: int,
    min_height_px: float = SPRING_EXTENSION_MARKER_ANNOTATION_MIN_HEIGHT_PX,
) -> List[float]:
    """Return a marker box with enough vertical target area for annotation."""

    x0, y0, x1, y1 = [float(value) for value in bbox]
    current_height = float(y1 - y0)
    if current_height >= float(min_height_px):
        return [float(x0), float(y0), float(x1), float(y1)]
    center_y = float((y0 + y1) / 2.0)
    half_height = float(min_height_px) / 2.0
    expanded_y0 = max(0.0, center_y - half_height)
    expanded_y1 = min(float(canvas_height), center_y + half_height)
    if expanded_y1 - expanded_y0 < float(min_height_px):
        if expanded_y0 <= 0.0:
            expanded_y1 = min(float(canvas_height), float(min_height_px))
        elif expanded_y1 >= float(canvas_height):
            expanded_y0 = max(0.0, float(canvas_height) - float(min_height_px))
    return [
        max(0.0, float(x0)),
        round(float(expanded_y0), 3),
        min(float(canvas_width), float(x1)),
        round(float(expanded_y1), 3),
    ]


def spring_missing_value_annotation_map(rendered_scene: RenderedSpringScene) -> Dict[str, List[float]]:
    """Return role-keyed annotation boxes for a missing-value spring task."""

    entity_bbox_map = dict(rendered_scene.render_map.get("entity_bbox_map_px", {}))
    canvas_width, canvas_height = rendered_scene.image.size
    left_extension = expanded_spring_extension_marker_bbox(
        entity_bbox_map["left_extension_marker"],
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )
    right_extension = expanded_spring_extension_marker_bbox(
        entity_bbox_map["right_extension_marker"],
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )
    return {
        "reference_weight": [float(value) for value in entity_bbox_map["left_weight_block"]],
        "reference_extension": [float(value) for value in left_extension],
        "query_weight": [float(value) for value in entity_bbox_map["right_weight_block"]],
        "query_extension": [float(value) for value in right_extension],
    }


__all__ = [
    "SPRING_EXTENSION_MARKER_ANNOTATION_MIN_HEIGHT_PX",
    "expanded_spring_extension_marker_bbox",
    "spring_missing_value_annotation_map",
]
