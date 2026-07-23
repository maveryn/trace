"""Annotation serialization helpers for logic-gate circuit scenes."""

from __future__ import annotations

from typing import Mapping, Sequence


def round_points(points: Sequence[Sequence[float]]) -> list[list[float]]:
    """Round point coordinates for stable trace serialization."""

    return [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in points]


def round_bbox_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Round bbox coordinates for stable trace serialization."""

    return {str(key): [round(float(value), 3) for value in bbox] for key, bbox in mapping.items()}


def projected_point_set(points: Sequence[Sequence[float]]) -> dict[str, object]:
    """Return the projected annotation payload for unordered point witnesses."""

    point_set = round_points(points)
    return {
        "type": "point_set",
        "point_set": list(point_set),
        "pixel_point_set": list(point_set),
        "value": list(point_set),
    }


def projected_bbox_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, object]:
    """Return the projected annotation payload for keyed bbox witnesses."""

    bbox_map = round_bbox_map(mapping)
    return {
        "type": "bbox_map",
        "bbox_map": dict(bbox_map),
        "pixel_bbox_map": dict(bbox_map),
        "value": dict(bbox_map),
    }


def projected_bbox(bbox: Sequence[float]) -> dict[str, object]:
    """Return the projected annotation payload for one bbox witness."""

    value = [round(float(item), 3) for item in bbox]
    return {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
        "value": list(value),
    }


def projected_bbox_set(bboxes: Sequence[Sequence[float]]) -> dict[str, object]:
    """Return the projected annotation payload for unordered bbox witnesses."""

    value = [[round(float(item), 3) for item in bbox] for bbox in bboxes]
    return {
        "type": "bbox_set",
        "bbox_set": list(value),
        "pixel_bbox_set": list(value),
        "value": list(value),
    }
