"""Annotation projection helpers for maze-exit puzzle tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue

def _round_bbox(bbox: Sequence[float]) -> list[float]:
    """Normalize one image-pixel bbox into stable float coordinates."""

    return [round(float(value), 3) for value in bbox]


def _round_point(point: Sequence[float]) -> list[float]:
    """Normalize one image-pixel point into stable float coordinates."""

    return [round(float(value), 3) for value in point]


def item_bbox_set(
    item_bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Return bbox_set annotation artifacts for an ordered set of maze exits."""

    bboxes = [_round_bbox(item_bbox_map[str(item_id)]) for item_id in item_ids]
    projected = {
        "bbox_set": list(bboxes),
        "pixel_bbox_set": list(bboxes),
        "value": list(bboxes),
    }
    witness = {
        "type": "bbox_set",
        "value": list(bboxes),
        "ordered_item_ids": [str(item_id) for item_id in item_ids],
    }
    return TypedValue(type="bbox_set", value=list(bboxes)), projected, witness


def item_point_set(
    item_point_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Return point_set annotation artifacts for an ordered set of maze exits."""

    points = [_round_point(item_point_map[str(item_id)]) for item_id in item_ids]
    projected = {
        "point_set": list(points),
        "pixel_point_set": list(points),
        "value": list(points),
    }
    witness = {
        "type": "point_set",
        "value": list(points),
        "ordered_item_ids": [str(item_id) for item_id in item_ids],
    }
    return TypedValue(type="point_set", value=list(points)), projected, witness


def single_item_point(
    item_point_map: Mapping[str, Sequence[float]],
    item_id: str,
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Return scalar point annotation artifacts for one maze exit."""

    point = _round_point(item_point_map[str(item_id)])
    projected = {
        "point": list(point),
        "pixel_point": list(point),
        "value": list(point),
    }
    witness = {
        "type": "point",
        "value": list(point),
        "ordered_item_ids": [str(item_id)],
    }
    return TypedValue(type="point", value=list(point)), projected, witness


def single_item_bbox(
    item_bbox_map: Mapping[str, Sequence[float]],
    item_id: str,
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Return scalar bbox annotation artifacts for one maze exit."""

    bbox = _round_bbox(item_bbox_map[str(item_id)])
    projected = {
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
        "value": list(bbox),
    }
    witness = {
        "type": "bbox",
        "value": list(bbox),
        "ordered_item_ids": [str(item_id)],
    }
    return TypedValue(type="bbox", value=list(bbox)), projected, witness


__all__ = [
    "item_bbox_set",
    "item_point_set",
    "single_item_bbox",
    "single_item_point",
]
