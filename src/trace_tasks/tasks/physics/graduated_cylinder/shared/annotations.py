"""Annotation helpers for graduated-cylinder diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def bbox(values: Sequence[float]) -> list[float]:
    """Return one JSON-stable pixel bounding box."""

    return [round(float(value), 3) for value in values]


def projected_bbox_map(annotation_value: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Return trace metadata for a bbox-map annotation."""

    boxes = {str(key): list(value) for key, value in annotation_value.items()}
    return {
        "type": "bbox_map",
        "bbox_map": dict(boxes),
        "pixel_bbox_map": dict(boxes),
    }


def projected_bbox(annotation_value: Sequence[float]) -> dict[str, Any]:
    """Return trace metadata for one scalar bbox annotation."""

    box = list(annotation_value)
    return {
        "type": "bbox",
        "bbox": list(box),
        "pixel_bbox": list(box),
    }
