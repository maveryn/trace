"""Annotation projection helpers for fluid-flow diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def clamp_bbox(values: Sequence[float], *, width: int, height: int) -> list[float]:
    """Clamp one bbox to image bounds."""

    x0, y0, x1, y1 = [float(value) for value in values[:4]]
    return bbox(
        (
            max(0.0, min(float(width - 1), x0)),
            max(0.0, min(float(height - 1), y0)),
            max(1.0, min(float(width), x1)),
            max(1.0, min(float(height), y1)),
        )
    )


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
