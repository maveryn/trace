"""Annotation projection helpers for free-body-force diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def clip_bbox(values: Sequence[float], *, width: int, height: int) -> list[float]:
    """Clamp one bbox to image bounds."""

    x0, y0, x1, y1 = [float(value) for value in values[:4]]
    return bbox(
        (
            max(0.0, min(float(width), min(x0, x1))),
            max(0.0, min(float(height), min(y0, y1))),
            max(0.0, min(float(width), max(x0, x1))),
            max(0.0, min(float(height), max(y0, y1))),
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
