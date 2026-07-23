"""Annotation helpers for piston-cylinder diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ANNOTATION_KEYS: tuple[str, ...] = ("pressure_readout", "initial_cylinder", "final_cylinder")


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


__all__ = ["ANNOTATION_KEYS", "bbox", "projected_bbox_map"]
