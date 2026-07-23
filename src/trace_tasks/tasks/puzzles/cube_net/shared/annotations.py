"""Annotation projection helpers for cube-net puzzle tasks."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from trace_tasks.core.types import TypedValue


def round_annotation_bbox(bbox: Sequence[float]) -> list[float]:
    """Round a pixel bbox into the JSON-stable annotation representation."""

    return [round(float(value), 3) for value in bbox]


def bbox_typed_value(bbox: Sequence[float]) -> TypedValue:
    """Build a typed scalar bbox annotation from one visual witness."""

    return TypedValue(type="bbox", value=round_annotation_bbox(bbox))


def projected_bbox(bbox: Sequence[float]) -> Dict[str, Any]:
    """Build the projected-annotation payload for one bbox witness."""

    value = round_annotation_bbox(bbox)
    return {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
        "value": list(value),
    }


__all__ = [
    "bbox_typed_value",
    "projected_bbox",
    "round_annotation_bbox",
]
