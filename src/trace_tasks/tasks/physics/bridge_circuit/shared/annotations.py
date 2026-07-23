"""Annotation helpers for bridge-circuit visual witnesses."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .state import RESISTOR_LABELS


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def clip_bbox(values: Sequence[float], *, width: int, height: int) -> list[float]:
    """Clip a bbox to the image bounds."""

    x0, y0, x1, y1 = [float(value) for value in values]
    return bbox(
        (
            max(0.0, min(float(width), min(x0, x1))),
            max(0.0, min(float(height), min(y0, y1))),
            max(0.0, min(float(width), max(x0, x1))),
            max(0.0, min(float(height), max(y0, y1))),
        )
    )


def union_bbox(*bboxes: Sequence[float]) -> list[float]:
    """Return the union of one or more bboxes."""

    usable = [tuple(float(value) for value in box) for box in bboxes if len(box) == 4]
    if not usable:
        return [0.0, 0.0, 0.0, 0.0]
    return bbox(
        (
            min(box[0] for box in usable),
            min(box[1] for box in usable),
            max(box[2] for box in usable),
            max(box[3] for box in usable),
        )
    )


def normalize_annotation_bbox_map(
    values: Mapping[str, Sequence[float]],
    *,
    missing_resistor: str,
    width: int,
    height: int,
) -> Dict[str, list[float]]:
    """Normalize the role-keyed bridge-circuit witness boxes."""

    normalized = {
        str(key): clip_bbox(value, width=int(width), height=int(height))
        for key, value in values.items()
    }
    known_labels = tuple(label for label in RESISTOR_LABELS if label != str(missing_resistor))
    required = (*known_labels, "target_resistor", "zero_meter")
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"bridge-circuit annotation is missing keys: {missing}")
    return {str(key): list(normalized[str(key)]) for key in required}


__all__ = ["bbox", "clip_bbox", "normalize_annotation_bbox_map", "union_bbox"]
