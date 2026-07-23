"""Annotation helpers for thermometer diagrams."""

from __future__ import annotations

from typing import List, Sequence


def bbox(values: Sequence[float]) -> List[float]:
    """Return a stable rounded pixel bbox."""

    return [round(float(value), 3) for value in values]


def bbox_union(boxes: Sequence[Sequence[float]]) -> List[float]:
    """Return the union of one or more pixel bboxes."""

    if not boxes:
        raise ValueError("cannot union an empty bbox sequence")
    return bbox(
        (
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        )
    )


__all__ = ["bbox", "bbox_union"]
