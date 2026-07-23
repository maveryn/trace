"""Annotation helpers for switch-circuit diagrams."""

from __future__ import annotations

from typing import List, Sequence


def bbox(values: Sequence[float]) -> List[float]:
    """Return a stable rounded pixel bbox."""

    return [round(float(value), 3) for value in values]


def union_bbox(*bboxes: Sequence[float]) -> List[float]:
    """Return the union of non-empty bboxes."""

    usable = [tuple(float(value) for value in item) for item in bboxes if len(item) == 4]
    if not usable:
        return [0.0, 0.0, 0.0, 0.0]
    return bbox(
        (
            min(item[0] for item in usable),
            min(item[1] for item in usable),
            max(item[2] for item in usable),
            max(item[3] for item in usable),
        )
    )


def clip_bbox(values: Sequence[float], *, width: int, height: int) -> List[float]:
    """Clip a bbox to image bounds."""

    x0, y0, x1, y1 = [float(value) for value in values]
    return bbox(
        (
            max(0.0, min(float(width), min(x0, x1))),
            max(0.0, min(float(height), min(y0, y1))),
            max(0.0, min(float(width), max(x0, x1))),
            max(0.0, min(float(height), max(y0, y1))),
        )
    )


__all__ = ["bbox", "clip_bbox", "union_bbox"]
