"""Annotation helpers for stack-stability diagrams."""

from __future__ import annotations

from typing import List, Sequence


def normalize_stack_annotation_bbox(annotation_bbox: Sequence[float]) -> List[float]:
    """Return a normalized scalar bbox for the selected stack witness."""

    values = [round(float(value), 3) for value in annotation_bbox]
    if len(values) != 4:
        raise ValueError(f"stack annotation bbox must have four values, got {values}")
    x0, y0, x1, y1 = values
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"invalid stack annotation bbox: {values}")
    return values


__all__ = ["normalize_stack_annotation_bbox"]
