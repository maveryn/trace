"""Annotation projection helpers for wave-interference diagrams."""

from __future__ import annotations

from typing import List, Sequence


def point(values: Sequence[float]) -> List[float]:
    """Return a stable rounded pixel point."""

    if len(values) != 2:
        raise ValueError("point annotation requires exactly two coordinates")
    return [round(float(values[0]), 3), round(float(values[1]), 3)]


def bbox(values: Sequence[float]) -> List[float]:
    """Return a stable rounded pixel bbox."""

    if len(values) != 4:
        raise ValueError("bbox requires exactly four coordinates")
    return [round(float(value), 3) for value in values]


def segment(start: Sequence[float], end: Sequence[float]) -> List[List[float]]:
    """Return one stable undirected pixel segment witness."""

    return [point(start), point(end)]


def segment_set(values: Sequence[Sequence[Sequence[float]]]) -> List[List[List[float]]]:
    """Return a stable list of segment witnesses."""

    return [segment(raw[0], raw[1]) for raw in values]


def bbox_union(*boxes: Sequence[float], padding: float = 0.0) -> List[float]:
    """Return the union of one or more pixel bboxes."""

    if not boxes:
        raise ValueError("cannot union an empty bbox sequence")
    pad = float(padding)
    return bbox(
        (
            min(float(box[0]) for box in boxes) - pad,
            min(float(box[1]) for box in boxes) - pad,
            max(float(box[2]) for box in boxes) + pad,
            max(float(box[3]) for box in boxes) + pad,
        )
    )


__all__ = ["bbox", "bbox_union", "point", "segment", "segment_set"]
