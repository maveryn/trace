"""Annotation helpers for buoyancy-density visual witnesses."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .state import ANNOTATION_KEYS


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def clip_bbox(values: Sequence[float], *, width: int, height: int) -> list[float]:
    """Clip a bbox to image bounds."""

    if len(values) != 4:
        raise ValueError("bbox must contain four values")
    x0, y0, x1, y1 = [float(value) for value in values]
    return bbox(
        (
            max(0.0, min(float(width), min(x0, x1))),
            max(0.0, min(float(height), min(y0, y1))),
            max(0.0, min(float(width), max(x0, x1))),
            max(0.0, min(float(height), max(y0, y1))),
        )
    )


def normalize_annotation_bbox_map(
    values: Mapping[str, Sequence[float]],
    *,
    width: int,
    height: int,
) -> Dict[str, list[float]]:
    """Normalize role-keyed witness boxes in stable public key order."""

    normalized = {
        str(key): clip_bbox(value, width=int(width), height=int(height))
        for key, value in values.items()
    }
    missing = [key for key in ANNOTATION_KEYS if key not in normalized]
    if missing:
        raise ValueError(f"buoyancy-density annotation is missing keys: {missing}")
    return {str(key): list(normalized[str(key)]) for key in ANNOTATION_KEYS}


__all__ = ["bbox", "clip_bbox", "normalize_annotation_bbox_map"]
