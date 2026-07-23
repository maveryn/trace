"""Annotation helpers for Vernier-caliper visual witnesses."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence


ANNOTATION_KEYS: tuple[str, str] = (
    "vernier_zero_tick",
    "aligned_vernier_tick",
)


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def point(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel point."""

    if len(values) != 2:
        raise ValueError("point must have exactly two coordinates")
    return [round(float(value), 3) for value in values]


def normalize_annotation_point_map(values: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    """Normalize the role-keyed caliper witness points."""

    normalized = {str(key): point(value) for key, value in values.items()}
    missing = [key for key in ANNOTATION_KEYS if key not in normalized]
    if missing:
        raise ValueError(f"Vernier-caliper annotation is missing keys: {missing}")
    return {key: list(normalized[key]) for key in ANNOTATION_KEYS}


__all__ = ["ANNOTATION_KEYS", "bbox", "normalize_annotation_point_map", "point"]
