"""Annotation helpers for analog-meter visual witnesses."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence


ANNOTATION_KEYS: tuple[str, str, str] = ("needle", "scale_region", "unit_label")


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def normalize_bbox_map(values: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    """Normalize the role-keyed analog-meter witness boxes."""

    normalized = {str(key): bbox(value) for key, value in values.items()}
    missing = [key for key in ANNOTATION_KEYS if key not in normalized]
    if missing:
        raise ValueError(f"analog meter annotation is missing keys: {missing}")
    return {key: list(normalized[key]) for key in ANNOTATION_KEYS}


__all__ = ["ANNOTATION_KEYS", "bbox", "normalize_bbox_map"]
