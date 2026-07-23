"""Annotation helpers for equivalent-circuit component witnesses."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence


def bbox(values: Sequence[float]) -> list[float]:
    """Return one JSON-stable pixel bbox."""

    return [round(float(value), 3) for value in values]


def clip_bbox(values: Sequence[float], *, width: int, height: int) -> list[float]:
    """Clip a component bbox to image bounds."""

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


def normalize_component_bbox_map(
    values: Mapping[str, Sequence[float]],
    *,
    component_prefix: str,
    width: int,
    height: int,
) -> Dict[str, list[float]]:
    """Normalize visible component boxes in numeric label order."""

    normalized = {
        str(key): clip_bbox(value, width=int(width), height=int(height))
        for key, value in values.items()
    }
    if not normalized:
        raise ValueError("equivalent-circuit annotation requires at least one component")
    bad_labels = [
        key
        for key in normalized
        if not key.startswith(str(component_prefix))
    ]
    if bad_labels:
        raise ValueError(f"unexpected equivalent-circuit component labels: {bad_labels}")

    def sort_key(label: str) -> tuple[int, str]:
        suffix = str(label)[len(str(component_prefix)) :]
        return (int(suffix) if suffix.isdigit() else 10_000, str(label))

    return {key: list(normalized[key]) for key in sorted(normalized, key=sort_key)}


__all__ = ["bbox", "clip_bbox", "normalize_component_bbox_map"]
