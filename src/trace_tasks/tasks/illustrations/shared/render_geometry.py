"""Shared pixel-scaling geometry helpers for illustration renderers."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def scale_bbox(bbox: Sequence[float], scale: int) -> Tuple[int, int, int, int]:
    """Scale a design-space bbox to integer pixel coordinates."""

    return tuple(int(round(float(value) * int(scale))) for value in bbox)  # type: ignore[return-value]


def scale_points(points: Iterable[Sequence[float]], scale: int) -> list[tuple[int, int]]:
    """Scale design-space points to integer pixel coordinates."""

    return [
        (int(round(float(point[0]) * int(scale))), int(round(float(point[1]) * int(scale))))
        for point in points
    ]


__all__ = ["scale_bbox", "scale_points"]
