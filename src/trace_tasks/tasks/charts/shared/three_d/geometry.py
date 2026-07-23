"""Pure screen-space geometry helpers for 3D chart renderers."""

from __future__ import annotations

from typing import Sequence, Tuple

BBox = list[float]
Point2D = Tuple[float, float]


def round_bbox(values: Sequence[float], *, ndigits: int = 3) -> BBox:
    """Return one rounded bbox from the first four coordinate values."""

    return [round(float(value), int(ndigits)) for value in values[:4]]


def bbox_center(bbox: Sequence[float], *, ndigits: int = 3) -> list[float]:
    """Return the rounded center of one bbox."""

    values = [float(value) for value in list(bbox)[:4]]
    return [
        round((values[0] + values[2]) / 2.0, int(ndigits)),
        round((values[1] + values[3]) / 2.0, int(ndigits)),
    ]


def point_bbox(px: float, py: float, radius: float, *, ndigits: int = 3) -> BBox:
    """Return a rounded bbox centered on one screen-space point."""

    return round_bbox(
        [
            float(px) - float(radius),
            float(py) - float(radius),
            float(px) + float(radius),
            float(py) + float(radius),
        ],
        ndigits=int(ndigits),
    )


def polygon_bbox(points: Sequence[Sequence[float]], *, ndigits: int = 3) -> BBox:
    """Return a rounded bbox enclosing screen-space polygon points."""

    return round_bbox(
        (
            min(float(point[0]) for point in points),
            min(float(point[1]) for point in points),
            max(float(point[0]) for point in points),
            max(float(point[1]) for point in points),
        ),
        ndigits=int(ndigits),
    )


__all__ = ["BBox", "Point2D", "bbox_center", "point_bbox", "polygon_bbox", "round_bbox"]
