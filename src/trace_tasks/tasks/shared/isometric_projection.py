"""Shared isometric projection helpers reused across multiple domains."""

from __future__ import annotations

from typing import Sequence, Tuple

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def iso_project_point_3d(point_3d: Sequence[float]) -> Point2:
    """Project one 3D point into the canonical Trace isometric 2D plane."""

    x_value = float(point_3d[0])
    y_value = float(point_3d[1])
    z_value = float(point_3d[2])
    projected_x = (float(x_value) - float(y_value)) * 0.8660254
    projected_y = ((float(x_value) + float(y_value)) * 0.5) - float(z_value)
    return (float(projected_x), float(projected_y))


__all__ = ["Point2", "Point3", "iso_project_point_3d"]
