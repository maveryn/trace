"""Pure projection helpers for synthetic 3D chart renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Point2D = Tuple[float, float]
Vector2D = Tuple[float, float]


@dataclass(frozen=True)
class ProjectionBasis2D:
    """Screen-space basis for projecting normalized 3D coordinates."""

    origin: Point2D
    x_vec: Vector2D
    y_vec: Vector2D
    z_vec: Vector2D

    def project_unit(self, x_unit: float, y_unit: float, z_unit: float) -> Point2D:
        """Project normalized x/y/z values to screen coordinates."""

        return (
            float(self.origin[0])
            + float(x_unit) * float(self.x_vec[0])
            + float(y_unit) * float(self.y_vec[0])
            + float(z_unit) * float(self.z_vec[0]),
            float(self.origin[1])
            + float(x_unit) * float(self.x_vec[1])
            + float(y_unit) * float(self.y_vec[1])
            + float(z_unit) * float(self.z_vec[1]),
        )


def value_to_unit(value: float, value_range: tuple[float, float]) -> float:
    """Normalize ``value`` into [0, 1] for one numeric axis range."""

    low, high = float(value_range[0]), float(value_range[1])
    if math.isclose(low, high):
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def surface_plot_basis(plot_bbox: Sequence[float]) -> ProjectionBasis2D:
    """Return the current Trace surface-chart oblique projection basis."""

    left, top, right, bottom = [float(value) for value in plot_bbox[:4]]
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    return ProjectionBasis2D(
        origin=(left + 0.18 * width, bottom - 0.10 * height),
        x_vec=(0.52 * width, -0.14 * height),
        y_vec=(0.24 * width, -0.24 * height),
        z_vec=(0.0, -0.56 * height),
    )


def project_ranged_point_3d(
    x_value: float,
    y_value: float,
    z_value: float,
    *,
    plot_bbox: Sequence[float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> Point2D:
    """Project ranged x/y/z chart values to screen coordinates."""

    return surface_plot_basis(plot_bbox).project_unit(
        value_to_unit(float(x_value), x_range),
        value_to_unit(float(y_value), y_range),
        value_to_unit(float(z_value), z_range),
    )


def axis_line_position(
    start: Point2D,
    end: Point2D,
    *,
    fraction: float,
) -> Point2D:
    """Return a clamped interpolation point on a projected axis line."""

    t = max(0.0, min(1.0, float(fraction)))
    return (
        float(start[0]) + t * (float(end[0]) - float(start[0])),
        float(start[1]) + t * (float(end[1]) - float(start[1])),
    )


__all__ = [
    "Point2D",
    "ProjectionBasis2D",
    "Vector2D",
    "axis_line_position",
    "project_ranged_point_3d",
    "surface_plot_basis",
    "value_to_unit",
]
