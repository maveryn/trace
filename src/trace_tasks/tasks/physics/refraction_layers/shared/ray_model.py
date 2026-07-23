"""Ray-geometry helpers for the refraction-layers scene."""

from __future__ import annotations

import math
from typing import Tuple

from trace_tasks.core.seed import hash64

from .state import RayGeometry, RefractionScenario


def ray_geometry(
    *,
    scenario: RefractionScenario,
    medium_box: Tuple[float, float, float, float],
) -> RayGeometry:
    """Project the ray path through the three media."""

    left, top, right, bottom = [float(value) for value in medium_box]
    margin = 26.0
    if scenario.orientation == "horizontal":
        y1 = top + ((bottom - top) / 3.0)
        y2 = top + (2.0 * (bottom - top) / 3.0)
        if scenario.entry_side == "top":
            border_values = (top, y1, y2, bottom)
            segment_mediums = ("M1", "M2", "M3")
        else:
            border_values = (bottom, y2, y1, top)
            segment_mediums = ("M3", "M2", "M1")
        distances = [abs(border_values[idx + 1] - border_values[idx]) for idx in range(3)]
        tangents = [math.tan(math.radians(float(scenario.angle_by_medium_deg[label]))) for label in segment_mediums]
        sign = float(scenario.transverse_sign)
        coeffs = (
            -sign * distances[0] * tangents[0],
            0.0,
            sign * distances[1] * tangents[1],
            sign * ((distances[1] * tangents[1]) + (distances[2] * tangents[2])),
        )
        lower = left + margin
        upper = right - margin
        x_low = max(lower - coeff for coeff in coeffs)
        x_high = min(upper - coeff for coeff in coeffs)
        if x_high < x_low:
            x_mid = (left + right) * 0.5
        else:
            jitter_span = min(48.0, max(0.0, (x_high - x_low) * 0.34))
            raw = (float(hash64(int(round(left + right + top + bottom)), f"{scenario.orientation}.x_jitter", 0) % 10_000) / 10_000.0) - 0.5
            x_mid = ((x_low + x_high) * 0.5) + (raw * 2.0 * jitter_span)
        points = tuple((float(x_mid + coeffs[idx]), float(border_values[idx])) for idx in range(4))
    else:
        x1 = left + ((right - left) / 3.0)
        x2 = left + (2.0 * (right - left) / 3.0)
        if scenario.entry_side == "left":
            border_values = (left, x1, x2, right)
            segment_mediums = ("M1", "M2", "M3")
        else:
            border_values = (right, x2, x1, left)
            segment_mediums = ("M3", "M2", "M1")
        distances = [abs(border_values[idx + 1] - border_values[idx]) for idx in range(3)]
        tangents = [math.tan(math.radians(float(scenario.angle_by_medium_deg[label]))) for label in segment_mediums]
        sign = float(scenario.transverse_sign)
        coeffs = (
            -sign * distances[0] * tangents[0],
            0.0,
            sign * distances[1] * tangents[1],
            sign * ((distances[1] * tangents[1]) + (distances[2] * tangents[2])),
        )
        lower = top + margin
        upper = bottom - margin
        y_low = max(lower - coeff for coeff in coeffs)
        y_high = min(upper - coeff for coeff in coeffs)
        if y_high < y_low:
            y_mid = (top + bottom) * 0.5
        else:
            jitter_span = min(36.0, max(0.0, (y_high - y_low) * 0.30))
            raw = (float(hash64(int(round(left + right + top + bottom)), f"{scenario.orientation}.y_jitter", 0) % 10_000) / 10_000.0) - 0.5
            y_mid = ((y_low + y_high) * 0.5) + (raw * 2.0 * jitter_span)
        points = tuple((float(border_values[idx]), float(y_mid + coeffs[idx])) for idx in range(4))
    return RayGeometry(
        points=points,  # type: ignore[arg-type]
        bend_points=(points[1], points[2]),
        segment_mediums=tuple(segment_mediums),
        segment_angles_deg=tuple(float(scenario.angle_by_medium_deg[label]) for label in segment_mediums),
    )
