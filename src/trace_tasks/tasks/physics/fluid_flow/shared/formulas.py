"""Formula helpers for fluid-flow continuity diagrams."""

from __future__ import annotations


def continuity_product(*, area_cm2: int, speed_m_s: int) -> int:
    """Return A*v for one station."""

    return int(area_cm2) * int(speed_m_s)


def pipe_size(area_cm2: int) -> float:
    """Return the visual pipe size for a cross-section area label."""

    return float(54.0 + 8.0 * (float(area_cm2) ** 0.5))
