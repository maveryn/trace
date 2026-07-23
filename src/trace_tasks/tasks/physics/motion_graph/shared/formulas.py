"""Kinematics helpers for motion-graph tasks."""

from __future__ import annotations


def classify_speed_change(v_start: int, v_end: int) -> str:
    """Return the speed-change state implied by one velocity-time segment."""

    start_abs = abs(int(v_start))
    end_abs = abs(int(v_end))
    if end_abs > start_abs:
        return "speeding_up"
    if end_abs < start_abs:
        return "slowing_down"
    return "constant_speed"


def interval_displacement(v_start: int, v_end: int, delta_t: int) -> int:
    """Return signed displacement under a linear velocity-time segment."""

    numerator = (int(v_start) + int(v_end)) * int(delta_t)
    if numerator % 2 != 0:
        raise ValueError("linear velocity interval does not have integer area")
    return int(numerator // 2)


def average_speed(distance_start: int, distance_end: int, delta_t: int) -> int:
    """Return integer average speed from a distance-time interval."""

    if int(delta_t) <= 0:
        raise ValueError("delta_t must be positive")
    distance_delta = int(distance_end) - int(distance_start)
    if int(distance_delta) < 0:
        raise ValueError("distance-time interval cannot decrease")
    if int(distance_delta) % int(delta_t) != 0:
        raise ValueError("distance-time interval does not have integer average speed")
    return int(distance_delta // int(delta_t))


__all__ = [
    "average_speed",
    "classify_speed_change",
    "interval_displacement",
]
