"""Formula helpers for gear-train reasoning."""

from __future__ import annotations

from typing import Any


def normalize_direction(value: Any) -> str | None:
    """Normalize common clockwise/counterclockwise aliases."""

    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cw": "clockwise",
        "clockwise": "clockwise",
        "counterclockwise": "counterclockwise",
        "counter_clockwise": "counterclockwise",
        "ccw": "counterclockwise",
        "anticlockwise": "counterclockwise",
        "anti_clockwise": "counterclockwise",
    }
    if text not in aliases:
        raise ValueError(f"unsupported gear rotation direction: {value}")
    return str(aliases[text])


def opposite_direction(direction: str) -> str:
    """Return the opposite rotation direction."""

    return "counterclockwise" if str(direction) == "clockwise" else "clockwise"


def propagate_output_direction(input_direction: str, gear_count: int) -> str:
    """Propagate adjacent meshed-gear reversals from input to output."""

    return str(input_direction) if (int(gear_count) - 1) % 2 == 0 else opposite_direction(str(input_direction))


def speed_relation(input_rpm: int, output_rpm: int) -> str:
    """Return whether the output gear is faster or slower than the input gear."""

    if int(output_rpm) == int(input_rpm):
        return "same"
    return "faster" if int(output_rpm) > int(input_rpm) else "slower"


def radius_from_teeth(teeth: int) -> float:
    """Return a visually proportional gear radius from tooth count."""

    return float(34.0 + 0.72 * float(teeth))


__all__ = [
    "normalize_direction",
    "opposite_direction",
    "propagate_output_direction",
    "radius_from_teeth",
    "speed_relation",
]
