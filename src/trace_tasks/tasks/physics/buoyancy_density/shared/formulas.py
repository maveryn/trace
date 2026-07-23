"""Formula helpers for buoyancy-density scenes."""

from __future__ import annotations

from typing import Tuple


def object_density_tenths(
    *,
    liquid_density_tenths: int,
    fraction_num: int,
    fraction_den: int,
) -> int | None:
    """Return object density in tenths when the configured values are integral."""

    if int(fraction_num) <= 0 or int(fraction_den) <= 0:
        return None
    if int(fraction_num) >= int(fraction_den):
        return None
    numerator = int(liquid_density_tenths) * int(fraction_num)
    if numerator % int(fraction_den) != 0:
        return None
    resolved = int(numerator // int(fraction_den))
    if resolved <= 0:
        return None
    return int(resolved)


def parse_fraction(value: object) -> Tuple[int, int]:
    """Parse a fraction supplied as 'a/b' or a length-two sequence."""

    if isinstance(value, str) and "/" in value:
        left, right = value.split("/", 1)
        return int(left), int(right)
    try:
        sequence = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("submerged_fraction must be 'a/b' or a length-two sequence") from exc
    if len(sequence) != 2:
        raise ValueError("submerged_fraction must contain exactly two values")
    return int(sequence[0]), int(sequence[1])


def format_density(tenths: int) -> str:
    """Format a density stored in tenths as one decimal place."""

    return f"{float(int(tenths)) / 10.0:.1f}"


__all__ = ["format_density", "object_density_tenths", "parse_fraction"]
