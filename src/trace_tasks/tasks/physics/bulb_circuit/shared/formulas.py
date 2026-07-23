"""Bulb-circuit power formulas."""

from __future__ import annotations

from typing import Sequence, Tuple


def power_values(scene_variant: str, resistances: Sequence[int]) -> Tuple[float, ...]:
    """Return relative bulb powers for the resolved circuit topology."""

    values = tuple(float(value) for value in resistances)
    if str(scene_variant) == "series_unequal":
        total = sum(values)
        return tuple(value / (total * total) for value in values)
    if str(scene_variant) == "parallel_unequal":
        return tuple(1.0 / value for value in values)
    if str(scene_variant) == "mixed_branch":
        first_branch = values[:2]
        second_branch = values[2:]
        first_total = sum(first_branch)
        second_total = sum(second_branch)
        return tuple(
            [value / (first_total * first_total) for value in first_branch]
            + [value / (second_total * second_total) for value in second_branch]
        )
    raise ValueError(f"unsupported bulb-circuit scene_variant: {scene_variant}")


def has_unique_powers(powers: Sequence[float]) -> bool:
    """Return whether every bulb has a distinct computed power."""

    for index, first in enumerate(powers):
        for second in powers[index + 1 :]:
            if abs(float(first) - float(second)) <= 1e-9:
                return False
    return True


__all__ = ["has_unique_powers", "power_values"]
