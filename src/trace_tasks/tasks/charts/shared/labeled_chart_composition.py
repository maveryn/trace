"""Composition sampling helpers for labeled chart task families."""

from __future__ import annotations

from typing import List

from trace_tasks.core.seed import spawn_rng


def sample_composition_with_sum(
    rng,
    *,
    target_sum: int,
    count: int,
    value_min: int,
    value_max: int,
) -> List[int]:
    """Sample one bounded integer composition with sum preserved exactly."""

    if int(count) <= 0:
        raise ValueError("composition count must be positive")
    if int(target_sum) < int(count) * int(value_min) or int(target_sum) > int(count) * int(value_max):
        raise ValueError("target_sum outside feasible support for bounded composition")

    values = [int(value_min)] * int(count)
    remaining = int(target_sum) - (int(count) * int(value_min))
    max_increment = int(value_max) - int(value_min)
    for index in range(int(count)):
        remaining_slots = int(count) - int(index) - 1
        max_possible_for_rest = int(remaining_slots) * int(max_increment)
        add_min = max(0, int(remaining) - int(max_possible_for_rest))
        add_max = min(int(max_increment), int(remaining))
        add_value = int(remaining) if int(index) == int(count) - 1 else int(rng.randint(int(add_min), int(add_max)))
        values[int(index)] += int(add_value)
        remaining -= int(add_value)
    if int(sum(values)) != int(target_sum):
        raise RuntimeError("bounded composition drifted from requested sum")
    return [int(value) for value in values]


def sample_percentage_composition(
    *,
    count: int,
    instance_seed: int,
    namespace: str,
) -> List[int]:
    """Sample one positive-integer percentage composition that sums to 100."""

    composition_rng = spawn_rng(int(instance_seed), str(namespace))
    values = sample_composition_with_sum(
        composition_rng,
        target_sum=100,
        count=int(count),
        value_min=1,
        value_max=100,
    )
    composition_rng.shuffle(values)
    return [int(value) for value in values]


__all__ = [
    "sample_composition_with_sum",
    "sample_percentage_composition",
]
