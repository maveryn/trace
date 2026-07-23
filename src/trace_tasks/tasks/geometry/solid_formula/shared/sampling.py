"""Neutral deterministic sampling helpers for solid-formula tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

T = TypeVar("T")


def select_support_value(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    support: Sequence[T],
) -> T:
    """Select one value from a public task's answer support."""

    values = tuple(support)
    if not values:
        raise ValueError("answer support must be non-empty")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return uniform_choice(rng, values)


def select_case_option(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    options: Sequence[T],
) -> tuple[T, int]:
    """Select one construction option for an already selected answer."""

    values = tuple(options)
    if not values:
        raise ValueError("construction option support must be non-empty")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return uniform_choice(rng, values), len(values)


__all__ = ["select_case_option", "select_support_value"]
