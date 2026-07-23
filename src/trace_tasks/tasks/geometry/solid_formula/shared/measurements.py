"""Measurement formatting and formula support helpers for solid-formula tasks."""

from __future__ import annotations

from typing import Iterable, Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map


def decimal_support(start: float, stop: float, *, step: float = 0.5) -> tuple[float, ...]:
    """Return a one-decimal support interval with inclusive endpoints."""

    values: list[float] = []
    current = float(start)
    epsilon = float(step) / 10.0
    while current <= float(stop) + epsilon:
        values.append(round1(current))
        current += float(step)
    return tuple(values)


def format_measure(value: float) -> str:
    """Return a compact geometry measurement label."""

    return fmt_measure(float(value))


def format_pi_multiple(value: float) -> str:
    """Return a compact coefficient of pi for volume labels."""

    pi_symbol = "\u03c0"
    coefficient = format_measure(float(value))
    if coefficient == "1":
        return pi_symbol
    return f"{coefficient}{pi_symbol}"


def unique_support(values: Iterable[float]) -> tuple[float, ...]:
    """Return sorted one-decimal support values."""

    return tuple(sorted({round1(float(value)) for value in values}))


def answer_support_probability_map(support: Sequence[float], answer: float) -> dict[str, float]:
    """Return a one-hot probability map over answer support values."""

    return geometry_selected_probability_map(
        tuple(float(value) for value in support),
        float(answer),
        key_fn=format_measure,
        sort_unique=True,
    )


__all__ = [
    "decimal_support",
    "answer_support_probability_map",
    "format_measure",
    "format_pi_multiple",
    "round1",
    "unique_support",
]
