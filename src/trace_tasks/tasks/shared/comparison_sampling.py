"""Shared numeric winner-gap helpers for comparison-style tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ComparisonGapMetrics:
    """Comparison winner/runner-up summary for one scalar-valued scene."""

    query_type: str
    winner_index: int
    runner_up_index: int
    winner_value: float
    runner_up_value: float
    scene_span: float
    gap_abs: float
    gap_normalized: float


def compute_comparison_gap_metrics(
    values: Sequence[float],
    *,
    query_type: str,
    epsilon: float = 1e-9,
) -> ComparisonGapMetrics:
    """Return winner/runner-up gap metrics for one comparison scene.

    `query_type` must be `largest` or `smallest`. The metric is domain-agnostic:
    tasks can reuse the same normalized-gap rule regardless of whether the
    compared quantity is angle, length, area, or another scalar.
    """

    numeric_values = [float(value) for value in values]
    if len(numeric_values) < 2:
        raise ValueError("comparison scenes require at least two values")

    query = str(query_type).strip().lower()
    if query not in {"largest", "smallest"}:
        raise ValueError(f"unsupported query_type: {query_type}")

    ordered_indices = sorted(
        range(len(numeric_values)),
        key=lambda index: (numeric_values[index], index),
        reverse=(query == "largest"),
    )
    winner_index = int(ordered_indices[0])
    runner_up_index = int(ordered_indices[1])
    winner_value = float(numeric_values[winner_index])
    runner_up_value = float(numeric_values[runner_up_index])
    scene_min = float(min(numeric_values))
    scene_max = float(max(numeric_values))
    scene_span = float(scene_max - scene_min)
    gap_abs = abs(float(winner_value) - float(runner_up_value))
    span_denom = max(float(scene_span), float(epsilon))
    gap_normalized = float(gap_abs) / float(span_denom)
    return ComparisonGapMetrics(
        query_type=query,
        winner_index=int(winner_index),
        runner_up_index=int(runner_up_index),
        winner_value=float(winner_value),
        runner_up_value=float(runner_up_value),
        scene_span=float(scene_span),
        gap_abs=float(gap_abs),
        gap_normalized=float(gap_normalized),
    )


def comparison_gap_is_valid(
    values: Sequence[float],
    *,
    query_type: str,
    min_normalized_gap: float,
    min_absolute_gap: float = 0.0,
    epsilon: float = 1e-9,
) -> bool:
    """Return whether one comparison scene satisfies configured winner-gap floors."""

    metrics = compute_comparison_gap_metrics(
        values,
        query_type=str(query_type),
        epsilon=float(epsilon),
    )
    return bool(
        float(metrics.gap_normalized) >= float(min_normalized_gap)
        and float(metrics.gap_abs) >= float(min_absolute_gap)
    )


__all__ = [
    "ComparisonGapMetrics",
    "comparison_gap_is_valid",
    "compute_comparison_gap_metrics",
]
