"""Neutral sampling helpers for the dumbbell chart scene."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.charts.dumbbell.shared.defaults import GEN_DEFAULTS, SCENE_NAMESPACE, balanced_choice, balanced_int, generation_int
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellRow


def rank_phrase(rank_order: str, rank_n: int) -> str:
    """Return a human-readable rank phrase."""

    if int(rank_n) == 1:
        return str(rank_order)
    ordinal = {2: "second"}.get(int(rank_n), f"{int(rank_n)}th")
    return f"{ordinal} {str(rank_order)}"


def value_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the numeric value bounds for row values."""

    return resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=0,
        fallback_max=100,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )


def gap_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the absolute gap bounds for paired row values."""

    gap_min = generation_int(params, "gap_min", 4)
    gap_max = generation_int(params, "gap_max", 56)
    if int(gap_min) < 1 or int(gap_max) <= int(gap_min):
        raise ValueError("gap range must be positive and ordered")
    return int(gap_min), int(gap_max)


def sample_row_count(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> tuple[int, dict[str, float]]:
    """Sample a balanced row count for one dumbbell scene."""

    row_min, row_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="row_count_min",
        max_key="row_count_max",
        fallback_min=10,
        fallback_max=16,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        low=int(row_min),
        high=int(row_max),
    )


def sample_rank_n(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> tuple[int, dict[str, float]]:
    """Sample a balanced supported rank position."""

    return balanced_choice(
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        values=(1, 2),
    )


def sample_threshold(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    min_key: str,
    max_key: str,
    step_key: str,
    fallback_min: int,
    fallback_max: int,
    fallback_step: int,
) -> tuple[int, dict[str, float]]:
    """Sample a balanced threshold from configured stepped support."""

    threshold_min = generation_int(params, str(min_key), int(fallback_min))
    threshold_max = generation_int(params, str(max_key), int(fallback_max))
    threshold_step = max(1, generation_int(params, str(step_key), int(fallback_step)))
    support = tuple(range(int(threshold_min), int(threshold_max) + 1, int(threshold_step)))
    return balanced_choice(params, instance_seed=int(instance_seed), namespace=str(namespace), values=support)


def sample_target_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    row_count: int,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, dict[str, float]]:
    """Sample a feasible target count for count-style objectives."""

    count_min = generation_int(params, str(min_key), int(fallback_min))
    count_max = min(generation_int(params, str(max_key), int(fallback_max)), int(row_count) - 1)
    if int(count_min) > int(count_max):
        raise ValueError("target count support is empty")
    return balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        low=int(count_min),
        high=int(count_max),
    )


def sample_labels_and_series(
    *,
    row_count: int,
    instance_seed: int,
    namespace: str,
) -> tuple[tuple[str, ...], str, str]:
    """Sample row labels and two legend labels without collisions."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{namespace}.labels")
    resolved = resolve_chart_entity_labels(
        rng,
        count=int(row_count) + 2,
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    row_labels = tuple(str(label) for label in resolved[: int(row_count)])
    series_a_name = str(resolved[int(row_count)])
    series_b_name = str(resolved[int(row_count) + 1])
    return row_labels, series_a_name, series_b_name


def build_rows_from_gap_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    labels: Sequence[str],
    gap_by_index: Mapping[int, int],
    signed_direction_by_index: Mapping[int, int] | None = None,
) -> tuple[DumbbellRow, ...]:
    """Materialize row values from explicit per-row gap and direction plans."""

    value_min, value_max = value_bounds(params)
    direction_plan = dict(signed_direction_by_index or {})
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{namespace}.rows")
    rows: list[DumbbellRow] = []
    for index, label in enumerate(labels):
        gap = int(gap_by_index[int(index)])
        if int(gap) < 1:
            raise ValueError("row gap must be positive")
        lo = int(value_min)
        hi = int(value_max) - int(gap)
        if int(lo) > int(hi):
            raise ValueError("gap exceeds feasible value range")
        lower_value = int(rng.randint(int(lo), int(hi)))
        direction = int(direction_plan.get(int(index), 1 if rng.random() < 0.5 else -1))
        if int(direction) > 0:
            value_a = int(lower_value + int(gap))
            value_b = int(lower_value)
        else:
            value_a = int(lower_value)
            value_b = int(lower_value + int(gap))
        rows.append(DumbbellRow(row_id=f"row_{index}", label=str(label), value_a=int(value_a), value_b=int(value_b)))
    return tuple(rows)


__all__ = [
    "build_rows_from_gap_plan",
    "gap_bounds",
    "rank_phrase",
    "sample_labels_and_series",
    "sample_rank_n",
    "sample_row_count",
    "sample_target_count",
    "sample_threshold",
    "value_bounds",
]
