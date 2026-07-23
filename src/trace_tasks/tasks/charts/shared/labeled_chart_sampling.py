"""Low-level sampling helpers for labeled chart task families."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .labeled_chart_values import balanced_choice_from_values


def choose_rank_n(
    feasible_ranks: Sequence[int],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Choose one feasible order-statistic rank deterministically."""

    values = [int(value) for value in feasible_ranks]
    if not values:
        raise ValueError(f"no feasible ranks for {namespace}")
    explicit_rank = params.get("rank_n")
    if explicit_rank is not None:
        selected = int(explicit_rank)
        if int(selected) not in set(values):
            raise ValueError("explicit rank_n is outside feasible support")
        return int(selected)
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, values, sort_keys=True))


def choose_mark_count(
    feasible_counts: Sequence[int],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Choose one feasible mark count deterministically."""

    values = [int(value) for value in feasible_counts]
    if not values:
        raise ValueError(f"no feasible mark counts for {namespace}")
    explicit_count = params.get("mark_count")
    if explicit_count is not None:
        selected = int(explicit_count)
        if int(selected) not in set(values):
            raise ValueError("explicit mark_count is outside feasible support")
        return int(selected)
    return balanced_choice_from_values(values, params=params, instance_seed=int(instance_seed), namespace=str(namespace))


def _sample_int_values(
    rng,
    *,
    count: int,
    min_value: int,
    max_value: int,
) -> List[int]:
    """Sample integer values uniformly from one inclusive range."""

    if int(count) < 0:
        raise ValueError("count must be non-negative")
    if int(min_value) > int(max_value):
        raise ValueError("min_value must be <= max_value")
    return [int(rng.randint(int(min_value), int(max_value))) for _ in range(int(count))]


def _sample_values_from_pool(
    rng,
    *,
    count: int,
    pool: Sequence[int],
) -> List[int]:
    """Sample integer values uniformly with replacement from one explicit pool."""

    values = [int(value) for value in pool]
    if not values:
        raise ValueError("pool must not be empty")
    return [int(values[rng.randint(0, len(values) - 1)]) for _ in range(int(count))]


__all__ = [
    "_sample_int_values",
    "_sample_values_from_pool",
    "choose_mark_count",
    "choose_rank_n",
]
