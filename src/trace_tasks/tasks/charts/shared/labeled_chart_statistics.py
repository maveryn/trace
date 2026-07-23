"""Statistic construction helpers for labeled chart task families."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng

from .labeled_chart_values import StatisticKind, cyclic_pair_deltas, max_symmetric_delta, shuffle_values


def build_values_for_median(
    target_answer: int,
    *,
    count: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
) -> List[int]:
    """Construct values with unique median equal to `target_answer`."""

    if int(count) % 2 == 0:
        raise ValueError("median construction requires an odd mark count")
    pair_count = (int(count) - 1) // 2
    max_delta = max_symmetric_delta(int(target_answer), value_min=int(value_min), value_max=int(value_max))
    deltas = cyclic_pair_deltas(pair_count=int(pair_count), max_delta=int(max_delta))
    values: List[int] = [int(target_answer)]
    for delta in deltas:
        values.append(int(target_answer) - int(delta))
        values.append(int(target_answer) + int(delta))
    if len(values) != int(count):
        raise RuntimeError("median construction produced the wrong number of values")
    if sorted(values)[len(values) // 2] != int(target_answer):
        raise RuntimeError("median construction does not preserve requested median")
    return shuffle_values(values, instance_seed=int(instance_seed), namespace="charts.value_order.median")


def build_values_for_nth_rank(
    target_answer: int,
    *,
    count: int,
    rank_n: int,
    direction: str,
    value_min: int,
    value_max: int,
    instance_seed: int,
) -> List[int]:
    """Construct values where `target_answer` is the unique nth distinct ranked value."""

    if int(rank_n) < 2:
        raise ValueError("rank_n must be at least 2")
    if int(count) < int(rank_n):
        raise ValueError("count must be at least rank_n")
    rng = spawn_rng(int(instance_seed), f"charts.values.nth_rank:{str(direction)}:{int(rank_n)}:{int(target_answer)}")
    if str(direction) == "highest":
        above_candidates = [int(value) for value in range(int(target_answer) + 1, int(value_max) + 1)]
        if len(above_candidates) < int(rank_n) - 1:
            raise ValueError("not enough distinct values above target_answer for nth-highest construction")
        ranked_values = [int(value) for value in rng.sample(above_candidates, k=int(rank_n) - 1)]
        fill_pool = ranked_values + [int(value) for value in range(int(value_min), int(target_answer))]
    elif str(direction) == "lowest":
        below_candidates = [int(value) for value in range(int(value_min), int(target_answer))]
        if len(below_candidates) < int(rank_n) - 1:
            raise ValueError("not enough distinct values below target_answer for nth-lowest construction")
        ranked_values = [int(value) for value in rng.sample(below_candidates, k=int(rank_n) - 1)]
        fill_pool = ranked_values + [int(value) for value in range(int(target_answer) + 1, int(value_max) + 1)]
    else:
        raise ValueError(f"unsupported rank direction: {direction}")
    if not fill_pool and int(count) > int(rank_n):
        raise ValueError("rank construction has no non-target fill values")
    values = [int(value) for value in ranked_values] + [int(target_answer)]
    while len(values) < int(count):
        values.append(int(fill_pool[int(rng.randint(0, len(fill_pool) - 1))]))
    rng.shuffle(values)
    if values.count(int(target_answer)) != 1:
        raise RuntimeError("rank construction must keep target_answer unique")
    return [int(value) for value in values]


def summarize_statistic_from_values(
    *,
    statistic_kind: StatisticKind,
    labels: Sequence[str],
    values: Sequence[int],
    rank_n: int | None = None,
) -> Tuple[int, List[str], Dict[str, Any]]:
    """Resolve the statistic answer and supporting labels from one labeled value list."""

    resolved_labels = [str(label) for label in labels]
    resolved_values = [int(value) for value in values]
    if len(resolved_labels) != len(resolved_values):
        raise ValueError("labels and values must have the same length")

    marks = {str(label): int(value) for label, value in zip(resolved_labels, resolved_values)}
    if str(statistic_kind) == "median":
        if int(len(resolved_values)) % 2 == 0:
            raise ValueError("median requires an odd number of values")
        sorted_pairs = sorted(((int(value), str(label)) for label, value in marks.items()), key=lambda item: (item[0], item[1]))
        median_index = len(sorted_pairs) // 2
        median_value = int(sorted_pairs[median_index][0])
        support_labels = [str(label) for label, value in marks.items() if int(value) == int(median_value)]
        if len(support_labels) != 1:
            raise ValueError("median requires one unique median label")
        return (
            int(median_value),
            [str(support_labels[0])],
            {
                "support_label": str(support_labels[0]),
                "sorted_values": [int(value) for value, _ in sorted_pairs],
            },
        )
    if str(statistic_kind) in {"nth_highest", "nth_lowest"}:
        if rank_n is None:
            raise ValueError(f"{statistic_kind} requires rank_n")
        resolved_rank = int(rank_n)
        unique_values = sorted(set(int(value) for value in resolved_values), reverse=str(statistic_kind) == "nth_highest")
        if int(resolved_rank) < 1 or int(resolved_rank) > len(unique_values):
            raise ValueError(f"rank_n outside distinct value support for {statistic_kind}")
        target_value = int(unique_values[int(resolved_rank) - 1])
        ranked_labels = [str(label) for label, value in marks.items() if int(value) == int(target_value)]
        if len(ranked_labels) != 1:
            raise ValueError(f"{statistic_kind} requires one unique ranked label")
        return (
            int(target_value),
            [str(ranked_labels[0])],
            {
                "rank_n": int(resolved_rank),
                "rank_direction": "highest" if str(statistic_kind) == "nth_highest" else "lowest",
                "ranked_label": str(ranked_labels[0]),
                "ranked_distinct_values": [int(value) for value in unique_values],
            },
        )
    raise ValueError(f"unsupported statistic_kind: {statistic_kind}")


__all__ = [
    "build_values_for_median",
    "build_values_for_nth_rank",
    "summarize_statistic_from_values",
]
