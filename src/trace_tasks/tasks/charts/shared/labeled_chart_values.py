"""Value and count helpers for labeled chart task families."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds

from .labeled_chart_defaults import LabeledChartDefaults

StatisticKind = str


def sorted_labels(labels: Sequence[str]) -> List[str]:
    """Return labels in deterministic alphabetical order."""

    return [str(label) for label in sorted(str(label) for label in labels)]


def resolve_value_bounds(
    params: Mapping[str, object],
    *,
    gen_defaults: Mapping[str, object],
    defaults: LabeledChartDefaults,
    task_id: str,
    instance_seed: int | None = None,
) -> Tuple[int, int]:
    """Resolve inclusive per-mark value bounds."""

    value_min, value_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="value_min",
        max_key="value_max",
        fallback_min=int(defaults.value_min),
        fallback_max=int(defaults.value_max),
        context=f"generation defaults for {task_id}",
    )
    enabled_raw = params.get("value_window_enabled", group_default(gen_defaults, "value_window_enabled", False))
    enabled = bool(enabled_raw)
    if isinstance(enabled_raw, str):
        enabled = str(enabled_raw).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled or instance_seed is None:
        return int(value_min), int(value_max)

    hard_max = int(params.get("value_hard_max", group_default(gen_defaults, "value_hard_max", 99)))
    span_min = int(params.get("value_window_span_min", group_default(gen_defaults, "value_window_span_min", 10)))
    span_max = int(params.get("value_window_span_max", group_default(gen_defaults, "value_window_span_max", 25)))
    hard_max = min(int(value_max), int(hard_max))
    span_min = max(1, int(span_min))
    span_max = max(int(span_min), int(span_max))
    max_feasible_span = int(hard_max) - int(value_min)
    if int(max_feasible_span) <= 0:
        return int(value_min), int(value_max)
    low_span = min(int(span_min), int(max_feasible_span))
    high_span = min(int(span_max), int(max_feasible_span))
    if int(low_span) > int(high_span):
        low_span = int(high_span)
    span = hashed_choice_from_values(
        [int(value) for value in range(int(low_span), int(high_span) + 1)],
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.value_window_span",
    )
    start_max = int(hard_max) - int(span)
    if int(start_max) < int(value_min):
        return int(value_min), int(hard_max)
    start = hashed_choice_from_values(
        [int(value) for value in range(int(value_min), int(start_max) + 1)],
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.value_window_start:{int(span)}",
    )
    return int(start), int(start) + int(span)


def resolve_mark_count_bounds(
    params: Mapping[str, object],
    *,
    gen_defaults: Mapping[str, object],
    defaults: LabeledChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    """Resolve inclusive chart mark-count bounds."""

    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="mark_count_min",
        max_key="mark_count_max",
        fallback_min=int(defaults.mark_count_min),
        fallback_max=int(defaults.mark_count_max),
        context=f"generation defaults for {task_id}",
    )


def resolve_target_answer_range(
    params: Mapping[str, object],
    *,
    value_min: int,
    value_max: int,
    target_answer_ranges: Mapping[str, Tuple[int, int]],
    statistic_kind: StatisticKind,
) -> Tuple[int, int]:
    """Resolve supported target-answer bounds for one statistic."""

    default_min, default_max = target_answer_ranges[str(statistic_kind)]
    explicit_min = params.get("target_answer_min", None)
    explicit_max = params.get("target_answer_max", None)
    if explicit_min is None and explicit_max is None:
        return int(default_min), int(default_max)
    min_value = int(default_min if explicit_min is None else explicit_min)
    max_value = int(default_max if explicit_max is None else explicit_max)
    if int(min_value) > int(max_value):
        raise ValueError("target_answer_min must be <= target_answer_max")
    min_value = max(int(min_value), int(value_min))
    max_value = min(int(max_value), int(value_max))
    if int(min_value) > int(max_value):
        raise ValueError("target answer bounds leave no feasible support")
    return int(min_value), int(max_value)


def balanced_choice_from_values(
    values: Sequence[int],
    *,
    params: Mapping[str, object],
    instance_seed: int,
    namespace: str,
) -> int:
    """Select one supported integer value using seeded RNG."""

    ordered = [int(value) for value in values]
    if not ordered:
        raise ValueError(f"no feasible values for {namespace}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, ordered, sort_keys=True))


def hashed_choice_from_values(
    values: Sequence[int],
    *,
    instance_seed: int,
    namespace: str,
) -> int:
    """Select one supported integer value without sampling-index coupling."""

    ordered = [int(value) for value in values]
    if not ordered:
        raise ValueError(f"no feasible values for {namespace}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, ordered, sort_keys=True))


def shuffle_values(values: Sequence[int], *, instance_seed: int, namespace: str) -> List[int]:
    """Shuffle numeric values independently of labels and chart type."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    ordered = [int(value) for value in values]
    rng.shuffle(ordered)
    return ordered


def max_symmetric_delta(target_value: int, *, value_min: int, value_max: int) -> int:
    """Return the largest +/- delta that stays within the value bounds."""

    return int(min(int(target_value) - int(value_min), int(value_max) - int(target_value)))


def cyclic_pair_deltas(*, pair_count: int, max_delta: int) -> List[int]:
    """Return one deterministic list of positive deltas for symmetric value construction."""

    if int(pair_count) <= 0:
        return []
    if int(max_delta) <= 0:
        raise ValueError("symmetric construction requires at least one positive available delta")
    return [1 + (int(index) % int(max_delta)) for index in range(int(pair_count))]


__all__ = [
    "StatisticKind",
    "balanced_choice_from_values",
    "cyclic_pair_deltas",
    "hashed_choice_from_values",
    "max_symmetric_delta",
    "resolve_mark_count_bounds",
    "resolve_target_answer_range",
    "resolve_value_bounds",
    "shuffle_values",
    "sorted_labels",
]
