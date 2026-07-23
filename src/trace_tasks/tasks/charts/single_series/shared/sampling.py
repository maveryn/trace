"""Neutral data sampling primitives for single-series chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.charts.shared.labeled_chart_count_datasets import build_value_count_dataset_for_variant
from trace_tasks.tasks.charts.shared.labeled_chart_composition import sample_composition_with_sum
from trace_tasks.tasks.charts.shared.labeled_chart_sampling import choose_mark_count
from trace_tasks.tasks.charts.shared.labeled_chart_summary_datasets import build_summary_statistics_dataset_for_variant
from trace_tasks.tasks.charts.shared.labeled_chart_trend_datasets import (
    build_trend_interval_change_dataset_for_variant,
    build_trend_structure_dataset_for_variant,
    build_trend_threshold_crossing_dataset_for_variant,
)
from trace_tasks.tasks.charts.shared.labeled_chart_values import (
    balanced_choice_from_values,
    resolve_mark_count_bounds,
    resolve_value_bounds,
    sorted_labels,
)
from trace_tasks.tasks.charts.shared.labeled_chart_variants import apply_scene_variant_mark_count_cap
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import DEFAULTS, GEN_DEFAULTS
from .state import SingleSeriesDataset, as_int_values, as_label_tuple


def count_dataset(
    *,
    count_variant: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> SingleSeriesDataset:
    values, answer, annotation_labels, extras = build_value_count_dataset_for_variant(
        count_variant=str(count_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        task_id=str(namespace),
    )
    ordered = tuple(str(label) for label in annotation_labels)
    return SingleSeriesDataset(
        labels=as_label_tuple(extras["labels"]),
        values=as_int_values(values),
        answer_value=int(answer),
        answer_type="integer",
        annotation_labels=tuple(sorted_labels(ordered)),
        ordered_annotation_labels=tuple(ordered),
        trace=dict(extras),
    )


def trend_structure_dataset(
    *,
    trend_variant: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> SingleSeriesDataset:
    values, answer, annotation_labels, extras = build_trend_structure_dataset_for_variant(
        trend_variant=str(trend_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        task_id=str(namespace),
    )
    ordered = tuple(str(label) for label in extras.get("ordered_annotation_labels", annotation_labels))
    return SingleSeriesDataset(
        labels=as_label_tuple(extras["labels"]),
        values=as_int_values(values),
        answer_value=int(answer),
        answer_type="integer",
        annotation_labels=tuple(sorted_labels(annotation_labels)),
        ordered_annotation_labels=tuple(ordered),
        trace=dict(extras),
    )


def interval_change_dataset(
    *,
    interval_variant: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> SingleSeriesDataset:
    values, answer, annotation_labels, extras = build_trend_interval_change_dataset_for_variant(
        interval_variant=str(interval_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        task_id=str(namespace),
    )
    ordered = tuple(str(label) for label in extras.get("ordered_annotation_labels", annotation_labels))
    return SingleSeriesDataset(
        labels=as_label_tuple(extras["labels"]),
        values=as_int_values(values),
        answer_value=int(answer),
        answer_type="integer",
        annotation_labels=tuple(sorted_labels(annotation_labels)),
        ordered_annotation_labels=tuple(ordered),
        trace=dict(extras),
    )


def threshold_crossing_dataset(
    *,
    crossing_variant: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> SingleSeriesDataset:
    values, answer, annotation_labels, extras = build_trend_threshold_crossing_dataset_for_variant(
        crossing_variant=str(crossing_variant),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        task_id=str(namespace),
    )
    ordered = tuple(str(label) for label in extras.get("ordered_annotation_labels", annotation_labels))
    return SingleSeriesDataset(
        labels=as_label_tuple(extras["labels"]),
        values=as_int_values(values),
        answer_value=str(answer),
        answer_type="string",
        annotation_labels=tuple(sorted_labels(annotation_labels)),
        ordered_annotation_labels=tuple(ordered),
        trace=dict(extras),
    )


def summary_dataset(
    *,
    statistic_kind: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    answer_target: str,
) -> SingleSeriesDataset:
    values, answer, annotation_labels, extras = build_summary_statistics_dataset_for_variant(
        statistic_kind=str(statistic_kind),
        scene_variant=str(scene_variant),
        params=params,
        instance_seed=int(instance_seed),
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        target_answer_ranges={"median": (1, 99)},
        task_id=str(namespace),
    )
    labels = as_label_tuple(extras["labels"])
    selected_label = str(annotation_labels[0])
    if str(answer_target) == "label":
        answer_value: int | str = selected_label
        answer_type = "string"
    else:
        answer_value = int(answer)
        answer_type = "integer"
    return SingleSeriesDataset(
        labels=labels,
        values=as_int_values(values),
        answer_value=answer_value,
        answer_type=answer_type,
        annotation_labels=(selected_label,),
        ordered_annotation_labels=(selected_label,),
        trace={**dict(extras), "answer_target": str(answer_target), "answer_label": selected_label},
    )


def counterfactual_dataset(
    *,
    operation: str,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> SingleSeriesDataset:
    """Sample counterfactual values using neutral scene-local operation codes."""

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        task_id=str(namespace),
        instance_seed=int(instance_seed),
    )
    mark_count = _choose_mark_count(
        params,
        scene_variant=str(scene_variant),
        operation=str(operation),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    labels = tuple(
        sample_chart_labels(
            count=int(mark_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(operation)}:{int(mark_count)}",
        )
    )
    if str(operation) == "remaining_mean":
        dataset = _remaining_mean_after_removal(
            labels=labels,
            params=params,
            instance_seed=int(instance_seed),
            value_min=int(value_min),
            value_max=int(value_max),
            namespace=str(namespace),
        )
    elif str(operation) == "target_share":
        dataset = _target_share_after_removal(
            labels=labels,
            params=params,
            instance_seed=int(instance_seed),
            value_min=int(value_min),
            value_max=int(value_max),
            namespace=str(namespace),
        )
    else:
        raise ValueError(f"unsupported counterfactual operation: {operation}")

    mark_min, mark_max = resolve_mark_count_bounds(params, gen_defaults=GEN_DEFAULTS, defaults=DEFAULTS, task_id=str(namespace))
    return SingleSeriesDataset(
        labels=tuple(dataset.labels),
        values=tuple(int(value) for value in dataset.values),
        answer_value=int(dataset.answer_value),
        answer_type="integer",
        annotation_labels=tuple(sorted_labels(dataset.annotation_labels)),
        ordered_annotation_labels=tuple(sorted_labels(dataset.annotation_labels)),
        trace={
            "value_min": int(value_min),
            "value_max": int(value_max),
            "mark_count": int(mark_count),
            "mark_count_range": [int(mark_min), int(mark_max)],
            "labels": [str(label) for label in dataset.labels],
            "values_by_label": {str(label): int(value) for label, value in zip(dataset.labels, dataset.values)},
            **dict(dataset.trace),
        },
    )


class _CounterfactualSample:
    def __init__(
        self,
        *,
        labels: Sequence[str],
        values: Sequence[int],
        answer_value: int,
        annotation_labels: Sequence[str],
        trace: Mapping[str, Any],
    ) -> None:
        self.labels = tuple(str(label) for label in labels)
        self.values = tuple(int(value) for value in values)
        self.answer_value = int(answer_value)
        self.annotation_labels = tuple(str(label) for label in annotation_labels)
        self.trace = dict(trace)


def _choose_mark_count(
    params: Mapping[str, Any],
    *,
    scene_variant: str,
    operation: str,
    instance_seed: int,
    namespace: str,
) -> int:
    mark_count_min, mark_count_max = resolve_mark_count_bounds(
        params,
        gen_defaults=GEN_DEFAULTS,
        defaults=DEFAULTS,
        task_id=str(namespace),
    )
    mark_count_min, mark_count_max = apply_scene_variant_mark_count_cap(
        scene_variant=str(scene_variant),
        mark_count_min=int(mark_count_min),
        mark_count_max=int(mark_count_max),
    )
    return choose_mark_count(
        [int(value) for value in range(int(mark_count_min), int(mark_count_max) + 1)],
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.mark_count.{str(operation)}",
    )


def _choose_count(
    params: Mapping[str, Any],
    *,
    explicit_key: str,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    max_allowed: int,
    instance_seed: int,
    namespace: str,
) -> int:
    min_count = int(params.get(str(min_key), group_default(GEN_DEFAULTS, str(min_key), int(fallback_min))))
    max_count = int(params.get(str(max_key), group_default(GEN_DEFAULTS, str(max_key), int(fallback_max))))
    max_count = min(int(max_count), int(max_allowed))
    candidates = [int(value) for value in range(int(min_count), int(max_count) + 1) if int(value) >= 0]
    if not candidates:
        raise ValueError(f"no feasible count support for {namespace}")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(candidates):
            raise ValueError(f"{explicit_key} outside feasible support")
        return int(value)
    return balanced_choice_from_values(candidates, params=params, instance_seed=int(instance_seed), namespace=str(namespace))


def _split_labels(
    labels: Sequence[str],
    *,
    selected_count: int,
    instance_seed: int,
    namespace: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    rng = spawn_rng(int(instance_seed), str(namespace))
    shuffled = [str(label) for label in labels]
    rng.shuffle(shuffled)
    return tuple(shuffled[: int(selected_count)]), tuple(shuffled[int(selected_count) :])


def _resolve_int_list(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> tuple[int, ...]:
    raw = params.get(str(key), group_default(GEN_DEFAULTS, str(key), list(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of integers")
    values = tuple(int(value) for value in raw)
    if not values:
        raise ValueError(f"{key} must not be empty")
    return values


def _random_values(
    *,
    count: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
    namespace: str,
) -> list[int]:
    rng = spawn_rng(int(instance_seed), str(namespace))
    return [int(rng.randint(int(value_min), int(value_max))) for _ in range(int(count))]


def _compose_values_with_sum(
    *,
    total: int,
    count: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
    namespace: str,
) -> list[int]:
    if int(total) < int(count) * int(value_min) or int(total) > int(count) * int(value_max):
        raise ValueError("total outside feasible value bounds")
    rng = spawn_rng(int(instance_seed), str(namespace))
    values = sample_composition_with_sum(
        rng,
        target_sum=int(total),
        count=int(count),
        value_min=int(value_min),
        value_max=int(value_max),
    )
    rng.shuffle(values)
    return [int(value) for value in values]


def _values_from_label_map(labels: Sequence[str], values_by_label: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(values_by_label[str(label)]) for label in labels)


def _remaining_mean_after_removal(
    *,
    labels: Sequence[str],
    params: Mapping[str, Any],
    instance_seed: int,
    value_min: int,
    value_max: int,
    namespace: str,
) -> _CounterfactualSample:
    """Construct a removable subset whose retained values have an integer mean."""

    removed_count = _choose_count(
        params,
        explicit_key="removed_count",
        min_key="removed_count_min",
        max_key="removed_count_max",
        fallback_min=1,
        fallback_max=3,
        max_allowed=len(labels) - 3,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.removed_count.remaining_mean",
    )
    removed_labels, retained_labels = _split_labels(
        labels,
        selected_count=int(removed_count),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.labels.remaining_mean",
    )
    retained_count = len(retained_labels)
    answer_min = int(params.get("target_answer_min", group_default(GEN_DEFAULTS, "target_answer_min", 15)))
    answer_max = int(params.get("target_answer_max", group_default(GEN_DEFAULTS, "target_answer_max", 65)))
    candidates = [
        int(value)
        for value in range(max(int(value_min), int(answer_min)), min(int(value_max), int(answer_max)) + 1)
        if int(value) * int(retained_count) >= int(retained_count) * int(value_min)
        and int(value) * int(retained_count) <= int(retained_count) * int(value_max)
    ]
    answer_value = balanced_choice_from_values(
        candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer.remaining_mean",
    )
    retained_values = _compose_values_with_sum(
        total=int(answer_value) * int(retained_count),
        count=int(retained_count),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.values.remaining_mean.retained",
    )
    removed_values = _random_values(
        count=int(removed_count),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.values.remaining_mean.removed",
    )
    values_by_label = {
        **{str(label): int(value) for label, value in zip(retained_labels, retained_values)},
        **{str(label): int(value) for label, value in zip(removed_labels, removed_values)},
    }
    return _CounterfactualSample(
        labels=labels,
        values=_values_from_label_map(labels, values_by_label),
        answer_value=int(answer_value),
        annotation_labels=sorted_labels(retained_labels),
        trace={
            "removed_labels": list(sorted_labels(removed_labels)),
            "retained_labels": list(sorted_labels(retained_labels)),
            "removed_count": int(removed_count),
            "retained_count": int(retained_count),
            "retained_sum": int(sum(retained_values)),
            "counterfactual_operation": "remove_labels_then_mean",
        },
    )


def _target_share_after_removal(
    *,
    labels: Sequence[str],
    params: Mapping[str, Any],
    instance_seed: int,
    value_min: int,
    value_max: int,
    namespace: str,
) -> _CounterfactualSample:
    """Construct removals where the target retained mark has an integer share."""

    removed_count = _choose_count(
        params,
        explicit_key="removed_count",
        min_key="removed_count_min",
        max_key="removed_count_max",
        fallback_min=1,
        fallback_max=3,
        max_allowed=len(labels) - 3,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.removed_count.target_share",
    )
    removed_labels, retained_labels = _split_labels(
        labels,
        selected_count=int(removed_count),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.labels.target_share",
    )
    target_label = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.target_label.target_share"),
            tuple(str(label) for label in retained_labels),
        )
    )
    other_retained = [str(label) for label in retained_labels if str(label) != str(target_label)]
    share_options = _resolve_int_list(params, "share_percent_values", (10, 15, 20, 25, 30, 40, 50, 60))
    feasible: list[tuple[int, int, int, int]] = []
    for percent in share_options:
        for target_value in range(int(value_min), int(value_max) + 1):
            if int(target_value) * 100 % int(percent) != 0:
                continue
            remaining_total = int(target_value) * 100 // int(percent)
            other_sum = int(remaining_total) - int(target_value)
            if len(other_retained) * int(value_min) <= int(other_sum) <= len(other_retained) * int(value_max):
                feasible.append((int(percent), int(target_value), int(remaining_total), int(other_sum)))
    if not feasible:
        raise ValueError("no feasible target-share support")
    answer_value = balanced_choice_from_values(
        sorted(set(item[0] for item in feasible)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer.target_share",
    )
    choices = [item for item in feasible if int(item[0]) == int(answer_value)]
    _, target_value, remaining_total, other_sum = uniform_choice(
        spawn_rng(int(instance_seed), f"{namespace}.target_share.choice.{answer_value}"),
        tuple(choices),
    )
    other_values = _compose_values_with_sum(
        total=int(other_sum),
        count=len(other_retained),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.values.target_share.retained_other",
    )
    removed_values = _random_values(
        count=int(removed_count),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.values.target_share.removed",
    )
    values_by_label = {
        str(target_label): int(target_value),
        **{str(label): int(value) for label, value in zip(other_retained, other_values)},
        **{str(label): int(value) for label, value in zip(removed_labels, removed_values)},
    }
    return _CounterfactualSample(
        labels=labels,
        values=_values_from_label_map(labels, values_by_label),
        answer_value=int(answer_value),
        annotation_labels=sorted_labels(retained_labels),
        trace={
            "removed_labels": list(sorted_labels(removed_labels)),
            "retained_labels": list(sorted_labels(retained_labels)),
            "target_label": str(target_label),
            "target_value": int(target_value),
            "remaining_total": int(remaining_total),
            "percent_value": int(answer_value),
            "counterfactual_operation": "remove_labels_then_target_share_percent",
        },
    )


__all__ = [
    "count_dataset",
    "counterfactual_dataset",
    "interval_change_dataset",
    "summary_dataset",
    "threshold_crossing_dataset",
    "trend_structure_dataset",
]
