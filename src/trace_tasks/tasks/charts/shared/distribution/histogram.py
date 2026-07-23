"""Histogram dataset builders for distribution-style chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .....core.seed import spawn_rng
from ..chart_scene_types import HistogramBinSpec
from .config import (
    DistributionChartDefaults,
    HistogramQueryVariant,
    _resolve_bin_frequency_bounds,
    _resolve_bin_start_bounds,
    _resolve_bin_width_bounds,
    _resolve_histogram_bin_count_bounds,
    _resolve_interval_bin_span_bounds,
    _resolve_outside_interval_bin_count_bounds,
)
from ..labeled_chart_values import balanced_choice_from_values
from ..labeled_chart_sampling import choose_mark_count


_CUMULATIVE_HISTOGRAM_VARIANTS = {
    "rank_item_bin_label",
}

def _build_histogram_labels(*, start_value: int, bin_width: int, bin_count: int) -> List[str]:
    labels: List[str] = []
    for index in range(int(bin_count)):
        value = int(start_value) + int(index) * int(bin_width)
        labels.append(str(int(value)))
    return labels


def _histogram_answer_range(
    *,
    query_id: HistogramQueryVariant,
    bin_count_max: int,
) -> Tuple[int, int]:
    if str(query_id) in _CUMULATIVE_HISTOGRAM_VARIANTS:
        return 1, 99
    if str(query_id) in {"interval_mass", "outside_interval_mass"}:
        raise ValueError(f"{query_id} derives its answer from sampled bar counts")
    raise ValueError(f"unsupported histogram query_id: {query_id}")


def _resolve_histogram_target_answer(
    params: Mapping[str, Any],
    *,
    query_id: HistogramQueryVariant,
    instance_seed: int,
    bin_count_max: int,
    task_id: str,
) -> int:
    default_min, default_max = _histogram_answer_range(
        query_id=str(query_id),
        bin_count_max=int(bin_count_max),
    )
    target_min = int(params.get("target_answer_min", default_min))
    target_max = int(params.get("target_answer_max", default_max))
    if int(target_min) > int(target_max):
        raise ValueError("target_answer_min must be <= target_answer_max")
    return balanced_choice_from_values(
        list(range(int(target_min), int(target_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:target_answer:{str(query_id)}",
    )


def build_histogram_dataset_for_variant(
    *,
    query_id: HistogramQueryVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
    mark_style: Mapping[str, Any],
) -> Tuple[List[HistogramBinSpec], int, List[str], Dict[str, Any]]:
    """Build one histogram dataset and query for the requested variant."""

    bin_count_min, bin_count_max = _resolve_histogram_bin_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    bin_width_min, bin_width_max = _resolve_bin_width_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    bin_start_min, bin_start_max = _resolve_bin_start_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    bin_frequency_min, bin_frequency_max = _resolve_bin_frequency_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    interval_bin_span_min, interval_bin_span_max = _resolve_interval_bin_span_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    outside_interval_bin_count_min, outside_interval_bin_count_max = _resolve_outside_interval_bin_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )

    rng = spawn_rng(int(instance_seed), f"{task_id}.histogram.{str(query_id)}")
    annotation_is_outside_interval = False

    if str(query_id) == "interval_mass":
        feasible_bin_counts = [
            int(value)
            for value in range(int(bin_count_min), int(bin_count_max) + 1)
            if int(value) >= int(interval_bin_span_min)
        ]
        if not feasible_bin_counts:
            raise ValueError("no feasible histogram interval_mass construction for requested interval span range")
        bin_count = choose_mark_count(
            feasible_bin_counts,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:histogram_bin_count:{str(query_id)}",
        )
        max_interval_span = min(int(interval_bin_span_max), int(bin_count))
        relevant_count = balanced_choice_from_values(
            list(range(int(interval_bin_span_min), int(max_interval_span) + 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:interval_bin_span:{str(query_id)}",
        )
        annotation_start = int(rng.randint(0, int(bin_count) - int(relevant_count)))
        annotation_stop = int(annotation_start) + int(relevant_count)
        relevant_values = [
            int(rng.randint(int(bin_frequency_min), int(bin_frequency_max)))
            for _ in range(int(relevant_count))
        ]
        target_answer = int(sum(int(value) for value in relevant_values))
        background_values = [
            int(rng.randint(int(bin_frequency_min), int(bin_frequency_max)))
            for _ in range(int(bin_count) - int(relevant_count))
        ]
        query_meta = {"interval_bin_span": int(relevant_count)}
    elif str(query_id) == "outside_interval_mass":
        feasible_bin_counts = [
            int(value)
            for value in range(int(bin_count_min), int(bin_count_max) + 1)
            if min(int(outside_interval_bin_count_max), int(value) - 2) >= int(outside_interval_bin_count_min)
        ]
        if not feasible_bin_counts:
            raise ValueError("no feasible histogram outside_interval_mass construction for requested outside-bin range")
        bin_count = choose_mark_count(
            feasible_bin_counts,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:histogram_bin_count:{str(query_id)}",
        )
        max_outside_count = min(int(outside_interval_bin_count_max), int(bin_count) - 2)
        outside_count = balanced_choice_from_values(
            list(range(int(outside_interval_bin_count_min), int(max_outside_count) + 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:outside_interval_bin_count:{str(query_id)}",
        )
        left_outside_count = int(rng.randint(1, int(outside_count) - 1))
        right_outside_count = int(outside_count) - int(left_outside_count)
        annotation_start = int(left_outside_count)
        annotation_stop = int(bin_count) - int(right_outside_count)
        relevant_count = int(annotation_stop) - int(annotation_start)
        relevant_values = [
            int(rng.randint(int(bin_frequency_min), int(bin_frequency_max)))
            for _ in range(int(relevant_count))
        ]
        background_values = [
            int(rng.randint(int(bin_frequency_min), int(bin_frequency_max)))
            for _ in range(int(outside_count))
        ]
        target_answer = int(sum(int(value) for value in background_values))
        annotation_is_outside_interval = True
        query_meta = {
            "interval_bin_span": int(relevant_count),
            "excluded_interval_bin_span": int(relevant_count),
            "outside_bin_count": int(outside_count),
            "outside_left_bin_count": int(left_outside_count),
            "outside_right_bin_count": int(right_outside_count),
        }
    elif str(query_id) in _CUMULATIVE_HISTOGRAM_VARIANTS:
        feasible_bin_counts = [
            int(value)
            for value in range(int(bin_count_min), int(bin_count_max) + 1)
            if int(value) >= 5
        ]
        if not feasible_bin_counts:
            raise ValueError("no feasible histogram cumulative-rank construction for requested bin-count range")
        bin_count = choose_mark_count(
            feasible_bin_counts,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:histogram_bin_count:{str(query_id)}",
        )
        sampled_values = [
            int(rng.randint(int(bin_frequency_min), int(bin_frequency_max)))
            for _ in range(int(bin_count))
        ]
        total_count = int(sum(sampled_values))
        min_answer_index = min(2, int(bin_count) - 1)
        max_answer_index = max(int(min_answer_index), int(bin_count) - 3)
        answer_index = balanced_choice_from_values(
            list(range(int(min_answer_index), int(max_answer_index) + 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:answer_bin_index:{str(query_id)}",
        )
        previous_cumulative = int(sum(sampled_values[: int(answer_index)]))
        answer_bin_count = int(sampled_values[int(answer_index)])
        target_rank = balanced_choice_from_values(
            list(range(int(previous_cumulative) + 1, int(previous_cumulative) + int(answer_bin_count) + 1)),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:target_rank:{str(query_id)}",
        )
        rank_fraction_numerator = 0
        rank_fraction_denominator = 0
        annotation_start = int(answer_index)
        annotation_stop = int(answer_index) + 1
        relevant_count = 1
        relevant_values = [int(sampled_values[int(answer_index)])]
        background_values = [
            int(value)
            for index, value in enumerate(sampled_values)
            if int(index) != int(answer_index)
        ]
        target_answer = -1
        query_meta = {
            "target_rank": int(target_rank),
            "total_count": int(total_count),
            "rank_fraction_numerator": int(rank_fraction_numerator),
            "rank_fraction_denominator": int(rank_fraction_denominator),
            "answer_bin_index": int(answer_index),
            "answer_prefix_bin_count": int(annotation_stop),
            "cumulative_count_before_answer_bin": int(previous_cumulative),
            "answer_bin_count": int(answer_bin_count),
            "cumulative_count_through_answer_bin": int(previous_cumulative) + int(answer_bin_count),
        }
    else:
        raise ValueError(f"unsupported histogram query_id: {query_id}")

    if int(bin_width_min) < 1 or int(bin_width_max) < int(bin_width_min):
        raise ValueError("histogram bin-width range must be positive and ordered")
    bin_width = balanced_choice_from_values(
        list(range(int(bin_width_min), int(bin_width_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:bin_width:{str(query_id)}",
    )
    capped_bin_start_max = min(int(bin_start_max), 100 - (int(bin_count) * int(bin_width)))
    if int(capped_bin_start_max) < int(bin_start_min):
        raise ValueError("histogram bin_start range cannot keep the largest x-axis value <= 99")
    start_value = balanced_choice_from_values(
        list(range(int(bin_start_min), int(capped_bin_start_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:bin_start:{str(query_id)}",
    )
    labels = _build_histogram_labels(
        start_value=int(start_value),
        bin_width=int(bin_width),
        bin_count=int(bin_count),
    )
    values: List[int] = []
    background_iter = iter(background_values)
    for index in range(int(bin_count)):
        if int(annotation_start) <= int(index) < int(annotation_stop):
            values.append(int(relevant_values[int(index) - int(annotation_start)]))
        else:
            values.append(int(next(background_iter)))

    if bool(annotation_is_outside_interval):
        annotation_labels = [str(label) for label in labels[: int(annotation_start)]]
        annotation_labels.extend(str(label) for label in labels[int(annotation_stop) :])
    else:
        annotation_labels = [str(label) for label in labels[int(annotation_start) : int(annotation_stop)]]
    bins: List[HistogramBinSpec] = []
    fill_rgb = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])
    for index, (label, count_value) in enumerate(zip(labels, values)):
        lower = int(start_value) + int(index) * int(bin_width)
        upper = int(lower) + int(bin_width) - 1
        bins.append(
            HistogramBinSpec(
                label=str(label),
                count=int(count_value),
                interval_start=int(lower),
                interval_end=int(upper),
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
            )
        )

    query_interval_label = ""
    query_bin_label = ""
    if str(query_id) in {"interval_mass", "outside_interval_mass"}:
        query_interval_label = f"{bins[int(annotation_start)].interval_start}-{bins[int(annotation_stop) - 1].interval_end}"
        query_meta["query_interval_start_value"] = int(bins[int(annotation_start)].interval_start)
        query_meta["query_interval_end_value"] = int(bins[int(annotation_stop) - 1].interval_end)
    if str(query_id) in _CUMULATIVE_HISTOGRAM_VARIANTS:
        query_bin_label = str(bins[int(query_meta["answer_bin_index"])].label)
        target_answer = int(query_bin_label)
        query_meta["answer_bin_label"] = str(query_bin_label)
        query_meta["answer_bin_value"] = int(target_answer)

    trace_extras = {
        "scene_variant": "histogram",
        "bin_count": int(bin_count),
        "bin_count_range": [int(bin_count_min), int(bin_count_max)],
        "bin_width": int(bin_width),
        "bin_width_range": [int(bin_width_min), int(bin_width_max)],
        "bin_start": int(start_value),
        "bin_start_range": [int(bin_start_min), int(capped_bin_start_max)],
        "bin_axis_value_max": 99,
        "bin_frequency_range": [int(bin_frequency_min), int(bin_frequency_max)],
        "interval_bin_span_range": [int(interval_bin_span_min), int(interval_bin_span_max)],
        "outside_interval_bin_count_range": [
            int(outside_interval_bin_count_min),
            int(outside_interval_bin_count_max),
        ],
        "target_answer": int(target_answer),
        "target_answer_range": (
            []
            if str(query_id) in {"interval_mass", "outside_interval_mass"}
            else (
                [int(labels[0]), int(labels[-1])]
                if str(query_id) in _CUMULATIVE_HISTOGRAM_VARIANTS
                else list(_histogram_answer_range(query_id=str(query_id), bin_count_max=int(bin_count_max)))
            )
        ),
        "labels": [str(label) for label in labels],
        "bin_counts": [int(value) for value in values],
        "annotation_labels": [str(label) for label in annotation_labels],
        "query_interval_label": str(query_interval_label),
        "query_bin_label": str(query_bin_label),
        **{str(key): value for key, value in query_meta.items()},
    }
    return bins, int(target_answer), [str(label) for label in annotation_labels], trace_extras


__all__ = ["build_histogram_dataset_for_variant"]
