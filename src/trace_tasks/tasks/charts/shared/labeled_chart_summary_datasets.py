"""Summary-statistic dataset builders for labeled charts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from ...shared.config_defaults import group_default
from .label_assets import sample_chart_labels
from .labeled_chart_defaults import LabeledChartDefaults
from .labeled_chart_values import (
    StatisticKind,
    balanced_choice_from_values,
    max_symmetric_delta,
    resolve_mark_count_bounds,
    resolve_target_answer_range,
    resolve_value_bounds,
)
from .labeled_chart_variants import SceneVariant, apply_scene_variant_mark_count_cap
from .labeled_chart_sampling import (
    choose_mark_count,
    choose_rank_n,
)
from .labeled_chart_statistics import (
    build_values_for_median,
    build_values_for_nth_rank,
    summarize_statistic_from_values,
)

def build_summary_statistics_dataset_for_variant(
    *,
    statistic_kind: StatisticKind,
    scene_variant: SceneVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    target_answer_ranges: Mapping[str, Tuple[int, int]],
    task_id: str,
) -> Tuple[List[int], int, List[str], Dict[str, Any]]:
    """Construct values, answer, annotation labels, and trace extras for one statistic variant."""

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    mark_count_min, mark_count_max = resolve_mark_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    mark_count_min, mark_count_max = apply_scene_variant_mark_count_cap(
        scene_variant=str(scene_variant),
        mark_count_min=int(mark_count_min),
        mark_count_max=int(mark_count_max),
    )

    if str(statistic_kind) in {"nth_highest", "nth_lowest"}:
        rank_n_min = int(params.get("rank_n_min", group_default(gen_defaults, "rank_n_min", 3)))
        rank_n_max = int(params.get("rank_n_max", group_default(gen_defaults, "rank_n_max", 8)))
        if int(rank_n_min) > int(rank_n_max):
            raise ValueError("rank_n_min must be <= rank_n_max")
        feasible_counts = [
            int(count)
            for count in range(int(mark_count_min), int(mark_count_max) + 1)
            if int(count) >= int(rank_n_min)
        ]
        mark_count = choose_mark_count(
            feasible_counts,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.mark_count:{str(statistic_kind)}:rank",
        )
        max_distinct_rank = int(value_max) - int(value_min) + 1
        rank_candidates = [
            int(rank)
            for rank in range(int(rank_n_min), min(int(rank_n_max), int(mark_count), int(max_distinct_rank)) + 1)
            if (
                (str(statistic_kind) == "nth_highest" and int(value_min) <= int(value_max) - (int(rank) - 1))
                or (str(statistic_kind) == "nth_lowest" and int(value_min) + (int(rank) - 1) <= int(value_max))
            )
        ]
        rank_n = choose_rank_n(
            rank_candidates,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.rank_n:{str(statistic_kind)}:{int(mark_count)}",
        )
        explicit_min = params.get("target_answer_min", None)
        explicit_max = params.get("target_answer_max", None)
        target_min = int(value_min if explicit_min is None else explicit_min)
        target_max = int(value_max if explicit_max is None else explicit_max)
        if str(statistic_kind) == "nth_highest":
            target_max = min(int(target_max), int(value_max) - (int(rank_n) - 1))
            direction = "highest"
        else:
            target_min = max(int(target_min), int(value_min) + (int(rank_n) - 1))
            direction = "lowest"
        target_candidates = [
            int(value)
            for value in range(int(target_min), int(target_max) + 1)
            if int(value_min) <= int(value) <= int(value_max)
        ]
        target_answer = balanced_choice_from_values(
            target_candidates,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.target_answer:{str(statistic_kind)}:{int(rank_n)}",
        )
        labels = list(
            sample_chart_labels(
                count=int(mark_count),
                instance_seed=int(instance_seed),
                namespace=f"{task_id}.labels:{str(statistic_kind)}:{int(mark_count)}",
            )
        )
        values = build_values_for_nth_rank(
            int(target_answer),
            count=int(mark_count),
            rank_n=int(rank_n),
            direction=str(direction),
            value_min=int(value_min),
            value_max=int(value_max),
            instance_seed=int(instance_seed),
        )
        answer_value, annotation_labels, summary_trace = summarize_statistic_from_values(
            statistic_kind=str(statistic_kind),
            labels=labels,
            values=values,
            rank_n=int(rank_n),
        )
        if int(answer_value) != int(target_answer):
            raise RuntimeError("constructed ranked values do not match the requested target answer")
        trace_extras = {
            "value_min": int(value_min),
            "value_max": int(value_max),
            "target_answer_range": [int(min(target_candidates)), int(max(target_candidates))],
            "mark_count_range": [int(mark_count_min), int(mark_count_max)],
            "rank_n_range": [int(rank_n_min), int(rank_n_max)],
            "target_answer": int(target_answer),
            "mark_count": int(mark_count),
            "labels": [str(label) for label in labels],
            "values_by_label": {str(label): int(value) for label, value in zip(labels, values)},
            **dict(summary_trace),
        }
        return [int(value) for value in values], int(answer_value), annotation_labels, trace_extras

    supported_answer_min, supported_answer_max = resolve_target_answer_range(
        params,
        value_min=int(value_min),
        value_max=int(value_max),
        target_answer_ranges=target_answer_ranges,
        statistic_kind=str(statistic_kind),
    )
    answer_candidates = [int(value) for value in range(int(supported_answer_min), int(supported_answer_max) + 1)]
    candidate_counts_for_target = [int(count) for count in range(int(mark_count_min), int(mark_count_max) + 1)]
    explicit_mark_count = params.get("mark_count")
    if explicit_mark_count is not None:
        candidate_counts_for_target = [
            int(count)
            for count in candidate_counts_for_target
            if int(count) == int(explicit_mark_count)
        ]
    if str(statistic_kind) != "median":
        raise ValueError(f"unsupported statistic_kind: {statistic_kind}")
    answer_candidates = [
        int(value)
        for value in answer_candidates
        if any(
            int(count) % 2 == 1
            and max_symmetric_delta(int(value), value_min=int(value_min), value_max=int(value_max))
            >= (int(count) - 1) // 2
            for count in candidate_counts_for_target
        )
    ]
    target_answer = balanced_choice_from_values(
        answer_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.target_answer:{str(statistic_kind)}",
    )

    feasible_counts = [
        int(count)
        for count in range(int(mark_count_min), int(mark_count_max) + 1)
        if int(count) % 2 == 1
        and max_symmetric_delta(int(target_answer), value_min=int(value_min), value_max=int(value_max))
        >= (int(count) - 1) // 2
    ]

    mark_count = choose_mark_count(
        feasible_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.mark_count:{str(statistic_kind)}:{int(target_answer)}",
    )
    labels = list(
        sample_chart_labels(
            count=int(mark_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(statistic_kind)}:{int(mark_count)}",
        )
    )

    values = build_values_for_median(
        int(target_answer),
        count=int(mark_count),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
    )

    answer_value, annotation_labels, summary_trace = summarize_statistic_from_values(
        statistic_kind=str(statistic_kind),
        labels=labels,
        values=values,
    )
    if int(answer_value) != int(target_answer):
        raise RuntimeError("constructed summary values do not match the requested target answer")

    trace_extras = {
        "value_min": int(value_min),
        "value_max": int(value_max),
        "target_answer_range": [int(supported_answer_min), int(supported_answer_max)],
        "mark_count_range": [int(mark_count_min), int(mark_count_max)],
        "target_answer": int(target_answer),
        "mark_count": int(mark_count),
        "labels": [str(label) for label in labels],
        "values_by_label": {str(label): int(value) for label, value in zip(labels, values)},
        **dict(summary_trace),
    }
    return [int(value) for value in values], int(answer_value), annotation_labels, trace_extras


__all__ = ["build_summary_statistics_dataset_for_variant"]
