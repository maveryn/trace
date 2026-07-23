"""Dataset builders for multiseries chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ...shared.label_assets import sample_chart_labels
from ...shared.labeled_chart_values import (
    balanced_choice_from_values,
    resolve_value_bounds,
    sorted_labels,
)
from ...shared.labeled_chart_composition import (
    sample_composition_with_sum,
)
from .defaults import resolve_category_count_bounds, resolve_series_count_bounds
from .sampling import _sample_distinct_values, sample_series_labels
from .state import MultiseriesChartDefaults


def build_category_total_extremum_label_dataset(
    *,
    variant_key: str,
    extremum_direction: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    namespace: str,
) -> Tuple[Dict[str, Dict[str, int]], str, List[int], Dict[str, Any]]:
    """Construct one category-total extremum label dataset."""

    if str(variant_key) != "category_total_extremum":
        raise ValueError(f"unsupported multiseries category-total variant: {variant_key}")
    if str(extremum_direction) not in {"largest", "smallest"}:
        raise ValueError(f"unsupported extremum_direction: {extremum_direction}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=namespace,
        instance_seed=int(instance_seed),
    )
    category_count_min, category_count_max = resolve_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    series_count_min, series_count_max = resolve_series_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    feasible_series_counts = [int(count) for count in range(int(series_count_min), int(series_count_max) + 1)]
    series_count = balanced_choice_from_values(
        feasible_series_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.series_count:{str(variant_key)}",
    )

    total_min = int(series_count) * int(value_min)
    total_max = int(series_count) * int(value_max)
    feasible_category_counts = [
        int(count)
        for count in range(int(category_count_min), int(category_count_max) + 1)
        if int(count) <= (int(total_max) - int(total_min) + 1)
    ]
    if not feasible_category_counts:
        raise ValueError("category-total task requires enough unique total support")
    category_count = balanced_choice_from_values(
        feasible_category_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.category_count:{str(variant_key)}:{int(series_count)}",
    )

    rank_min = int(
        params.get(
            "category_total_rank_min",
            group_default(gen_defaults, "category_total_rank_min", group_default(gen_defaults, "rank_min", 1)),
        )
    )
    rank_max = int(
        params.get(
            "category_total_rank_max",
            group_default(gen_defaults, "category_total_rank_max", group_default(gen_defaults, "rank_max", 2)),
        )
    )
    if int(rank_min) < 1:
        raise ValueError("category_total_rank_min must be >= 1")
    if int(rank_min) > int(rank_max):
        raise ValueError("category_total_rank_min must be <= category_total_rank_max")
    rank_candidates = [
        int(rank)
        for rank in range(int(rank_min), int(rank_max) + 1)
        if 1 <= int(rank) <= int(category_count)
    ]
    if not rank_candidates:
        raise ValueError("category-total task requires a feasible rank")
    answer_rank = balanced_choice_from_values(
        rank_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer_rank:{str(variant_key)}:{str(extremum_direction)}",
    )

    category_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(variant_key)}:{int(category_count)}",
        )
    )
    series_labels = list(sample_series_labels(count=int(series_count), instance_seed=int(instance_seed)))
    total_rng = spawn_rng(int(instance_seed), f"{namespace}.category_totals:{str(variant_key)}")
    category_totals = total_rng.sample(
        [int(value) for value in range(int(total_min), int(total_max) + 1)],
        int(category_count),
    )
    assignment_rng = spawn_rng(int(instance_seed), f"{namespace}.category_total_assignment:{str(variant_key)}")
    assignment_rng.shuffle(category_totals)
    totals_by_category = {
        str(category_label): int(total)
        for category_label, total in zip(category_labels, category_totals)
    }
    if str(extremum_direction) == "smallest":
        sorted_total_items = sorted(totals_by_category.items(), key=lambda item: (int(item[1]), str(item[0])))
        rank_order = "ascending"
    else:
        sorted_total_items = sorted(totals_by_category.items(), key=lambda item: (-int(item[1]), str(item[0])))
        rank_order = "descending"
    answer_label = str(sorted_total_items[int(answer_rank) - 1][0])
    answer_total = int(totals_by_category[str(answer_label)])

    value_rng = spawn_rng(int(instance_seed), f"{namespace}.category_total_values:{str(variant_key)}")
    values_by_category: Dict[str, Dict[str, int]] = {}
    annotation_values: List[int] = []
    for category_label in category_labels:
        series_values = sample_composition_with_sum(
            value_rng,
            target_sum=int(totals_by_category[str(category_label)]),
            count=int(series_count),
            value_min=int(value_min),
            value_max=int(value_max),
        )
        value_rng.shuffle(series_values)
        values_by_category[str(category_label)] = {
            str(series_label): int(value)
            for series_label, value in zip(series_labels, series_values)
        }
        if str(category_label) == str(answer_label):
            annotation_values = [int(value) for value in series_values]

    trace_extras: Dict[str, Any] = {
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "series_count": int(series_count),
        "series_count_range": [int(series_count_min), int(series_count_max)],
        "category_labels": [str(label) for label in category_labels],
        "series_labels": [str(label) for label in series_labels],
        "queried_series_labels": [str(label) for label in series_labels],
        "value_min": int(value_min),
        "value_max": int(value_max),
        "value_range": [int(value_min), int(value_max)],
        "category_total_range": [int(total_min), int(total_max)],
        "answer_label": str(answer_label),
        "answer_rank": int(answer_rank),
        "answer_score": int(answer_total),
        "annotation_values": [int(value) for value in annotation_values],
        "derived_metric": "category_total",
        "calculation_scope": "category_total",
        "rank_order": str(rank_order),
        "extremum_direction": str(extremum_direction),
        "ranked_category_labels": [str(label) for label, _score in sorted_total_items],
        "category_totals_by_category": {
            str(category_label): int(value)
            for category_label, value in sorted(totals_by_category.items())
        },
        "derived_values_by_category": {
            str(category_label): int(value)
            for category_label, value in sorted(totals_by_category.items())
        },
        "values_by_category": {
            str(category_label): {
                str(series_label): int(value)
                for series_label, value in sorted(series_values.items())
            }
            for category_label, series_values in values_by_category.items()
        },
    }
    return values_by_category, str(answer_label), [int(value) for value in annotation_values], trace_extras


def build_pairwise_comparison_count_dataset(
    *,
    variant_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    namespace: str,
) -> Tuple[Dict[str, Dict[str, int]], int, List[str], Dict[str, Any]]:
    """Construct comparison-count values with controlled satisfying categories.

    This builder owns the invariant that exactly ``target_answer`` category
    groups satisfy the selected greater-than or less-than relation. The public
    task decides the comparison axis; this helper only returns neutral
    category/series values and trace fields for rendering.
    """

    if str(variant_key) not in {"series_a_gt_b_count", "series_a_lt_b_count"}:
        raise ValueError(f"unsupported multiseries variant_key: {variant_key}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=namespace,
        instance_seed=int(instance_seed),
    )
    category_count_min, category_count_max = resolve_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    series_count_min, series_count_max = resolve_series_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    if int(series_count_min) < 2:
        raise ValueError("multiseries comparison requires at least two series")

    answer_min = int(params.get("target_answer_min", group_default(gen_defaults, "target_answer_min", defaults.target_answer_min)))
    answer_max = int(params.get("target_answer_max", group_default(gen_defaults, "target_answer_max", defaults.target_answer_max)))
    if int(answer_min) > int(answer_max):
        raise ValueError("target_answer_min must be <= target_answer_max")

    answer_candidates = [
        int(value)
        for value in range(int(answer_min), int(answer_max) + 1)
        if int(value) <= int(category_count_max)
    ]
    target_answer = balanced_choice_from_values(
        answer_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_answer:{str(variant_key)}",
    )
    feasible_category_counts = [
        int(count)
        for count in range(int(category_count_min), int(category_count_max) + 1)
        if int(count) >= int(target_answer)
    ]
    category_count = balanced_choice_from_values(
        feasible_category_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.category_count:{str(variant_key)}:{int(target_answer)}",
    )
    feasible_series_counts = [int(count) for count in range(int(series_count_min), int(series_count_max) + 1)]
    series_count = balanced_choice_from_values(
        feasible_series_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.series_count:{str(variant_key)}",
    )

    category_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(variant_key)}:{int(category_count)}",
        )
    )
    series_labels = list(sample_series_labels(count=int(series_count), instance_seed=int(instance_seed)))
    query_rng = spawn_rng(int(instance_seed), f"{namespace}.query_pair")
    pair_candidates = [
        (str(series_labels[left_index]), str(series_labels[right_index]))
        for left_index in range(int(series_count))
        for right_index in range(int(left_index) + 1, int(series_count))
    ]
    query_pair = pair_candidates[query_rng.randrange(len(pair_candidates))]
    left_series, right_series = (str(query_pair[0]), str(query_pair[1]))

    category_indices = list(range(int(category_count)))
    satisfy_rng = spawn_rng(int(instance_seed), f"{namespace}.satisfying_categories")
    satisfy_rng.shuffle(category_indices)
    satisfying_indices = set(category_indices[: int(target_answer)])

    values_rng = spawn_rng(int(instance_seed), f"{namespace}.values")
    values_by_category: Dict[str, Dict[str, int]] = {}
    annotation_labels: List[str] = []
    for category_index, category_label in enumerate(category_labels):
        sampled_values = sorted(
            _sample_distinct_values(
                values_rng,
                count=int(series_count),
                value_min=int(value_min),
                value_max=int(value_max),
            )
        )
        low_value = int(sampled_values[0])
        high_value = int(sampled_values[-1])
        remaining_values = [int(value) for value in sampled_values[1:-1]]
        relation_holds = bool(int(category_index) in satisfying_indices)
        if str(variant_key) == "series_a_gt_b_count":
            left_value = int(high_value if relation_holds else low_value)
            right_value = int(low_value if relation_holds else high_value)
        else:
            left_value = int(low_value if relation_holds else high_value)
            right_value = int(high_value if relation_holds else low_value)
        category_values: Dict[str, int] = {
            str(left_series): int(left_value),
            str(right_series): int(right_value),
        }
        distractor_series_labels = [
            str(series_label)
            for series_label in series_labels
            if str(series_label) not in {str(left_series), str(right_series)}
        ]
        for distractor_series, distractor_value in zip(distractor_series_labels, remaining_values):
            category_values[str(distractor_series)] = int(distractor_value)
        values_by_category[str(category_label)] = {
            str(series_label): int(category_values[str(series_label)])
            for series_label in series_labels
        }
        if relation_holds:
            annotation_labels.append(str(category_label))

    annotation_labels = sorted_labels(annotation_labels)
    trace_extras: Dict[str, Any] = {
        "target_answer": int(target_answer),
        "target_answer_range": [int(answer_min), int(answer_max)],
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "series_count": int(series_count),
        "series_count_range": [int(series_count_min), int(series_count_max)],
        "category_labels": [str(label) for label in category_labels],
        "series_labels": [str(label) for label in series_labels],
        "queried_series_labels": [str(left_series), str(right_series)],
        "left_series_label": str(left_series),
        "right_series_label": str(right_series),
        "value_min": int(value_min),
        "value_max": int(value_max),
        "values_by_category": {
            str(category_label): {
                str(series_label): int(value)
                for series_label, value in sorted(series_values.items())
            }
            for category_label, series_values in values_by_category.items()
        },
        "comparison": "greater_than" if str(variant_key) == "series_a_gt_b_count" else "less_than",
    }
    return values_by_category, int(target_answer), annotation_labels, trace_extras


def build_pair_equality_label_dataset(
    *,
    variant_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    namespace: str,
) -> Tuple[Dict[str, Dict[str, int]], str, List[int], Dict[str, Any]]:
    """Construct one multiseries exact-equality label dataset."""

    if str(variant_key) != "pair_equality":
        raise ValueError(f"unsupported multiseries equality variant: {variant_key}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=namespace,
        instance_seed=int(instance_seed),
    )
    if int(value_min) >= int(value_max):
        raise ValueError("pair equality task requires at least two possible values")
    category_count_min, category_count_max = resolve_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    series_count_min, series_count_max = resolve_series_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    if int(series_count_min) < 2:
        raise ValueError("multiseries equality task requires at least two series")

    feasible_category_counts = [int(count) for count in range(int(category_count_min), int(category_count_max) + 1)]
    category_count = balanced_choice_from_values(
        feasible_category_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.category_count:{str(variant_key)}",
    )
    feasible_series_counts = [int(count) for count in range(int(series_count_min), int(series_count_max) + 1)]
    series_count = balanced_choice_from_values(
        feasible_series_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.series_count:{str(variant_key)}",
    )

    category_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(variant_key)}:{int(category_count)}",
        )
    )
    series_labels = list(sample_series_labels(count=int(series_count), instance_seed=int(instance_seed)))
    query_rng = spawn_rng(int(instance_seed), f"{namespace}.query_pair:{str(variant_key)}")
    pair_candidates = [
        (str(series_labels[left_index]), str(series_labels[right_index]))
        for left_index in range(int(series_count))
        for right_index in range(int(left_index) + 1, int(series_count))
    ]
    left_series, right_series = pair_candidates[query_rng.randrange(len(pair_candidates))]

    answer_index = balanced_choice_from_values(
        list(range(int(category_count))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer_index:{str(variant_key)}",
    )
    answer_label = str(category_labels[int(answer_index)])
    value_rng = spawn_rng(int(instance_seed), f"{namespace}.values:{str(variant_key)}")
    equal_value = int(value_rng.randint(int(value_min), int(value_max)))
    non_equal_values = [int(value) for value in range(int(value_min), int(value_max) + 1)]

    values_by_category: Dict[str, Dict[str, int]] = {}
    equality_by_category: Dict[str, bool] = {}
    gap_by_category: Dict[str, int] = {}
    for category_label in category_labels:
        if str(category_label) == str(answer_label):
            left_value = int(equal_value)
            right_value = int(equal_value)
        else:
            left_value = int(value_rng.randint(int(value_min), int(value_max)))
            right_candidates = [int(value) for value in non_equal_values if int(value) != int(left_value)]
            right_value = int(right_candidates[value_rng.randrange(len(right_candidates))])

        category_values: Dict[str, int] = {
            str(left_series): int(left_value),
            str(right_series): int(right_value),
        }
        for series_label in series_labels:
            if str(series_label) in category_values:
                continue
            category_values[str(series_label)] = int(value_rng.randint(int(value_min), int(value_max)))
        values_by_category[str(category_label)] = {
            str(series_label): int(category_values[str(series_label)])
            for series_label in series_labels
        }
        is_equal = int(left_value) == int(right_value)
        equality_by_category[str(category_label)] = bool(is_equal)
        gap_by_category[str(category_label)] = abs(int(left_value) - int(right_value))

    equality_labels = [str(label) for label, is_equal in equality_by_category.items() if bool(is_equal)]
    if equality_labels != [str(answer_label)]:
        raise RuntimeError("pair equality task failed to construct a unique equality label")

    trace_extras: Dict[str, Any] = {
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "series_count": int(series_count),
        "series_count_range": [int(series_count_min), int(series_count_max)],
        "category_labels": [str(label) for label in category_labels],
        "series_labels": [str(label) for label in series_labels],
        "queried_series_labels": [str(left_series), str(right_series)],
        "left_series_label": str(left_series),
        "right_series_label": str(right_series),
        "value_min": int(value_min),
        "value_max": int(value_max),
        "value_range": [int(value_min), int(value_max)],
        "answer_label": str(answer_label),
        "answer_equal_value": int(equal_value),
        "answer_score": 0,
        "answer_rank": 1,
        "annotation_values": [int(equal_value), int(equal_value)],
        "derived_metric": "exact_equality",
        "direction": "none",
        "rank_order": "unique_zero_gap",
        "ranked_category_labels": [str(answer_label)],
        "derived_values_by_category": {
            str(label): int(value)
            for label, value in sorted(gap_by_category.items())
        },
        "equality_by_category": {
            str(label): bool(value)
            for label, value in sorted(equality_by_category.items())
        },
        "values_by_category": {
            str(category_label): {
                str(series_label): int(value)
                for series_label, value in sorted(series_values.items())
            }
            for category_label, series_values in values_by_category.items()
        },
    }
    return values_by_category, str(answer_label), [int(equal_value), int(equal_value)], trace_extras


def build_series_rank_at_category_label_dataset(
    *,
    variant_key: str,
    extremum_direction: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    namespace: str,
) -> Tuple[Dict[str, Dict[str, int]], str, List[int], Dict[str, Any]]:
    """Construct one dataset for ranking series values within a target category."""

    if str(variant_key) != "series_rank_at_category":
        raise ValueError(f"unsupported multiseries series-rank variant: {variant_key}")
    if str(extremum_direction) not in {"largest", "smallest"}:
        raise ValueError(f"unsupported extremum_direction: {extremum_direction}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=namespace,
        instance_seed=int(instance_seed),
    )
    category_count_min, category_count_max = resolve_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    series_count_min, series_count_max = resolve_series_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    if int(series_count_min) < 2:
        raise ValueError("multiseries series-rank task requires at least two series")

    feasible_category_counts = [int(count) for count in range(int(category_count_min), int(category_count_max) + 1)]
    category_count = balanced_choice_from_values(
        feasible_category_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.category_count:{str(variant_key)}",
    )
    feasible_series_counts = [int(count) for count in range(int(series_count_min), int(series_count_max) + 1)]
    series_count = balanced_choice_from_values(
        feasible_series_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.series_count:{str(variant_key)}",
    )
    if int(series_count) > (int(value_max) - int(value_min) + 1):
        raise ValueError("series-rank task requires enough distinct values for the target category")

    rank_min = int(
        params.get(
            "series_rank_rank_min",
            group_default(gen_defaults, "series_rank_rank_min", group_default(gen_defaults, "rank_min", 1)),
        )
    )
    rank_max = int(
        params.get(
            "series_rank_rank_max",
            group_default(gen_defaults, "series_rank_rank_max", group_default(gen_defaults, "rank_max", 3)),
        )
    )
    if int(rank_min) < 1:
        raise ValueError("series_rank_rank_min must be >= 1")
    if int(rank_min) > int(rank_max):
        raise ValueError("series_rank_rank_min must be <= series_rank_rank_max")
    rank_candidates = [
        int(rank)
        for rank in range(int(rank_min), int(rank_max) + 1)
        if 1 <= int(rank) <= int(series_count)
    ]
    if not rank_candidates:
        raise ValueError("series-rank task requires a feasible rank")
    answer_rank = balanced_choice_from_values(
        rank_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer_rank:{str(variant_key)}:{str(extremum_direction)}",
    )

    category_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(variant_key)}:{int(category_count)}",
        )
    )
    series_labels = list(sample_series_labels(count=int(series_count), instance_seed=int(instance_seed)))
    target_category_index = balanced_choice_from_values(
        list(range(int(category_count))),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.target_category_index:{str(variant_key)}",
    )
    target_category_label = str(category_labels[int(target_category_index)])

    value_rng = spawn_rng(int(instance_seed), f"{namespace}.values:{str(variant_key)}")
    target_values = _sample_distinct_values(
        value_rng,
        count=int(series_count),
        value_min=int(value_min),
        value_max=int(value_max),
    )
    value_rng.shuffle(target_values)

    values_by_category: Dict[str, Dict[str, int]] = {}
    for category_label in category_labels:
        if str(category_label) == str(target_category_label):
            category_values = {
                str(series_label): int(value)
                for series_label, value in zip(series_labels, target_values)
            }
        else:
            category_values = {
                str(series_label): int(value_rng.randint(int(value_min), int(value_max)))
                for series_label in series_labels
            }
        values_by_category[str(category_label)] = {
            str(series_label): int(category_values[str(series_label)])
            for series_label in series_labels
        }

    values_by_series = {
        str(series_label): int(values_by_category[str(target_category_label)][str(series_label)])
        for series_label in series_labels
    }
    if str(extremum_direction) == "smallest":
        ranked_series_items = sorted(values_by_series.items(), key=lambda item: (int(item[1]), str(item[0])))
        rank_order = "ascending"
    else:
        ranked_series_items = sorted(values_by_series.items(), key=lambda item: (-int(item[1]), str(item[0])))
        rank_order = "descending"
    answer_series_label = str(ranked_series_items[int(answer_rank) - 1][0])
    answer_score = int(values_by_series[str(answer_series_label)])
    annotation_values = [
        int(values_by_series[str(series_label)])
        for series_label in series_labels
    ]

    trace_extras: Dict[str, Any] = {
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "series_count": int(series_count),
        "series_count_range": [int(series_count_min), int(series_count_max)],
        "category_labels": [str(label) for label in category_labels],
        "series_labels": [str(label) for label in series_labels],
        "queried_series_labels": [str(label) for label in series_labels],
        "target_category_label": str(target_category_label),
        "target_category_index": int(target_category_index),
        "value_min": int(value_min),
        "value_max": int(value_max),
        "value_range": [int(value_min), int(value_max)],
        "answer_label": str(answer_series_label),
        "answer_series_label": str(answer_series_label),
        "answer_rank": int(answer_rank),
        "answer_score": int(answer_score),
        "annotation_values": [int(value) for value in annotation_values],
        "derived_metric": "series_value_at_target_category",
        "calculation_scope": "target_category_series_rank",
        "rank_order": str(rank_order),
        "extremum_direction": str(extremum_direction),
        "ranked_series_labels": [str(label) for label, _value in ranked_series_items],
        "ranked_category_labels": [str(target_category_label)],
        "values_by_series_at_target_category": {
            str(label): int(value)
            for label, value in sorted(values_by_series.items())
        },
        "derived_values_by_category": {
            str(label): int(values_by_series[str(label)])
            for label in series_labels
        },
        "values_by_category": {
            str(category_label): {
                str(series_label): int(value)
                for series_label, value in sorted(series_values.items())
            }
            for category_label, series_values in values_by_category.items()
        },
    }
    return values_by_category, str(answer_series_label), [int(value) for value in annotation_values], trace_extras


def _sample_score_window(
    rng,
    *,
    count: int,
    max_score: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> List[int]:
    """Sample a unique, usually tight derived-score support."""

    score_min = int(params.get("derived_score_min", group_default(gen_defaults, "derived_score_min", 1)))
    score_max = int(params.get("derived_score_max", group_default(gen_defaults, "derived_score_max", int(max_score))))
    if int(score_min) < 1:
        raise ValueError("derived_score_min must be positive")
    if int(score_max) > int(max_score):
        score_max = int(max_score)
    if int(score_min) > int(score_max):
        raise ValueError("derived_score_min must be <= derived_score_max")
    score_support_size = int(score_max) - int(score_min) + 1
    if int(count) > int(score_support_size):
        raise ValueError("delta-extremum task requires enough distinct derived scores")
    extra_min = int(params.get("score_spread_extra_min", group_default(gen_defaults, "score_spread_extra_min", 0)))
    extra_max = int(params.get("score_spread_extra_max", group_default(gen_defaults, "score_spread_extra_max", 4)))
    if int(extra_min) < 0 or int(extra_max) < 0:
        raise ValueError("score_spread_extra bounds must be non-negative")
    if int(extra_min) > int(extra_max):
        raise ValueError("score_spread_extra_min must be <= score_spread_extra_max")
    feasible_extra_max = min(int(extra_max), int(score_support_size) - int(count))
    feasible_extra_min = min(int(extra_min), int(feasible_extra_max))
    extra = rng.randint(int(feasible_extra_min), int(feasible_extra_max))
    window_size = int(count) + int(extra)
    start = rng.randint(int(score_min), int(score_max) - int(window_size) + 1)
    window = [int(value) for value in range(int(start), int(start) + int(window_size))]
    return [int(value) for value in rng.sample(window, int(count))]


def _sample_ratio_percent_window(
    rng,
    *,
    count: int,
    feasible_scores: Sequence[int],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> List[int]:
    """Sample a unique, usually tight ratio-percent score support."""

    support = sorted({int(score) for score in feasible_scores})
    if int(count) > len(support):
        raise ValueError("ratio-extremum task requires enough distinct feasible ratio scores")
    extra_min = int(params.get("ratio_score_spread_extra_min", group_default(gen_defaults, "ratio_score_spread_extra_min", 0)))
    extra_max = int(params.get("ratio_score_spread_extra_max", group_default(gen_defaults, "ratio_score_spread_extra_max", 6)))
    if int(extra_min) < 0 or int(extra_max) < 0:
        raise ValueError("ratio_score_spread_extra bounds must be non-negative")
    if int(extra_min) > int(extra_max):
        raise ValueError("ratio_score_spread_extra_min must be <= ratio_score_spread_extra_max")
    feasible_extra_max = min(int(extra_max), len(support) - int(count))
    feasible_extra_min = min(int(extra_min), int(feasible_extra_max))
    extra = rng.randint(int(feasible_extra_min), int(feasible_extra_max))
    window_size = int(count) + int(extra)
    start = rng.randint(0, len(support) - int(window_size))
    window = support[int(start) : int(start) + int(window_size)]
    return [int(value) for value in rng.sample(window, int(count))]


def _share_feasible_totals(
    *,
    score_percent: int,
    series_count: int,
    value_min: int,
    value_max: int,
    total_min: int,
    total_max: int,
) -> List[int]:
    """Return category totals that make an exact target-series share feasible."""

    totals: List[int] = []
    for total in range(int(total_min), int(total_max) + 1):
        if (int(score_percent) * int(total)) % 100 != 0:
            continue
        target_value = (int(score_percent) * int(total)) // 100
        remaining_total = int(total) - int(target_value)
        if not (int(value_min) <= int(target_value) <= int(value_max)):
            continue
        if remaining_total < (int(series_count) - 1) * int(value_min):
            continue
        if remaining_total > (int(series_count) - 1) * int(value_max):
            continue
        totals.append(int(total))
    return totals


def _pair_ratio_feasible_denominators(
    *,
    score_percent: int,
    value_min: int,
    value_max: int,
) -> List[int]:
    """Return denominator values that make an exact numerator/denominator percent feasible."""

    denominators: List[int] = []
    for denominator in range(int(value_min), int(value_max) + 1):
        if (int(score_percent) * int(denominator)) % 100 != 0:
            continue
        numerator = (int(score_percent) * int(denominator)) // 100
        if int(value_min) <= int(numerator) <= int(value_max):
            denominators.append(int(denominator))
    return denominators


def _sample_queried_pair_for_score(
    rng,
    *,
    score: int,
    value_min: int,
    value_max: int,
    direction: str,
) -> Tuple[int, int]:
    """Sample two queried-series values whose derived score is `score`."""

    if int(score) <= 0:
        raise ValueError("derived score must be positive")
    if int(value_min) + int(score) > int(value_max):
        raise ValueError("derived score does not fit within the value bounds")
    low_value = rng.randint(int(value_min), int(value_max) - int(score))
    high_value = int(low_value) + int(score)
    if str(direction) == "increase":
        return int(low_value), int(high_value)
    if str(direction) == "decrease":
        return int(high_value), int(low_value)
    if str(direction) == "absolute":
        if rng.randrange(2) == 0:
            return int(low_value), int(high_value)
        return int(high_value), int(low_value)
    raise ValueError(f"unsupported delta direction: {direction}")


def build_delta_extremum_label_dataset(
    *,
    variant_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    namespace: str,
) -> Tuple[Dict[str, Dict[str, int]], str, List[int], Dict[str, Any]]:
    """Construct one multiseries derived-delta extremum label dataset."""

    supported_variants = {
        "ranked_largest_increase",
        "ranked_largest_decrease",
        "ranked_largest_gap",
        "ranked_smallest_gap",
    }
    if str(variant_key) not in supported_variants:
        raise ValueError(f"unsupported multiseries delta variant_key: {variant_key}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=namespace,
        instance_seed=int(instance_seed),
    )
    category_count_min, category_count_max = resolve_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    series_count_min, series_count_max = resolve_series_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    if int(series_count_min) < 2:
        raise ValueError("multiseries delta task requires at least two series")

    feasible_category_counts = [
        int(count)
        for count in range(int(category_count_min), int(category_count_max) + 1)
        if int(count) <= (int(value_max) - int(value_min))
    ]
    if not feasible_category_counts:
        raise ValueError("category count range exceeds available unique delta support")
    category_count = balanced_choice_from_values(
        feasible_category_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.category_count:{str(variant_key)}",
    )
    feasible_series_counts = [int(count) for count in range(int(series_count_min), int(series_count_max) + 1)]
    series_count = balanced_choice_from_values(
        feasible_series_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.series_count:{str(variant_key)}",
    )

    rank_min = int(params.get("rank_min", group_default(gen_defaults, "rank_min", 1)))
    rank_max = int(params.get("rank_max", group_default(gen_defaults, "rank_max", 3)))
    if int(rank_min) < 1:
        raise ValueError("rank_min must be >= 1")
    if int(rank_min) > int(rank_max):
        raise ValueError("rank_min must be <= rank_max")
    rank_candidates = [
        int(rank)
        for rank in range(int(rank_min), int(rank_max) + 1)
        if 1 <= int(rank) <= int(category_count)
    ]
    if not rank_candidates:
        raise ValueError("delta-extremum task requires a feasible rank")
    answer_rank = balanced_choice_from_values(
        rank_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer_rank:{str(variant_key)}",
    )

    category_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(variant_key)}:{int(category_count)}",
        )
    )
    series_labels = list(sample_series_labels(count=int(series_count), instance_seed=int(instance_seed)))
    query_rng = spawn_rng(int(instance_seed), f"{namespace}.query_pair")
    pair_candidates = [
        (str(series_labels[left_index]), str(series_labels[right_index]))
        for left_index in range(int(series_count))
        for right_index in range(int(left_index) + 1, int(series_count))
    ]
    left_series, right_series = pair_candidates[query_rng.randrange(len(pair_candidates))]

    score_rng = spawn_rng(int(instance_seed), f"{namespace}.derived_scores:{str(variant_key)}")
    max_score = int(value_max) - int(value_min)
    scores = _sample_score_window(
        score_rng,
        count=int(category_count),
        max_score=int(max_score),
        params=params,
        gen_defaults=gen_defaults,
    )
    assignment_rng = spawn_rng(int(instance_seed), f"{namespace}.score_assignment:{str(variant_key)}")
    assignment_rng.shuffle(scores)
    scores_by_category = {
        str(category_label): int(score)
        for category_label, score in zip(category_labels, scores)
    }
    if str(variant_key) == "ranked_smallest_gap":
        sorted_score_items = sorted(
            scores_by_category.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )
        rank_order = "ascending"
    else:
        sorted_score_items = sorted(
            scores_by_category.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        rank_order = "descending"
    answer_label = str(sorted_score_items[int(answer_rank) - 1][0])
    answer_score = int(scores_by_category[str(answer_label)])

    if str(variant_key) == "ranked_largest_increase":
        direction = "increase"
        derived_metric = "increase"
    elif str(variant_key) == "ranked_largest_decrease":
        direction = "decrease"
        derived_metric = "decrease"
    else:
        direction = "absolute"
        derived_metric = "absolute_gap"

    value_rng = spawn_rng(int(instance_seed), f"{namespace}.values:{str(variant_key)}")
    values_by_category: Dict[str, Dict[str, int]] = {}
    derived_values_by_category: Dict[str, int] = {}
    annotation_values: List[int] = []
    for category_label in category_labels:
        score = int(scores_by_category[str(category_label)])
        left_value, right_value = _sample_queried_pair_for_score(
            value_rng,
            score=int(score),
            value_min=int(value_min),
            value_max=int(value_max),
            direction=str(direction),
        )
        category_values: Dict[str, int] = {
            str(left_series): int(left_value),
            str(right_series): int(right_value),
        }
        for series_label in series_labels:
            if str(series_label) in category_values:
                continue
            category_values[str(series_label)] = int(value_rng.randint(int(value_min), int(value_max)))
        values_by_category[str(category_label)] = {
            str(series_label): int(category_values[str(series_label)])
            for series_label in series_labels
        }
        if str(derived_metric) == "increase":
            derived_value = int(right_value) - int(left_value)
        elif str(derived_metric) == "decrease":
            derived_value = int(left_value) - int(right_value)
        else:
            derived_value = abs(int(right_value) - int(left_value))
        derived_values_by_category[str(category_label)] = int(derived_value)
        if str(category_label) == str(answer_label):
            annotation_values = [int(left_value), int(right_value), int(derived_value)]

    trace_extras: Dict[str, Any] = {
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "series_count": int(series_count),
        "series_count_range": [int(series_count_min), int(series_count_max)],
        "category_labels": [str(label) for label in category_labels],
        "series_labels": [str(label) for label in series_labels],
        "queried_series_labels": [str(left_series), str(right_series)],
        "left_series_label": str(left_series),
        "right_series_label": str(right_series),
        "value_min": int(value_min),
        "value_max": int(value_max),
        "value_range": [int(value_min), int(value_max)],
        "answer_label": str(answer_label),
        "answer_rank": int(answer_rank),
        "answer_score": int(answer_score),
        "annotation_values": [int(value) for value in annotation_values],
        "derived_metric": str(derived_metric),
        "direction": str(direction),
        "rank_order": str(rank_order),
        "ranked_category_labels": [str(label) for label, _score in sorted_score_items],
        "derived_values_by_category": {
            str(category_label): int(value)
            for category_label, value in sorted(derived_values_by_category.items())
        },
        "values_by_category": {
            str(category_label): {
                str(series_label): int(value)
                for series_label, value in sorted(series_values.items())
            }
            for category_label, series_values in values_by_category.items()
        },
    }
    return values_by_category, str(answer_label), [int(value) for value in annotation_values], trace_extras


def build_ratio_extremum_label_dataset(
    *,
    variant_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    namespace: str,
) -> Tuple[Dict[str, Dict[str, int]], str, List[int], Dict[str, Any]]:
    """Construct one multiseries ratio/share extremum label dataset."""

    supported_variants = {
        "ranked_largest_series_share",
        "ranked_smallest_series_share",
        "ranked_largest_pair_ratio",
        "ranked_smallest_pair_ratio",
    }
    if str(variant_key) not in supported_variants:
        raise ValueError(f"unsupported multiseries ratio variant_key: {variant_key}")

    is_share_variant = str(variant_key) in {
        "ranked_largest_series_share",
        "ranked_smallest_series_share",
    }
    is_smallest_variant = str(variant_key) in {
        "ranked_smallest_series_share",
        "ranked_smallest_pair_ratio",
    }

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=namespace,
        instance_seed=int(instance_seed),
    )
    category_count_min, category_count_max = resolve_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    series_count_min, series_count_max = resolve_series_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        context_label=namespace,
    )
    if int(series_count_min) < 2:
        raise ValueError("multiseries ratio task requires at least two series")

    feasible_series_counts = [int(count) for count in range(int(series_count_min), int(series_count_max) + 1)]
    series_count = balanced_choice_from_values(
        feasible_series_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.series_count:{str(variant_key)}",
    )

    if bool(is_share_variant):
        score_min = int(params.get("share_percent_min", group_default(gen_defaults, "share_percent_min", 10)))
        score_max = int(params.get("share_percent_max", group_default(gen_defaults, "share_percent_max", 75)))
        value_window_raw = params.get("value_window_enabled", group_default(gen_defaults, "value_window_enabled", False))
        value_window_enabled = bool(value_window_raw)
        if isinstance(value_window_raw, str):
            value_window_enabled = str(value_window_raw).strip().lower() in {"1", "true", "yes", "on"}
        if bool(value_window_enabled) and "category_total_min" not in params and "category_total_max" not in params:
            total_min = int(series_count) * int(value_min)
            total_max = int(series_count) * int(value_max)
        else:
            total_min = int(params.get("category_total_min", group_default(gen_defaults, "category_total_min", 35)))
            total_max = int(params.get("category_total_max", group_default(gen_defaults, "category_total_max", 160)))
        if int(total_min) > int(total_max):
            raise ValueError("category_total_min must be <= category_total_max")
        feasible_scores = [
            int(score)
            for score in range(int(score_min), int(score_max) + 1)
            if _share_feasible_totals(
                score_percent=int(score),
                series_count=int(series_count),
                value_min=int(value_min),
                value_max=int(value_max),
                total_min=int(total_min),
                total_max=int(total_max),
            )
        ]
        score_range = [int(score_min), int(score_max)]
        category_total_range = [int(total_min), int(total_max)]
    else:
        score_min = int(params.get("pair_ratio_percent_min", group_default(gen_defaults, "pair_ratio_percent_min", 40)))
        score_max = int(params.get("pair_ratio_percent_max", group_default(gen_defaults, "pair_ratio_percent_max", 260)))
        feasible_scores = [
            int(score)
            for score in range(int(score_min), int(score_max) + 1)
            if _pair_ratio_feasible_denominators(
                score_percent=int(score),
                value_min=int(value_min),
                value_max=int(value_max),
            )
        ]
        score_range = [int(score_min), int(score_max)]
        category_total_range = None
    if int(score_min) < 1:
        raise ValueError("ratio percent minimum must be positive")
    if int(score_min) > int(score_max):
        raise ValueError("ratio percent minimum must be <= maximum")

    feasible_category_counts = [
        int(count)
        for count in range(int(category_count_min), int(category_count_max) + 1)
        if int(count) <= len(feasible_scores)
    ]
    if not feasible_category_counts:
        raise ValueError("category count range exceeds feasible ratio-percent support")
    category_count = balanced_choice_from_values(
        feasible_category_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.category_count:{str(variant_key)}:{int(series_count)}",
    )

    rank_min = int(params.get("rank_min", group_default(gen_defaults, "rank_min", 1)))
    rank_max = int(params.get("rank_max", group_default(gen_defaults, "rank_max", 3)))
    if int(rank_min) < 1:
        raise ValueError("rank_min must be >= 1")
    if int(rank_min) > int(rank_max):
        raise ValueError("rank_min must be <= rank_max")
    rank_candidates = [
        int(rank)
        for rank in range(int(rank_min), int(rank_max) + 1)
        if 1 <= int(rank) <= int(category_count)
    ]
    if not rank_candidates:
        raise ValueError("ratio-extremum task requires a feasible rank")
    answer_rank = balanced_choice_from_values(
        rank_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer_rank:{str(variant_key)}",
    )

    category_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.labels:{str(variant_key)}:{int(category_count)}",
        )
    )
    series_labels = list(sample_series_labels(count=int(series_count), instance_seed=int(instance_seed)))
    query_rng = spawn_rng(int(instance_seed), f"{namespace}.query_series:{str(variant_key)}")
    if bool(is_share_variant):
        target_series = str(series_labels[query_rng.randrange(len(series_labels))])
        numerator_series = str(target_series)
        denominator_series = "category_total"
        queried_series_labels = [str(target_series)]
    else:
        pair_candidates = [
            (str(series_labels[numerator_index]), str(series_labels[denominator_index]))
            for numerator_index in range(int(series_count))
            for denominator_index in range(int(series_count))
            if int(numerator_index) != int(denominator_index)
        ]
        numerator_series, denominator_series = pair_candidates[query_rng.randrange(len(pair_candidates))]
        target_series = ""
        queried_series_labels = [str(numerator_series), str(denominator_series)]

    score_rng = spawn_rng(int(instance_seed), f"{namespace}.ratio_scores:{str(variant_key)}")
    scores = _sample_ratio_percent_window(
        score_rng,
        count=int(category_count),
        feasible_scores=feasible_scores,
        params=params,
        gen_defaults=gen_defaults,
    )
    assignment_rng = spawn_rng(int(instance_seed), f"{namespace}.ratio_score_assignment:{str(variant_key)}")
    assignment_rng.shuffle(scores)
    scores_by_category = {
        str(category_label): int(score)
        for category_label, score in zip(category_labels, scores)
    }
    if bool(is_smallest_variant):
        sorted_score_items = sorted(scores_by_category.items(), key=lambda item: (int(item[1]), str(item[0])))
        rank_order = "ascending"
    else:
        sorted_score_items = sorted(scores_by_category.items(), key=lambda item: (-int(item[1]), str(item[0])))
        rank_order = "descending"
    answer_label = str(sorted_score_items[int(answer_rank) - 1][0])
    answer_score = int(scores_by_category[str(answer_label)])

    value_rng = spawn_rng(int(instance_seed), f"{namespace}.values:{str(variant_key)}")
    values_by_category: Dict[str, Dict[str, int]] = {}
    denominator_values_by_category: Dict[str, int] = {}
    ratio_percent_by_category: Dict[str, int] = {}
    annotation_values: List[int] = []
    for category_label in category_labels:
        score = int(scores_by_category[str(category_label)])
        if bool(is_share_variant):
            totals = _share_feasible_totals(
                score_percent=int(score),
                series_count=int(series_count),
                value_min=int(value_min),
                value_max=int(value_max),
                total_min=int(category_total_range[0] if category_total_range is not None else 0),
                total_max=int(category_total_range[1] if category_total_range is not None else 0),
            )
            category_total = int(totals[value_rng.randrange(len(totals))])
            numerator_value = (int(score) * int(category_total)) // 100
            remaining_total = int(category_total) - int(numerator_value)
            distractor_series_labels = [
                str(series_label)
                for series_label in series_labels
                if str(series_label) != str(target_series)
            ]
            distractor_values = sample_composition_with_sum(
                value_rng,
                target_sum=int(remaining_total),
                count=len(distractor_series_labels),
                value_min=int(value_min),
                value_max=int(value_max),
            )
            value_rng.shuffle(distractor_values)
            category_values = {
                str(target_series): int(numerator_value),
                **{
                    str(series_label): int(value)
                    for series_label, value in zip(distractor_series_labels, distractor_values)
                },
            }
            denominator_value = int(category_total)
        else:
            denominators = _pair_ratio_feasible_denominators(
                score_percent=int(score),
                value_min=int(value_min),
                value_max=int(value_max),
            )
            denominator_value = int(denominators[value_rng.randrange(len(denominators))])
            numerator_value = (int(score) * int(denominator_value)) // 100
            category_values = {
                str(numerator_series): int(numerator_value),
                str(denominator_series): int(denominator_value),
            }
            for series_label in series_labels:
                if str(series_label) in category_values:
                    continue
                category_values[str(series_label)] = int(value_rng.randint(int(value_min), int(value_max)))

        values_by_category[str(category_label)] = {
            str(series_label): int(category_values[str(series_label)])
            for series_label in series_labels
        }
        denominator_values_by_category[str(category_label)] = int(denominator_value)
        ratio_percent_by_category[str(category_label)] = int(score)
        if str(category_label) == str(answer_label):
            annotation_values = [int(numerator_value), int(denominator_value), int(score)]

    trace_extras: Dict[str, Any] = {
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "series_count": int(series_count),
        "series_count_range": [int(series_count_min), int(series_count_max)],
        "category_labels": [str(label) for label in category_labels],
        "series_labels": [str(label) for label in series_labels],
        "queried_series_labels": [str(label) for label in queried_series_labels],
        "target_series_label": str(target_series),
        "numerator_series_label": str(numerator_series),
        "denominator_series_label": str(denominator_series),
        "value_min": int(value_min),
        "value_max": int(value_max),
        "value_range": [int(value_min), int(value_max)],
        "score_percent_range": list(score_range),
        "category_total_range": list(category_total_range) if category_total_range is not None else None,
        "answer_label": str(answer_label),
        "answer_rank": int(answer_rank),
        "answer_score": int(answer_score),
        "answer_score_percent": int(answer_score),
        "annotation_values": [int(value) for value in annotation_values],
        "derived_metric": "series_share_percent" if bool(is_share_variant) else "pair_ratio_percent",
        "calculation_scope": "category_total_share" if bool(is_share_variant) else "queried_pair_ratio",
        "rank_order": str(rank_order),
        "ranked_category_labels": [str(label) for label, _score in sorted_score_items],
        "denominator_values_by_category": {
            str(category_label): int(value)
            for category_label, value in sorted(denominator_values_by_category.items())
        },
        "ratio_percent_by_category": {
            str(category_label): int(value)
            for category_label, value in sorted(ratio_percent_by_category.items())
        },
        "derived_values_by_category": {
            str(category_label): int(value)
            for category_label, value in sorted(ratio_percent_by_category.items())
        },
        "values_by_category": {
            str(category_label): {
                str(series_label): int(value)
                for series_label, value in sorted(series_values.items())
            }
            for category_label, series_values in values_by_category.items()
        },
    }
    return values_by_category, str(answer_label), [int(value) for value in annotation_values], trace_extras


__all__ = [
    "build_category_total_extremum_label_dataset",
    "build_delta_extremum_label_dataset",
    "build_pair_equality_label_dataset",
    "build_pairwise_comparison_count_dataset",
    "build_ratio_extremum_label_dataset",
    "build_series_rank_at_category_label_dataset",
]
