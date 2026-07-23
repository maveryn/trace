"""Boxplot dataset builders for distribution-style chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from .....core.seed import spawn_rng
from ..chart_scene_types import BoxPlotSpec
from .config import (
    BoxPlotQueryVariant,
    DistributionChartDefaults,
    _build_boxplot_spec_for_median,
    _resolve_boxplot_category_count_bounds,
    _resolve_boxplot_value_bounds,
)
from ..label_assets import sample_chart_labels
from ..labeled_chart_values import balanced_choice_from_values
from ..labeled_chart_sampling import choose_mark_count

def build_boxplot_median_rank_difference_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
    mark_style: Mapping[str, Any],
) -> Tuple[List[BoxPlotSpec], int, List[str], Dict[str, Any]]:
    """Build a boxplot scene for a ranked-median numeric difference query."""

    category_count_min, category_count_max = _resolve_boxplot_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    value_min, value_max = _resolve_boxplot_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    upper_rank = max(1, int(params.get("median_rank_upper_rank", gen_defaults.get("median_rank_upper_rank", 1))))
    lower_rank_raw = params.get("median_rank_lower_rank", gen_defaults.get("median_rank_lower_rank", 3))
    lower_rank_fixed: int | None
    if str(lower_rank_raw).strip().lower() in {"lowest", "bottom", "last"}:
        lower_rank_fixed = None
    else:
        lower_rank_fixed = max(int(upper_rank) + 1, int(lower_rank_raw))
    category_count_min = max(int(category_count_min), int(lower_rank_fixed or (int(upper_rank) + 1)))
    category_count = choose_mark_count(
        list(range(int(category_count_min), int(category_count_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:category_count:median_rank_difference_value",
    )
    lower_rank = int(lower_rank_fixed or int(category_count))
    median_min = int(value_min) + 2
    median_max = int(value_max) - 2
    candidate_medians = list(range(int(median_min), int(median_max) + 1))
    if len(candidate_medians) < int(category_count):
        raise ValueError("boxplot median support is too small for requested category count")
    answer_min = max(
        1,
        int(params.get("median_rank_difference_min", gen_defaults.get("median_rank_difference_min", 1))),
    )
    answer_max_raw = params.get("median_rank_difference_max", gen_defaults.get("median_rank_difference_max"))
    answer_max = None if answer_max_raw is None else max(int(answer_min), int(answer_max_raw))

    feasible_lows_by_answer: Dict[int, List[int]] = {}
    upper_slots = max(0, int(upper_rank) - 1)
    between_slots = max(0, int(lower_rank) - int(upper_rank) - 1)
    below_slots = max(0, int(category_count) - int(lower_rank))
    answer_upper = int(answer_max if answer_max is not None else int(median_max) - int(median_min))
    for candidate_answer in range(int(answer_min), int(answer_upper) + 1):
        lows: List[int] = []
        for low_value in range(int(median_min), int(median_max) - int(candidate_answer) + 1):
            high_value = int(low_value) + int(candidate_answer)
            if int(median_max) - int(high_value) < int(upper_slots):
                continue
            if int(high_value) - int(low_value) - 1 < int(between_slots):
                continue
            if int(low_value) - int(median_min) < int(below_slots):
                continue
            lows.append(int(low_value))
        if lows:
            feasible_lows_by_answer[int(candidate_answer)] = lows
    if not feasible_lows_by_answer:
        raise ValueError("unable to construct ranked-median boxplot difference within requested answer bounds")
    answer = int(
        balanced_choice_from_values(
            sorted(feasible_lows_by_answer),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:median_rank_difference_answer:{int(lower_rank)}",
        )
    )
    low_median = int(
        balanced_choice_from_values(
            feasible_lows_by_answer[int(answer)],
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}:median_rank_difference_lower_median:{int(lower_rank)}:{int(answer)}",
        )
    )
    high_median = int(low_median) + int(answer)
    rng = spawn_rng(int(instance_seed), f"{task_id}.boxplot.median_rank_difference_value")
    above_values = sorted(
        [int(value) for value in rng.sample(list(range(int(high_median) + 1, int(median_max) + 1)), int(upper_slots))],
        reverse=True,
    )
    between_values = sorted(
        [int(value) for value in rng.sample(list(range(int(low_median) + 1, int(high_median))), int(between_slots))],
        reverse=True,
    )
    below_values = sorted(
        [int(value) for value in rng.sample(list(range(int(median_min), int(low_median))), int(below_slots))],
        reverse=True,
    )
    ranked_medians = [*above_values, int(high_median), *between_values, int(low_median), *below_values]
    if len(ranked_medians) != int(category_count):
        raise RuntimeError("constructed ranked median list has the wrong length")
    medians = [int(value) for value in rng.sample(ranked_medians, int(category_count))]

    labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:median_rank_difference_value:{int(category_count)}",
            max_chars=3,
        )
    )
    fill_rgb = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])
    specs = [
        _build_boxplot_spec_for_median(
            label=str(label),
            median=int(median),
            value_min=int(value_min),
            value_max=int(value_max),
            rng=rng,
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
        )
        for label, median in zip(labels, medians)
    ]
    label_by_median = {int(spec.median): str(spec.label) for spec in specs}
    upper_median = int(ranked_medians[int(upper_rank) - 1])
    lower_median = int(ranked_medians[int(lower_rank) - 1])
    upper_label = str(label_by_median[int(upper_median)])
    lower_label = str(label_by_median[int(lower_median)])
    trace_extras = {
        "scene_variant": "boxplot",
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "value_range": [int(value_min), int(value_max)],
        "labels": [str(spec.label) for spec in specs],
        "answer_value": int(answer),
        "upper_rank": int(upper_rank),
        "lower_rank": int(lower_rank),
        "rank_pair": (
            "top_bottom"
            if int(lower_rank) == int(category_count)
            else f"top_{int(lower_rank)}"
        ),
        "upper_rank_label": str(upper_label),
        "lower_rank_label": str(lower_label),
        "upper_rank_median": int(upper_median),
        "lower_rank_median": int(lower_median),
        "annotation_labels": [str(upper_label), str(lower_label)],
        "quartiles_by_label": {
            str(spec.label): {
                "whisker_min": int(spec.whisker_min),
                "q1": int(spec.q1),
                "median": int(spec.median),
                "q3": int(spec.q3),
                "whisker_max": int(spec.whisker_max),
                "iqr": int(spec.q3) - int(spec.q1),
            }
            for spec in specs
        },
    }
    return specs, int(answer), [str(upper_label), str(lower_label)], trace_extras


def build_boxplot_paired_median_shift_dataset(
    *,
    query_id: str,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
    mark_style: Mapping[str, Any],
) -> Tuple[List[BoxPlotSpec], List[BoxPlotSpec], str, List[str], Dict[str, Any]]:
    """Build matched before/after boxplot panels and a median-shift label query."""

    category_count_min, category_count_max = _resolve_boxplot_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    value_min, value_max = _resolve_boxplot_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    category_count = choose_mark_count(
        list(range(int(category_count_min), int(category_count_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:category_count:{str(query_id)}",
    )
    median_min = int(value_min) + 2
    median_max = int(value_max) - 2
    max_shift_support = max(1, (int(median_max) - int(median_min)) // 2)
    shift_min = max(1, int(params.get("paired_median_shift_min", gen_defaults.get("paired_median_shift_min", 2))))
    shift_max = min(
        int(max_shift_support),
        max(int(shift_min), int(params.get("paired_median_shift_max", gen_defaults.get("paired_median_shift_max", 12)))),
    )
    shift_support = list(range(int(shift_min), int(shift_max) + 1))
    if len(shift_support) < int(category_count):
        raise ValueError("paired boxplot shift support is too small for requested category count")

    rng = spawn_rng(int(instance_seed), f"{task_id}.boxplot.{str(query_id)}")
    base_labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(query_id)}:{int(category_count)}",
            max_chars=3,
        )
    )
    shift_magnitudes = [int(value) for value in rng.sample(shift_support, int(category_count))]
    rng.shuffle(shift_magnitudes)
    if str(query_id) == "paired_median_greatest_increase_label":
        signed_shifts = [int(value) for value in shift_magnitudes]
        answer_index = max(range(int(category_count)), key=lambda idx: int(signed_shifts[idx]))
        shift_direction = "increase"
    elif str(query_id) == "paired_median_greatest_decrease_label":
        signed_shifts = [-int(value) for value in shift_magnitudes]
        answer_index = max(range(int(category_count)), key=lambda idx: -int(signed_shifts[idx]))
        shift_direction = "decrease"
    elif str(query_id) == "paired_median_greatest_absolute_change_label":
        signed_shifts = [
            int(value) if int(rng.randint(0, 1)) == 1 else -int(value)
            for value in shift_magnitudes
        ]
        answer_index = max(range(int(category_count)), key=lambda idx: abs(int(signed_shifts[idx])))
        shift_direction = "absolute_change"
    else:
        raise ValueError(f"unsupported paired boxplot query id: {query_id}")

    before_fill = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])
    after_fill = tuple(
        int(max(20, min(235, round(0.55 * float(channel) + 70.0))))
        for channel in reversed(before_fill)
    )

    before_specs: List[BoxPlotSpec] = []
    after_specs: List[BoxPlotSpec] = []
    pairs: Dict[str, Dict[str, Any]] = {}
    for base_label, shift in zip(base_labels, signed_shifts):
        if int(shift) >= 0:
            before_min = int(median_min)
            before_max = int(median_max) - int(shift)
        else:
            before_min = int(median_min) - int(shift)
            before_max = int(median_max)
        if int(before_min) > int(before_max):
            raise ValueError("no feasible paired median support for requested shift")
        before_median = int(rng.randint(int(before_min), int(before_max)))
        after_median = int(before_median) + int(shift)
        before_label = f"{str(base_label)}__before"
        after_label = f"{str(base_label)}__after"
        before_specs.append(
            _build_boxplot_spec_for_median(
                label=str(base_label),
                median=int(before_median),
                value_min=int(value_min),
                value_max=int(value_max),
                rng=rng,
                fill_rgb=before_fill,
                outline_rgb=outline_rgb,
            )
        )
        after_specs.append(
            _build_boxplot_spec_for_median(
                label=str(base_label),
                median=int(after_median),
                value_min=int(value_min),
                value_max=int(value_max),
                rng=rng,
                fill_rgb=after_fill,
                outline_rgb=outline_rgb,
            )
        )
        pairs[str(base_label)] = {
            "before_label": str(before_label),
            "after_label": str(after_label),
            "display_label": str(base_label),
            "before_panel": "before",
            "after_panel": "after",
            "before_median": int(before_median),
            "after_median": int(after_median),
            "signed_shift": int(shift),
            "absolute_shift": abs(int(shift)),
        }

    answer_label = str(base_labels[int(answer_index)])
    annotation_labels = [
        str(pairs[str(answer_label)]["before_label"]),
        str(pairs[str(answer_label)]["after_label"]),
    ]
    trace_extras = {
        "scene_variant": "boxplot",
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "rendered_boxplot_count": int(len(before_specs) + len(after_specs)),
        "rendered_boxplot_count_range": [int(category_count_min) * 2, int(category_count_max) * 2],
        "value_range": [int(value_min), int(value_max)],
        "base_labels": [str(label) for label in base_labels],
        "labels": [str(label) for label in base_labels],
        "rendered_labels": [
            *(f"{str(label)}__before" for label in base_labels),
            *(f"{str(label)}__after" for label in base_labels),
        ],
        "answer_label": str(answer_label),
        "answer_shift": int(pairs[str(answer_label)]["signed_shift"]),
        "answer_absolute_shift": int(pairs[str(answer_label)]["absolute_shift"]),
        "shift_direction": str(shift_direction),
        "annotation_labels": list(annotation_labels),
        "paired_panels": {"before": "Before", "after": "After"},
        "pairs_by_base_label": dict(pairs),
        "quartiles_by_label": {
            **{
                f"{str(spec.label)}__before": {
                    "display_label": str(spec.label),
                    "panel": "before",
                    "whisker_min": int(spec.whisker_min),
                    "q1": int(spec.q1),
                    "median": int(spec.median),
                    "q3": int(spec.q3),
                    "whisker_max": int(spec.whisker_max),
                    "iqr": int(spec.q3) - int(spec.q1),
                }
                for spec in before_specs
            },
            **{
                f"{str(spec.label)}__after": {
                    "display_label": str(spec.label),
                    "panel": "after",
                    "whisker_min": int(spec.whisker_min),
                    "q1": int(spec.q1),
                    "median": int(spec.median),
                    "q3": int(spec.q3),
                    "whisker_max": int(spec.whisker_max),
                    "iqr": int(spec.q3) - int(spec.q1),
                }
                for spec in after_specs
            }
        },
        "quartiles_by_base_label": {
            str(base_label): {
                "before": {
                    "whisker_min": int(before_spec.whisker_min),
                    "q1": int(before_spec.q1),
                    "median": int(before_spec.median),
                    "q3": int(before_spec.q3),
                    "whisker_max": int(before_spec.whisker_max),
                    "iqr": int(before_spec.q3) - int(before_spec.q1),
                },
                "after": {
                    "whisker_min": int(after_spec.whisker_min),
                    "q1": int(after_spec.q1),
                    "median": int(after_spec.median),
                    "q3": int(after_spec.q3),
                    "whisker_max": int(after_spec.whisker_max),
                    "iqr": int(after_spec.q3) - int(after_spec.q1),
                },
            }
            for base_label, before_spec, after_spec in zip(base_labels, before_specs, after_specs)
        },
        "before_medians_by_label": {
            str(spec.label): int(spec.median)
            for spec in before_specs
        },
        "after_medians_by_label": {
            str(spec.label): int(spec.median)
            for spec in after_specs
        },
    }
    return before_specs, after_specs, str(answer_label), list(annotation_labels), trace_extras


def build_boxplot_dataset_for_variant(
    *,
    query_id: BoxPlotQueryVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
    mark_style: Mapping[str, Any],
) -> Tuple[List[BoxPlotSpec], str, int, Dict[str, Any]]:
    """Build one categorical boxplot scene and label-answer query."""

    category_count_min, category_count_max = _resolve_boxplot_category_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    value_min, value_max = _resolve_boxplot_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    category_count = choose_mark_count(
        list(range(int(category_count_min), int(category_count_max) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}:category_count:{str(query_id)}",
    )
    labels = list(
        sample_chart_labels(
            count=int(category_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(query_id)}:{int(category_count)}",
            max_chars=3,
        )
    )
    rng = spawn_rng(int(instance_seed), f"{task_id}.boxplot.{str(query_id)}")
    fill_rgb = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])

    def _sample_clustered_unique(pool_min: int, pool_max: int, count: int) -> List[int]:
        if int(count) <= 0:
            return []
        window_size = max(int(count) + 2, 4)
        lower = max(int(pool_min), int(pool_max) - int(window_size) + 1)
        pool = list(range(int(lower), int(pool_max) + 1))
        if len(pool) < int(count):
            pool = list(range(int(pool_min), int(pool_max) + 1))
        if len(pool) < int(count):
            raise ValueError("insufficient clustered support for unique sampling")
        rng.shuffle(pool)
        return [int(value) for value in pool[: int(count)]]

    def _sample_clustered_unique_low(pool_min: int, pool_max: int, count: int) -> List[int]:
        if int(count) <= 0:
            return []
        window_size = max(int(count) + 2, 4)
        upper = min(int(pool_max), int(pool_min) + int(window_size) - 1)
        pool = list(range(int(pool_min), int(upper) + 1))
        if len(pool) < int(count):
            pool = list(range(int(pool_min), int(pool_max) + 1))
        if len(pool) < int(count):
            raise ValueError("insufficient clustered support for unique low-end sampling")
        rng.shuffle(pool)
        return [int(value) for value in pool[: int(count)]]

    def _sample_unique_top_gap_margins(*, support_max: int, count: int, gap_min: int, gap_max: int) -> List[int]:
        if int(count) < 2:
            raise ValueError("reference-median boxplot tasks require at least two queried-side candidates")
        if int(support_max) < int(count):
            raise ValueError("reference-median margin support is too small for requested category count")
        feasible_pairs: List[Tuple[int, int]] = []
        for winner in range(1, int(support_max) + 1):
            for gap in range(int(gap_min), int(gap_max) + 1):
                runner_up = int(winner) - int(gap)
                if int(runner_up) < 1:
                    continue
                if int(runner_up) - 1 < int(count) - 2:
                    continue
                feasible_pairs.append((int(winner), int(runner_up)))
        if not feasible_pairs:
            raise ValueError("unable to construct unique reference-median winner margins")
        winner_margin, runner_up_margin = feasible_pairs[int(rng.randint(0, len(feasible_pairs) - 1))]
        other_pool = list(range(1, int(runner_up_margin)))
        other_margins = rng.sample(other_pool, int(count) - 2) if int(count) > 2 else []
        return sorted([int(winner_margin), int(runner_up_margin), *[int(value) for value in other_margins]])

    def _build_box_for_median(label: str, median: int) -> BoxPlotSpec:
        return _build_boxplot_spec_for_median(
            label=str(label),
            median=int(median),
            value_min=int(value_min),
            value_max=int(value_max),
            rng=rng,
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
        )

    def _resolve_reference_side_count(
        *,
        direction: str,
        candidate_count: int,
        median_min: int,
        median_max: int,
    ) -> int:
        min_key = "median_reference_above_count_min" if str(direction) == "above" else "median_reference_below_count_min"
        max_key = "median_reference_above_count_max" if str(direction) == "above" else "median_reference_below_count_max"
        explicit_min = params.get(min_key)
        explicit_max = params.get(max_key)
        if explicit_min is None and explicit_max is None:
            count_min = max(2, int(candidate_count) // 2)
            count_max = min(int(candidate_count), int(count_min) + 1)
        else:
            count_min = max(2, int(2 if explicit_min is None else explicit_min))
            count_max = min(
                int(candidate_count),
                max(int(count_min), int(candidate_count if explicit_max is None else explicit_max)),
            )
        feasible_counts: List[int] = []
        for active_count in range(int(count_min), int(count_max) + 1):
            passive_count = int(candidate_count) - int(active_count)
            if int(passive_count) < 0:
                continue
            if str(direction) == "above":
                threshold_min = max(int(value_min) + 3, int(median_min) + int(passive_count))
                threshold_max = int(median_max) - int(active_count)
            else:
                threshold_min = int(median_min) + int(active_count)
                threshold_max = int(median_max) - int(passive_count)
            if int(threshold_min) <= int(threshold_max):
                feasible_counts.append(int(active_count))
        if not feasible_counts:
            raise ValueError("no feasible reference-median side counts for requested category count")
        return int(rng.choice(feasible_counts))

    def _build_reference_median_specs(*, direction: str) -> Tuple[List[BoxPlotSpec], str, int, Dict[str, Any]]:
        if int(category_count) < 4:
            raise ValueError("boxplot reference-median task requires at least four categories")
        gap_min = max(1, int(params.get("median_reference_winner_gap_min", 1)))
        gap_max = max(int(gap_min), int(params.get("median_reference_winner_gap_max", gap_min)))
        median_min = int(value_min) + 2
        median_max = int(value_max) - 2
        candidate_count = int(category_count) - 1
        active_count = _resolve_reference_side_count(
            direction=str(direction),
            candidate_count=int(candidate_count),
            median_min=int(median_min),
            median_max=int(median_max),
        )
        passive_count = int(candidate_count) - int(active_count)
        if str(direction) == "above":
            threshold_min = max(int(value_min) + 3, int(median_min) + int(passive_count))
            threshold_max = int(median_max) - int(active_count)
        elif str(direction) == "below":
            threshold_min = int(median_min) + int(active_count)
            threshold_max = int(median_max) - int(passive_count)
        else:
            raise ValueError(f"unsupported reference-median direction: {direction}")
        if int(threshold_min) > int(threshold_max):
            raise ValueError("boxplot reference-median support is too small for requested category count")
        reference_threshold = int(rng.randint(int(threshold_min), int(threshold_max)))
        if str(direction) == "above":
            active_support = int(median_max) - int(reference_threshold)
            passive_medians = _sample_clustered_unique_low(
                int(median_min),
                int(reference_threshold) - 1,
                int(passive_count),
            )
        else:
            active_support = int(reference_threshold) - int(median_min)
            passive_medians = _sample_clustered_unique(
                int(reference_threshold) + 1,
                int(median_max),
                int(passive_count),
            )
        active_margins = _sample_unique_top_gap_margins(
            support_max=int(active_support),
            count=int(active_count),
            gap_min=int(gap_min),
            gap_max=int(gap_max),
        )

        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)
        reference_label = str(shuffled_labels[0])
        candidate_labels = [str(label) for label in shuffled_labels[1:]]
        active_labels = list(candidate_labels[: int(active_count)])
        passive_labels = list(candidate_labels[int(active_count) :])
        label_to_margin = {
            str(label): int(margin)
            for label, margin in zip(active_labels, active_margins)
        }
        winner_margin = int(max(active_margins))
        winner_label = next(
            str(label)
            for label, margin in label_to_margin.items()
            if int(margin) == int(winner_margin)
        )

        label_to_box: Dict[str, BoxPlotSpec] = {}
        if str(direction) == "above":
            reference_q3 = int(reference_threshold)
            reference_q1 = int(rng.randint(int(value_min) + 1, int(reference_q3) - 2))
            reference_median = int(rng.randint(int(reference_q1) + 1, int(reference_q3) - 1))
            reference_whisker_min = max(
                int(value_min),
                int(reference_q1) - int(rng.randint(0, min(2, int(reference_q1) - int(value_min)))),
            )
            reference_whisker_max = min(
                int(value_max),
                int(reference_q3) + int(rng.randint(0, min(2, int(value_max) - int(reference_q3)))),
            )
            label_to_box[str(reference_label)] = BoxPlotSpec(
                label=str(reference_label),
                whisker_min=int(reference_whisker_min),
                q1=int(reference_q1),
                median=int(reference_median),
                q3=int(reference_q3),
                whisker_max=int(reference_whisker_max),
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
            )
            for label, margin in label_to_margin.items():
                label_to_box[str(label)] = _build_box_for_median(str(label), int(reference_threshold) + int(margin))
            reference_meta = {"reference_q3": int(reference_q3)}
        else:
            reference_q1 = int(reference_threshold)
            reference_median = int(rng.randint(int(reference_q1) + 1, int(value_max) - 1))
            reference_q3 = int(rng.randint(int(reference_median) + 1, int(value_max)))
            reference_whisker_min = max(
                int(value_min),
                int(reference_q1) - int(rng.randint(0, min(2, int(reference_q1) - int(value_min)))),
            )
            reference_whisker_max = min(
                int(value_max),
                int(reference_q3) + int(rng.randint(0, min(2, int(value_max) - int(reference_q3)))),
            )
            label_to_box[str(reference_label)] = BoxPlotSpec(
                label=str(reference_label),
                whisker_min=int(reference_whisker_min),
                q1=int(reference_q1),
                median=int(reference_median),
                q3=int(reference_q3),
                whisker_max=int(reference_whisker_max),
                fill_rgb=fill_rgb,
                outline_rgb=outline_rgb,
            )
            for label, margin in label_to_margin.items():
                label_to_box[str(label)] = _build_box_for_median(str(label), int(reference_threshold) - int(margin))
            reference_meta = {"reference_q1": int(reference_q1)}

        for label, median in zip(passive_labels, passive_medians):
            label_to_box[str(label)] = _build_box_for_median(str(label), int(median))
        return [label_to_box[str(label)] for label in labels], str(winner_label), int(winner_margin), {
            "reference_label": str(reference_label),
            "winner_margin": int(winner_margin),
            "reference_side_count": int(active_count),
            "reference_other_side_count": int(passive_count),
            **reference_meta,
        }

    specs: List[BoxPlotSpec] = []
    generation_meta: Dict[str, Any] = {}
    if str(query_id) == "median_above_reference_q3":
        specs, answer_label, annotation_value, generation_meta = _build_reference_median_specs(direction="above")
    elif str(query_id) == "median_below_reference_q1":
        specs, answer_label, annotation_value, generation_meta = _build_reference_median_specs(direction="below")
    elif str(query_id) in {"largest_iqr", "smallest_iqr"}:
        support_min = 2
        support_max = int(value_max) - int(value_min) - 3
        feasible_iqrs = list(range(int(support_min), int(support_max) + 1))
        if len(feasible_iqrs) < int(category_count):
            raise ValueError("boxplot IQR support is too small for requested category count")
        gap_min = int(params.get("iqr_winner_gap_min", 0))
        gap_max = int(params.get("iqr_winner_gap_max", gap_min))
        if int(gap_max) > 0:
            gap_min = max(1, int(gap_min))
            gap_max = max(int(gap_min), int(gap_max))
            if str(query_id) == "largest_iqr":
                answer_min = int(support_min) + int(gap_min) + int(category_count) - 2
                if int(answer_min) > int(support_max):
                    raise ValueError("boxplot largest-IQR gap support is too small for requested category count")
                winner_iqr = int(rng.randint(int(answer_min), int(support_max)))
                gap_cap = min(int(gap_max), int(winner_iqr) - int(support_min) - int(category_count) + 2)
                winner_gap = int(rng.randint(int(gap_min), int(gap_cap)))
                runner_up = int(winner_iqr) - int(winner_gap)
                other_iqrs = _sample_clustered_unique(
                    int(support_min),
                    int(runner_up) - 1,
                    int(category_count) - 2,
                )
                iqrs = [int(winner_iqr), int(runner_up), *other_iqrs]
            else:
                answer_max = int(support_max) - int(gap_min) - int(category_count) + 2
                if int(answer_max) < int(support_min):
                    raise ValueError("boxplot smallest-IQR gap support is too small for requested category count")
                winner_iqr = int(rng.randint(int(support_min), int(answer_max)))
                gap_cap = min(int(gap_max), int(support_max) - int(winner_iqr) - int(category_count) + 2)
                winner_gap = int(rng.randint(int(gap_min), int(gap_cap)))
                runner_up = int(winner_iqr) + int(winner_gap)
                other_iqrs = _sample_clustered_unique_low(
                    int(runner_up) + 1,
                    int(support_max),
                    int(category_count) - 2,
                )
                iqrs = [int(winner_iqr), int(runner_up), *other_iqrs]
            rng.shuffle(iqrs)
        else:
            rng.shuffle(feasible_iqrs)
            iqrs = feasible_iqrs[: int(category_count)]
            rng.shuffle(iqrs)
        for label, iqr in zip(labels, iqrs):
            q1_min = int(value_min) + 1
            q1_max = int(value_max) - int(iqr) - 2
            if int(q1_min) > int(q1_max):
                raise ValueError("no feasible q1 support for boxplot IQR construction")
            q1 = int(rng.randint(int(q1_min), int(q1_max)))
            q3 = int(q1) + int(iqr)
            median = int(rng.randint(int(q1) + 1, int(q3) - 1))
            whisker_min = max(int(value_min), int(q1) - int(rng.randint(0, min(2, int(q1) - int(value_min)))))
            whisker_max = min(int(value_max), int(q3) + int(rng.randint(0, min(2, int(value_max) - int(q3)))))
            specs.append(
                BoxPlotSpec(
                    label=str(label),
                    whisker_min=int(whisker_min),
                    q1=int(q1),
                    median=int(median),
                    q3=int(q3),
                    whisker_max=int(whisker_max),
                    fill_rgb=fill_rgb,
                    outline_rgb=outline_rgb,
                )
            )
        if str(query_id) == "largest_iqr":
            answer_label = max(specs, key=lambda spec: int(spec.q3) - int(spec.q1)).label
            annotation_value = max(int(spec.q3) - int(spec.q1) for spec in specs)
        else:
            answer_label = min(specs, key=lambda spec: int(spec.q3) - int(spec.q1)).label
            annotation_value = min(int(spec.q3) - int(spec.q1) for spec in specs)
    else:
        raise ValueError(f"unsupported boxplot query_id: {query_id}")

    trace_extras = {
        "scene_variant": "boxplot",
        "category_count": int(category_count),
        "category_count_range": [int(category_count_min), int(category_count_max)],
        "value_range": [int(value_min), int(value_max)],
        "labels": [str(spec.label) for spec in specs],
        "answer_label": str(answer_label),
        "annotation_value": int(annotation_value),
        "quartiles_by_label": {
            str(spec.label): {
                "whisker_min": int(spec.whisker_min),
                "q1": int(spec.q1),
                "median": int(spec.median),
                "q3": int(spec.q3),
                "whisker_max": int(spec.whisker_max),
                "iqr": int(spec.q3) - int(spec.q1),
            }
            for spec in specs
        },
        **generation_meta,
    }
    return specs, str(answer_label), int(annotation_value), trace_extras


__all__ = [
    "build_boxplot_dataset_for_variant",
    "build_boxplot_median_rank_difference_dataset",
    "build_boxplot_paired_median_shift_dataset",
]
