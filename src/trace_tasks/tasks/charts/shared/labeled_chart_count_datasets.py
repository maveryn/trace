"""Value-count dataset builders for labeled charts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from ....core.seed import spawn_rng
from ...shared.config_defaults import group_default
from .label_assets import sample_chart_labels
from .labeled_chart_defaults import LabeledChartDefaults
from .labeled_chart_values import (
    balanced_choice_from_values,
    resolve_mark_count_bounds,
    resolve_value_bounds,
    shuffle_values,
    sorted_labels,
)
from .labeled_chart_variants import SceneVariant, is_pie_like_scene_variant
from .labeled_chart_composition import sample_composition_with_sum
from .labeled_chart_sampling import (
    _sample_int_values,
    _sample_values_from_pool,
    choose_mark_count,
)

def _find_pie_count_query(
    *,
    count_variant: str,
    target_answer: int,
    mark_count: int,
    instance_seed: int,
    task_id: str,
) -> Tuple[List[int], Dict[str, Any]]:
    """Sample one percentage composition and compatible count query."""

    query_rng = spawn_rng(int(instance_seed), f"{task_id}.pie_query:{str(count_variant)}")
    for _ in range(256):
        values = sample_composition_with_sum(
            query_rng,
            target_sum=100,
            count=int(mark_count),
            value_min=1,
            value_max=100,
        )
        query_rng.shuffle(values)
        if str(count_variant) == "above_threshold":
            candidates = [int(threshold) for threshold in range(0, 100) if sum(1 for value in values if int(value) > int(threshold)) == int(target_answer)]
            if candidates:
                threshold = int(candidates[query_rng.randint(0, len(candidates) - 1)])
                return [int(value) for value in values], {"threshold": int(threshold), "comparison": "greater_than"}
        elif str(count_variant) == "below_threshold":
            candidates = [int(threshold) for threshold in range(1, 101) if sum(1 for value in values if int(value) < int(threshold)) == int(target_answer)]
            if candidates:
                threshold = int(candidates[query_rng.randint(0, len(candidates) - 1)])
                return [int(value) for value in values], {"threshold": int(threshold), "comparison": "less_than"}
        elif str(count_variant) == "in_interval":
            candidates: List[Tuple[int, int]] = []
            for interval_min in range(0, 101):
                for interval_max in range(int(interval_min), 101):
                    count = sum(
                        1
                        for value in values
                        if int(interval_min) <= int(value) <= int(interval_max)
                    )
                    if int(count) == int(target_answer):
                        candidates.append((int(interval_min), int(interval_max)))
            if candidates:
                interval_min, interval_max = candidates[query_rng.randint(0, len(candidates) - 1)]
                return [
                    int(value) for value in values
                ], {
                    "interval_min": int(interval_min),
                    "interval_max": int(interval_max),
                    "interval_inclusive": True,
                }
        else:
            raise ValueError(f"unsupported count_variant: {count_variant}")
    raise RuntimeError(f"unable to construct pie-like count query for {count_variant}")

def build_value_count_dataset_for_variant(
    *,
    count_variant: str,
    scene_variant: SceneVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    task_id: str,
) -> Tuple[List[int], int, List[str], Dict[str, Any]]:
    """Construct one labeled chart dataset for threshold/interval counting tasks."""

    pie_like = bool(is_pie_like_scene_variant(str(scene_variant)))
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
    default_answer_max = min(int(mark_count_max), 10)
    supported_answer_min = int(params.get("target_answer_min", group_default(gen_defaults, "target_answer_min", 0)))
    supported_answer_max = int(params.get("target_answer_max", group_default(gen_defaults, "target_answer_max", default_answer_max)))
    if int(supported_answer_min) > int(supported_answer_max):
        raise ValueError("target_answer_min must be <= target_answer_max")

    answer_candidates = [
        int(value)
        for value in range(int(supported_answer_min), int(supported_answer_max) + 1)
        if int(value) <= int(mark_count_max)
    ]
    target_answer = balanced_choice_from_values(
        answer_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.target_answer:{str(count_variant)}",
    )
    feasible_counts = [
        int(count)
        for count in range(int(mark_count_min), int(mark_count_max) + 1)
        if int(count) >= int(target_answer)
    ]
    mark_count = choose_mark_count(
        feasible_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.mark_count:{str(count_variant)}:{int(target_answer)}",
    )
    labels = list(
        sample_chart_labels(
            count=int(mark_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(count_variant)}:{int(mark_count)}",
        )
    )
    if pie_like:
        values, query_trace = _find_pie_count_query(
            count_variant=str(count_variant),
            target_answer=int(target_answer),
            mark_count=int(mark_count),
            instance_seed=int(instance_seed),
            task_id=task_id,
        )
        if str(count_variant) == "above_threshold":
            threshold = int(query_trace["threshold"])
            annotation_rule = lambda value: int(value) > int(threshold)
        elif str(count_variant) == "below_threshold":
            threshold = int(query_trace["threshold"])
            annotation_rule = lambda value: int(value) < int(threshold)
        else:
            interval_min = int(query_trace["interval_min"])
            interval_max = int(query_trace["interval_max"])
            annotation_rule = lambda value: int(interval_min) <= int(value) <= int(interval_max)
        marks = {str(label): int(value) for label, value in zip(labels, values)}
        annotation_labels = sorted_labels(
            [str(label) for label in labels if bool(annotation_rule(int(marks[str(label)])))]
        )
        if int(len(annotation_labels)) != int(target_answer):
            raise RuntimeError("constructed pie-like counting dataset does not match requested answer")
        trace_extras: Dict[str, Any] = {
            "value_min": 1,
            "value_max": 99,
            "value_semantics": "percentage",
            "composition_total": 100,
            "target_answer_range": [int(supported_answer_min), int(supported_answer_max)],
            "mark_count_range": [int(mark_count_min), int(mark_count_max)],
            "target_answer": int(target_answer),
            "mark_count": int(mark_count),
            "labels": [str(label) for label in labels],
            "values_by_label": {str(label): int(marks[str(label)]) for label in labels},
            "annotation_labels": list(annotation_labels),
            **query_trace,
        }
        return [int(value) for value in values], int(target_answer), annotation_labels, trace_extras

    query_rng = spawn_rng(int(instance_seed), f"{task_id}.query:{str(count_variant)}")

    if str(count_variant) == "above_threshold":
        if int(target_answer) == 0:
            values = _sample_int_values(
                query_rng,
                count=int(mark_count),
                min_value=int(value_min),
                max_value=int(value_max),
            )
            threshold = int(max(values))
        elif int(target_answer) == int(mark_count):
            values = _sample_int_values(
                query_rng,
                count=int(mark_count),
                min_value=int(value_min),
                max_value=int(value_max),
            )
            threshold = int(min(values)) - 1
        else:
            threshold = int(query_rng.randint(int(value_min), int(value_max) - 1))
            low_values = _sample_int_values(
                query_rng,
                count=int(mark_count) - int(target_answer),
                min_value=int(value_min),
                max_value=int(threshold),
            )
            high_values = _sample_int_values(
                query_rng,
                count=int(target_answer),
                min_value=int(threshold) + 1,
                max_value=int(value_max),
            )
            values = shuffle_values(
                [*low_values, *high_values],
                instance_seed=int(instance_seed),
                namespace=f"{task_id}.value_order:{str(count_variant)}",
            )
        annotation_rule = lambda value: int(value) > int(threshold)
        query_trace = {
            "threshold": int(threshold),
            "comparison": "greater_than",
        }
    elif str(count_variant) == "below_threshold":
        if int(target_answer) == 0:
            values = _sample_int_values(
                query_rng,
                count=int(mark_count),
                min_value=int(value_min),
                max_value=int(value_max),
            )
            threshold = int(min(values))
        elif int(target_answer) == int(mark_count):
            values = _sample_int_values(
                query_rng,
                count=int(mark_count),
                min_value=int(value_min),
                max_value=int(value_max),
            )
            threshold = int(max(values)) + 1
        else:
            threshold = int(query_rng.randint(int(value_min) + 1, int(value_max)))
            low_values = _sample_int_values(
                query_rng,
                count=int(target_answer),
                min_value=int(value_min),
                max_value=int(threshold) - 1,
            )
            high_values = _sample_int_values(
                query_rng,
                count=int(mark_count) - int(target_answer),
                min_value=int(threshold),
                max_value=int(value_max),
            )
            values = shuffle_values(
                [*low_values, *high_values],
                instance_seed=int(instance_seed),
                namespace=f"{task_id}.value_order:{str(count_variant)}",
            )
        annotation_rule = lambda value: int(value) < int(threshold)
        query_trace = {
            "threshold": int(threshold),
            "comparison": "less_than",
        }
    elif str(count_variant) == "in_interval":
        if int(target_answer) == 0:
            values = _sample_int_values(
                query_rng,
                count=int(mark_count),
                min_value=int(value_min),
                max_value=int(value_max),
            )
            present_values = {int(value) for value in values}
            missing_values = [
                int(value)
                for value in range(int(value_min), int(value_max) + 1)
                if int(value) not in present_values
            ]
            interval_value = int(
                missing_values[query_rng.randint(0, len(missing_values) - 1)]
                if missing_values
                else int(value_max) + 1
            )
            interval_min = int(interval_value)
            interval_max = int(interval_value)
        elif int(target_answer) == int(mark_count):
            values = _sample_int_values(
                query_rng,
                count=int(mark_count),
                min_value=int(value_min),
                max_value=int(value_max),
            )
            interval_min = int(min(values))
            interval_max = int(max(values))
        else:
            interval_mode = ("left", "right", "both")[int(query_rng.randint(0, 2))]
            if str(interval_mode) == "left":
                interval_min = int(query_rng.randint(int(value_min) + 1, int(value_max)))
                interval_max = int(value_max)
                outside_pool = [int(value) for value in range(int(value_min), int(interval_min))]
            elif str(interval_mode) == "right":
                interval_min = int(value_min)
                interval_max = int(query_rng.randint(int(value_min), int(value_max) - 1))
                outside_pool = [int(value) for value in range(int(interval_max) + 1, int(value_max) + 1)]
            else:
                interval_min = int(query_rng.randint(int(value_min) + 1, int(value_max) - 1))
                interval_max = int(query_rng.randint(int(interval_min), int(value_max) - 1))
                outside_pool = [
                    *[int(value) for value in range(int(value_min), int(interval_min))],
                    *[int(value) for value in range(int(interval_max) + 1, int(value_max) + 1)],
                ]
            inside_values = _sample_int_values(
                query_rng,
                count=int(target_answer),
                min_value=int(interval_min),
                max_value=int(interval_max),
            )
            outside_values = _sample_values_from_pool(
                query_rng,
                count=int(mark_count) - int(target_answer),
                pool=outside_pool,
            )
            values = shuffle_values(
                [*inside_values, *outside_values],
                instance_seed=int(instance_seed),
                namespace=f"{task_id}.value_order:{str(count_variant)}",
            )
        annotation_rule = lambda value: int(interval_min) <= int(value) <= int(interval_max)
        query_trace = {
            "interval_min": int(interval_min),
            "interval_max": int(interval_max),
            "interval_inclusive": True,
        }
    else:
        raise ValueError(f"unsupported count_variant: {count_variant}")

    marks = {str(label): int(value) for label, value in zip(labels, values)}
    annotation_labels = sorted_labels(
        [str(label) for label in labels if bool(annotation_rule(int(marks[str(label)])))]
    )
    if int(len(annotation_labels)) != int(target_answer):
        raise RuntimeError("constructed counting dataset does not match requested answer")

    trace_extras: Dict[str, Any] = {
        "value_min": int(value_min),
        "value_max": int(value_max),
        "target_answer_range": [int(supported_answer_min), int(supported_answer_max)],
        "mark_count_range": [int(mark_count_min), int(mark_count_max)],
        "target_answer": int(target_answer),
        "mark_count": int(mark_count),
        "labels": [str(label) for label in labels],
        "values_by_label": {str(label): int(marks[str(label)]) for label in labels},
        "annotation_labels": list(annotation_labels),
        **query_trace,
    }
    return [int(value) for value in values], int(target_answer), annotation_labels, trace_extras


__all__ = ["build_value_count_dataset_for_variant"]
