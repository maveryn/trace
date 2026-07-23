"""Public task for `task_charts__boxplot__median_rank_difference_value`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.boxplot._lifecycle import (
    SingleBoxplotTaskPlan,
    boxplot_attempt_seed,
    materialize_single_boxplot_plan,
)
from trace_tasks.tasks.charts.boxplot.shared.defaults import DOMAIN, SCENE_ID, SCENE_NAMESPACE, GENERATION_DEFAULTS, merge_task_defaults
from trace_tasks.tasks.charts.boxplot.shared.prompts import SINGLE_SCENE_PROMPT_KEY, build_prompt_artifacts
from trace_tasks.tasks.charts.boxplot.shared.rendering import resolve_mark_style
from trace_tasks.tasks.charts.boxplot.shared.sampling import (
    balanced_int_choice,
    build_boxplot_for_median,
    choose_category_count,
    quartiles_by_label,
    resolve_value_bounds,
    sample_labels,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions


TASK_ID = "task_charts__boxplot__median_rank_difference_value"
OBJECTIVE_CONTRACT = "median_rank_difference_value"
TOP_SECOND_QUERY_ID = "median_top_second_difference_value"
TOP_THIRD_QUERY_ID = "median_top_third_difference_value"
TOP_BOTTOM_QUERY_ID = "median_top_bottom_difference_value"
TASK_PARAM_DEFAULTS = {
    "category_count_min": 6,
    "category_count_max": 12,
    "median_rank_difference_min": 2,
    "median_rank_difference_max": 18,
}


def _rank_parameters(selected_query_id: str) -> tuple[int, int | None, tuple[str, str]]:
    if str(selected_query_id) == TOP_SECOND_QUERY_ID:
        return 1, 2, ("highest_median_boxplot", "second_highest_median_boxplot")
    if str(selected_query_id) == TOP_THIRD_QUERY_ID:
        return 1, 3, ("highest_median_boxplot", "third_highest_median_boxplot")
    if str(selected_query_id) == TOP_BOTTOM_QUERY_ID:
        return 1, None, ("highest_median_boxplot", "lowest_median_boxplot")
    raise ValueError(f"unsupported median-rank query id: {selected_query_id}")


def _ranked_medians(
    *,
    upper_rank: int,
    lower_rank: int,
    category_count: int,
    median_min: int,
    median_max: int,
    answer: int,
    low_median: int,
    rng: Any,
) -> list[int]:
    """Construct median values so the requested ranked pair has the answer gap."""

    high_median = int(low_median) + int(answer)
    upper_slots = max(0, int(upper_rank) - 1)
    between_slots = max(0, int(lower_rank) - int(upper_rank) - 1)
    below_slots = max(0, int(category_count) - int(lower_rank))
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
    ranked = [*above_values, int(high_median), *between_values, int(low_median), *below_values]
    if len(ranked) != int(category_count):
        raise RuntimeError("constructed ranked median list has the wrong length")
    return [int(value) for value in ranked]


def _feasible_lows_by_answer(
    *,
    upper_rank: int,
    lower_rank: int,
    category_count: int,
    median_min: int,
    median_max: int,
    answer_min: int,
    answer_max: int,
) -> dict[int, list[int]]:
    """Return feasible lower medians for each requested rank-gap answer."""

    upper_slots = max(0, int(upper_rank) - 1)
    between_slots = max(0, int(lower_rank) - int(upper_rank) - 1)
    below_slots = max(0, int(category_count) - int(lower_rank))
    feasible: dict[int, list[int]] = {}
    for candidate_answer in range(int(answer_min), int(answer_max) + 1):
        lows: list[int] = []
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
            feasible[int(candidate_answer)] = lows
    return feasible


def _build_rank_difference_plan(
    params: dict[str, Any],
    instance_seed: int,
    selected_query_id: str,
) -> SingleBoxplotTaskPlan:
    """Bind ranked median-pair selection and numeric-difference answer."""

    upper_rank, requested_lower_rank, role_keys = _rank_parameters(str(selected_query_id))
    effective_params = merge_task_defaults(params, TASK_PARAM_DEFAULTS)
    mark_style = resolve_mark_style(effective_params, instance_seed=int(instance_seed), mark_count=1)
    minimum_count = int(requested_lower_rank or (int(upper_rank) + 1))
    category_count, category_range = choose_category_count(
        params=effective_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.median_rank.category_count.{selected_query_id}",
        minimum=int(minimum_count),
    )
    lower_rank = int(requested_lower_rank or category_count)
    value_min, value_max = resolve_value_bounds(effective_params, instance_seed=int(instance_seed))
    median_min = int(value_min) + 2
    median_max = int(value_max) - 2
    if len(range(int(median_min), int(median_max) + 1)) < int(category_count):
        raise ValueError("boxplot median support is too small for requested category count")
    answer_min = max(1, int(effective_params.get("median_rank_difference_min", GENERATION_DEFAULTS.get("median_rank_difference_min", 1))))
    raw_answer_max = effective_params.get("median_rank_difference_max", GENERATION_DEFAULTS.get("median_rank_difference_max"))
    answer_max = int(raw_answer_max if raw_answer_max is not None else int(median_max) - int(median_min))
    answer_max = max(int(answer_min), int(answer_max))
    feasible_lows = _feasible_lows_by_answer(
        upper_rank=int(upper_rank),
        lower_rank=int(lower_rank),
        category_count=int(category_count),
        median_min=int(median_min),
        median_max=int(median_max),
        answer_min=int(answer_min),
        answer_max=int(answer_max),
    )
    if not feasible_lows:
        raise ValueError("unable to construct ranked-median boxplot difference within requested answer bounds")
    answer_value = balanced_int_choice(
        sorted(feasible_lows),
        params=effective_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.median_rank.answer.{int(lower_rank)}",
    )
    low_median = balanced_int_choice(
        feasible_lows[int(answer_value)],
        params=effective_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.median_rank.lower_median.{int(lower_rank)}.{int(answer_value)}",
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.median_rank.values.{int(lower_rank)}")
    ranked_medians = _ranked_medians(
        upper_rank=int(upper_rank),
        lower_rank=int(lower_rank),
        category_count=int(category_count),
        median_min=int(median_min),
        median_max=int(median_max),
        answer=int(answer_value),
        low_median=int(low_median),
        rng=rng,
    )
    medians = [int(value) for value in rng.sample(ranked_medians, int(category_count))]
    labels = sample_labels(
        count=int(category_count),
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.median_rank.labels.{int(category_count)}",
    )
    fill_rgb = tuple(int(channel) for channel in mark_style["mark_fill_rgb"])
    outline_rgb = tuple(int(channel) for channel in mark_style["mark_outline_rgb"])
    boxplots = tuple(
        build_boxplot_for_median(
            label=str(label),
            median=int(median),
            value_min=int(value_min),
            value_max=int(value_max),
            rng=rng,
            fill_rgb=fill_rgb,
            outline_rgb=outline_rgb,
        )
        for label, median in zip(labels, medians)
    )
    label_by_median = {int(spec.median): str(spec.label) for spec in boxplots}
    upper_median = int(ranked_medians[int(upper_rank) - 1])
    lower_median = int(ranked_medians[int(lower_rank) - 1])
    upper_label = str(label_by_median[int(upper_median)])
    lower_label = str(label_by_median[int(lower_median)])
    prompt_artifacts = build_prompt_artifacts(
        scene_key=SINGLE_SCENE_PROMPT_KEY,
        prompt_query_key=str(selected_query_id),
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    rank_pair = "top_bottom" if int(lower_rank) == int(category_count) else f"top_{int(lower_rank)}"
    relations = {
        "query_id": str(selected_query_id),
        "scene_variant": "boxplot",
        "category_count": int(category_count),
        "category_count_range": [int(category_range[0]), int(category_range[1])],
        "value_range": [int(value_min), int(value_max)],
        "labels": [str(spec.label) for spec in boxplots],
        "answer_value": int(answer_value),
        "upper_rank": int(upper_rank),
        "lower_rank": int(lower_rank),
        "rank_pair": str(rank_pair),
        "upper_rank_label": str(upper_label),
        "lower_rank_label": str(lower_label),
        "upper_rank_median": int(upper_median),
        "lower_rank_median": int(lower_median),
        "annotation_labels": [str(upper_label), str(lower_label)],
        "quartiles_by_label": quartiles_by_label(boxplots),
    }
    return SingleBoxplotTaskPlan(
        boxplots=boxplots,
        params=effective_params,
        mark_style=mark_style,
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        answer_value=int(answer_value),
        question_format="numeric_open",
        role_to_label={str(role): str(label) for role, label in zip(role_keys, (upper_label, lower_label))},
        relations=relations,
        prompt_artifacts=prompt_artifacts,
    )


@register_task
class ChartsDistributionBoxplotMedianRankDifferenceValueTask:
    """Compute the difference between two ranked boxplot medians."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (TOP_SECOND_QUERY_ID, TOP_THIRD_QUERY_ID, TOP_BOTTOM_QUERY_ID)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        selected_query_id, _probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=self.supported_query_ids,
            default_query_id=TOP_THIRD_QUERY_ID,
            task_id=self.task_id,
        )
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            attempt_seed = boxplot_attempt_seed(int(instance_seed), int(attempt))
            try:
                plan = _build_rank_difference_plan(dict(task_params), int(attempt_seed), str(selected_query_id))
                materialized = materialize_single_boxplot_plan(
                    instance_seed=int(attempt_seed),
                    selected_query_id=str(selected_query_id),
                    plan=plan,
                )
                return TaskOutput(
                    prompt=materialized.prompt,
                    answer_gt=materialized.answer_gt,
                    annotation_gt=materialized.annotation_gt,
                    image=materialized.image,
                    image_id="img0",
                    trace_payload=materialized.trace_payload,
                    task_versions=default_task_versions(),
                    scene_id=SCENE_ID,
                    query_id=materialized.query_id,
                    prompt_variants=materialized.prompt_variants,
                )
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"failed to generate {self.task_id}: {last_error}")


__all__ = ["ChartsDistributionBoxplotMedianRankDifferenceValueTask"]
