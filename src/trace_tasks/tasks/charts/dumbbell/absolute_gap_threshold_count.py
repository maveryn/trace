"""Public task for `task_charts__dumbbell__absolute_gap_threshold_count`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dumbbell._lifecycle import DumbbellTaskPlan, run_dumbbell_plan
from trace_tasks.tasks.charts.dumbbell.shared.defaults import DOMAIN, SCENE_NAMESPACE
from trace_tasks.tasks.charts.dumbbell.shared.prompts import base_dynamic_slots, build_prompt_artifacts
from trace_tasks.tasks.charts.dumbbell.shared.sampling import (
    build_rows_from_gap_plan,
    gap_bounds,
    sample_labels_and_series,
    sample_row_count,
    sample_target_count,
    sample_threshold,
)
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellDataset, DumbbellQuery, DumbbellRow
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__dumbbell__absolute_gap_threshold_count"
OBJECTIVE_CONTRACT = "absolute_gap_threshold_count"
AT_LEAST_QUERY_ID = "absolute_gap_at_least_threshold_count"
AT_MOST_QUERY_ID = "absolute_gap_at_most_threshold_count"
GAP_RELATION_BY_QUERY_ID = {
    AT_LEAST_QUERY_ID: "at_least",
    AT_MOST_QUERY_ID: "at_most",
}
TASK_PARAM_DEFAULTS: dict[str, Any] = {}


def _gap_supports_for_relation(*, relation: str, threshold: int, gap_min: int, gap_max: int) -> tuple[list[int], list[int]]:
    """Return target and distractor absolute-gap supports for one relation."""

    if str(relation) == "at_least":
        return (
            [gap for gap in range(int(threshold), int(gap_max) + 1)],
            [gap for gap in range(int(gap_min), int(threshold))],
        )
    if str(relation) == "at_most":
        return (
            [gap for gap in range(int(gap_min), int(threshold) + 1)],
            [gap for gap in range(int(threshold) + 1, int(gap_max) + 1)],
        )
    raise ValueError(f"unsupported gap relation: {relation}")


def _absolute_gap_plan(
    *,
    row_count: int,
    target_count: int,
    target_support: list[int],
    distractor_support: list[int],
    instance_seed: int,
    relation: str,
) -> tuple[dict[int, int], tuple[int, ...]]:
    """Construct absolute gap values from disjoint target and distractor supports."""

    if not target_support or not distractor_support:
        raise ValueError("empty gap-threshold support")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.gap_threshold.absolute_gap_plan.{relation}")
    target_indices = tuple(sorted(rng.sample(list(range(int(row_count))), int(target_count))))
    target_index_set = set(int(index) for index in target_indices)
    gap_by_index: dict[int, int] = {}
    for index in range(int(row_count)):
        support = target_support if int(index) in target_index_set else distractor_support
        gap_by_index[index] = int(rng.choice(support))
    return gap_by_index, target_indices


def _gap_matches_relation(gap: int, relation: str, threshold: int) -> bool:
    """Return whether one absolute row gap satisfies the relation."""

    if str(relation) == "at_least":
        return int(gap) >= int(threshold)
    if str(relation) == "at_most":
        return int(gap) <= int(threshold)
    raise ValueError(f"unsupported gap relation: {relation}")


def _validate_absolute_gap_targets(
    *,
    rows: tuple[DumbbellRow, ...],
    relation: str,
    threshold: int,
    annotation_row_ids: tuple[str, ...],
) -> None:
    """Verify that relation-matching rows are exactly the annotated rows."""

    expected = {
        str(row.row_id)
        for row in rows
        if _gap_matches_relation(int(row.gap), str(relation), int(threshold))
    }
    actual = {str(row_id) for row_id in annotation_row_ids}
    if expected != actual:
        raise ValueError("absolute-gap threshold construction produced inconsistent annotation rows")


def _build_dataset(
    *,
    params: dict[str, Any],
    instance_seed: int,
    selected_query_id: str,
) -> DumbbellDataset:
    """Sample rows whose absolute pairwise gap satisfies a threshold relation."""

    gap_relation = GAP_RELATION_BY_QUERY_ID[str(selected_query_id)]
    row_count, row_count_probabilities = sample_row_count(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.gap_threshold.row_count.{gap_relation}",
    )
    row_labels, series_a_name, series_b_name = sample_labels_and_series(
        row_count=int(row_count),
        instance_seed=int(instance_seed),
        namespace=f"gap_threshold.{gap_relation}",
    )
    threshold, threshold_probabilities = sample_threshold(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.gap_threshold.threshold.{gap_relation}",
        min_key="gap_threshold_min",
        max_key="gap_threshold_max",
        step_key="gap_threshold_step",
        fallback_min=12,
        fallback_max=28,
        fallback_step=4,
    )
    target_count, target_count_probabilities = sample_target_count(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.gap_threshold.answer.{gap_relation}",
        row_count=int(row_count),
        min_key="gap_threshold_count_min",
        max_key="gap_threshold_count_max",
        fallback_min=2,
        fallback_max=10,
    )
    gap_min, gap_max = gap_bounds(params)
    target_support, distractor_support = _gap_supports_for_relation(
        relation=str(gap_relation),
        threshold=int(threshold),
        gap_min=int(gap_min),
        gap_max=int(gap_max),
    )
    gap_by_index, target_indices = _absolute_gap_plan(
        row_count=int(row_count),
        target_count=int(target_count),
        target_support=target_support,
        distractor_support=distractor_support,
        instance_seed=int(instance_seed),
        relation=str(gap_relation),
    )

    rows = build_rows_from_gap_plan(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"gap_threshold.{gap_relation}",
        labels=row_labels,
        gap_by_index=gap_by_index,
    )
    annotation_row_ids = tuple(f"row_{index}" for index in target_indices)
    _validate_absolute_gap_targets(
        rows=tuple(rows),
        relation=str(gap_relation),
        threshold=int(threshold),
        annotation_row_ids=annotation_row_ids,
    )
    params_out = {
        "scene_variant": "horizontal_dumbbell",
        "row_count": int(row_count),
        "row_count_probabilities": dict(row_count_probabilities),
        "series_a_name": str(series_a_name),
        "series_b_name": str(series_b_name),
        "gap_threshold_relation": str(gap_relation),
        "gap_threshold_value": int(threshold),
        "target_gap_support": [int(min(target_support)), int(max(target_support))],
        "distractor_gap_support": [int(min(distractor_support)), int(max(distractor_support))],
        "threshold_value_probabilities": dict(threshold_probabilities),
        "target_count": int(target_count),
        "target_count_probabilities": dict(target_count_probabilities),
    }
    return DumbbellDataset(
        scene_variant="horizontal_dumbbell",
        series_a_name=str(series_a_name),
        series_b_name=str(series_b_name),
        rows=tuple(rows),
        query=DumbbellQuery(
            answer=int(target_count),
            answer_type="integer",
            annotation_row_ids=annotation_row_ids,
            params=params_out,
        ),
    )


def _prompt_slots(dataset: DumbbellDataset) -> dict[str, Any]:
    gap_relation = str(dataset.query.params["gap_threshold_relation"])
    return base_dynamic_slots(
        winner_series=str(dataset.series_a_name),
        loser_series=str(dataset.series_b_name),
        gap_relation_phrase="at least" if gap_relation == "at_least" else "at most",
        gap_threshold_value=int(dataset.query.params["gap_threshold_value"]),
    )


@register_task
class ChartsDumbbellAbsoluteGapThresholdCountTask:
    """Count rows whose absolute pairwise gap satisfies a threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (AT_LEAST_QUERY_ID, AT_MOST_QUERY_ID)
    default_dataset_enabled = True

    def _build_plan(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> DumbbellTaskPlan:
        """Bind the absolute-gap threshold count objective before rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        dataset = _build_dataset(
            params=effective_params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
        )
        answer_gt = TypedValue(type=str(dataset.query.answer_type), value=int(dataset.query.answer))
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slot_values=_prompt_slots(dataset),
            instance_seed=int(instance_seed),
        )
        return DumbbellTaskPlan(
            dataset=dataset,
            params=effective_params,
            answer_gt=answer_gt,
            question_format="dumbbell_absolute_gap_threshold_count",
            reasoning_load=0.64,
            prompt_artifacts=prompt_artifacts,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_dumbbell_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=AT_LEAST_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsDumbbellAbsoluteGapThresholdCountTask"]
