"""Public task for `task_charts__dumbbell__side_winner_count`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dumbbell._lifecycle import DumbbellTaskPlan, run_dumbbell_plan
from trace_tasks.tasks.charts.dumbbell.shared.defaults import DOMAIN, SCENE_NAMESPACE
from trace_tasks.tasks.charts.dumbbell.shared import prompts as dumbbell_prompts
from trace_tasks.tasks.charts.dumbbell.shared import sampling as dumbbell_sampling
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellDataset, DumbbellQuery
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellRow
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__dumbbell__side_winner_count"
OBJECTIVE_CONTRACT = "side_winner_count"
SERIES_A_GREATER_QUERY_ID = "series_a_greater_threshold_count"
SERIES_B_GREATER_QUERY_ID = "series_b_greater_threshold_count"
SIDE_DIRECTION_BY_QUERY_ID = {
    SERIES_A_GREATER_QUERY_ID: "series_a_greater",
    SERIES_B_GREATER_QUERY_ID: "series_b_greater",
}
TASK_PARAM_DEFAULTS: dict[str, Any] = {}


def _side_gap_and_direction_plan(
    *,
    row_count: int,
    target_count: int,
    threshold: int,
    gap_min: int,
    gap_max: int,
    requested_direction: int,
    instance_seed: int,
) -> tuple[dict[int, int], dict[int, int], tuple[int, ...]]:
    """Construct row gaps where only target rows satisfy the signed side threshold."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.side_winner.signed_gap_plan.{requested_direction}")
    target_indices = tuple(sorted(rng.sample(list(range(int(row_count))), int(target_count))))
    target_index_set = set(int(index) for index in target_indices)
    gap_by_index: dict[int, int] = {}
    signed_direction_by_index: dict[int, int] = {}
    for index in range(int(row_count)):
        if int(index) in target_index_set:
            gap_by_index[index] = int(rng.randint(int(threshold), min(int(gap_max), int(threshold) + 22)))
            signed_direction_by_index[index] = int(requested_direction)
            continue
        if rng.random() < 0.55:
            gap_by_index[index] = int(rng.randint(int(gap_min), max(int(gap_min), int(threshold) - 3)))
            signed_direction_by_index[index] = int(requested_direction)
        else:
            gap_by_index[index] = int(rng.randint(int(gap_min), min(int(gap_max), int(threshold) + 18)))
            signed_direction_by_index[index] = -int(requested_direction)
    return gap_by_index, signed_direction_by_index, target_indices


def _signed_margin_for_direction(row: DumbbellRow, side_direction: str) -> int:
    """Return the winner-minus-loser margin for the requested side direction."""

    if str(side_direction) == "series_a_greater":
        return int(row.signed_delta_a_minus_b)
    if str(side_direction) == "series_b_greater":
        return -int(row.signed_delta_a_minus_b)
    raise ValueError(f"unsupported side direction: {side_direction}")


def _side_direction_trace_label(*, side_direction: str, series_a_name: str, series_b_name: str) -> str:
    """Return a readable trace label for the signed pairwise comparison."""

    if str(side_direction) == "series_a_greater":
        return f"{series_a_name} minus {series_b_name}"
    if str(side_direction) == "series_b_greater":
        return f"{series_b_name} minus {series_a_name}"
    raise ValueError(f"unsupported side direction: {side_direction}")


def _validate_side_targets(
    *,
    rows: tuple[DumbbellRow, ...],
    side_direction: str,
    threshold: int,
    annotation_row_ids: tuple[str, ...],
) -> None:
    """Verify that the signed-margin target set matches the annotation rows."""

    expected = {
        str(row.row_id)
        for row in rows
        if _signed_margin_for_direction(row, str(side_direction)) >= int(threshold)
    }
    actual = {str(row_id) for row_id in annotation_row_ids}
    if expected != actual:
        raise ValueError("side-winner count construction produced inconsistent annotation rows")


def _build_dataset(
    *,
    params: dict[str, Any],
    instance_seed: int,
    selected_query_id: str,
) -> DumbbellDataset:
    """Sample rows where one series exceeds the other by a threshold."""

    side_direction = SIDE_DIRECTION_BY_QUERY_ID[str(selected_query_id)]
    row_count, row_count_probabilities = dumbbell_sampling.sample_row_count(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.side_winner.row_count.{side_direction}",
    )
    row_labels, series_a_name, series_b_name = dumbbell_sampling.sample_labels_and_series(
        row_count=int(row_count),
        instance_seed=int(instance_seed),
        namespace=f"side_winner.{side_direction}",
    )
    threshold, threshold_probabilities = dumbbell_sampling.sample_threshold(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.side_winner.threshold.{side_direction}",
        min_key="side_threshold_min",
        max_key="side_threshold_max",
        step_key="side_threshold_step",
        fallback_min=10,
        fallback_max=24,
        fallback_step=2,
    )
    target_count, target_count_probabilities = dumbbell_sampling.sample_target_count(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.side_winner.answer.{side_direction}",
        row_count=int(row_count),
        min_key="side_winner_count_min",
        max_key="side_winner_count_max",
        fallback_min=2,
        fallback_max=10,
    )
    gap_min, gap_max = dumbbell_sampling.gap_bounds(params)
    requested_direction = 1 if str(side_direction) == "series_a_greater" else -1
    gap_by_index, signed_direction_by_index, target_indices = _side_gap_and_direction_plan(
        row_count=int(row_count),
        target_count=int(target_count),
        threshold=int(threshold),
        gap_min=int(gap_min),
        gap_max=int(gap_max),
        requested_direction=int(requested_direction),
        instance_seed=int(instance_seed),
    )

    rows = dumbbell_sampling.build_rows_from_gap_plan(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"side_winner.{side_direction}",
        labels=row_labels,
        gap_by_index=gap_by_index,
        signed_direction_by_index=signed_direction_by_index,
    )
    annotation_row_ids = tuple(f"row_{index}" for index in target_indices)
    _validate_side_targets(
        rows=tuple(rows),
        side_direction=str(side_direction),
        threshold=int(threshold),
        annotation_row_ids=annotation_row_ids,
    )
    target_row_labels = [str(rows[int(row_id.removeprefix("row_"))].label) for row_id in annotation_row_ids]
    params_out = {
        "scene_variant": "horizontal_dumbbell",
        "row_count": int(row_count),
        "row_count_probabilities": dict(row_count_probabilities),
        "series_a_name": str(series_a_name),
        "series_b_name": str(series_b_name),
        "side_direction": str(side_direction),
        "requested_direction_sign": int(requested_direction),
        "signed_margin_expression": _side_direction_trace_label(
            side_direction=str(side_direction),
            series_a_name=str(series_a_name),
            series_b_name=str(series_b_name),
        ),
        "threshold_value": int(threshold),
        "threshold_value_probabilities": dict(threshold_probabilities),
        "target_count": int(target_count),
        "target_count_probabilities": dict(target_count_probabilities),
        "target_row_labels": list(target_row_labels),
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
    side_direction = str(dataset.query.params["side_direction"])
    winner_series = str(dataset.series_a_name)
    loser_series = str(dataset.series_b_name)
    if side_direction == "series_b_greater":
        winner_series = str(dataset.series_b_name)
        loser_series = str(dataset.series_a_name)
    return dumbbell_prompts.base_dynamic_slots(
        winner_series=str(winner_series),
        loser_series=str(loser_series),
        threshold_value=int(dataset.query.params["threshold_value"]),
    )


@register_task
class ChartsDumbbellSideWinnerCountTask:
    """Count rows where one legend series exceeds the other by a threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (SERIES_A_GREATER_QUERY_ID, SERIES_B_GREATER_QUERY_ID)
    default_dataset_enabled = True

    def _build_plan(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> DumbbellTaskPlan:
        """Bind the side-comparison count objective before neutral rendering."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        dataset = _build_dataset(
            params=effective_params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
        )
        answer_gt = TypedValue(type=str(dataset.query.answer_type), value=int(dataset.query.answer))
        prompt_artifacts = dumbbell_prompts.build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slot_values=_prompt_slots(dataset),
            instance_seed=int(instance_seed),
        )
        return DumbbellTaskPlan(
            dataset=dataset,
            params=effective_params,
            answer_gt=answer_gt,
            question_format="dumbbell_side_winner_count",
            reasoning_load=0.70,
            prompt_artifacts=prompt_artifacts,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_dumbbell_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=SERIES_A_GREATER_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsDumbbellSideWinnerCountTask"]
