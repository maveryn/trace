"""Public task for `task_charts__dumbbell__gap_rank_row_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.dumbbell._lifecycle import DumbbellTaskPlan, run_dumbbell_plan
from trace_tasks.tasks.charts.dumbbell.shared.defaults import DOMAIN, SCENE_NAMESPACE
from trace_tasks.tasks.charts.dumbbell.shared.prompts import base_dynamic_slots, build_prompt_artifacts
from trace_tasks.tasks.charts.dumbbell.shared.sampling import (
    build_rows_from_gap_plan,
    gap_bounds,
    rank_phrase,
    sample_labels_and_series,
    sample_rank_n,
    sample_row_count,
)
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellDataset, DumbbellQuery
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__dumbbell__gap_rank_row_label"
OBJECTIVE_CONTRACT = "gap_rank_row_label"
LARGEST_GAP_QUERY_ID = "largest_gap_rank_row_label"
SMALLEST_GAP_QUERY_ID = "smallest_gap_rank_row_label"
RANK_ORDER_BY_QUERY_ID = {
    LARGEST_GAP_QUERY_ID: "largest",
    SMALLEST_GAP_QUERY_ID: "smallest",
}
TASK_PARAM_DEFAULTS: dict[str, Any] = {"row_count_min": 10, "row_count_max": 12}


def _build_dataset(
    *,
    params: dict[str, Any],
    instance_seed: int,
    selected_query_id: str,
) -> DumbbellDataset:
    """Sample the ranked-gap objective and bind its answer row."""

    rank_order = RANK_ORDER_BY_QUERY_ID[str(selected_query_id)]
    row_count, row_count_probabilities = sample_row_count(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.gap_rank.row_count.{rank_order}",
    )
    row_labels, series_a_name, series_b_name = sample_labels_and_series(
        row_count=int(row_count),
        instance_seed=int(instance_seed),
        namespace=f"gap_rank.{rank_order}",
    )
    rank_n, rank_n_probabilities = sample_rank_n(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.gap_rank.rank_n.{rank_order}",
    )
    target_row_index = int(
        uniform_choice(
            spawn_rng(
                int(instance_seed),
                f"{SCENE_NAMESPACE}.gap_rank.answer_row.{rank_order}",
            ),
            tuple(range(int(row_count))),
            sort_keys=True,
        )
    )

    gap_min, gap_max = gap_bounds(params)
    support = [gap for gap in range(int(gap_min), int(gap_max) + 1)]
    if len(support) < int(row_count):
        raise ValueError("not enough distinct gaps for ranked-gap query")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.gap_rank.{rank_order}.gaps")
    gaps_sorted = sorted(rng.sample(support, int(row_count)))
    ranked = sorted(gaps_sorted, reverse=str(rank_order) == "largest")
    answer_gap = int(ranked[int(rank_n) - 1])
    remaining_gaps = [gap for gap in gaps_sorted if int(gap) != int(answer_gap)]
    gap_by_index: dict[int, int] = {}
    for index in range(int(row_count)):
        gap_by_index[int(index)] = int(answer_gap if int(index) == int(target_row_index) else remaining_gaps.pop())

    rows = build_rows_from_gap_plan(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"gap_rank.{rank_order}",
        labels=row_labels,
        gap_by_index=gap_by_index,
    )
    answer_row = rows[int(target_row_index)]
    params_out = {
        "scene_variant": "horizontal_dumbbell",
        "row_count": int(row_count),
        "row_count_probabilities": dict(row_count_probabilities),
        "series_a_name": str(series_a_name),
        "series_b_name": str(series_b_name),
        "rank_order": str(rank_order),
        "rank_n": int(rank_n),
        "rank_phrase": rank_phrase(str(rank_order), int(rank_n)),
        "rank_n_probabilities": dict(rank_n_probabilities),
        "answer_gap": int(answer_gap),
        "answer_row_id": str(answer_row.row_id),
        "answer_label": str(answer_row.label),
    }
    return DumbbellDataset(
        scene_variant="horizontal_dumbbell",
        series_a_name=str(series_a_name),
        series_b_name=str(series_b_name),
        rows=tuple(rows),
        query=DumbbellQuery(
            answer=str(answer_row.label),
            answer_type="string",
            annotation_row_ids=(str(answer_row.row_id),),
            params=params_out,
        ),
    )


def _prompt_slots(dataset: DumbbellDataset) -> dict[str, Any]:
    return base_dynamic_slots(
        rank_phrase=str(dataset.query.params["rank_phrase"]),
        winner_series=str(dataset.series_a_name),
        loser_series=str(dataset.series_b_name),
    )


@register_task
class ChartsDumbbellGapRankRowLabelTask:
    """Return the row label at a requested absolute gap rank."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (LARGEST_GAP_QUERY_ID, SMALLEST_GAP_QUERY_ID)
    default_dataset_enabled = True

    def _build_plan(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any],
        selected_query_id: str,
    ) -> DumbbellTaskPlan:
        """Bind the ranked-gap objective, then use neutral scene materialization."""

        effective_params = {**TASK_PARAM_DEFAULTS, **dict(params)}
        dataset = _build_dataset(
            params=effective_params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
        )
        answer_gt = TypedValue(type=str(dataset.query.answer_type), value=str(dataset.query.answer))
        prompt_artifacts = build_prompt_artifacts(
            prompt_query_key=str(selected_query_id),
            dynamic_slot_values=_prompt_slots(dataset),
            instance_seed=int(instance_seed),
        )
        return DumbbellTaskPlan(
            dataset=dataset,
            params=effective_params,
            answer_gt=answer_gt,
            question_format="dumbbell_gap_rank_row_label",
            reasoning_load=0.68,
            prompt_artifacts=prompt_artifacts,
            annotation_style="row_pair_segment",
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_dumbbell_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=LARGEST_GAP_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsDumbbellGapRankRowLabelTask"]
