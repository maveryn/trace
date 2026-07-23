from __future__ import annotations

from trace_tasks.tasks.charts.scatter_points._lifecycle import (
    ScatterPointsTaskPlan,
    run_scatter_points_lifecycle,
)
from trace_tasks.tasks.charts.scatter_points.shared.prompts import dynamic_slots
from trace_tasks.tasks.charts.scatter_points.shared.sampling import build_category_threshold_dataset
from trace_tasks.tasks.charts.scatter_points.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scatter_points__category_threshold_point_count"
QUESTION_FORMAT = "scatter_points_category_threshold_count"
PROGRAM_CODE = "count(point, category(point)=target_category and compare(coord(point, axis), threshold, direction))"
QUERY_IDS = (
    "category_x_above_threshold_count",
    "category_x_below_threshold_count",
    "category_y_above_threshold_count",
    "category_y_below_threshold_count",
)
QUERY_ARGS = {
    "category_x_above_threshold_count": {"threshold_axis": "x", "threshold_direction": "above"},
    "category_x_below_threshold_count": {"threshold_axis": "x", "threshold_direction": "below"},
    "category_y_above_threshold_count": {"threshold_axis": "y", "threshold_direction": "above"},
    "category_y_below_threshold_count": {"threshold_axis": "y", "threshold_direction": "below"},
}
REASONING_LOAD = 0.74
DEFAULT_QUERY_ID = QUERY_IDS[0]


def _build_category_threshold_plan(
    params,
    instance_seed,
    selected_query_id,
    query_probabilities,
) -> ScatterPointsTaskPlan:
    if str(selected_query_id) not in QUERY_ARGS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")
    semantic_args = dict(QUERY_ARGS[str(selected_query_id)])
    dataset = build_category_threshold_dataset(
        params=params,
        instance_seed=int(instance_seed),
        threshold_axis=str(semantic_args["threshold_axis"]),
        threshold_direction=str(semantic_args["threshold_direction"]),
    )
    return ScatterPointsTaskPlan(
        dataset=dataset,
        params=dict(params),
        prompt_query_key=str(selected_query_id),
        dynamic_slots=dynamic_slots(dataset=dataset),
        question_format=QUESTION_FORMAT,
        program_code=PROGRAM_CODE,
        query_params={
            **dict(semantic_args),
            "target_category_label": str(dataset.query.trace["target_category_label"]),
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=REASONING_LOAD,
    )


@register_task
class ChartsScatterPointsCategoryThresholdPointCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'logical_composition')
    domain = DOMAIN
    objective_contract = "category_threshold_point_count"
    supported_query_ids = QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_scatter_points_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_category_threshold_plan,
        )
