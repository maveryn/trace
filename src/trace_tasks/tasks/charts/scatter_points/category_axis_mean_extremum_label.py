from __future__ import annotations

from trace_tasks.tasks.charts.scatter_points._lifecycle import (
    ScatterPointsTaskPlan,
    run_scatter_points_lifecycle,
)
from trace_tasks.tasks.charts.scatter_points.shared.prompts import dynamic_slots
from trace_tasks.tasks.charts.scatter_points.shared.sampling import build_category_mean_dataset
from trace_tasks.tasks.charts.scatter_points.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scatter_points__category_axis_mean_extremum_label"
QUESTION_FORMAT = "scatter_points_category_mean_extremum"
PROGRAM_CODE = "argextreme_label(category, mean(coord(points(category), axis)), direction)"
QUERY_IDS = (
    "largest_mean_x_category_label",
    "smallest_mean_x_category_label",
    "largest_mean_y_category_label",
    "smallest_mean_y_category_label",
)
QUERY_ARGS = {
    "largest_mean_x_category_label": {"mean_axis": "x", "mean_extremum": "largest"},
    "smallest_mean_x_category_label": {"mean_axis": "x", "mean_extremum": "smallest"},
    "largest_mean_y_category_label": {"mean_axis": "y", "mean_extremum": "largest"},
    "smallest_mean_y_category_label": {"mean_axis": "y", "mean_extremum": "smallest"},
}
REASONING_LOAD = 0.70
DEFAULT_QUERY_ID = QUERY_IDS[0]


def _build_category_mean_plan(
    params,
    instance_seed,
    selected_query_id,
    query_probabilities,
) -> ScatterPointsTaskPlan:
    if str(selected_query_id) not in QUERY_ARGS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")
    semantic_args = dict(QUERY_ARGS[str(selected_query_id)])
    dataset = build_category_mean_dataset(
        params=params,
        instance_seed=int(instance_seed),
        mean_axis=str(semantic_args["mean_axis"]),
        mean_extremum=str(semantic_args["mean_extremum"]),
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
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=REASONING_LOAD,
        annotation_kind="bbox",
    )


@register_task
class ChartsScatterPointsCategoryAxisMeanExtremumLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation')
    domain = DOMAIN
    objective_contract = "category_axis_mean_extremum_label"
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
            build_plan=_build_category_mean_plan,
        )
