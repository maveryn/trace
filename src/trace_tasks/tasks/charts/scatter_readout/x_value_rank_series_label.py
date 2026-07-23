"""Select the series with the highest or lowest value at one x-axis label."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.charts.scatter_readout._lifecycle import (
    ScatterReadoutTaskPlan,
    run_scatter_readout_lifecycle,
    single_point_readout_binding,
    single_point_readout_plan,
)
from trace_tasks.tasks.charts.scatter_readout.shared.sampling import build_base_dataset, select_x_extreme_point
from trace_tasks.tasks.charts.scatter_readout.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scatter_readout__x_value_rank_series_label"
QUESTION_FORMAT = "scatter_series_readout_query"
PROGRAM_CODE = (
    "select_label(arg_extreme(series, value(series,x_label), direction)); "
    "output=string_label; annotation=point(target_mark); "
    "scene=scatter_readout; scope=x_value_rank_series_label"
)
QUERY_IDS = ("x_highest_series_label", "x_lowest_series_label")
QUERY_ARGS = {
    "x_highest_series_label": {"extremum": "highest"},
    "x_lowest_series_label": {"extremum": "lowest"},
}
DEFAULT_QUERY_ID = QUERY_IDS[0]
REASONING_LOAD = 0.58


def _build_x_rank_plan(
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> ScatterReadoutTaskPlan:
    """Bind one x-axis label and select the extremum series at that position."""

    if str(selected_query_id) not in QUERY_ARGS:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")
    semantic_args = dict(QUERY_ARGS[str(selected_query_id)])
    dataset = build_base_dataset(params=params, instance_seed=int(instance_seed))
    x_label, target_point = select_x_extreme_point(
        dataset=dataset,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target",
        extremum=str(semantic_args["extremum"]),
    )
    binding = single_point_readout_binding(
        answer=str(target_point.series_label),
        answer_type="string",
        target_series_label=str(target_point.series_label),
        target_point_id=str(target_point.point_id),
        target_x_label=str(x_label),
        target_y_value=int(target_point.y_value),
        operation="x_value_rank_selection",
        extra_trace={
            **dict(semantic_args),
        },
    )
    return single_point_readout_plan(
        dataset=dataset,
        binding=binding,
        params=dict(params),
        prompt_query_key=str(selected_query_id),
        question_format=QUESTION_FORMAT,
        program_code=PROGRAM_CODE,
        query_params={
            **dict(semantic_args),
            "operation": "x_value_rank_selection",
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=REASONING_LOAD,
    )


@register_task
class ChartsScatterXValueRankSeriesLabelTask:
    """Select the highest or lowest series at one x-axis label."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "x_value_rank_series_label"
    supported_query_ids = QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_scatter_readout_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_x_rank_plan,
        )


__all__ = ["ChartsScatterXValueRankSeriesLabelTask"]
