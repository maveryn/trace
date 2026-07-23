from __future__ import annotations

from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.scatter_readout._lifecycle import (
    run_scatter_readout_lifecycle,
    single_point_readout_binding,
    single_point_readout_plan,
)
from trace_tasks.tasks.charts.scatter_readout.shared.sampling import build_base_dataset, select_series_point
from trace_tasks.tasks.charts.scatter_readout.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scatter_readout__series_value_at_x_value"
PROMPT_QUERY_KEY = "series_value_at_x_value"
QUESTION_FORMAT = "scatter_series_readout_query"
PROGRAM_CODE = (
    "value(series, x_label); output=integer_value; annotation=point(target_mark); "
    "scene=scatter_readout; scope=series_value_at_x_value"
)
QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
REASONING_LOAD = 0.46


def _build_value_at_x_plan(
    params,
    instance_seed,
    selected_query_id,
    query_probabilities,
):
    if str(selected_query_id) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")

    dataset = build_base_dataset(params=params, instance_seed=int(instance_seed))
    target_series, target_point = select_series_point(
        dataset=dataset,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target",
    )
    binding = single_point_readout_binding(
        answer=int(target_point.y_value),
        answer_type="integer",
        target_series_label=str(target_series.label),
        target_point_id=str(target_point.point_id),
        target_x_label=str(target_point.x_label),
        target_y_value=int(target_point.y_value),
        operation="direct_value_readout",
    )
    return single_point_readout_plan(
        dataset=dataset,
        binding=binding,
        params=dict(params),
        prompt_query_key=PROMPT_QUERY_KEY,
        question_format=QUESTION_FORMAT,
        program_code=PROGRAM_CODE,
        query_params={
            "operation": "direct_value_readout",
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=REASONING_LOAD,
    )


@register_task
class ChartsScatterSeriesValueAtXValueTask:
    task_id = TASK_ID
    reasoning_operations = ('direct_retrieval',)
    domain = DOMAIN
    objective_contract = "series_value_at_x_value"
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
            build_plan=_build_value_at_x_plan,
        )


__all__ = ["ChartsScatterSeriesValueAtXValueTask"]
