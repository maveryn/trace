"""Compute a visible numeric axis span in a scientific plot frame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.charts.scientific_axis_frame._lifecycle import (
    AxisFrameTaskPlan,
    run_axis_frame_lifecycle,
)
from trace_tasks.tasks.charts.scientific_axis_frame.shared.prompts import dynamic_slots
from trace_tasks.tasks.charts.scientific_axis_frame.shared.sampling import build_axis_span_dataset
from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scientific_axis_frame__axis_span_value"
PROGRAM_CODE = (
    "difference(max_visible_tick(axis), min_visible_tick(axis)); "
    "output=integer_value; annotation=segment(axis_visible_span); "
    "scene=scientific_axis_frame; scope=axis_span_value"
)
AXIS_SPAN_QUERY_IDS = (
    "x_axis_span_value",
    "y_axis_span_value",
)
REASONING_LOAD = 0.44
DEFAULT_QUERY_ID = AXIS_SPAN_QUERY_IDS[0]


def _build_axis_span_plan(
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> AxisFrameTaskPlan:
    """Bind one public query id to an axis-span objective."""

    if str(selected_query_id) not in set(AXIS_SPAN_QUERY_IDS):
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query_id}")
    axis = str(selected_query_id)[0]
    dataset = build_axis_span_dataset(
        params=params,
        instance_seed=int(instance_seed),
        axis=str(axis),
    )
    return AxisFrameTaskPlan(
        dataset=dataset,
        params=dict(params),
        prompt_key=str(selected_query_id),
        dynamic_slots=dynamic_slots(axis_name=str(dataset.binding.trace["axis_name"])),
        question_format="scientific_axis_frame",
        program_code=PROGRAM_CODE,
        query_params={
            "axis": str(axis),
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=REASONING_LOAD,
        highlight_tick_keys=(),
    )


@register_task
class ChartsScientificAxisFrameAxisSpanValueTask:
    """Compute the visible numeric span of one axis."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    objective_contract = "axis_span_value"
    supported_query_ids = AXIS_SPAN_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_axis_frame_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_axis_span_plan,
        )


__all__ = [
    "AXIS_SPAN_QUERY_IDS",
    "ChartsScientificAxisFrameAxisSpanValueTask",
]
