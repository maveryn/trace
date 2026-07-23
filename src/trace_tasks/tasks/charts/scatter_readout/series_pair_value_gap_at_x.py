"""Compute the absolute value gap between two scatter series at one x-axis label."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.scatter_readout._lifecycle import (
    ScatterReadoutTaskPlan,
    numeric_pair_readout_plan,
    run_scatter_readout_lifecycle,
)
from trace_tasks.tasks.charts.scatter_readout.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__scatter_readout__series_pair_value_gap_at_x"
PROMPT_QUERY_KEY = "series_pair_value_gap_at_x"
QUESTION_FORMAT = "scatter_series_readout_query"
PROGRAM_CODE = (
    "abs(value(series_a,x_label)-value(series_b,x_label)); "
    "output=integer_value; annotation=segment(series_a_mark,series_b_mark); "
    "scene=scatter_readout; scope=series_pair_value_gap_at_x"
)
QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
REASONING_LOAD = 0.70


def _build_pair_gap_plan(
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> ScatterReadoutTaskPlan:
    """Bind the same-x pair as an absolute gap objective."""

    return numeric_pair_readout_plan(
        params=params,
        instance_seed=int(instance_seed),
        query_probabilities=query_probabilities,
        namespace=f"{TASK_ID}.target",
        prompt_query_key=PROMPT_QUERY_KEY,
        question_format=QUESTION_FORMAT,
        program_code=PROGRAM_CODE,
        annotation_kind="target_comparison_segment",
        operation="absolute_difference",
        reasoning_load=REASONING_LOAD,
        answer_fn=lambda target, comparison: abs(int(target.y_value) - int(comparison.y_value)),
    )


@register_task
class ChartsScatterSeriesPairValueGapAtXTask:
    """Compute the absolute value gap between two series at the same x-axis label."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    objective_contract = "series_pair_value_gap_at_x"
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
            build_plan=_build_pair_gap_plan,
        )


__all__ = ["ChartsScatterSeriesPairValueGapAtXTask"]
