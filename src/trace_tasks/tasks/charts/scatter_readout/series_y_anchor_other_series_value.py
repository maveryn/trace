"""Transfer an x-axis anchor from one scatter series to another series value."""

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


TASK_ID = "task_charts__scatter_readout__series_y_anchor_other_series_value"
PROMPT_QUERY_KEY = "series_y_anchor_other_series_value"
QUESTION_FORMAT = "scatter_series_readout_query"
PROGRAM_CODE = (
    "value(comparison_series, x_label(point in anchor_series where y_value=anchor_value)); "
    "output=integer_value; annotation=point(comparison_mark); "
    "scene=scatter_readout; scope=series_y_anchor_other_series_value"
)
QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID
REASONING_LOAD = 0.74


def _build_anchor_transfer_plan(
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> ScatterReadoutTaskPlan:
    """Bind the same-x pair as an anchor-to-comparison value transfer objective."""

    return numeric_pair_readout_plan(
        params=params,
        instance_seed=int(instance_seed),
        query_probabilities=query_probabilities,
        namespace=f"{TASK_ID}.target",
        prompt_query_key=PROMPT_QUERY_KEY,
        question_format=QUESTION_FORMAT,
        program_code=PROGRAM_CODE,
        annotation_kind="comparison_point",
        operation="same_x_transfer_value",
        reasoning_load=REASONING_LOAD,
        answer_fn=lambda _target, comparison: int(comparison.y_value),
    )


@register_task
class ChartsScatterSeriesYAnchorOtherSeriesValueTask:
    """Use a value in one series to find the same-x value in a second series."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    objective_contract = "series_y_anchor_other_series_value"
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
            build_plan=_build_anchor_transfer_plan,
        )


__all__ = ["ChartsScatterSeriesYAnchorOtherSeriesValueTask"]
