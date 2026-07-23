"""Compute average rate of change over an ordered chart interval."""
from __future__ import annotations
from dataclasses import replace
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from ._lifecycle import build_interval_change_plan as B, run_single_series_lifecycle as R
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__interval_rate_value"
D = dict(mark_count_min=8, mark_count_max=14, value_min=1, value_max=80, value_window_enabled=True, value_window_span_min=25, value_window_span_max=25, interval_gap_min=2, interval_gap_max=6, average_rate_abs_min=2, average_rate_abs_max=12)
PGM = "abs(quotient(value(end_mark)-value(start_mark), step_count(start_mark,end_mark))); output=integer_value; annotation=point_map(start_mark,end_mark); scene=single_series; scope=interval_rate_value"
def _build_plan(params, seed, query_id, _):
    if query_id != SINGLE_QUERY_ID: raise ValueError(f"unsupported query_id for {T}: {query_id}")
    plan = B(params=params, seed=seed, namespace=T, interval_variant="average_rate_over_interval", prompt_key="average_rate_over_interval", dynamic_slots={}, relation_params={"interval_variant":"absolute_average_rate_over_interval"}, program_code=PGM, reasoning_load=0.72)
    answer_value = abs(int(plan.dataset.answer_value))
    return replace(
        plan,
        dataset=replace(plan.dataset, answer_value=int(answer_value), trace={**dict(plan.dataset.trace), "answer_value": int(answer_value)}),
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
    )

@register_task
class ChartsTrendIntervalRateValueTask:
    task_id = T
    reasoning_operations = ('counting', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "interval_rate_value"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
