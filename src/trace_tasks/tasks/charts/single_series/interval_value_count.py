"""Count marks whose values fall inside an inclusive interval."""
from __future__ import annotations
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from ._lifecycle import build_count_plan as B, run_single_series_lifecycle as R
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__interval_value_count"
D = dict(mark_count_max=16)
PGM = "count(filter(marks, lower_bound <= value(mark) <= upper_bound)); output=integer_count; annotation=point_set(matching_marks); scene=single_series; scope=interval_value_count"
def _build_plan(params, seed, query_id, _):
    if query_id != SINGLE_QUERY_ID: raise ValueError(f"unsupported query_id for {T}: {query_id}")
    return B(params=params, seed=seed, namespace=T, count_variant="in_interval", prompt_key="in_interval", dynamic_slots={"interval_min":"trace:str:interval_min", "interval_max":"trace:str:interval_max"}, relation_params={"count_variant":"in_interval", "interval_min":"trace:int:interval_min", "interval_max":"trace:int:interval_max", "interval_inclusive":"trace:raw:interval_inclusive"}, program_code=PGM, reasoning_load=0.58)

@register_task
class ChartsCountingIntervalValueCountTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "interval_value_count"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
