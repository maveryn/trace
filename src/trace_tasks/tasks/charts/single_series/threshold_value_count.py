"""Count marks above or below one threshold in a labeled chart."""
from __future__ import annotations
from ._lifecycle import build_count_plan as B, run_single_series_lifecycle as R
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__threshold_value_count"
Q = {"above_threshold_count": ("above_threshold", "greater than"), "below_threshold_count": ("below_threshold", "less than")}
D = dict(mark_count_max=20)
PGM = "count(filter(marks, value(mark) comparison threshold)); output=integer_count; annotation=point_set(matching_marks); scene=single_series; scope=threshold_value_count"
def _build_plan(params, seed, query_id, _):
    variant, phrase = Q[str(query_id)]
    return B(params=params, seed=seed, namespace=T, count_variant=variant, prompt_key=query_id, dynamic_slots={"threshold":"trace:str:threshold", "threshold_comparison_phrase":phrase}, relation_params={"count_variant":variant, "comparison":"trace:str:comparison", "threshold":"trace:int:threshold"}, program_code=PGM, reasoning_load=0.35)

@register_task
class ChartsCountingThresholdValueCountTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "threshold_value_count"
    supported_query_ids = tuple(Q)
    default_query_id = "above_threshold_count"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
