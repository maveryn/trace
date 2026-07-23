"""Compute endpoint-to-endpoint change over an ordered chart interval."""
from __future__ import annotations
from ._lifecycle import build_interval_change_plan as B, run_single_series_lifecycle as R
from .shared.prompts import endpoint_slots
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__endpoint_change_value"
Q = {"absolute_endpoint_change_value": ("absolute_change_between_labels", "absolute"), "signed_endpoint_change_value": ("signed_change_between_labels", "signed"), "percent_endpoint_change_value": ("percent_change_between_labels", "percent")}
D = dict(mark_count_min=8, mark_count_max=14, value_min=1, value_max=80, value_window_enabled=True, value_window_span_min=25, value_window_span_max=25, interval_gap_min=2, interval_gap_max=6, change_abs_min=5, change_abs_max=60, percent_change_min=-75, percent_change_max=150, percent_change_step=5)
PGM = "difference(value(end_mark), value(start_mark), mode); output=integer_value; annotation=point_map(start_mark,end_mark); scene=single_series; scope=endpoint_change_value"
L = {"absolute":0.38, "signed":0.48, "percent":0.82}
def _build_plan(params, seed, query_id, _):
    variant, mode = Q[str(query_id)]
    return B(params=params, seed=seed, namespace=T, interval_variant=variant, prompt_key=query_id, dynamic_slots=endpoint_slots(mode), relation_params={"interval_variant":variant, "endpoint_change_kind":mode}, program_code=PGM, reasoning_load=L[mode])

@register_task
class ChartsTrendEndpointChangeValueTask:
    task_id = T
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    objective_contract = "endpoint_change_value"
    supported_query_ids = tuple(Q)
    default_query_id = "absolute_endpoint_change_value"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
