"""Turning Point Count for an ordered single-series chart."""
from __future__ import annotations
from ._lifecycle import build_trend_structure_plan as B, run_single_series_lifecycle as R
from .shared.prompts import turning_slots
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__turning_point_count"
Q = {"peak_turning_point_count": ("peak_count", "peak"), "trough_turning_point_count": ("trough_count", "trough")}
D = dict(mark_count_min=6, mark_count_max=10, value_min=1, value_max=99, value_window_enabled=True, value_window_span_min=20, value_window_span_max=25)
PGM = "count(turning_points(sequence(values), turning_kind)); output=integer_count; annotation=point_set(turning_points); scene=single_series; scope=turning_point_count"
def _build_plan(params, seed, query_id, _):
    variant, mode = Q[str(query_id)]
    return B(params=params, seed=seed, namespace=T, trend_variant=variant, prompt_key=query_id, dynamic_slots=turning_slots(mode), relation_params={"trend_variant":variant, "turning_point_type":mode}, program_code=PGM, reasoning_load=0.42)

@register_task
class ChartsTrendTurningPointCountTask:
    task_id = T
    reasoning_operations = ('counting',)
    domain = DOMAIN
    objective_contract = "turning_point_count"
    supported_query_ids = tuple(Q)
    default_query_id = "peak_turning_point_count"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
