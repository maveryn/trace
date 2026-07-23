"""Monotone Streak Length for an ordered single-series chart."""
from __future__ import annotations
from ._lifecycle import build_trend_structure_plan as B, run_single_series_lifecycle as R
from .shared.prompts import streak_slots
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__monotone_streak_length"
Q = {"longest_increasing_streak_length": ("longest_increasing_streak", "increasing"), "longest_decreasing_streak_length": ("longest_decreasing_streak", "decreasing")}
D = dict(mark_count_min=6, mark_count_max=10, value_min=1, value_max=99, value_window_enabled=True, value_window_span_min=20, value_window_span_max=25)
PGM = "length(unique_longest_monotone_run(sequence(values), direction)); output=integer_value; annotation=point_set(run_marks); scene=single_series; scope=monotone_streak_length"
def _build_plan(params, seed, query_id, _):
    variant, mode = Q[str(query_id)]
    return B(params=params, seed=seed, namespace=T, trend_variant=variant, prompt_key=query_id, dynamic_slots=streak_slots(mode), relation_params={"trend_variant":variant, "streak_direction":mode}, program_code=PGM, reasoning_load=0.64)

@register_task
class ChartsTrendMonotoneStreakLengthTask:
    task_id = T
    reasoning_operations = ('filtering', 'counting', 'ranking')
    domain = DOMAIN
    objective_contract = "monotone_streak_length"
    supported_query_ids = tuple(Q)
    default_query_id = "longest_increasing_streak_length"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
