"""Return the value of an order-statistic mark from a chart."""
from __future__ import annotations
from ._lifecycle import build_summary_plan as B, run_single_series_lifecycle as R
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__order_statistic_value"
Q = {"median_order_statistic_value":"median", "nth_highest_order_statistic_value":"nth_highest", "nth_lowest_order_statistic_value":"nth_lowest"}
D = dict(mark_count_min=15, mark_count_max=17, value_max=99, value_window_enabled=True, value_window_span_min=24, value_window_span_max=25, rank_n_min=3, rank_n_max=3, scene_variant_weights={"area":0.0,"bar":1.0,"line":1.0,"scatter":0.0,"horizontal_bar":0.0,"dot_plot":1.0,"lollipop":0.0})
PGM = "value(select_ranked_mark(marks, statistic_kind)); output=integer_value; annotation=point(selected_mark); scene=single_series; scope=order_statistic_value"
def _build_plan(params, seed, query_id, _):
    return B(params=params, seed=seed, namespace=T, statistic_kind=Q[str(query_id)], prompt_key=query_id, answer_target="value", program_code=PGM, reasoning_load=0.70)

@register_task
class ChartsSingleSeriesOrderStatisticValueTask:
    task_id = T
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "order_statistic_value"
    supported_query_ids = tuple(Q)
    default_query_id = "median_order_statistic_value"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
