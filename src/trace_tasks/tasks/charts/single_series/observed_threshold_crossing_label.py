"""Find the first observed threshold crossing label in an ordered chart."""
from __future__ import annotations
from ._lifecycle import build_crossing_plan as B, run_single_series_lifecycle as R
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__observed_threshold_crossing_label"
Q = {"observed_above_threshold_crossing_label": ("first_crosses_above_threshold", "above"), "observed_below_threshold_crossing_label": ("first_crosses_below_threshold", "below")}
D = dict(mark_count_min=8, mark_count_max=14, value_min=1, value_max=80, threshold_edge_margin=8, threshold_min=20, threshold_max=60, crossing_index_min=2, crossing_index_max=10)
PGM = "first_label(filter_prefix(sequence(values), value(label) comparison threshold)); output=string_label; annotation=point(crossing_mark); scene=single_series; scope=observed_threshold_crossing_label"
def _build_plan(params, seed, query_id, _):
    variant, direction = Q[str(query_id)]
    return B(params=params, seed=seed, namespace=T, crossing_variant=variant, crossing_mode="observed", direction=direction, prompt_key=query_id, projected=False, annotation_kind="point", program_code=PGM, reasoning_load=0.52)

@register_task
class ChartsTrendObservedThresholdCrossingLabelTask:
    task_id = T
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = DOMAIN
    objective_contract = "observed_threshold_crossing_label"
    supported_query_ids = tuple(Q)
    default_query_id = "observed_above_threshold_crossing_label"
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
