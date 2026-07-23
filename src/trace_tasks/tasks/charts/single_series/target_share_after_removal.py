"""Compute target_share_after_removal over labeled chart values."""
from __future__ import annotations
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from ._lifecycle import build_counterfactual_plan as B, run_single_series_lifecycle as R
from .shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task
T = "task_charts__single_series__target_share_after_removal"
D = dict(mark_count_min=4, mark_count_max=10, value_min=5, value_max=80, removed_count_min=1, removed_count_max=2)
PGM = "percent_share(value(target_label), sum(values(marks excluding removed_labels))); output=integer_value; annotation=point_set(retained_marks); scene=single_series; scope=target_share_after_removal"
def _build_plan(params, seed, query_id, _):
    if query_id != SINGLE_QUERY_ID: raise ValueError(f"unsupported query_id for {T}: {query_id}")
    return B(params=params, seed=seed, namespace=T, operation="target_share", prompt_key="target_share_after_removal", dynamic_slots={"removed_labels_text":"trace:quoted:removed_labels", "retained_labels_text":"trace:quoted:retained_labels", "target_label":"trace:str:target_label"}, relation_params={"counterfactual_operation":"remove_labels_then_target_share_percent", "removed_labels":"trace:list:removed_labels", "retained_labels":"trace:list:retained_labels", "target_label":"trace:str:target_label"}, program_code=PGM, reasoning_load=0.78)

@register_task
class ChartsHypotheticalTargetShareAfterRemovalPublicTask:
    task_id = T
    reasoning_operations = ('aggregation', 'state_update', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "target_share_after_removal"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return R(task=self, instance_seed=instance_seed, params={**D, **params}, max_attempts=max_attempts, default_query_id=self.default_query_id, build_plan=_build_plan)
