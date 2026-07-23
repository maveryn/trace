"""Public task for `task_charts__waterfall__reverse_step_final_total`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.waterfall._lifecycle import build_counterfactual_plan, run_waterfall_lifecycle
from trace_tasks.tasks.charts.waterfall.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__waterfall__reverse_step_final_total"
OBJECTIVE_CONTRACT = "reverse_step_final_total"
PROMPT_QUERY_KEY = "reverse_step_final_total"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind the reverse-step counterfactual before neutral rendering."""

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    return build_counterfactual_plan(
        params=params,
        instance_seed=int(instance_seed),
        prompt_query_key=PROMPT_QUERY_KEY,
        operation_name="reverse_step_sign",
        counterfactual_phrase="reversed",
        answer_from_final_and_delta=lambda final_value, delta: int(final_value) - (2 * int(delta)),
        query_probabilities=query_probabilities,
        step_count_min=5,
        step_count_max=7,
        target_delta_abs_min=5,
        target_delta_abs_max=15,
        answer_min=0,
        answer_max=100,
    )


class ChartsWaterfallReverseStepFinalTotalTask:
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'state_update', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        return run_waterfall_lifecycle(
            task=self,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            default_query_id=DEFAULT_QUERY_ID,
            build_plan=_build_plan,
        )


register_task(ChartsWaterfallReverseStepFinalTotalTask)


__all__ = ["ChartsWaterfallReverseStepFinalTotalTask"]
