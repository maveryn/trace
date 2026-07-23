"""Public task for `task_charts__waterfall__running_total_value`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.waterfall._lifecycle import WaterfallTaskPlan, run_waterfall_lifecycle
from trace_tasks.tasks.charts.waterfall.shared.annotations import bbox_set_artifacts
from trace_tasks.tasks.charts.waterfall.shared.defaults import DOMAIN
from trace_tasks.tasks.charts.waterfall.shared.sampling import choose_step_index, sample_waterfall_dataset
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__waterfall__running_total_value"
OBJECTIVE_CONTRACT = "running_total_value"
PROMPT_QUERY_KEY = "running_total_after_step"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
DEFAULT_QUERY_ID = SINGLE_QUERY_ID


def _build_plan(params, instance_seed, selected_branch, query_probabilities):
    """Bind the running-total objective before neutral rendering."""

    if str(selected_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")
    dataset = sample_waterfall_dataset(
        params,
        instance_seed=int(instance_seed),
        step_count_min=8,
        step_count_max=10,
    )
    target_index = choose_step_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.target_step",
        step_count=len(dataset.steps),
        min_index=4,
        max_from_end=1,
    )
    target_step = dataset.steps[int(target_index)]
    running_ids = ("start",) + tuple(step.step_id for step in dataset.steps[: int(target_index) + 1])

    def _bind_annotation(rendered):
        return bbox_set_artifacts([rendered.bar_bboxes_px[str(bar_id)] for bar_id in running_ids])

    return WaterfallTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(target_step.running_after)),
        annotation_builder=_bind_annotation,
        prompt_query_key=PROMPT_QUERY_KEY,
        dynamic_slots={"target_step_label": str(target_step.label)},
        query_params={
            "target_step_id": str(target_step.step_id),
            "target_step_label": str(target_step.label),
            "target_step_index": int(target_index),
            "step_count": int(len(dataset.steps)),
            "start_value": int(dataset.start_value),
            "final_value": int(dataset.final_value),
            "annotation_roles": list(running_ids),
        },
        relations={
            "target_step_id": str(target_step.step_id),
            "target_step_label": str(target_step.label),
            "target_step_index": int(target_index),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "waterfall_running_total_witness",
            "running_value_bar_ids": list(running_ids),
            "target_step_id": str(target_step.step_id),
            "answer": int(target_step.running_after),
        },
        question_format="waterfall_running_total_value",
    )


class ChartsWaterfallRunningTotalValueTask:
    task_id = TASK_ID
    reasoning_operations = ('aggregation',)
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


register_task(ChartsWaterfallRunningTotalValueTask)


__all__ = ["ChartsWaterfallRunningTotalValueTask"]
