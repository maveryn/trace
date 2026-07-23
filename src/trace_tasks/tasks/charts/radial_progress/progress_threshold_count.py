from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_count_dataset_from_frame, build_count_plan, run_radial_progress_task
from .shared.sampling import (
    sample_answer_count,
    sample_condition_values,
    sample_progress_frame,
    sample_threshold,
)
from .shared.state import DOMAIN


TASK_ID = "task_charts__radial_progress__progress_threshold_count"
AT_LEAST_QUERY_ID = "at_least_threshold_count"
BELOW_QUERY_ID = "below_threshold_count"
SUPPORTED_QUERY_IDS = (AT_LEAST_QUERY_ID, BELOW_QUERY_ID)
DEFAULT_QUERY_ID = AT_LEAST_QUERY_ID


def _build_plan(params, instance_seed, selected, probabilities):
    """Bind the threshold direction, then construct exactly that many matching widgets."""

    frame = sample_progress_frame(params, instance_seed=int(instance_seed))
    answer_count, answer_support, answer_probabilities = sample_answer_count(
        params,
        item_count=int(frame.item_count),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.answer_count.{selected}",
    )
    threshold = sample_threshold(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.threshold.{selected}",
    )
    if str(selected) == AT_LEAST_QUERY_ID:
        predicate = lambda value: int(value) >= int(threshold)
        threshold_phrase = f"at least {threshold}%"
        condition_kind = "progress_at_least_threshold"
    else:
        predicate = lambda value: int(value) < int(threshold)
        threshold_phrase = f"below {threshold}%"
        condition_kind = "progress_below_threshold"
    values, annotation_item_ids = sample_condition_values(
        item_count=int(frame.item_count),
        answer_count=int(answer_count),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.values.{selected}",
        target_predicate=predicate,
    )
    dataset = build_count_dataset_from_frame(
        frame=frame,
        values=values,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        answer_count=int(answer_count),
        answer_support=list(answer_support),
        answer_probabilities=dict(answer_probabilities),
        annotation_type="bbox_set",
        annotation_item_ids=tuple(annotation_item_ids),
        question_params={
            "threshold_value": int(threshold),
            "threshold_phrase": str(threshold_phrase),
            "count_condition": str(condition_kind),
        },
    )
    return build_count_plan(
        dataset=dataset,
        prompt_key=str(selected),
        scene_probabilities=dict(frame.scene_probabilities),
    )


@register_task
class ChartsRadialProgressThresholdCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "progress_threshold_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_radial_progress_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadialProgressThresholdCountTask", "SUPPORTED_QUERY_IDS"]
