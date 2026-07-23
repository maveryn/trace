from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_count_dataset_from_frame, build_count_plan, run_radial_progress_task
from .shared.sampling import (
    sample_answer_count,
    sample_condition_values,
    sample_progress_frame,
    sample_range_pair,
)
from .shared.state import DOMAIN


TASK_ID = "task_charts__radial_progress__progress_interval_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _build_plan(params, instance_seed, selected, probabilities):
    """Bind the interval bounds and build values with an exact in-range count."""

    frame = sample_progress_frame(params, instance_seed=int(instance_seed))
    answer_count, answer_support, answer_probabilities = sample_answer_count(
        params,
        item_count=int(frame.item_count),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.answer_count",
    )
    lower, upper = sample_range_pair(params, instance_seed=int(instance_seed), namespace=f"{TASK_ID}.range_pair")
    values, annotation_item_ids = sample_condition_values(
        item_count=int(frame.item_count),
        answer_count=int(answer_count),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.values",
        target_predicate=lambda value: int(lower) <= int(value) <= int(upper),
    )
    dataset = build_count_dataset_from_frame(
        frame=frame,
        values=values,
        branch_id=SINGLE_QUERY_ID,
        branch_probabilities=dict(probabilities),
        answer_count=int(answer_count),
        answer_support=list(answer_support),
        answer_probabilities=dict(answer_probabilities),
        annotation_type="bbox_set",
        annotation_item_ids=tuple(annotation_item_ids),
        question_params={
            "range_lower": int(lower),
            "range_upper": int(upper),
            "range_phrase": f"from {lower}% through {upper}%",
            "count_condition": "progress_within_range",
        },
    )
    return build_count_plan(
        dataset=dataset,
        prompt_key="within_range_count",
        scene_probabilities=dict(frame.scene_probabilities),
    )


@register_task
class ChartsRadialProgressIntervalCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "progress_interval_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_radial_progress_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadialProgressIntervalCountTask", "SUPPORTED_QUERY_IDS"]
