from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_label_dataset_from_frame, build_label_plan, run_radial_progress_task
from .shared.sampling import (
    sample_distinct_values,
    sample_progress_frame,
)
from .shared.state import DOMAIN


TASK_ID = "task_charts__radial_progress__extremum_remaining_label"
HIGHEST_REMAINING_QUERY_ID = "highest_remaining_label"
LOWEST_REMAINING_QUERY_ID = "lowest_remaining_label"
SUPPORTED_QUERY_IDS = (HIGHEST_REMAINING_QUERY_ID, LOWEST_REMAINING_QUERY_ID)
DEFAULT_QUERY_ID = HIGHEST_REMAINING_QUERY_ID


def _build_plan(params, instance_seed, selected, probabilities):
    """Bind remaining-progress extremum direction and annotate the selected widget."""

    frame = sample_progress_frame(params, instance_seed=int(instance_seed))
    values = sample_distinct_values(
        params,
        item_count=int(frame.item_count),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.values.{selected}",
    )
    if str(selected) == HIGHEST_REMAINING_QUERY_ID:
        target_index = min(range(int(frame.item_count)), key=lambda index: int(values[index]))
        extremum_phrase = "the most remaining progress"
        extremum_kind = "highest_remaining"
    else:
        target_index = max(range(int(frame.item_count)), key=lambda index: int(values[index]))
        extremum_phrase = "the least remaining progress"
        extremum_kind = "lowest_remaining"
    answer = str(frame.labels[int(target_index)])
    dataset = build_label_dataset_from_frame(
        frame=frame,
        values=values,
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        answer=str(answer),
        annotation_type="bbox",
        annotation_item_ids=(f"i{int(target_index)}",),
        question_params={
            "extremum_phrase": str(extremum_phrase),
            "remaining_extremum": str(extremum_kind),
            "target_value": int(values[int(target_index)]),
            "target_remaining": int(100 - int(values[int(target_index)])),
        },
    )
    return build_label_plan(
        dataset=dataset,
        prompt_key=str(selected),
        scene_probabilities=dict(frame.scene_probabilities),
    )


@register_task
class ChartsRadialProgressExtremumRemainingLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "extremum_remaining_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed, *, params, max_attempts):
        return run_radial_progress_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsRadialProgressExtremumRemainingLabelTask", "SUPPORTED_QUERY_IDS"]
