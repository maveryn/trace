"""Sokoban closest matching box-goal label task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SokobanLifecycleTask, SokobanObjective, run_sokoban_lifecycle
from .shared.annotations import option_cell_bbox
from .shared.sampling import (
    sample_closest_box_goal_dataset,
    select_box_goal_distance_option_count,
)


TASK_ID = "task_games__sokoban__closest_box_goal_label"
QUERY_ID = "closest_box_goal_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_closest_box_goal_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    public_query: str,
) -> SokobanObjective:
    """Construct a labeled box-goal board and bind the unique closest box."""

    if str(public_query) != QUERY_ID:
        raise ValueError(f"unsupported Sokoban closest box-goal query: {public_query}")
    option_count, option_support, option_probabilities = select_box_goal_distance_option_count(
        task_params,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    dataset = sample_closest_box_goal_dataset(
        option_count=int(option_count),
        params=task_params,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    answer_label = str(dataset["answer_option_label"])
    option_specs = tuple(dict(option) for option in dataset.get("option_specs", []))
    return SokobanObjective(
        dataset=dict(dataset),
        answer_gt=TypedValue(type="option_letter", value=answer_label),
        prompt_query_key=QUERY_ID,
        object_description_key="object_description_closest_box_goal_label",
        annotation_hint_key="annotation_hint_selected_box_bbox",
        json_example_key="json_example_selected_box_bbox",
        annotation_source="cell_bboxes_px",
        option_count_support=[int(value) for value in option_support],
        option_count_probabilities=dict(option_probabilities),
        build_annotation=lambda rendered: option_cell_bbox(
            rendered,
            option_specs=option_specs,
            answer_label=answer_label,
        ),
        answer_hint_key="answer_hint_option_letter",
        json_example_answer_only_key="json_example_answer_only_option_label",
        trace_extra_params={
            "distance_kind": "manhattan",
            "option_count_support": [int(value) for value in option_support],
            "option_count_probabilities": dict(option_probabilities),
        },
        execution_extra={
            "distance_kind": "manhattan",
            "answer_box_label": answer_label,
        },
    )


@register_task
class GamesSokobanClosestBoxGoalLabelTask(SokobanLifecycleTask):
    """Select the labeled box closest to its matching colored goal dot."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'matching')
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_sokoban_lifecycle(
            namespace=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_closest_box_goal_objective,
        )


__all__ = ["GamesSokobanClosestBoxGoalLabelTask"]
