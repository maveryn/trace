"""Sokoban push stand-cell label task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SokobanLifecycleTask, SokobanObjective, run_sokoban_lifecycle
from .shared.annotations import option_cell_bbox
from .shared.sampling import sample_push_stand_cell_dataset
from .shared.state import PUSH_STAND_OPTION_COUNT_SUPPORT


TASK_ID = "task_games__sokoban__push_stand_cell_label"
QUERY_ID = "push_stand_cell_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_push_stand_cell_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    public_query: str,
) -> SokobanObjective:
    """Construct a straight-push board and bind the required stand cell."""

    if str(public_query) != QUERY_ID:
        raise ValueError(f"unsupported Sokoban push stand-cell query: {public_query}")
    dataset = sample_push_stand_cell_dataset(
        params=task_params,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    answer_label = str(dataset["answer_option_label"])
    option_specs = tuple(dict(option) for option in dataset.get("option_specs", []))
    color_label = str(dataset["target_color_label"])
    return SokobanObjective(
        dataset=dict(dataset),
        answer_gt=TypedValue(type="option_letter", value=answer_label),
        prompt_query_key=QUERY_ID,
        object_description_key="object_description_push_stand_cell_label",
        annotation_hint_key="annotation_hint_selected_stand_cell_bbox",
        json_example_key="json_example_selected_stand_cell_bbox",
        annotation_source="cell_bboxes_px",
        option_count_support=[int(value) for value in PUSH_STAND_OPTION_COUNT_SUPPORT],
        option_count_probabilities={"4": 1.0},
        build_annotation=lambda rendered: option_cell_bbox(
            rendered,
            option_specs=option_specs,
            answer_label=answer_label,
        ),
        answer_hint_key="answer_hint_option_letter_stand_cell",
        json_example_answer_only_key="json_example_answer_only_option_label",
        prompt_dynamic_values={"target_color_label": color_label},
        trace_extra_params={
            "target_color_label": color_label,
            "target_color_name": str(dataset["target_color_name"]),
        },
        execution_extra={
            "target_color_label": color_label,
            "target_color_name": str(dataset["target_color_name"]),
        },
    )


@register_task
class GamesSokobanPushStandCellLabelTask(SokobanLifecycleTask):
    """Select where the player stands to push a box straight to its goal."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'state_update')
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_sokoban_lifecycle(
            namespace=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_push_stand_cell_objective,
        )


__all__ = ["GamesSokobanPushStandCellLabelTask"]
