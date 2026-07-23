from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SlidingBlockObjective, run_sliding_block_lifecycle
from .shared.annotations import source_board_and_option_bbox_map
from .shared.sampling import build_board_for_move_result, move_result_option_labels, select_target_from_support
from .shared.state import DOMAIN


TASK_ID = "task_games__sliding_block__sliding_block_move_result_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
PROMPT_QUERY_KEY = "move_result_label"


def _prepare_result_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    _exit_side: str,
    _selected_query: str,
) -> SlidingBlockObjective:
    """Bind one final-board option label after constructing a valid slide sequence."""

    option_labels, option_count_probabilities = move_result_option_labels(task_params, instance_seed=int(attempt_seed))
    correct_label = str(
        select_target_from_support(
            task_params,
            support=option_labels,
            instance_seed=int(attempt_seed),
            namespace=f"{TASK_ID}.answer_label",
        )
    )
    dataset = build_board_for_move_result(
        params=task_params,
        instance_seed=int(attempt_seed),
        option_labels=option_labels,
        correct_option_label=str(correct_label),
    )
    moved_block_ids = [str(block_id) for block_id in dataset["moved_block_ids"]]
    correct_option_id = str(dataset["correct_option_id"])
    return SlidingBlockObjective(
        dataset=dataset,
        answer_gt=TypedValue(type="option_letter", value=str(correct_label)),
        answer_block_ids=moved_block_ids,
        render_mode="final_board_options",
        annotation_source="board_bbox_px+option_panel_bboxes_px",
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_default_prefix=PROMPT_QUERY_KEY,
        build_annotation=lambda rendered: source_board_and_option_bbox_map(
            rendered,
            option_id=correct_option_id,
        ),
        prompt_dynamic_values={"move_sequence_description": str(dataset["move_sequence_description"])},
        trace_extra_params={
            "answer_support": [str(value) for value in option_labels],
            "target_answer": str(correct_label),
            "option_count_probabilities": dict(option_count_probabilities),
        },
        execution_extra={"correct_option_label": str(correct_label)},
    )


@register_task
class GamesSlidingBlockMoveResultLabelTask:
    """Choose the option panel matching the board after applying the shown slides."""

    task_id = TASK_ID
    reasoning_operations = ('state_update', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate the final-board option task through a local objective hook."""

        return run_sliding_block_lifecycle(
            namespace=TASK_ID,
            domain=DOMAIN,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=DEFAULT_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_result_objective,
        )


__all__ = ["GamesSlidingBlockMoveResultLabelTask"]
