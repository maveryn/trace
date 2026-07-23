"""Count horizontal or vertical blocks on a neutral sliding-block board."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SlidingBlockObjective, run_sliding_block_lifecycle
from .shared.annotations import block_bbox_set
from .shared.sampling import build_board_for_orientation_target, integer_support, select_target_from_support
from .shared.state import DOMAIN


TASK_ID = "task_games__sliding_block__block_orientation_count"
HORIZONTAL_QUERY_ID = "horizontal_block_count"
VERTICAL_QUERY_ID = "vertical_block_count"
SUPPORTED_QUERY_IDS = (HORIZONTAL_QUERY_ID, VERTICAL_QUERY_ID)
PROMPT_DEFAULT_PREFIX = "block_orientation_count"


def _orientation_for_query(query_id: str) -> str:
    """Map the semantic query id to the rectangular block orientation."""

    if str(query_id) == HORIZONTAL_QUERY_ID:
        return "horizontal"
    if str(query_id) == VERTICAL_QUERY_ID:
        return "vertical"
    raise ValueError(f"unsupported sliding-block orientation query: {query_id}")


def _prepare_orientation_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    _exit_side: str,
    selected_query: str,
) -> SlidingBlockObjective:
    """Bind the requested orientation count and every matching block bbox."""

    orientation = _orientation_for_query(str(selected_query))
    support = integer_support(
        task_params,
        min_key="block_orientation_count_min",
        max_key="block_orientation_count_max",
        fallback_min=1,
        fallback_max=6,
    )
    target = int(
        select_target_from_support(
            task_params,
            support=support,
            instance_seed=int(attempt_seed),
            namespace=f"{TASK_ID}.{orientation}.target_answer",
        )
    )
    dataset = build_board_for_orientation_target(
        params=task_params,
        instance_seed=int(attempt_seed),
        orientation=str(orientation),
        orientation_target=int(target),
    )
    answer_block_ids = [str(block_id) for block_id in dataset["orientation_block_ids"]]
    return SlidingBlockObjective(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(len(answer_block_ids))),
        answer_block_ids=answer_block_ids,
        render_mode="single_board",
        annotation_source="block_bboxes_px",
        prompt_query_key=str(selected_query),
        prompt_default_prefix=PROMPT_DEFAULT_PREFIX,
        build_annotation=lambda rendered: block_bbox_set(rendered, block_ids=answer_block_ids),
        trace_extra_params={
            "answer_support": [int(value) for value in support],
            "target_answer": int(target),
            "orientation": str(orientation),
            "orientation_count": int(len(answer_block_ids)),
        },
        execution_extra={
            "orientation": str(orientation),
            "orientation_block_ids": [str(block_id) for block_id in answer_block_ids],
            "orientation_count": int(len(answer_block_ids)),
        },
    )


@register_task
class GamesSlidingBlockOrientationCountTask:
    """Count rectangular sliding blocks by horizontal or vertical orientation."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate the block-orientation count task through task-owned hooks."""

        return run_sliding_block_lifecycle(
            namespace=TASK_ID,
            domain=DOMAIN,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=HORIZONTAL_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_orientation_objective,
        )


__all__ = ["GamesSlidingBlockOrientationCountTask"]
