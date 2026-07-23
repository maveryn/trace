"""Count non-target blocks with at least one legal slide."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SlidingBlockObjective, run_sliding_block_lifecycle
from .shared.annotations import block_bbox_set
from .shared.sampling import build_board_for_movable_target, integer_support, select_target_from_support
from .shared.state import DOMAIN


TASK_ID = "task_games__sliding_block__movable_block_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
PROMPT_QUERY_KEY = "movable_block_count"


def _prepare_movable_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    _exit_side: str,
    _selected_query: str,
) -> SlidingBlockObjective:
    """Bind the target movable-block count and every counted block bbox."""

    support = integer_support(
        task_params,
        min_key="movable_block_count_min",
        max_key="movable_block_count_max",
        fallback_min=4,
        fallback_max=9,
    )
    target = int(
        select_target_from_support(
            task_params,
            support=support,
            instance_seed=int(attempt_seed),
            namespace=f"{TASK_ID}.target_answer",
        )
    )
    dataset = build_board_for_movable_target(
        params=task_params,
        instance_seed=int(attempt_seed),
        movable_target=int(target),
    )
    answer_block_ids = [str(block_id) for block_id in dataset["movable_block_ids"]]
    return SlidingBlockObjective(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(len(answer_block_ids))),
        answer_block_ids=answer_block_ids,
        render_mode="single_board",
        annotation_source="block_bboxes_px",
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_default_prefix=PROMPT_QUERY_KEY,
        build_annotation=lambda rendered: block_bbox_set(rendered, block_ids=answer_block_ids),
        trace_extra_params={
            "answer_support": [int(value) for value in support],
            "target_answer": int(target),
            "movable_block_count": int(len(answer_block_ids)),
        },
        execution_extra={"movable_block_count": int(len(answer_block_ids))},
    )


@register_task
class GamesSlidingBlockMovableBlockCountTask:
    """Count labeled blocks that can slide at least one cell."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate the movable-block count task through task-owned objective hooks."""

        return run_sliding_block_lifecycle(
            namespace=TASK_ID,
            domain=DOMAIN,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=DEFAULT_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_movable_objective,
        )


__all__ = ["GamesSlidingBlockMovableBlockCountTask"]
