from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SlidingBlockObjective, run_sliding_block_lifecycle
from .shared.annotations import block_bbox_set
from .shared.sampling import build_exit_path_board, integer_support, select_target_from_support
from .shared.state import DOMAIN


TASK_ID = "task_games__sliding_block__sliding_block_blocker_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
PROMPT_QUERY_KEY = "blocker_count"


def _prepare_blocker_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    exit_side: str,
    _selected_query: str,
) -> SlidingBlockObjective:
    support = integer_support(task_params, min_key="blocker_count_min", max_key="blocker_count_max", fallback_min=0, fallback_max=5)
    target = int(
        select_target_from_support(
            task_params,
            support=support,
            instance_seed=int(attempt_seed),
            namespace=f"{TASK_ID}.target_answer",
        )
    )
    dataset = build_exit_path_board(
        params=task_params,
        instance_seed=int(attempt_seed),
        exit_side=str(exit_side),
        blocker_target=int(target),
        namespace=f"{TASK_ID}.board",
    )
    answer_block_ids = [str(block_id) for block_id in dataset["blocking_block_ids"]]
    return SlidingBlockObjective(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(len(answer_block_ids))),
        answer_block_ids=answer_block_ids,
        render_mode="exit_path",
        annotation_source="block_bboxes_px",
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_default_prefix=PROMPT_QUERY_KEY,
        build_annotation=lambda rendered: block_bbox_set(rendered, block_ids=answer_block_ids),
        trace_extra_params={"answer_support": [int(value) for value in support], "target_answer": int(target)},
        execution_extra={"blocker_count": int(len(answer_block_ids))},
    )


@register_task
class GamesSlidingBlockBlockerCountTask:
    """Count rectangular blocks currently occupying the red target block's exit path."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate the path-blocker count task through task-owned objective hooks."""

        return run_sliding_block_lifecycle(
            namespace=TASK_ID,
            domain=DOMAIN,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=DEFAULT_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_blocker_objective,
        )


__all__ = ["GamesSlidingBlockBlockerCountTask"]
