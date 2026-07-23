"""Count full or one-gap rows in a static Tetris board."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import TetrisObjectivePlan, resolve_tetris_integer_target, run_tetris_lifecycle
from .shared.defaults import DEFAULTS
from .shared.prompts import format_json_examples
from .shared.rendering import RENDER_MODE_STATIC_BOARD
from .shared.sampling import build_row_occupancy_sample


TASK_ID = "task_games__tetris__row_occupancy_status_count"
SUPPORTED_QUERY_IDS = ("full_row_count", "one_gap_row_count")
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation=[[120, 360, 520, 392], [120, 430, 520, 462]],
    answer=2,
)


@dataclass(frozen=True)
class RowOccupancyRequest:
    """Task-local operands for counting rows by occupancy status."""

    prompt_query_key: str
    row_status: str
    support_key: str = "row_occupancy_status_count_support"
    explicit_key: str = "target_row_count"


def _row_request_from_branch(selected_query: str) -> RowOccupancyRequest:
    """Convert the public row-count query into a concrete occupancy predicate."""

    if str(selected_query) == "full_row_count":
        return RowOccupancyRequest(prompt_query_key="full_row_count", row_status="full")
    if str(selected_query) == "one_gap_row_count":
        return RowOccupancyRequest(prompt_query_key="one_gap_row_count", row_status="one_gap")
    raise ValueError(f"unsupported Tetris row-occupancy query: {selected_query}")


def _resolve_target_row_total(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    request: RowOccupancyRequest,
) -> tuple[int, dict[str, float], tuple[int, ...]]:
    """Sample the exact number of rows that should satisfy the row predicate."""

    return resolve_tetris_integer_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key=str(request.support_key),
        explicit_key=str(request.explicit_key),
        fallback_support=DEFAULTS.row_occupancy_status_count_support,
        namespace=f"{TASK_ID}.{request.row_status}.target_row_count",
    )


def _prepare_row_occupancy_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
    _axes,
) -> TetrisObjectivePlan:
    """Resolve the requested row status and exact target row count."""

    request = _row_request_from_branch(str(selected_query))
    target_row_count, target_probabilities, target_support = _resolve_target_row_total(
        instance_seed=int(instance_seed),
        task_params=task_params,
        request=request,
    )

    def construct_attempt(rng, axes):
        return build_row_occupancy_sample(
            rng,
            scene_variant=str(axes.scene_variant),
            board_rows=int(axes.board_rows),
            board_cols=int(axes.board_cols),
            row_status=str(request.row_status),
            target_row_count=int(target_row_count),
        )

    return TetrisObjectivePlan(
        attempt_namespace=f"games.tetris.row_status.{request.row_status}.{int(target_row_count)}",
        prompt_query_key=str(request.prompt_query_key),
        answer_hint_key="answer_hint_row_occupancy_status_count",
        annotation_hint_key="annotation_hint_row_occupancy_status_count",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        render_mode=RENDER_MODE_STATIC_BOARD,
        query_params={
            "row_occupancy_status": str(request.row_status),
            "target_row_count": int(target_row_count),
            "target_row_count_support": [int(value) for value in target_support],
            "target_row_count_probabilities": dict(target_probabilities),
            "row_occupancy_query_probabilities": dict(query_probabilities),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesTetrisRowOccupancyStatusCountTask:
    """Count board rows matching a selected occupancy status."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_tetris_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_row_occupancy_objective,
        )


__all__ = ["GamesTetrisRowOccupancyStatusCountTask"]
