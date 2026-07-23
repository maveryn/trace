"""Count captures for one marked chess-variant piece."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.chess_variant._lifecycle import (
    ChessVariantObjectivePlan,
    prepare_marked_piece_count_objective,
    run_chess_variant_public_entry,
)
from trace_tasks.tasks.games.chess_variant.shared.prompts import prompt_defaults
from trace_tasks.tasks.games.chess_variant.shared.state import ChessVariantSceneAxes
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_games__chess_variant__marked_piece_capture_count"
QUERY_ID = "marked_piece_capture_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
MARKED_CAPTURE_SUPPORT = (0, 1, 2, 3, 4)


def _prepare_marked_piece_capture_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    axes: ChessVariantSceneAxes,
    query_id: str,
) -> ChessVariantObjectivePlan:
    """Bind capture counting while excluding empty movement squares."""

    if str(query_id) != QUERY_ID:
        raise ValueError(f"unsupported chess-variant capture query: {query_id}")
    defaults = prompt_defaults()
    return prepare_marked_piece_count_objective(
        task_id=TASK_ID,
        instance_seed=int(instance_seed),
        task_params=task_params,
        axes=axes,
        query_id=str(query_id),
        support_key="marked_piece_capture_count_support",
        fallback_support=MARKED_CAPTURE_SUPPORT,
        destination_mode="capture",
        attempt_namespace="games.chess_variant.marked_piece_capture_count",
        landing_rule_text=str(defaults["capture_landing_rule_text"]),
        example_answer=2,
    )


@register_task
class GamesChessVariantMarkedPieceCaptureCountTask:
    """Count opponent pieces capturable by the marked variant piece."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = QUERY_ID
    prepare_objective = staticmethod(_prepare_marked_piece_capture_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate a marked-piece capture count."""

        return run_chess_variant_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessVariantMarkedPieceCaptureCountTask"]
