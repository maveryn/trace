"""Count empty destinations for one marked chess piece."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    ChessObjectivePlan,
    prepare_marked_piece_destination_family_objective,
    run_chess_public_entry,
)
from .shared.state import SCENE_ID


TASK_ID = "task_games__chess__marked_piece_destination_count"
QUERY_ID = "marked_piece_destination_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
MARKED_DESTINATION_SUPPORT = (1, 2, 3, 4, 5, 6)
MARKED_PIECE_KIND_SUPPORT = ("knight", "bishop", "rook", "queen")
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_marked_piece_destination_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ChessObjectivePlan:
    """Bind empty destination counting while excluding capture squares."""

    if str(query_id) != QUERY_ID:
        raise ValueError(f"unsupported chess marked-piece destination query: {query_id}")
    return prepare_marked_piece_destination_family_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        query_id=str(query_id),
        gen_defaults=_GEN_DEFAULTS,
        support_key="marked_piece_destination_count_support",
        fallback_support=MARKED_DESTINATION_SUPPORT,
        marked_piece_kind_support=MARKED_PIECE_KIND_SUPPORT,
        destination_mode="move",
        query_destination_mode="empty",
        attempt_namespace="games.chess.marked_piece_destination_count",
    )


@register_task
class GamesChessMarkedPieceDestinationCountTask:
    """Count empty one-move destinations for the marked chess piece."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    prepare_objective = staticmethod(_prepare_marked_piece_destination_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a marked-piece empty-destination count."""

        return run_chess_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessMarkedPieceDestinationCountTask"]
