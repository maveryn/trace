"""Count visible chess pieces of one named kind."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import ChessObjectivePlan, prepare_chess_piece_count_objective, run_chess_public_entry
from .shared.state import SCENE_ID


TASK_ID = "task_games__chess__piece_kind_count"
QUERY_ID = "piece_kind_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_piece_kind_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ChessObjectivePlan:
    """Bind visible piece-kind count semantics without color filtering."""

    return prepare_chess_piece_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        query_id=str(query_id),
        gen_defaults=_GEN_DEFAULTS,
        attempt_namespace="games.chess.piece_kind_count",
        include_color=False,
    )


@register_task
class GamesChessPieceKindCountTask:
    """Count all visible pieces of the named kind."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = QUERY_ID
    prepare_objective = staticmethod(_prepare_piece_kind_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a kind-only visible-piece count."""

        return run_chess_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessPieceKindCountTask"]
