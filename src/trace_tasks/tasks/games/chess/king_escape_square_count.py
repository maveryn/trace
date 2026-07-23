"""Count safe one-step destination squares for a marked chess king."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    ChessObjectivePlan,
    prepare_chess_bbox_count_objective,
    run_chess_public_entry,
)
from .shared.sampling import sample_king_escape_scene
from .shared.state import SCENE_ID


TASK_ID = "task_games__chess__king_escape_square_count"
QUERY_ID = "king_escape_square_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
KING_ESCAPE_SUPPORT = (0, 1, 2, 3, 4, 5)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_king_escape_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ChessObjectivePlan:
    """Bind marked-king safe-destination count semantics."""

    def construct_sample(rng, axes, target_answer):
        return sample_king_escape_scene(rng=rng, axes=axes, target_answer=int(target_answer))

    return prepare_chess_bbox_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        query_id=str(query_id),
        gen_defaults=_GEN_DEFAULTS,
        support_key="king_escape_square_count_support",
        fallback_support=KING_ESCAPE_SUPPORT,
        attempt_namespace="games.chess.king_escape_square_count",
        construct_sample=construct_sample,
        badge_text="Marked king",
        witness_type="cell_set",
    )


@register_task
class GamesChessKingEscapeSquareCountTask:
    """Count safe one-step destinations for the marked king."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = QUERY_ID
    prepare_objective = staticmethod(_prepare_king_escape_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a safe king-destination count."""

        return run_chess_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessKingEscapeSquareCountTask"]
