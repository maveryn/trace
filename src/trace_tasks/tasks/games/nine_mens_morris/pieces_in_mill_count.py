"""Count Nine Men's Morris pieces that belong to mills."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.style import SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    NineMensMorrisObjectivePlan,
    morris_piece_count_attempt,
    resolve_morris_count_target,
    run_morris_registered_task,
)
from .shared.sampling import (
    ALL_MILL_PIECE_COUNT_SUPPORT,
    resolve_nine_mens_morris_visual_axes,
    sample_all_mill_piece_board,
)


TASK_ID = "task_games__nine_mens_morris__pieces_in_mill_count"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "all_pieces_in_mill_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
TARGET_SUPPORT_KEY = "all_pieces_in_mill_count_support"
TARGET_FALLBACK_SUPPORT = ALL_MILL_PIECE_COUNT_SUPPORT


def _prepare_pieces_in_mill_objective(
    instance_seed,
    task_params,
    _selected_branch,
    branch_probabilities,
    gen_defaults,
):
    """Resolve the exact mill-piece count and bind board construction."""

    del branch_probabilities
    axes = resolve_nine_mens_morris_visual_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        params=task_params,
        namespace="games.nine_mens_morris.pieces_in_mill",
        supported_style_variants=SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS,
    )
    target = resolve_morris_count_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        support_key=TARGET_SUPPORT_KEY,
        fallback_support=TARGET_FALLBACK_SUPPORT,
        namespace="games.nine_mens_morris.pieces_in_mill.target_answer",
    )

    def construct_attempt(rng, _resolved_axes):
        board_state = sample_all_mill_piece_board(
            rng=rng,
            target_answer=int(target.target_answer),
        )
        piece_ids = tuple(str(piece_id) for piece_id in board_state.all_piece_ids_in_mill)
        return morris_piece_count_attempt(
            board_state=board_state,
            prompt_key=PROMPT_QUERY_KEY,
            annotation_entity_ids=piece_ids,
            target=target,
            extra_query_params={
                "target_answer": int(target.target_answer),
            },
        )

    return NineMensMorrisObjectivePlan(
        axes=axes,
        attempt_namespace="games.nine_mens_morris.pieces_in_mill",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesNineMensMorrisAllPiecesInMillCountTask:
    """Count all visible pieces that belong to at least one mill."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = QUERY_ID
    _namespace = "games.nine_mens_morris.pieces_in_mill"
    _prepare_objective = staticmethod(_prepare_pieces_in_mill_objective)

    def generate(self, instance_seed: int, *, params: dict | None = None, max_attempts: int = 100) -> TaskOutput:
        """Generate an all-pieces-in-mill count instance."""

        return run_morris_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesNineMensMorrisAllPiecesInMillCountTask"]
