"""Winning-move option label task for 3D Tic-Tac-Toe boards."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    prepare_option_move_objective_from_semantics,
    run_tic_tac_toe_3d_lifecycle,
)
from .shared.prompts import format_json_examples
from .shared.sampling import sample_winning_move_scene
from .shared.state import OPTION_LABELS

TASK_ID = "task_games__tic_tac_toe_3d__winning_move_cell_label"
QUERY_X_WIN_MOVE = "x_winning_move_label"
QUERY_O_WIN_MOVE = "o_winning_move_label"
SUPPORTED_QUERY_IDS = (QUERY_X_WIN_MOVE, QUERY_O_WIN_MOVE)
TARGET_PLAYER_BY_QUERY = {
    QUERY_X_WIN_MOVE: "X",
    QUERY_O_WIN_MOVE: "O",
}
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation=[[320, 190, 380, 250], [250, 190, 310, 250], [390, 190, 450, 250]],
    answer="B",
)


def _prepare_winning_move_objective(
    _instance_seed: int,
    _params: Mapping[str, Any],
    winning_branch: str,
    winning_branch_weights: Mapping[str, float],
):
    """Bind the target player and answer option for the winning-move task."""

    return prepare_option_move_objective_from_semantics(
        selected_branch=str(winning_branch),
        branch_probabilities=winning_branch_weights,
        target_player_by_branch=TARGET_PLAYER_BY_QUERY,
        sample_scene=sample_winning_move_scene,
        attempt_prefix="winning_move",
        branch_trace_key="winning_move_branch",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        option_labels=OPTION_LABELS,
    )


@register_task
class GamesTicTacToe3DWinningMoveCellLabelTask:
    """Choose the labeled cell that completes a 3D Tic-Tac-Toe line."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params=None, max_attempts=100):
        task_params = dict(params or {})
        output = run_tic_tac_toe_3d_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_X_WIN_MOVE,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_winning_move_objective,
        )
        return output


__all__ = ["GamesTicTacToe3DWinningMoveCellLabelTask", "TASK_ID"]
