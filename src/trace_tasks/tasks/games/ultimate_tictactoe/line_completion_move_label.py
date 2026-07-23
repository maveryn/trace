"""Choose a winning or blocking move in one Ultimate Tic-Tac-Toe board."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import LABEL_JSON_EXAMPLES, bind_ultimate_payload, run_ultimate_lifecycle, sample_with_retry
from .shared.sampling import sample_local_line_tactic
from .shared.state import PLAYER_O, PLAYER_X


TASK_ID = "task_games__ultimate_tictactoe__line_completion_move_label"
NAMESPACE = "ultimate_tictactoe.line_completion_move_label"
QUERY_X_WIN_MOVE = "x_winning_move_label"
QUERY_O_WIN_MOVE = "o_winning_move_label"
QUERY_X_BLOCK_MOVE = "x_blocking_move_label"
QUERY_O_BLOCK_MOVE = "o_blocking_move_label"
SUPPORTED_QUERY_IDS = (
    QUERY_X_WIN_MOVE,
    QUERY_O_WIN_MOVE,
    QUERY_X_BLOCK_MOVE,
    QUERY_O_BLOCK_MOVE,
)


_TACTIC_BY_BRANCH = {
    QUERY_X_WIN_MOVE: (PLAYER_X, PLAYER_X, False),
    QUERY_O_WIN_MOVE: (PLAYER_O, PLAYER_O, False),
    QUERY_X_BLOCK_MOVE: (PLAYER_X, PLAYER_O, True),
    QUERY_O_BLOCK_MOVE: (PLAYER_O, PLAYER_X, True),
}


def _prepare_tactic_payload(
    instance_seed: int,
    params: Mapping[str, Any],
    branch_key: str,
    branch_probabilities: Mapping[str, float],
    style_variant: str,
    style_variant_probabilities: Mapping[str, float],
    max_attempts: int,
):
    # Bind one selected empty cell; supporting line cells remain trace-only.
    acting_player, threat_player, blocking = _TACTIC_BY_BRANCH[str(branch_key)]
    sample = sample_with_retry(
        public_id=TASK_ID,
        namespace=NAMESPACE,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        build_attempt=lambda rng: sample_local_line_tactic(
            rng,
                instance_seed=int(instance_seed),
                params=params,
                acting_player=str(acting_player),
                threat_player=str(threat_player),
                blocking=bool(blocking),
                namespace=f"{NAMESPACE}.{str(branch_key)}",
        ),
    )
    return bind_ultimate_payload(
        sample=sample,
        answer_gt=TypedValue(type="string", value=str(sample.answer)),
        prompt_key=str(branch_key),
        branch_probabilities=dict(branch_probabilities),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_variant_probabilities),
        examples=LABEL_JSON_EXAMPLES,
        annotation_kind="bbox",
        semantic_params={
            "acting_player": str(acting_player),
            "threat_player": str(threat_player),
            "blocking_tactic": bool(blocking),
        },
    )


@register_task
class GamesUltimateTicTacToeLineCompletionMoveLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_ultimate_lifecycle(
            public_id=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=QUERY_X_WIN_MOVE,
            namespace=NAMESPACE,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            prepare_payload=_prepare_tactic_payload,
        )
