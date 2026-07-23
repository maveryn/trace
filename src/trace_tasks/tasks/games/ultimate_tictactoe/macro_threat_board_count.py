"""Count small boards containing an immediate local win."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import COUNT_JSON_EXAMPLES, bind_ultimate_payload, run_ultimate_lifecycle, sample_with_retry
from .shared.sampling import sample_macro_threat_board_count
from .shared.state import PLAYER_O, PLAYER_X


TASK_ID = "task_games__ultimate_tictactoe__macro_threat_board_count"
NAMESPACE = "ultimate_tictactoe.macro_threat_board_count"
QUERY_X_IMMEDIATE_WIN_BOARD_COUNT = "x_immediate_win_board_count"
QUERY_O_IMMEDIATE_WIN_BOARD_COUNT = "o_immediate_win_board_count"
SUPPORTED_QUERY_IDS = (
    QUERY_X_IMMEDIATE_WIN_BOARD_COUNT,
    QUERY_O_IMMEDIATE_WIN_BOARD_COUNT,
)
_PLAYER_BY_BRANCH = {
    QUERY_X_IMMEDIATE_WIN_BOARD_COUNT: PLAYER_X,
    QUERY_O_IMMEDIATE_WIN_BOARD_COUNT: PLAYER_O,
}


def _prepare_macro_payload(
    instance_seed: int,
    params: Mapping[str, Any],
    branch_key: str,
    branch_probabilities: Mapping[str, float],
    style_variant: str,
    style_variant_probabilities: Mapping[str, float],
    max_attempts: int,
):
    player = _PLAYER_BY_BRANCH[str(branch_key)]
    sample = sample_with_retry(
        public_id=TASK_ID,
        namespace=NAMESPACE,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        build_attempt=lambda rng: sample_macro_threat_board_count(
            rng,
                instance_seed=int(instance_seed),
                params=params,
                player=str(player),
                namespace=f"{NAMESPACE}.{str(branch_key)}",
                branch_count=len(SUPPORTED_QUERY_IDS),
        ),
    )
    return bind_ultimate_payload(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_key=str(branch_key),
        branch_probabilities=dict(branch_probabilities),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_variant_probabilities),
        examples=COUNT_JSON_EXAMPLES,
        semantic_params={"threat_player": str(player)},
    )


@register_task
class GamesUltimateTicTacToeMacroThreatBoardCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_ultimate_lifecycle(
            public_id=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=QUERY_X_IMMEDIATE_WIN_BOARD_COUNT,
            namespace=NAMESPACE,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            prepare_payload=_prepare_macro_payload,
        )
