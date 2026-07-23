"""Layer piece-count task for 3D Tic-Tac-Toe boards."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from ._lifecycle import TicTacToe3DObjectivePlan, run_tic_tac_toe_3d_lifecycle
from .shared.defaults import GEN_DEFAULTS
from .shared.prompts import format_json_examples
from .shared.sampling import sample_layer_piece_count_scene

TASK_ID = "task_games__tic_tac_toe_3d__layer_piece_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "piece_count_in_layer"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
TARGET_PLAYERS = ("X", "O")
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation=[[212, 318], [285, 391], [358, 318]],
    answer=3,
)


def _prepare_layer_piece_count_objective(
    _instance_seed: int,
    _params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> TicTacToe3DObjectivePlan:
    """Bind the target mark and layer count for the public count task."""

    if str(selected_branch) != QUERY_ID:
        raise ValueError(
            f"unsupported 3D Tic-Tac-Toe layer-count branch: {selected_branch}"
        )
    target_player, target_player_probabilities = resolve_games_named_axis(
        instance_seed=int(_instance_seed),
        params=_params,
        gen_defaults=GEN_DEFAULTS,
        namespace=f"{TASK_ID}.target_player",
        explicit_key="target_player",
        weights_key="target_player_weights",
        balance_flag_key="balanced_target_player_sampling",
        supported_variants=TARGET_PLAYERS,
    )

    def construct_attempt(rng, axes):
        return sample_layer_piece_count_scene(
            rng=rng,
            target_player=target_player,
            target_layer=str(axes.target_layer),
            target_answer=int(axes.target_answer),
        )

    return TicTacToe3DObjectivePlan(
        attempt_namespace=f"games.tic_tac_toe_3d.layer_count.{target_player}",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
        annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
        annotation_kind="cell_point_set",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        construct_attempt=construct_attempt,
        prompt_dynamic_slots={"target_player": str(target_player)},
        trace_params={
            "target_player": target_player,
            "target_player_probabilities": dict(target_player_probabilities),
            "layer_count_branch": str(selected_branch),
            "layer_count_branch_probabilities": dict(branch_probabilities),
        },
    )


@register_task
class GamesTicTacToe3DLayerPieceCountTask:
    """Count target-player pieces in one named 3D Tic-Tac-Toe layer."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any] | None = None,
        max_attempts: int = 100,
    ):
        task_params = dict(params or {})
        output = run_tic_tac_toe_3d_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_layer_piece_count_objective,
        )
        return output


__all__ = ["GamesTicTacToe3DLayerPieceCountTask", "TASK_ID"]
