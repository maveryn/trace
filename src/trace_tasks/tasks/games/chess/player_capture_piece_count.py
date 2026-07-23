"""Count opponent chess pieces capturable by one side."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.games.shared.piece_board_rules import color_name, opponent
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    ChessObjectivePlan,
    prepare_chess_bbox_count_objective,
    run_chess_public_entry,
)
from .shared.sampling import resolve_player_color, sample_player_capture_scene
from .shared.state import SCENE_ID


TASK_ID = "task_games__chess__player_capture_piece_count"
QUERY_ID = "player_capture_piece_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PLAYER_CAPTURE_SUPPORT = (1, 2, 3, 4, 5, 6)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_player_capture_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ChessObjectivePlan:
    """Bind side-to-move capture-count semantics for one Chess board."""

    def construct_sample(rng, axes, target_answer):
        player_color = resolve_player_color(rng, params=task_params)
        return sample_player_capture_scene(
            rng=rng,
            axes=axes,
            player_color=str(player_color),
            target_answer=int(target_answer),
        )

    def prompt_slots(sample) -> dict[str, str]:
        return {
            "player_color_name": color_name(sample.player_color),
            "opponent_color_name": color_name(opponent(sample.player_color)),
        }

    return prepare_chess_bbox_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        query_id=str(query_id),
        gen_defaults=_GEN_DEFAULTS,
        support_key="player_capture_piece_count_support",
        fallback_support=PLAYER_CAPTURE_SUPPORT,
        attempt_namespace="games.chess.player_capture_piece_count",
        construct_sample=construct_sample,
        badge_builder=lambda sample: f"{color_name(sample.player_color)} to move",
        witness_type="piece_set",
        build_prompt_dynamic_slots=prompt_slots,
        build_query_params=lambda sample: {"player_color": str(sample.player_color)},
    )


@register_task
class GamesChessPlayerCapturePieceCountTask:
    """Count opponent pieces that the named side can capture immediately."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = QUERY_ID
    prepare_objective = staticmethod(_prepare_player_capture_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a side capture-count scene."""

        return run_chess_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessPlayerCapturePieceCountTask"]
