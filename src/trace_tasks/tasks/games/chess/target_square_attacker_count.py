"""Count chess pieces attacking a marked square."""

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
from .shared.sampling import resolve_player_color, sample_target_square_attacker_scene
from .shared.state import SCENE_ID


TASK_ID = "task_games__chess__target_square_attacker_count"
SUPPORTED_QUERY_IDS = (
    "king_square_attacker_count",
    "white_piece_attacks_target_square_count",
    "black_piece_attacks_target_square_count",
)
TARGET_ATTACKER_SUPPORT = (0, 1, 2, 3, 4)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _attacker_query(selected: str) -> tuple[str, bool]:
    """Return fixed attacker color or king-target mode for the selected query."""

    if str(selected) == "white_piece_attacks_target_square_count":
        return "white", False
    if str(selected) == "black_piece_attacks_target_square_count":
        return "black", False
    return "", True


def _prepare_target_square_attacker_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ChessObjectivePlan:
    """Bind marked-square attacker-count semantics for one selected query."""

    fixed_attacker_color, target_has_king = _attacker_query(str(query_id))
    def construct_sample(rng, axes, target_answer):
        attacker_color = fixed_attacker_color
        if bool(target_has_king):
            attacker_color = opponent(resolve_player_color(rng, params=task_params))
        return sample_target_square_attacker_scene(
            rng=rng,
            axes=axes,
            attacker_color=str(attacker_color),
            target_answer=int(target_answer),
            target_has_king=bool(target_has_king),
        )

    def attacker_color_from_sample(sample) -> str:
        return opponent(sample.player_color) if bool(target_has_king) else str(sample.player_color)

    def sample_query_params(sample) -> dict[str, Any]:
        attacker_color = attacker_color_from_sample(sample)
        return {
            "attacker_color": str(attacker_color),
            "target_has_king": bool(target_has_king),
        }

    def execution_fields(sample) -> dict[str, Any]:
        attacker_color = attacker_color_from_sample(sample)
        return {
            "attacker_color": str(attacker_color),
            "target_has_king": bool(target_has_king),
            "attacker_color_name": color_name(attacker_color),
        }

    return prepare_chess_bbox_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        query_id=str(query_id),
        gen_defaults=_GEN_DEFAULTS,
        support_key="target_square_attacker_count_support",
        fallback_support=TARGET_ATTACKER_SUPPORT,
        attempt_namespace="games.chess.target_square_attacker_count",
        construct_sample=construct_sample,
        badge_text="Marked king" if bool(target_has_king) else "Target square",
        witness_type="piece_set",
        build_query_params=sample_query_params,
        build_execution_extra=execution_fields,
    )


@register_task
class GamesChessTargetSquareAttackerCountTask:
    """Count pieces attacking the marked king or marked target square."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'spatial_relations')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    prepare_objective = staticmethod(_prepare_target_square_attacker_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a target-square attacker count."""

        return run_chess_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessTargetSquareAttackerCountTask"]
