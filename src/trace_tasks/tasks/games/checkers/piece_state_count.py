"""Count visible Checkers pieces by color and board-edge state."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from ._lifecycle import (
    CheckersObjectivePlan,
    checkers_target_trace_params,
    resolve_checkers_task_target,
    run_checkers_lifecycle,
)
from .shared.rules import BLACK, RED
from .shared.sampling import (
    resolve_task_occupied_range,
    sample_piece_state_scene,
    scene_object_description,
)
from .shared.state import SCENE_ID, SampledCheckersScene

TASK_ID = "task_games__checkers__piece_state_count"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PLAYER_BY_NAME = {"red": RED, "black": BLACK}
PIECE_STATE_KIND_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "all": {
        "edge_only": False,
        "scope_phrase": "are on the board",
    },
    "edge": {
        "edge_only": True,
        "scope_phrase": "are on the outer border row or column",
    },
}
PIECE_STATE_COUNT_SUPPORT = (0, 1, 2, 3, 4, 5, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _prepare_piece_state_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> CheckersObjectivePlan:
    """Bind the selected color/perimeter piece-state query to exact-count sampling."""

    if str(query_id) != QUERY_ID:
        raise ValueError(f"unsupported Checkers piece-state query: {query_id}")
    target_player_name, target_player_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{TASK_ID}.target_player",
        explicit_key="target_player",
        weights_key="target_player_weights",
        balance_flag_key="balanced_target_player_sampling",
        supported_variants=tuple(PLAYER_BY_NAME),
    )
    piece_state_kind, piece_state_kind_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{TASK_ID}.piece_state_kind",
        explicit_key="piece_state_kind",
        weights_key="piece_state_kind_weights",
        balance_flag_key="balanced_piece_state_kind_sampling",
        supported_variants=tuple(PIECE_STATE_KIND_SETTINGS),
    )
    settings = PIECE_STATE_KIND_SETTINGS[str(piece_state_kind)]
    player = int(PLAYER_BY_NAME[str(target_player_name)])
    edge_only = bool(settings["edge_only"])
    target = resolve_checkers_task_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="piece_state_count_support",
        fallback_support=PIECE_STATE_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.target_answer",
    )
    occupied_range = resolve_task_occupied_range(
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
    )

    def construct_attempt(rng, axes):
        return sample_piece_state_scene(
            rng=rng,
            axes=axes,
            params=task_params,
            target_answer=int(target.target_answer),
            player=int(player),
            edge_only=bool(edge_only),
            occupied_range=occupied_range,
        )

    def prompt_slots(sample: SampledCheckersScene) -> dict[str, str]:
        return {
            "object_description": scene_object_description(str(sample.scene_variant))
        }

    target_player = "red" if int(player) == int(RED) else "black"
    return CheckersObjectivePlan(
        attempt_namespace=f"games.checkers.piece_state_count.{target_player}.{str(piece_state_kind)}",
        prompt_query_key="piece_state_count",
        target=target,
        query_params={
            **checkers_target_trace_params(target),
            "target_player": str(target_player),
            "target_player_probabilities": dict(target_player_probabilities),
            "piece_state_kind": str(piece_state_kind),
            "piece_state_kind_probabilities": dict(piece_state_kind_probabilities),
            "edge_only": bool(edge_only),
        },
        prompt_dynamic_slots={
            "target_player_name": str(target_player),
            "piece_state_scope_phrase": str(settings["scope_phrase"]),
        },
        execution_extra={
            "target_player": str(target_player),
            "piece_state_kind": str(piece_state_kind),
            "edge_only": bool(edge_only),
        },
        construct_attempt=construct_attempt,
        build_prompt_dynamic_slots=prompt_slots,
    )


@register_task
class GamesCheckersPieceStateCountTask:
    """Count visible Checkers pieces by color and board-edge state."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a color/perimeter piece count with point annotations."""

        return run_checkers_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_piece_state_objective,
        )


__all__ = ["GamesCheckersPieceStateCountTask"]
