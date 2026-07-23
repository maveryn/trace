"""Count black or white frontier discs on a Reversi board."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from ._lifecycle import ObjectiveReversiPlan, run_reversi_lifecycle
from .shared.rules import player_name
from .shared.defaults import DEFAULTS
from .shared.sampling import resolve_reversi_target_axis, sample_frontier_disc_scene
from .shared.state import BLACK, SCENE_ID, SCENE_NAMESPACE, WHITE

TASK_ID = "task_games__reversi__frontier_disc_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "frontier_disc_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
DEFAULT_QUERY_ID = QUERY_ID
PLAYER_BY_NAME = {"black": BLACK, "white": WHITE}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _prepare_frontier_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    query_id: str,
) -> ObjectiveReversiPlan:
    """Prepare target support and sampler hooks for frontier-disc counts."""

    if str(query_id) != QUERY_ID:
        raise ValueError(f"unsupported Reversi frontier query: {query_id}")
    query_player_name, query_player_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{SCENE_NAMESPACE}.frontier.target_player",
        explicit_key="target_player",
        weights_key="target_player_weights",
        balance_flag_key="balanced_target_player_sampling",
        supported_variants=tuple(PLAYER_BY_NAME),
    )
    query_player = int(PLAYER_BY_NAME[str(query_player_name)])
    target_axis = resolve_reversi_target_axis(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="frontier_disc_count_support",
        fallback_support=DEFAULTS.frontier_disc_count_support,
        namespace=f"{SCENE_NAMESPACE}.frontier.target_answer",
    )
    return ObjectiveReversiPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.frontier.{str(query_player_name)}",
        prompt_query_key=PROMPT_QUERY_KEY,
        target_axis=target_axis,
        annotation_kind="disc_point_set",
        query_params={
            "query_player": player_name(int(query_player)),
            "target_player": str(query_player_name),
            "target_player_probabilities": dict(query_player_probabilities),
        },
        construct_attempt=lambda rng, axes: sample_frontier_disc_scene(
            rng=rng,
            board_size=int(axes.board_size),
            query_player=int(query_player),
            target_answer=int(target_axis.target_answer),
        ),
    )


@register_task
class GamesReversiFrontierDiscCountTask:
    """Count queried-color discs touching at least one empty square."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any] | None = None,
        max_attempts: int = 100,
    ) -> TaskOutput:
        return run_reversi_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_frontier_objective,
        )


__all__ = ["GamesReversiFrontierDiscCountTask", "TASK_ID"]
