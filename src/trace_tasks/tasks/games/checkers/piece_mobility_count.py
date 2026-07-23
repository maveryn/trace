"""Count Checkers pieces with at least one move condition available."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    prepare_checkers_move_condition_objective,
    run_checkers_lifecycle,
)
from .shared.sampling import sample_piece_mobility_scene
from .shared.state import SCENE_ID


TASK_ID = "task_games__checkers__piece_mobility_count"
SUPPORTED_QUERY_IDS = (
    "piece_with_legal_move_count",
    "piece_with_capture_move_count",
)
QUERY_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "piece_with_legal_move_count": {
        "support_key": "piece_with_legal_move_count_support",
        "fallback_support": (0, 1, 2, 3, 4, 5),
        "capture_only": False,
    },
    "piece_with_capture_move_count": {
        "support_key": "piece_with_capture_move_count_support",
        "fallback_support": (0, 1, 2, 3, 4),
        "capture_only": True,
    },
}
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)[0]


def _prepare_piece_mobility_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> Any:
    """Bind the selected source-piece mobility query to sampler and annotation semantics."""

    settings = QUERY_SETTINGS[str(query_id)]
    return prepare_checkers_move_condition_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        query_id=str(query_id),
        support_key=str(settings["support_key"]),
        fallback_support=tuple(int(value) for value in settings["fallback_support"]),
        capture_only=bool(settings["capture_only"]),
        count_trace_keys=("piece_with_legal_move_count", "piece_with_capture_move_count"),
        gen_defaults=_GEN_DEFAULTS,
        attempt_namespace="games.checkers.piece_mobility_count",
        sample_scene=sample_piece_mobility_scene,
    )


@register_task
class GamesCheckersPieceMobilityCountTask:
    """Count current-player pieces with at least one sampled move condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a movable-piece count by binding source-piece witnesses locally."""

        return run_checkers_lifecycle(
            task_id=self.task_id,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_piece_mobility_objective,
        )


__all__ = ["GamesCheckersPieceMobilityCountTask"]
