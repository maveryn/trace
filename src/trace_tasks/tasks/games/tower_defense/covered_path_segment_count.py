"""Count path nodes covered by visible tower ranges."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import TowerDefenseObjectivePlan, run_tower_defense_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import sample_covered_path_scene


TASK_ID = "task_games__tower_defense__covered_path_segment_count"
PROMPT_QUERY_KEY = "covered_path_segment_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)


def _prepare_covered_path_objective(
    _instance_seed: int,
    _params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> TowerDefenseObjectivePlan:
    """Bind the covered-path-node count objective."""

    if str(selected_branch) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported tower-defense covered-path branch: {selected_branch}")

    def construct_attempt(rng, axes, render_params, task_params):
        return sample_covered_path_scene(
            rng=rng,
            axes=axes,
            render_params=render_params,
            params=task_params,
        )

    return TowerDefenseObjectivePlan(
        attempt_namespace="games.tower_defense.covered_path_segment_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_kind="path_point_set",
        tower_count_support_key="covered_path_tower_count_support",
        tower_count_fallback=DEFAULTS.covered_path_tower_count_support,
        target_answer_support_key="covered_path_target_answer_support",
        target_answer_fallback=DEFAULTS.covered_path_target_answer_support,
        construct_attempt=construct_attempt,
        path_count_must_cover_target=True,
        tower_count_must_cover_target=True,
        trace_params={"covered_path_branch_probabilities": dict(branch_probabilities)},
    )


@register_task
class GamesTowerDefenseCoveredPathSegmentCountTask:
    """Count visible path nodes covered by at least one tower range ring."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'spatial_relations', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100):
        return run_tower_defense_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_covered_path_objective,
        )


__all__ = ["GamesTowerDefenseCoveredPathSegmentCountTask", "TASK_ID"]
