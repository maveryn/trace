"""Select the candidate tower position covering the most path enemies."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from ._lifecycle import TowerDefenseObjectivePlan, run_tower_defense_lifecycle
from .shared.defaults import DEFAULTS, GEN_DEFAULTS
from .shared.sampling import sample_best_tower_position_scene


TASK_ID = "task_games__tower_defense__best_tower_position_label"
PROMPT_QUERY_KEY = "best_tower_position_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)


def _prepare_best_position_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> TowerDefenseObjectivePlan:
    """Bind the best candidate tower-position objective."""

    if str(selected_branch) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported tower-defense best-position branch: {selected_branch}")
    answer_option_index, answer_option_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="best_position_answer_option_index_support",
        explicit_key="answer_option_index",
        fallback_support=DEFAULTS.best_position_answer_option_index_support,
        namespace=f"{TASK_ID}.answer_option_index",
        balanced_flag_key="balanced_answer_option_sampling",
        namespace_support_permutation=True,
    )

    def construct_attempt(rng, axes, render_params, task_params):
        return sample_best_tower_position_scene(
            rng=rng,
            axes=axes,
            render_params=render_params,
            params={**dict(task_params), "answer_option_index": int(answer_option_index)},
        )

    return TowerDefenseObjectivePlan(
        attempt_namespace="games.tower_defense.best_tower_position_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_kind="tower_point",
        tower_count_support_key="best_position_candidate_count_support",
        tower_count_fallback=DEFAULTS.best_position_candidate_count_support,
        target_answer_support_key="best_position_target_answer_support",
        target_answer_fallback=DEFAULTS.best_position_target_answer_support,
        construct_attempt=construct_attempt,
        answer_type="string",
        json_example_answer="B",
        path_count_must_cover_target=True,
        trace_params={
            "best_position_branch_probabilities": dict(branch_probabilities),
            "answer_option_index": int(answer_option_index),
            "answer_option_index_probabilities": dict(answer_option_probabilities),
        },
    )


@register_task
class GamesTowerDefenseBestTowerPositionLabelTask:
    """Choose the candidate tower position covering the most path enemies."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'spatial_relations', 'topology', 'state_update')
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
            prepare_objective=_prepare_best_position_objective,
        )


__all__ = ["GamesTowerDefenseBestTowerPositionLabelTask", "TASK_ID"]
