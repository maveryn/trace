"""Space-shooter labeled enemy ship hit task."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from ._lifecycle import SpaceShooterLifecycleTask, SpaceShooterObjective, run_space_shooter_lifecycle
from .shared.annotations import single_entity_bbox
from .shared.defaults import DEFAULTS, GEN_DEFAULTS
from .shared.sampling import enemy_ship_ids_hit_by_player_shots, sample_enemy_ship_hit_scene
from .shared.state import ENEMY_LABELS, SceneAxes, validate_basic_space_shooter_sample


TASK_ID = "task_games__space_shooter__hit_enemy_ship_label"
PROMPT_QUERY_KEY = "hit_enemy_ship_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
OPTION_LABELS = ("A", "B", "C", "D")
JSON_EXAMPLE = '{"annotation":[420,250,482,298],"answer":"C"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"C"}'


def _resolve_correct_option_index(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve the balanced A-D position for the one correct labeled ship."""

    target_index, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="hit_enemy_ship_label_option_support",
        explicit_key="correct_option_index",
        fallback_support=DEFAULTS.hit_enemy_ship_label_option_support,
        namespace=f"{TASK_ID}.correct_option_index",
        balanced_flag_key="balanced_correct_option_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="hit_enemy_ship_label_option_support",
        fallback=DEFAULTS.hit_enemy_ship_label_option_support,
    )
    return int(target_index), tuple(int(value) for value in support), dict(probabilities)


def _relabel_candidate_enemies(sample, *, candidate_ids: tuple[str, ...]):
    """Give the four visible option ships A-D while keeping hidden labels unique."""

    candidate_label_by_id = {
        str(enemy_id): str(OPTION_LABELS[index])
        for index, enemy_id in enumerate(candidate_ids)
    }
    hidden_labels = [str(label) for label in ENEMY_LABELS if str(label) not in set(OPTION_LABELS)]
    hidden_index = 0
    relabeled_enemies = []
    for enemy in sample.enemies:
        if str(enemy.enemy_id) in candidate_label_by_id:
            relabeled_enemies.append(replace(enemy, label=candidate_label_by_id[str(enemy.enemy_id)]))
            continue
        if hidden_index >= len(hidden_labels):
            raise ValueError("not enough hidden labels for space-shooter enemies")
        relabeled_enemies.append(replace(enemy, label=hidden_labels[hidden_index]))
        hidden_index += 1
    return replace(sample, enemies=tuple(relabeled_enemies))


def _prepare_hit_enemy_ship_label_objective(
    rng,
    params: Mapping[str, Any],
    axes: SceneAxes,
    instance_seed: int,
) -> SpaceShooterObjective:
    """Construct a scene and select one labeled enemy ship hit by blue shots."""

    correct_index, support, probabilities = _resolve_correct_option_index(
        instance_seed=int(instance_seed),
        params=params,
    )
    max_scene_hits = min(3, max(1, int(axes.enemy_count) - 3))
    scene_hit_count = int(rng.randint(1, max_scene_hits + 1))
    sample = sample_enemy_ship_hit_scene(rng=rng, axes=axes, target_answer=scene_hit_count)
    hit_ids = tuple(str(enemy_id) for enemy_id in enemy_ship_ids_hit_by_player_shots(sample))
    hit_id_set = set(hit_ids)
    non_hit_ids = tuple(str(enemy.enemy_id) for enemy in sample.enemies if str(enemy.enemy_id) not in hit_id_set)
    if not hit_ids or len(non_hit_ids) < 3:
        raise ValueError("hit_enemy_ship_label needs one hit ship and three non-hit candidate ships")

    selected_hit_id = str(rng.choice(tuple(hit_ids)))
    distractor_ids = tuple(str(enemy_id) for enemy_id in rng.sample(tuple(non_hit_ids), 3))
    candidate_ids: list[str | None] = [None, None, None, None]
    candidate_ids[int(correct_index)] = selected_hit_id
    distractor_iter = iter(distractor_ids)
    for index in range(len(candidate_ids)):
        if candidate_ids[index] is None:
            candidate_ids[index] = str(next(distractor_iter))
    resolved_candidate_ids = tuple(str(enemy_id) for enemy_id in candidate_ids if enemy_id is not None)
    sample = _relabel_candidate_enemies(sample, candidate_ids=resolved_candidate_ids)
    sample = replace(
        sample,
        answer=str(OPTION_LABELS[int(correct_index)]),
        annotation_entity_ids=(selected_hit_id,),
        metadata={
            **dict(sample.metadata),
            "hit_enemy_ids": list(hit_ids),
            "candidate_enemy_ids": list(resolved_candidate_ids),
            "candidate_labels": {
                str(enemy_id): str(OPTION_LABELS[index])
                for index, enemy_id in enumerate(resolved_candidate_ids)
            },
            "selected_enemy_id": str(selected_hit_id),
            "correct_option": int(correct_index),
            "correct_option_index": int(correct_index),
            "correct_option_label": str(OPTION_LABELS[int(correct_index)]),
            "correct_option_support": [int(value) for value in support],
            "correct_option_probabilities": dict(probabilities),
            "scene_hit_count": int(scene_hit_count),
        },
    )
    validate_basic_space_shooter_sample(sample)
    return SpaceShooterObjective(
        sample=sample,
        answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        build_annotation=single_entity_bbox,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        show_enemy_labels=True,
        visible_enemy_label_ids=resolved_candidate_ids,
    )


@register_task
class GamesSpaceShooterHitEnemyShipLabelTask(SpaceShooterLifecycleTask):
    """Choose which labeled enemy ship is hit by an existing blue shot."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'state_update')
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_space_shooter_lifecycle(
            namespace=TASK_ID,
            prompt_query_key=PROMPT_QUERY_KEY,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=DEFAULT_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_hit_enemy_ship_label_objective,
        )


__all__ = ["GamesSpaceShooterHitEnemyShipLabelTask"]
