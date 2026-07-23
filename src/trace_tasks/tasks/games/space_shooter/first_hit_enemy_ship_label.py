"""Space-shooter first hit enemy-ship label task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from ._lifecycle import SpaceShooterLifecycleTask, SpaceShooterObjective, run_space_shooter_lifecycle
from .shared.annotations import single_entity_bbox
from .shared.defaults import DEFAULTS, GEN_DEFAULTS
from .shared.state import (
    ENEMY_LABELS,
    SceneAxes,
    SpaceEnemy,
    SpaceProjectile,
    SpaceShooterSample,
    validate_basic_space_shooter_sample,
)


TASK_ID = "task_games__space_shooter__first_hit_enemy_ship_label"
PROMPT_QUERY_KEY = "first_hit_enemy_ship_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
OPTION_LABELS = ("A", "B", "C", "D")
JSON_EXAMPLE = '{"annotation":[414,510,438,546],"answer":"B"}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"B"}'


def _small_dy(rng) -> float:
    """Return small visual jitter that cannot invert y-slot ordering."""

    return float(rng.uniform(-3.0, 3.0))


def _make_enemy(*, enemy_index: int, lane: int, y_slot: int, rng) -> SpaceEnemy:
    """Create one enemy ship at a fixed logical lane/height."""

    label_index = int(enemy_index)
    if label_index < 0 or label_index >= len(ENEMY_LABELS):
        raise ValueError("space-shooter enemy_index exceeds available visible labels")
    return SpaceEnemy(
        enemy_id=f"enemy_{int(enemy_index)}",
        label=str(ENEMY_LABELS[label_index]),
        lane=int(lane),
        y_slot=int(y_slot),
        dx_frac=0.0,
        dy_px=_small_dy(rng),
    )


def _make_projectile(*, projectile_index: int, lane: int, y_slot: int, owner: str, rng) -> SpaceProjectile:
    """Create one projectile at a fixed logical lane/height."""

    return SpaceProjectile(
        projectile_id=f"{str(owner)}_projectile_{int(projectile_index)}",
        owner=str(owner),
        lane=int(lane),
        y_slot=int(y_slot),
        dx_frac=0.0,
        dy_px=_small_dy(rng),
    )


def _resolve_correct_option_index(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve the balanced A-D position for the first-hit enemy ship."""

    target_index, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="first_hit_enemy_ship_label_option_support",
        explicit_key="correct_option_index",
        fallback_support=DEFAULTS.first_hit_enemy_ship_label_option_support,
        namespace=f"{TASK_ID}.correct_option_index",
        balanced_flag_key="balanced_correct_option_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="first_hit_enemy_ship_label_option_support",
        fallback=DEFAULTS.first_hit_enemy_ship_label_option_support,
    )
    return int(target_index), tuple(int(value) for value in support), dict(probabilities)


def _slot_pairs_for_distance(distance: int) -> tuple[tuple[int, int], ...]:
    """Return valid (blue_shot_slot, enemy_slot) pairs for one travel distance."""

    pairs = []
    for shot_slot in (3, 4, 5):
        enemy_slot = int(shot_slot) - int(distance)
        if 0 <= int(enemy_slot) <= 2:
            pairs.append((int(shot_slot), int(enemy_slot)))
    return tuple(pairs)


def _prepare_first_hit_enemy_ship_label_objective(
    rng,
    params: Mapping[str, Any],
    axes: SceneAxes,
    instance_seed: int,
) -> SpaceShooterObjective:
    """Construct four labeled enemy ships and choose the first one hit."""

    lane_count = int(axes.lane_count)
    if lane_count < 4:
        raise ValueError("first_hit_enemy_ship_label requires at least four lanes")
    correct_index, support, probabilities = _resolve_correct_option_index(
        instance_seed=int(instance_seed),
        params=params,
    )

    correct_distance = int(rng.choice((1, 2)))
    distractor_distances = [distance for distance in range(correct_distance + 1, 6) if _slot_pairs_for_distance(distance)]
    if len(distractor_distances) < 3:
        raise ValueError("not enough distinct first-hit distances for distractor shots")
    rng.shuffle(distractor_distances)
    distance_by_option = [0, 0, 0, 0]
    distance_by_option[int(correct_index)] = int(correct_distance)
    distance_iter = iter(distractor_distances[:3])
    for index in range(4):
        if int(index) != int(correct_index):
            distance_by_option[int(index)] = int(next(distance_iter))

    candidate_lanes = list(rng.sample(range(lane_count), 4))
    enemies: list[SpaceEnemy] = []
    projectiles: list[SpaceProjectile] = []
    candidate_projectile_ids: list[str] = []
    candidate_enemy_ids: list[str] = []
    hit_enemy_by_projectile: dict[str, str] = {}
    hit_projectile_by_enemy: dict[str, str] = {}
    first_hit_distance_by_enemy: dict[str, int] = {}

    for option_index, lane in enumerate(candidate_lanes):
        distance = int(distance_by_option[int(option_index)])
        shot_slot, enemy_slot = rng.choice(_slot_pairs_for_distance(distance))
        enemy = _make_enemy(enemy_index=len(enemies), lane=int(lane), y_slot=int(enemy_slot), rng=rng)
        enemies.append(enemy)
        projectile = _make_projectile(
            projectile_index=len(projectiles),
            lane=int(lane),
            y_slot=int(shot_slot),
            owner="player",
            rng=rng,
        )
        projectiles.append(projectile)
        label = str(OPTION_LABELS[int(option_index)])
        candidate_projectile_ids.append(str(projectile.projectile_id))
        candidate_enemy_ids.append(str(enemy.enemy_id))
        hit_enemy_by_projectile[str(projectile.projectile_id)] = str(enemy.enemy_id)
        hit_projectile_by_enemy[str(enemy.enemy_id)] = str(projectile.projectile_id)
        first_hit_distance_by_enemy[str(enemy.enemy_id)] = int(distance)

    unused_lanes = [lane for lane in range(lane_count) if lane not in set(candidate_lanes)]
    rng.shuffle(unused_lanes)
    enemy_target = min(int(axes.enemy_count), len(enemies) + (2 * len(unused_lanes)))
    occupied = {(int(enemy.lane), int(enemy.y_slot)) for enemy in enemies}
    occupied.update((int(projectile.lane), int(projectile.y_slot)) for projectile in projectiles)
    for lane in unused_lanes:
        if len(enemies) >= enemy_target:
            break
        free_enemy_slots = [slot for slot in (0, 1, 2) if (int(lane), int(slot)) not in occupied]
        rng.shuffle(free_enemy_slots)
        for slot in free_enemy_slots[: int(rng.randint(1, 3))]:
            if len(enemies) >= enemy_target:
                break
            occupied.add((int(lane), int(slot)))
            enemies.append(_make_enemy(enemy_index=len(enemies), lane=int(lane), y_slot=int(slot), rng=rng))

    red_projectile_lanes = list(unused_lanes)
    rng.shuffle(red_projectile_lanes)
    for lane in red_projectile_lanes[: max(0, min(2, len(red_projectile_lanes)))]:
        lane_enemy_slots = [int(enemy.y_slot) for enemy in enemies if int(enemy.lane) == int(lane)]
        if not lane_enemy_slots:
            continue
        free_red_slots = [
            slot
            for slot in (3, 4, 5)
            if (int(lane), int(slot)) not in occupied and min(lane_enemy_slots) < int(slot)
        ]
        if not free_red_slots:
            continue
        rng.shuffle(free_red_slots)
        max_red_count = min(len(free_red_slots), max(int(value) for value in axes.enemy_projectile_per_lane_support))
        red_count = int(rng.randint(1, max_red_count + 1))
        for slot in free_red_slots[:red_count]:
            occupied.add((int(lane), int(slot)))
            projectiles.append(
                _make_projectile(
                    projectile_index=len(projectiles),
                    lane=int(lane),
                    y_slot=int(slot),
                    owner="enemy",
                    rng=rng,
                )
            )

    selected_enemy_id = str(candidate_enemy_ids[int(correct_index)])
    selected_projectile_id = str(hit_projectile_by_enemy[selected_enemy_id])
    sample = SpaceShooterSample(
        lane_count=lane_count,
        scene_variant=str(axes.scene_variant),
        answer=str(OPTION_LABELS[int(correct_index)]),
        player_lane=int(rng.randrange(lane_count)),
        enemies=tuple(enemies),
        projectiles=tuple(projectiles),
        safe_lane_indices=tuple(),
        annotation_entity_ids=(selected_enemy_id,),
        target_answer=None,
        construction_mode="first_hit_enemy_ship",
        metadata={
            "candidate_projectile_ids": list(candidate_projectile_ids),
            "candidate_enemy_ids": list(candidate_enemy_ids),
            "candidate_labels": {
                str(enemy_id): str(OPTION_LABELS[index])
                for index, enemy_id in enumerate(candidate_enemy_ids)
            },
            "hit_enemy_by_projectile": dict(hit_enemy_by_projectile),
            "hit_projectile_by_enemy": dict(hit_projectile_by_enemy),
            "first_hit_distance_by_enemy": dict(first_hit_distance_by_enemy),
            "selected_projectile_id": selected_projectile_id,
            "selected_enemy_id": selected_enemy_id,
            "correct_option": int(correct_index),
            "correct_option_index": int(correct_index),
            "correct_option_label": str(OPTION_LABELS[int(correct_index)]),
            "correct_option_support": [int(value) for value in support],
            "correct_option_probabilities": dict(probabilities),
        },
    )
    validate_basic_space_shooter_sample(sample)
    if min(first_hit_distance_by_enemy.values()) != int(correct_distance):
        raise ValueError("first-hit-enemy construction has wrong minimum distance")
    if sum(1 for distance in first_hit_distance_by_enemy.values() if int(distance) == int(correct_distance)) != 1:
        raise ValueError("first-hit-enemy construction must have a unique earliest ship")
    return SpaceShooterObjective(
        sample=sample,
        answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        build_annotation=single_entity_bbox,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        show_enemy_labels=True,
        visible_enemy_label_ids=tuple(candidate_enemy_ids),
    )


@register_task
class GamesSpaceShooterFirstHitEnemyShipLabelTask(SpaceShooterLifecycleTask):
    """Choose which labeled enemy ship is hit first."""

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
            build_objective=_prepare_first_hit_enemy_ship_label_objective,
        )


__all__ = ["GamesSpaceShooterFirstHitEnemyShipLabelTask"]
