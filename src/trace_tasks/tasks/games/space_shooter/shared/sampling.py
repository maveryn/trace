"""Scene-local sampling primitives for space-shooter tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import DEFAULTS, GEN_DEFAULTS
from .state import (
    ENEMY_LABELS,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
    SceneAxes,
    SpaceEnemy,
    SpaceProjectile,
    SpaceShooterSample,
    lane_entity_id,
    validate_basic_space_shooter_sample,
)


def resolve_scene_axes(*, namespace: str, instance_seed: int, params: Mapping[str, Any]) -> SceneAxes:
    """Resolve visual and count axes shared by all space-shooter objectives."""

    from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

    scene_variant, scene_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_STYLE_VARIANTS,
    )
    lane_count, lane_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="lane_count_support",
        explicit_key="lane_count",
        fallback_support=DEFAULTS.lane_count_support,
        namespace=f"{namespace}.lane_count",
        balanced_flag_key="balanced_lane_count_sampling",
        namespace_support_permutation=True,
    )
    enemy_count, enemy_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key="enemy_count_support",
        explicit_key="enemy_count",
        fallback_support=DEFAULTS.enemy_count_support,
        namespace=f"{namespace}.enemy_count",
        balanced_flag_key="balanced_enemy_count_sampling",
        namespace_support_permutation=True,
    )
    enemy_projectile_per_lane_support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key="enemy_projectile_per_lane_support",
        fallback=DEFAULTS.enemy_projectile_per_lane_support,
    )
    return SceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        lane_count=int(lane_count),
        enemy_count=int(enemy_count),
        enemy_projectile_per_lane_support=tuple(int(value) for value in enemy_projectile_per_lane_support),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        lane_count_probabilities=dict(lane_count_probabilities),
        enemy_count_probabilities=dict(enemy_count_probabilities),
    )


def resolve_target_answer(
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve one balanced integer target answer for controlled count tasks."""

    target_answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{namespace}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return int(target_answer), tuple(int(value) for value in support), dict(probabilities)


def _entity_dy(rng) -> float:
    return float(rng.uniform(-8.0, 8.0))


def _make_enemy(*, enemy_index: int, lane: int, y_slot: int, rng, score_value: int | None = None) -> SpaceEnemy:
    label_index = int(enemy_index)
    if label_index < 0 or label_index >= len(ENEMY_LABELS):
        raise ValueError("space-shooter enemy_index exceeds available visible labels")
    return SpaceEnemy(
        enemy_id=f"enemy_{int(enemy_index)}",
        label=str(ENEMY_LABELS[label_index]),
        lane=int(lane),
        y_slot=int(y_slot),
        dx_frac=0.0,
        dy_px=_entity_dy(rng),
        score_value=None if score_value is None else int(score_value),
    )


def _make_projectile(
    *,
    projectile_index: int,
    lane: int,
    y_slot: int,
    owner: str,
    rng,
) -> SpaceProjectile:
    return SpaceProjectile(
        projectile_id=f"{str(owner)}_projectile_{int(projectile_index)}",
        owner=str(owner),
        lane=int(lane),
        y_slot=int(y_slot),
        dx_frac=0.0,
        dy_px=_entity_dy(rng),
    )


def _claim_position(
    *,
    rng,
    occupied: set[Tuple[int, int]],
    lane_candidates: Sequence[int],
    slot_candidates: Sequence[int],
) -> Tuple[int, int]:
    """Claim one lane/vertical-slot pair so visible objects do not stack."""

    candidates = [
        (int(lane), int(slot))
        for lane in lane_candidates
        for slot in slot_candidates
        if (int(lane), int(slot)) not in occupied
    ]
    if not candidates:
        raise ValueError("no free space-shooter lane/slot position remains")
    lane, slot = rng.choice(candidates)
    occupied.add((int(lane), int(slot)))
    return int(lane), int(slot)


def _add_enemy_projectile_line(
    *,
    rng,
    occupied: set[Tuple[int, int]],
    projectiles: list[SpaceProjectile],
    lane: int,
    enemies: Sequence[SpaceEnemy],
    support: Sequence[int],
    min_count: int = 1,
    slot_candidates: Sequence[int] = (3, 4, 5),
) -> int:
    """Add a visible vertical line of one to three red shots in a lane."""

    free_slots = [
        int(slot)
        for slot in slot_candidates
        if (int(lane), int(slot)) not in occupied
        and any(int(enemy.lane) == int(lane) and int(enemy.y_slot) < int(slot) for enemy in enemies)
    ]
    if len(free_slots) < int(min_count):
        raise ValueError("not enough slots for requested enemy projectile line")
    supported_counts = sorted(
        {
            int(count)
            for count in support
            if int(min_count) <= int(count) <= min(3, len(free_slots))
        }
    )
    if not supported_counts:
        raise ValueError("enemy projectile support does not fit available slots")
    shot_count = int(rng.choice(supported_counts))
    rng.shuffle(free_slots)
    for slot in free_slots[:shot_count]:
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
    return int(shot_count)


def _add_player_projectile(
    *,
    rng,
    occupied: set[Tuple[int, int]],
    projectiles: list[SpaceProjectile],
    lane_candidates: Sequence[int],
    enemies: Sequence[SpaceEnemy],
    slot_candidates: Sequence[int] = (5,),
) -> None:
    """Add one blue player shot below all same-lane enemies and red shots."""

    candidates = [
        (int(lane), int(slot))
        for lane in lane_candidates
        for slot in slot_candidates
        if (int(lane), int(slot)) not in occupied
        and any(int(enemy.lane) == int(lane) for enemy in enemies)
        and all(
            int(enemy.y_slot) < int(slot)
            for enemy in enemies
            if int(enemy.lane) == int(lane)
        )
        and all(
            int(projectile.y_slot) < int(slot)
            for projectile in projectiles
            if str(projectile.owner) == "enemy" and int(projectile.lane) == int(lane)
        )
    ]
    if not candidates:
        raise ValueError("no valid below-threat player projectile slot remains")
    lane, slot = rng.choice(candidates)
    occupied.add((int(lane), int(slot)))
    projectiles.append(
        _make_projectile(
            projectile_index=len(projectiles),
            lane=int(lane),
            y_slot=int(slot),
            owner="player",
            rng=rng,
        )
    )


def enemy_ship_ids_hit_by_player_shots(sample: SpaceShooterSample) -> Tuple[str, ...]:
    """Return enemy ids hit by blue shots under the per-lane lower-first rule."""

    player_counts_by_lane: dict[int, int] = {}
    for projectile in sample.projectiles:
        if str(projectile.owner) != "player":
            continue
        lane = int(projectile.lane)
        player_counts_by_lane[lane] = int(player_counts_by_lane.get(lane, 0)) + 1
    enemies_by_lane: dict[int, list[SpaceEnemy]] = {}
    for enemy in sample.enemies:
        enemies_by_lane.setdefault(int(enemy.lane), []).append(enemy)
    hit_ids: list[str] = []
    for lane in sorted(enemies_by_lane):
        player_count = int(player_counts_by_lane.get(int(lane), 0))
        lane_enemies = sorted(
            enemies_by_lane[int(lane)],
            key=lambda enemy: int(enemy.y_slot),
            reverse=True,
        )
        hit_ids.extend(str(enemy.enemy_id) for enemy in lane_enemies[: min(player_count, len(lane_enemies))])
    return tuple(hit_ids)


def sample_enemy_ship_count_scene(*, rng, axes: SceneAxes) -> SpaceShooterSample:
    """Construct a scene where every visible enemy ship is counted."""

    lane_count = int(axes.lane_count)
    enemy_count = int(axes.enemy_count)
    occupied: set[Tuple[int, int]] = set()
    enemies: list[SpaceEnemy] = []
    while len(enemies) < enemy_count:
        lane, slot = _claim_position(
            rng=rng,
            occupied=occupied,
            lane_candidates=range(lane_count),
            slot_candidates=(0, 1, 2, 3),
        )
        enemies.append(_make_enemy(enemy_index=len(enemies), lane=lane, y_slot=slot, rng=rng))

    projectiles: list[SpaceProjectile] = []
    projectile_lanes = list(range(lane_count))
    rng.shuffle(projectile_lanes)
    projectile_lines_added = 0
    multi_shot_line_added = False
    can_force_multi_shot_line = any(int(value) >= 2 for value in axes.enemy_projectile_per_lane_support)
    for lane in projectile_lanes:
        if projectile_lines_added >= max(1, lane_count // 3) and multi_shot_line_added:
            break
        try:
            added = _add_enemy_projectile_line(
                rng=rng,
                occupied=occupied,
                projectiles=projectiles,
                lane=int(lane),
                enemies=enemies,
                support=axes.enemy_projectile_per_lane_support,
                min_count=2 if can_force_multi_shot_line and not multi_shot_line_added else 1,
                slot_candidates=(2, 3, 4),
            )
        except ValueError:
            continue
        projectile_lines_added += 1
        multi_shot_line_added = bool(multi_shot_line_added or added >= 2)
    for _ in range(max(1, lane_count // 4)):
        try:
            _add_player_projectile(
                rng=rng,
                occupied=occupied,
                projectiles=projectiles,
                lane_candidates=range(lane_count),
                enemies=enemies,
            )
        except ValueError:
            continue

    annotation_ids = tuple(str(enemy.enemy_id) for enemy in enemies)
    sample = SpaceShooterSample(
        lane_count=lane_count,
        scene_variant=str(axes.scene_variant),
        answer=int(len(enemies)),
        player_lane=int(rng.randrange(lane_count)),
        enemies=tuple(enemies),
        projectiles=tuple(projectiles),
        safe_lane_indices=tuple(),
        annotation_entity_ids=annotation_ids,
        target_answer=int(len(enemies)),
        construction_mode="visible_enemy_ship_count",
        metadata={
            "target_answer": int(len(enemies)),
            "target_answer_support": list(DEFAULTS.enemy_count_support),
        },
    )
    validate_basic_space_shooter_sample(sample)
    return sample


def sample_enemy_ship_hit_scene(*, rng, axes: SceneAxes, target_answer: int) -> SpaceShooterSample:
    """Construct lanes where blue-shot/enemy counts realize the target hit count."""

    lane_count = int(axes.lane_count)
    target = int(target_answer)
    if target < 0 or target > lane_count * 3:
        raise ValueError("space-shooter hit count target does not fit lane capacity")
    occupied: set[Tuple[int, int]] = set()
    enemies: list[SpaceEnemy] = []
    projectiles: list[SpaceProjectile] = []
    annotation_ids: list[str] = []

    hit_counts_by_lane = {lane: 0 for lane in range(lane_count)}
    remaining_hits = int(target)
    if remaining_hits >= 2 and lane_count > 0:
        lane = int(rng.randrange(lane_count))
        first_count = min(3, remaining_hits, int(rng.choice((2, 3))))
        hit_counts_by_lane[lane] = int(first_count)
        remaining_hits -= int(first_count)
    for _ in range(remaining_hits):
        candidates = [lane for lane, count in hit_counts_by_lane.items() if int(count) < 3]
        lane = int(rng.choice(candidates))
        hit_counts_by_lane[lane] += 1

    lanes_with_hits = tuple(sorted(lane for lane, count in hit_counts_by_lane.items() if int(count) > 0))
    for lane in lanes_with_hits:
        hit_count = int(hit_counts_by_lane[int(lane)])
        count_options = [(hit_count, hit_count)]
        if hit_count < 3:
            count_options.append((hit_count + 1, hit_count))
            count_options.append((hit_count, hit_count + 1))
        enemy_count, player_count = rng.choice(count_options)
        enemy_slots = sorted(rng.sample((0, 1, 2), int(enemy_count)))
        lane_enemies: list[SpaceEnemy] = []
        for slot in enemy_slots:
            occupied.add((int(lane), int(slot)))
            enemy = _make_enemy(enemy_index=len(enemies), lane=int(lane), y_slot=int(slot), rng=rng)
            enemies.append(enemy)
            lane_enemies.append(enemy)
        for slot in sorted(rng.sample((3, 4, 5), int(player_count))):
            occupied.add((int(lane), int(slot)))
            projectiles.append(
                _make_projectile(
                    projectile_index=len(projectiles),
                    lane=int(lane),
                    y_slot=int(slot),
                    owner="player",
                    rng=rng,
                )
            )
        lower_enemies = sorted(lane_enemies, key=lambda enemy: int(enemy.y_slot), reverse=True)[:hit_count]
        annotation_ids.extend(str(enemy.enemy_id) for enemy in lower_enemies)

    distractor_lanes = [lane for lane in range(lane_count) if lane not in set(lanes_with_hits)]
    rng.shuffle(distractor_lanes)
    enemy_target = min(max(int(axes.enemy_count), target + 2), lane_count * 3)
    enemy_only_lanes: list[int] = []
    for lane in distractor_lanes:
        if len(enemies) >= enemy_target:
            break
        enemy_only_lanes.append(int(lane))
        enemy_count = int(rng.randint(1, 4))
        for slot in sorted(rng.sample((0, 1, 2), min(enemy_count, enemy_target - len(enemies)))):
            occupied.add((int(lane), int(slot)))
            enemies.append(_make_enemy(enemy_index=len(enemies), lane=int(lane), y_slot=int(slot), rng=rng))

    player_only_lanes = [
        lane
        for lane in range(lane_count)
        if lane not in set(lanes_with_hits)
        and lane not in set(enemy_only_lanes)
    ]
    rng.shuffle(player_only_lanes)
    for lane in player_only_lanes[: max(0, min(2, len(player_only_lanes)))]:
        player_count = int(rng.randint(1, 4))
        for slot in sorted(rng.sample((3, 4, 5), int(player_count))):
            occupied.add((int(lane), int(slot)))
            projectiles.append(
                _make_projectile(
                    projectile_index=len(projectiles),
                    lane=int(lane),
                    y_slot=int(slot),
                    owner="player",
                    rng=rng,
                )
            )

    red_lanes = list(enemy_only_lanes)
    rng.shuffle(red_lanes)
    can_force_multi_shot_line = any(int(value) >= 2 for value in axes.enemy_projectile_per_lane_support)
    red_line_added = False
    for lane in red_lanes:
        if red_line_added and rng.random() < 0.60:
            continue
        try:
            _add_enemy_projectile_line(
                rng=rng,
                occupied=occupied,
                projectiles=projectiles,
                lane=int(lane),
                enemies=enemies,
                support=axes.enemy_projectile_per_lane_support,
                min_count=2 if can_force_multi_shot_line and not red_line_added else 1,
                slot_candidates=(3, 4),
            )
        except ValueError:
            continue
        red_line_added = True

    sample = SpaceShooterSample(
        lane_count=lane_count,
        scene_variant=str(axes.scene_variant),
        answer=int(len(annotation_ids)),
        player_lane=int(rng.randrange(lane_count)),
        enemies=tuple(enemies),
        projectiles=tuple(projectiles),
        safe_lane_indices=tuple(),
        annotation_entity_ids=tuple(annotation_ids),
        target_answer=int(target),
        construction_mode="blue_shot_ship_alignment",
        metadata={
            "target_answer": int(target),
            "target_answer_support": list(DEFAULTS.enemy_ship_hit_count_support),
            "hit_lane_indices": [int(lane) for lane in lanes_with_hits],
        },
    )
    validate_basic_space_shooter_sample(sample)
    computed_annotation_ids = enemy_ship_ids_hit_by_player_shots(sample)
    if tuple(annotation_ids) != tuple(computed_annotation_ids):
        raise ValueError("space-shooter hit construction does not match lower-first lane rule")
    return sample


def sample_safe_lane_scene(*, rng, axes: SceneAxes, target_answer: int) -> SpaceShooterSample:
    """Construct a scene where target answer is the number of safe bottom lanes."""

    lane_count = max(int(target_answer), int(axes.lane_count))
    target = min(int(target_answer), lane_count)
    lanes = list(range(lane_count))
    rng.shuffle(lanes)
    safe_lanes = tuple(sorted(lanes[:target]))
    unsafe_lanes = [lane for lane in range(lane_count) if lane not in set(safe_lanes)]
    occupied: set[Tuple[int, int]] = set()
    enemies: list[SpaceEnemy] = []
    for lane in unsafe_lanes:
        _, slot = _claim_position(rng=rng, occupied=occupied, lane_candidates=(lane,), slot_candidates=(0, 1, 2))
        enemies.append(_make_enemy(enemy_index=len(enemies), lane=lane, y_slot=slot, rng=rng))
    projectiles: list[SpaceProjectile] = []
    can_force_multi_shot_line = any(int(value) >= 2 for value in axes.enemy_projectile_per_lane_support)
    for unsafe_index, lane in enumerate(unsafe_lanes):
        _add_enemy_projectile_line(
            rng=rng,
            occupied=occupied,
            projectiles=projectiles,
            lane=int(lane),
            enemies=enemies,
            support=axes.enemy_projectile_per_lane_support,
            min_count=2 if int(unsafe_index) == 0 and can_force_multi_shot_line else 1,
            slot_candidates=(2, 3, 4),
        )
    enemy_target = min(int(axes.enemy_count), lane_count + int(rng.randrange(0, 3)))
    while len(enemies) < enemy_target:
        lane, slot = _claim_position(rng=rng, occupied=occupied, lane_candidates=range(lane_count), slot_candidates=(0, 1, 2))
        enemies.append(_make_enemy(enemy_index=len(enemies), lane=lane, y_slot=slot, rng=rng))
    for lane in safe_lanes:
        if rng.random() < 0.45:
            try:
                _add_player_projectile(
                    rng=rng,
                    occupied=occupied,
                    projectiles=projectiles,
                    lane_candidates=(lane,),
                    enemies=enemies,
                )
            except ValueError:
                continue
    annotation_ids = tuple(lane_entity_id(lane) for lane in safe_lanes)
    sample = SpaceShooterSample(
        lane_count=lane_count,
        scene_variant=str(axes.scene_variant),
        answer=int(len(safe_lanes)),
        player_lane=int(rng.randrange(lane_count)),
        enemies=tuple(enemies),
        projectiles=tuple(projectiles),
        safe_lane_indices=safe_lanes,
        annotation_entity_ids=annotation_ids,
        target_answer=int(target),
        construction_mode="safe_bottom_lane_count",
        metadata={
            "target_answer": int(target),
            "target_answer_support": list(DEFAULTS.safe_lane_count_support),
        },
    )
    validate_basic_space_shooter_sample(sample)
    return sample
