"""Scene-neutral Mini-golf sampling primitives."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from .defaults import (
    DEFAULTS,
    FIRST_OBSTACLE_MODE,
    OBSTACLE_KINDS,
    OBSTACLE_LABELS,
    OBSTACLE_RADIUS_NORM,
    SCENE_VARIANTS,
    SHOT_MODES,
    SHOT_OPTIONS_MODE,
    STYLE_VARIANTS,
)
from .rules import distance, first_hit_obstacle_id, normalize_angle, target_angle_for_mode, trace_shot_path, unit_from_angle
from .state import (
    MinigolfObstacle,
    MinigolfSample,
    MinigolfShotOption,
    obstacle_entity_id,
    path_entity_id,
    path_label,
    validate_first_obstacle_sample,
    validate_shot_options_sample,
)


MIN_OBSTACLE_POINT_CLEARANCE_NORM = 0.15
MIN_OBSTACLE_OBSTACLE_CLEARANCE_NORM = 0.15
MIN_OBSTACLE_PATH_CLEARANCE_NORM = 0.12


@dataclass(frozen=True)
class MinigolfAxes:
    """Resolved scene and visual axes shared by Mini-golf objectives."""

    scene_variant: str
    style_variant: str
    obstacle_count: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    obstacle_count_probabilities: Dict[str, float]


def resolve_minigolf_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
    obstacle_count_support_key: str = "obstacle_count_support",
    obstacle_count_fallback_support: Sequence[int] = DEFAULTS.obstacle_count_support,
) -> MinigolfAxes:
    """Resolve scene, style, and obstacle-count axes for one Mini-golf task."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=STYLE_VARIANTS,
    )
    obstacle_count, obstacle_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(obstacle_count_support_key),
        explicit_key="obstacle_count",
        fallback_support=tuple(int(value) for value in obstacle_count_fallback_support),
        namespace=f"{namespace}.obstacle_count",
        balanced_flag_key="balanced_obstacle_count_sampling",
        namespace_support_permutation=True,
    )
    return MinigolfAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        obstacle_count=int(obstacle_count),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        obstacle_count_probabilities=dict(obstacle_count_probabilities),
    )


def resolve_minigolf_label_choice(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str],
    namespace: str,
    balanced_flag_key: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one label-valued support choice."""

    support = _string_support(params, gen_defaults=gen_defaults, key=str(support_key), fallback=fallback_support)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{explicit_key}={value!r} is not in {support_key}")
        return value, {str(item): (1.0 if str(item) == value else 0.0) for item in support}
    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(uniform_choice(rng, support)), probabilities


def resolve_minigolf_integer_choice(
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve one integer-valued Mini-golf task axis."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities)


def sample_first_obstacle_scene(
    *,
    rng: Any,
    axes: MinigolfAxes,
    target_label: str,
) -> MinigolfSample:
    """Construct a course where a short cue first hits one labeled obstacle."""

    obstacle_count = int(axes.obstacle_count)
    for _attempt in range(128):
        ball = (float(rng.uniform(0.34, 0.66)), float(rng.uniform(0.78, 0.88)))
        hole = (float(rng.uniform(0.36, 0.64)), float(rng.uniform(0.12, 0.24)))
        angle = float(rng.uniform(-2.42, -1.10))
        direction = unit_from_angle(angle)
        travel_distance = float(rng.uniform(0.42, 0.64))
        target = (ball[0] + (direction[0] * travel_distance), ball[1] + (direction[1] * travel_distance))
        if not (0.14 <= target[0] <= 0.86 and 0.14 <= target[1] <= 0.72):
            continue
        if distance(target, hole) < MIN_OBSTACLE_POINT_CLEARANCE_NORM:
            continue
        shown_path = (ball, target)
        obstacles = _obstacles_with_target_label(
            rng=rng,
            target_label=str(target_label),
            target_center=target,
            obstacle_count=obstacle_count,
            avoid_paths=(shown_path,),
            avoid_points=(ball, hole),
        )
        if obstacles is None:
            continue
        target_obstacle = next(obstacle for obstacle in obstacles if str(obstacle.label) == str(target_label))
        first_id = first_hit_obstacle_id(origin=ball, angle_rad=angle, obstacles=obstacles)
        if str(first_id) != str(target_obstacle.obstacle_id):
            continue
        sample = MinigolfSample(
            mode=FIRST_OBSTACLE_MODE,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=str(target_obstacle.label),
            ball_x_norm=float(ball[0]),
            ball_y_norm=float(ball[1]),
            hole_x_norm=float(hole[0]),
            hole_y_norm=float(hole[1]),
            obstacles=obstacles,
            shot_options=tuple(),
            target_obstacle_id=str(target_obstacle.obstacle_id),
            target_obstacle_label=str(target_obstacle.label),
            target_path_id=None,
            target_path_label=None,
            annotation_entity_ids=(str(target_obstacle.obstacle_id),),
            construction_mode="short_cue_first_obstacle_collision",
            cue_visible_fraction=float(rng.uniform(0.32, 0.43)),
            hidden_paths_norm={"shown_path": tuple(shown_path)},
        )
        validate_first_obstacle_sample(sample)
        return sample
    raise ValueError("failed to construct Mini-golf first-obstacle scene")


def sample_shot_options_scene(
    *,
    rng: Any,
    axes: MinigolfAxes,
    option_count: int,
    target_index: int,
) -> MinigolfSample:
    """Construct a course where exactly one numbered cue reaches the hole."""

    option_count = int(option_count)
    target_index = int(target_index)
    for _attempt in range(160):
        ball = (float(rng.uniform(0.38, 0.62)), float(rng.uniform(0.78, 0.88)))
        hole = (float(rng.uniform(0.30, 0.70)), float(rng.uniform(0.12, 0.25)))
        mode = str(rng.choice(SHOT_MODES))
        target_angle = target_angle_for_mode(ball=ball, hole=hole, mode=mode)
        success, _, target_path = trace_shot_path(ball_xy=ball, angle_rad=target_angle, hole_xy=hole, obstacles=tuple())
        if not success or len(target_path) < 2:
            continue

        obstacles: list[MinigolfObstacle] = []
        obstacle_kind_cycle = cycle(shuffled_support(rng, OBSTACLE_KINDS))
        for index in range(int(axes.obstacle_count)):
            if int(index) >= len(OBSTACLE_LABELS):
                raise ValueError("Mini-golf obstacle_count exceeds available visible labels")
            maybe = _safe_obstacle_position(
                rng=rng,
                existing=obstacles,
                avoid_points=(ball, hole),
                avoid_paths=(target_path,),
            )
            if maybe is None:
                break
            obstacles.append(
                MinigolfObstacle(
                    obstacle_id=obstacle_entity_id(index),
                    label=str(OBSTACLE_LABELS[int(index)]),
                    kind=str(next(obstacle_kind_cycle)),
                    x_norm=float(maybe[0]),
                    y_norm=float(maybe[1]),
                    radius_norm=float(OBSTACLE_RADIUS_NORM * rng.uniform(0.88, 1.08)),
                    color_index=int(index),
                )
            )
        success, _, target_path = trace_shot_path(ball_xy=ball, angle_rad=target_angle, hole_xy=hole, obstacles=obstacles)
        if not success:
            continue
        options = _make_shot_options(
            rng=rng,
            option_count=option_count,
            target_index=target_index,
            target_angle=target_angle,
            ball=ball,
            hole=hole,
            obstacles=obstacles,
        )
        if options is None:
            continue
        hidden_paths: Dict[str, Tuple[Tuple[float, float], ...]] = {}
        success_count = 0
        success_id = ""
        for option in options:
            reaches_hole, _, path = trace_shot_path(
                ball_xy=ball,
                angle_rad=float(option.angle_rad),
                hole_xy=hole,
                obstacles=obstacles,
            )
            hidden_paths[str(option.path_id)] = tuple(path)
            if reaches_hole:
                success_count += 1
                success_id = str(option.path_id)
        target_option = options[int(target_index)]
        if success_count != 1 or str(success_id) != str(target_option.path_id):
            continue
        sample = MinigolfSample(
            mode=SHOT_OPTIONS_MODE,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=str(target_option.label),
            ball_x_norm=float(ball[0]),
            ball_y_norm=float(ball[1]),
            hole_x_norm=float(hole[0]),
            hole_y_norm=float(hole[1]),
            obstacles=tuple(obstacles),
            shot_options=options,
            target_obstacle_id=None,
            target_obstacle_label=None,
            target_path_id=str(target_option.path_id),
            target_path_label=str(target_option.label),
            annotation_entity_ids=(str(target_option.path_id),),
            construction_mode=f"unique_numbered_{mode}_cue_reaches_hole",
            cue_visible_fraction=0.36,
            hidden_paths_norm=dict(hidden_paths),
        )
        validate_shot_options_sample(sample)
        return sample
    raise ValueError("failed to construct Mini-golf shot-options scene")


def _resolve_named_axis(
    *,
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named Mini-golf axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{namespace_root}.{namespace}",
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(value) for value in supported],
    )


def _string_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    """Resolve a string support list from params/defaults."""

    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    if raw is None:
        raw = tuple(fallback)
    values = (str(raw),) if isinstance(raw, str) else tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one label")
    return values


def _safe_obstacle_position(
    *,
    rng: Any,
    existing: Sequence[MinigolfObstacle],
    avoid_points: Sequence[Tuple[float, float]],
    avoid_paths: Sequence[Tuple[Tuple[float, float], ...]],
) -> Tuple[float, float] | None:
    """Sample an obstacle position away from key points and target paths."""

    for _ in range(160):
        x = float(rng.uniform(0.14, 0.86))
        y = float(rng.uniform(0.16, 0.76))
        point = (x, y)
        if any(distance(point, other) < MIN_OBSTACLE_POINT_CLEARANCE_NORM for other in avoid_points):
            continue
        if any(
            distance(point, (float(obs.x_norm), float(obs.y_norm))) < MIN_OBSTACLE_OBSTACLE_CLEARANCE_NORM
            for obs in existing
        ):
            continue
        too_close_to_path = False
        for path in avoid_paths:
            for a, b in zip(path, path[1:]):
                ax, ay = float(a[0]), float(a[1])
                bx, by = float(b[0]), float(b[1])
                vx, vy = bx - ax, by - ay
                denom = (vx * vx) + (vy * vy)
                if denom <= 1e-9:
                    continue
                t = max(0.0, min(1.0, (((x - ax) * vx) + ((y - ay) * vy)) / denom))
                closest = (ax + (t * vx), ay + (t * vy))
                if distance(point, closest) < MIN_OBSTACLE_PATH_CLEARANCE_NORM:
                    too_close_to_path = True
                    break
            if too_close_to_path:
                break
        if too_close_to_path:
            continue
        return point
    return None


def _obstacles_with_target_label(
    *,
    rng: Any,
    target_label: str,
    target_center: Tuple[float, float],
    obstacle_count: int,
    avoid_paths: Sequence[Tuple[Tuple[float, float], ...]],
    avoid_points: Sequence[Tuple[float, float]],
) -> Tuple[MinigolfObstacle, ...] | None:
    """Create obstacle set with the target label on the first-hit obstacle."""

    labels = [str(label) for label in OBSTACLE_LABELS[: int(obstacle_count)]]
    if str(target_label) not in labels:
        return None
    target_index = labels.index(str(target_label))
    if any(distance(target_center, avoid_point) < MIN_OBSTACLE_POINT_CLEARANCE_NORM for avoid_point in avoid_points):
        return None
    obstacles: list[MinigolfObstacle] = []
    obstacle_kind_cycle = cycle(shuffled_support(rng, OBSTACLE_KINDS))
    for index, label in enumerate(labels):
        if int(index) == int(target_index):
            center = (float(target_center[0]), float(target_center[1]))
            if any(
                distance(center, (float(obs.x_norm), float(obs.y_norm))) < MIN_OBSTACLE_OBSTACLE_CLEARANCE_NORM
                for obs in obstacles
            ):
                return None
        else:
            maybe = _safe_obstacle_position(
                rng=rng,
                existing=obstacles,
                avoid_points=tuple(avoid_points) + (target_center,),
                avoid_paths=avoid_paths,
            )
            if maybe is None:
                return None
            center = maybe
        obstacles.append(
            MinigolfObstacle(
                obstacle_id=obstacle_entity_id(index),
                label=str(label),
                kind=str(next(obstacle_kind_cycle)),
                x_norm=float(center[0]),
                y_norm=float(center[1]),
                radius_norm=float(OBSTACLE_RADIUS_NORM * rng.uniform(0.88, 1.10)),
                color_index=int(index),
            )
        )
    return tuple(obstacles)


def _make_shot_options(
    *,
    rng: Any,
    option_count: int,
    target_index: int,
    target_angle: float,
    ball: Tuple[float, float],
    hole: Tuple[float, float],
    obstacles: Sequence[MinigolfObstacle],
) -> Tuple[MinigolfShotOption, ...] | None:
    """Create shot options with exactly one hole-reaching option."""

    offsets = [-0.96, -0.76, -0.56, -0.38, 0.38, 0.56, 0.76, 0.96]
    rng.shuffle(offsets)
    angles: list[float | None] = [None for _ in range(int(option_count))]
    angles[int(target_index)] = float(target_angle)
    for index in range(int(option_count)):
        if angles[index] is not None:
            continue
        found = False
        for _ in range(80):
            if offsets:
                candidate = normalize_angle(float(target_angle) + float(offsets.pop()))
            else:
                candidate = float(rng.uniform(-2.82, -0.32))
            success, _, _ = trace_shot_path(ball_xy=ball, angle_rad=float(candidate), hole_xy=hole, obstacles=obstacles)
            if success:
                continue
            if any(existing is not None and abs(normalize_angle(float(existing) - float(candidate))) < 0.22 for existing in angles):
                continue
            angles[index] = float(candidate)
            found = True
            break
        if not found:
            return None
    return tuple(
        MinigolfShotOption(
            path_id=path_entity_id(index),
            label=path_label(index),
            angle_rad=float(angles[index]),
            color_index=int(index),
        )
        for index in range(int(option_count))
    )


__all__ = [
    "MIN_OBSTACLE_OBSTACLE_CLEARANCE_NORM",
    "MIN_OBSTACLE_PATH_CLEARANCE_NORM",
    "MIN_OBSTACLE_POINT_CLEARANCE_NORM",
    "MinigolfAxes",
    "resolve_minigolf_axes",
    "resolve_minigolf_integer_choice",
    "resolve_minigolf_label_choice",
    "sample_first_obstacle_scene",
    "sample_shot_options_scene",
]
