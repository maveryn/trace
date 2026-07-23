"""Identity-free sampling primitives for pool-table scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.games.shared.style import SUPPORTED_POOL_STYLE_VARIANTS

from .defaults import DEFAULTS
from .rules import (
    POOL_BALL_NUMBERS,
    SUPPORTED_POOL_SCENE_VARIANTS,
    ball_entity_id,
    ball_group,
    balls_on_segment,
    point_distance,
)
from .state import POOL_POCKETS, PoolBall, PoolSceneState, validate_pool_scene_state


@dataclass(frozen=True)
class PoolIntegerAxis:
    """Resolved integer target axis for one public pool objective."""

    target_value: int
    target_value_support: Tuple[int, ...]
    target_value_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PoolVisualAxes:
    """Resolved semantic and visual axes shared by pool objectives."""

    scene_variant: str
    style_variant: str
    object_ball_count: int
    line_clearance: float
    min_ball_distance: float
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    object_ball_count_probabilities: Dict[str, float]


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named pool visual axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(item) for item in supported],
    )


def resolve_pool_visual_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> PoolVisualAxes:
    """Resolve shared visual and table-placement axes for one pool instance."""

    cycle_params = dict(params)
    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=cycle_params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_POOL_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=cycle_params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_POOL_STYLE_VARIANTS,
    )
    object_ball_count, object_ball_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=cycle_params,
        gen_defaults=gen_defaults,
        support_key="object_ball_count_support",
        explicit_key="object_ball_count",
        fallback_support=DEFAULTS.object_ball_count_support,
        namespace=f"{str(namespace)}.object_ball_count",
        balanced_flag_key="balanced_object_ball_count_sampling",
        namespace_support_permutation=True,
    )
    return PoolVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        object_ball_count=int(object_ball_count),
        line_clearance=float(params.get("line_clearance", group_default(gen_defaults, "line_clearance", DEFAULTS.line_clearance))),
        min_ball_distance=float(params.get("min_ball_distance", group_default(gen_defaults, "min_ball_distance", DEFAULTS.min_ball_distance))),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        object_ball_count_probabilities=dict(object_ball_count_probabilities),
    )


def resolve_pool_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str = "balanced_target_answer_sampling",
) -> PoolIntegerAxis:
    """Resolve one task-owned integer target axis from config/defaults."""

    target_value, target_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return PoolIntegerAxis(
        target_value=int(target_value),
        target_value_support=tuple(int(value) for value in support),
        target_value_probabilities=dict(target_probabilities),
    )


def _is_inside_table(point: Tuple[float, float]) -> bool:
    """Return whether a normalized point is inside the playable table area."""

    x, y = point
    return 0.075 <= float(x) <= 0.925 and 0.095 <= float(y) <= 0.905


def _add_ball(
    balls: list[PoolBall],
    *,
    number: int,
    center: Tuple[float, float],
    is_marked: bool = False,
) -> None:
    """Append one ball to the generated layout."""

    balls.append(
        PoolBall(
            ball_id=ball_entity_id(int(number)),
            number=int(number),
            group=ball_group(int(number)),
            center=(round(float(center[0]), 5), round(float(center[1]), 5)),
            is_cue=bool(int(number) == 0),
            is_marked=bool(is_marked),
        )
    )


def _position_available(center: Tuple[float, float], balls: Sequence[PoolBall], *, min_distance: float) -> bool:
    """Return whether a new ball can be placed at a center."""

    return _is_inside_table(center) and all(point_distance(center, ball.center) >= float(min_distance) for ball in balls)


def _sample_free_position(rng: Any, balls: Sequence[PoolBall], *, min_distance: float) -> Tuple[float, float]:
    """Sample one non-overlapping object-ball position."""

    for _ in range(800):
        center = (float(rng.uniform(0.10, 0.90)), float(rng.uniform(0.13, 0.87)))
        if _position_available(center, balls, min_distance=float(min_distance)):
            return center
    raise ValueError("failed to sample non-overlapping pool ball")


def sample_numbered_pool_scene(
    *,
    rng: Any,
    axes: PoolVisualAxes,
    object_numbers: Sequence[int],
    current_player_group: str,
) -> PoolSceneState:
    """Lay out a cue ball plus a task-selected set of object-ball numbers."""

    group = str(current_player_group)
    if group not in {"solid", "stripe"}:
        raise ValueError(f"unsupported current_player_group={group!r}")
    selected = [int(number) for number in object_numbers]
    if len(selected) != len(set(selected)):
        raise ValueError("pool object-ball numbers must be unique")
    if any(number not in POOL_BALL_NUMBERS for number in selected):
        raise ValueError("pool object-ball number is outside the supported ball set")

    balls: list[PoolBall] = []
    cue_center = (float(rng.uniform(0.135, 0.235)), float(rng.uniform(0.42, 0.58)))
    _add_ball(balls, number=0, center=cue_center)
    for number in selected:
        _add_ball(
            balls,
            number=int(number),
            center=_sample_free_position(rng, balls, min_distance=float(axes.min_ball_distance)),
        )

    state = PoolSceneState(
        scene_variant=str(axes.scene_variant),
        balls=tuple(balls),
        pockets=POOL_POCKETS,
        cue_ball_id="cue_ball",
        marked_ball_id=None,
        marked_pocket_id=None,
        current_player_group=str(group),
        construction_mode=f"selected_numbered_balls_{group}",
    )
    validate_pool_scene_state(state)
    return state


def _point_on_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    t: float,
    offset: float,
) -> Tuple[float, float]:
    """Return one point near a shot segment."""

    sx, sy = start
    ex, ey = end
    vx = float(ex - sx)
    vy = float(ey - sy)
    length = max(1e-6, (vx * vx + vy * vy) ** 0.5)
    nx = -vy / length
    ny = vx / length
    return (float(sx + (t * vx) + (offset * nx)), float(sy + (t * vy) + (offset * ny)))


def sample_marked_shot_pool_scene(
    *,
    rng: Any,
    axes: PoolVisualAxes,
    target_answer: int,
) -> PoolSceneState:
    """Construct a marked two-segment shot with controlled blocker count."""

    min_distance = float(axes.min_ball_distance)
    clearance = float(axes.line_clearance)
    cue_center = (float(rng.uniform(0.145, 0.185)), float(rng.uniform(0.48, 0.56)))
    target_center = (float(rng.uniform(0.48, 0.56)), float(rng.uniform(0.36, 0.46)))
    pocket = POOL_POCKETS[2] if target_center[1] < 0.50 else POOL_POCKETS[5]
    target_number = int(rng.choice([2, 3, 4, 5, 6, 9, 10, 11, 12, 13]))
    balls: list[PoolBall] = []
    _add_ball(balls, number=0, center=cue_center)
    _add_ball(balls, number=target_number, center=target_center, is_marked=True)

    used_numbers = {0, target_number}
    blocker_ts = list(shuffled_support(rng, (0.34, 0.54, 0.72, 0.86)))
    for index in range(int(target_answer)):
        segment_start, segment_end = (cue_center, target_center) if index % 2 == 0 else (target_center, pocket.center)
        center = _point_on_segment(
            segment_start,
            segment_end,
            t=blocker_ts[int(index)],
            offset=float(rng.uniform(-0.010, 0.010)),
        )
        if not _position_available(center, balls, min_distance=float(min_distance) * 0.70):
            raise ValueError("failed to place pool blocker")
        number = next(value for value in POOL_BALL_NUMBERS if value not in used_numbers)
        used_numbers.add(int(number))
        _add_ball(balls, number=int(number), center=center)

    foil_count = int(rng.randint(5, 8))
    for _ in range(foil_count):
        number = next(value for value in POOL_BALL_NUMBERS if value not in used_numbers)
        used_numbers.add(int(number))
        for _attempt in range(400):
            center = _sample_free_position(rng, balls, min_distance=float(min_distance))
            candidate = PoolBall("candidate", number, ball_group(number), center)
            if (
                point_distance(center, cue_center) > 0.11
                and point_distance(center, target_center) > 0.11
                and point_distance(center, pocket.center) > 0.11
                and not balls_on_segment(
                    balls=[candidate],
                    start=cue_center,
                    end=target_center,
                    ignore_ball_ids=(),
                    clearance=float(clearance),
                )
                and not balls_on_segment(
                    balls=[candidate],
                    start=target_center,
                    end=pocket.center,
                    ignore_ball_ids=(),
                    clearance=float(clearance),
                )
            ):
                _add_ball(balls, number=int(number), center=center)
                break
        else:
            raise ValueError("failed to place pool foil")

    state = PoolSceneState(
        scene_variant=str(axes.scene_variant),
        balls=tuple(balls),
        pockets=POOL_POCKETS,
        cue_ball_id="cue_ball",
        marked_ball_id=ball_entity_id(target_number),
        marked_pocket_id=str(pocket.pocket_id),
        current_player_group=None,
        construction_mode="marked_two_segment_shot_controlled_blockers",
    )
    validate_pool_scene_state(state)
    return state


__all__ = [
    "PoolIntegerAxis",
    "PoolVisualAxes",
    "resolve_pool_integer_axis",
    "resolve_pool_visual_axes",
    "sample_marked_shot_pool_scene",
    "sample_numbered_pool_scene",
]
