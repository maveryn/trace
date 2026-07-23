"""Identity-free sampling helpers for pinball-table game scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import shuffled_support, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice

from .defaults import (
    BUMPER_RADIUS_NORM,
    DEFAULTS,
    DROP_TARGET_HEIGHT_NORM,
    DROP_TARGET_WIDTH_NORM,
    OBJECT_LABELS,
    PATH_SCORE_VALUES,
    ROLLOVER_HEIGHT_NORM,
    ROLLOVER_WIDTH_NORM,
    STANDUP_RADIUS_NORM,
)
from .state import (
    SUPPORTED_PINBALL_OBJECT_KINDS,
    SUPPORTED_PINBALL_SCENE_VARIANTS,
    SUPPORTED_PINBALL_STYLE_VARIANTS,
    PinballObject,
    PinballSceneState,
    pinball_object_id,
    validate_pinball_scene_state,
)


@dataclass(frozen=True)
class PinballVisualAxes:
    """Resolved scene/render axes shared by pinball-table tasks."""

    scene_variant: str
    style_variant: str
    object_count: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    object_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PinballTargetLabelAxis:
    """Resolved visible-object label target for label-selection tasks."""

    target_object_label: str
    target_object_label_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PinballFirstHitConstruction:
    """Constructed launch scene with the unique first-hit object bound."""

    scene: PinballSceneState
    target_object_id: str
    target_object_label: str


@dataclass(frozen=True)
class PinballScoreableObjectCountConstruction:
    """Constructed mixed-score scene with scoreable object ids bound."""

    scene: PinballSceneState
    annotation_entity_ids: Tuple[str, ...]
    scoreable_count: int
    score_values: Tuple[int, ...]


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
    """Resolve one balanced named pinball axis."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(value) for value in supported],
    )


def resolve_pinball_visual_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> PinballVisualAxes:
    """Resolve shared visual axes without public task or query routing."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_PINBALL_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_PINBALL_STYLE_VARIANTS,
    )
    object_count, object_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="object_count_support",
        explicit_key="object_count",
        fallback_support=DEFAULTS.object_count_support,
        namespace=f"{str(namespace)}.object_count",
        balanced_flag_key="balanced_object_count_sampling",
        namespace_support_permutation=True,
    )
    return PinballVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        object_count=int(object_count),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        object_count_probabilities=dict(object_count_probabilities),
    )


def resolve_pinball_target_label(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    object_count: int,
) -> PinballTargetLabelAxis:
    """Resolve a target label from labels visible in the sampled scene."""

    support = tuple(OBJECT_LABELS[: int(object_count)])
    explicit = params.get("target_object_label")
    probabilities = {str(label): 1.0 / float(len(support)) for label in support}
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError("target_object_label must be visible for the sampled object_count")
        return PinballTargetLabelAxis(
            target_object_label=value,
            target_object_label_probabilities={str(label): (1.0 if str(label) == value else 0.0) for label in support},
        )
    rng = spawn_rng(int(instance_seed), str(namespace))
    return PinballTargetLabelAxis(
        target_object_label=str(uniform_choice(rng, support)),
        target_object_label_probabilities=probabilities,
    )


def _unit_from_angle(angle_rad: float) -> Tuple[float, float]:
    """Return a unit vector for an angle in normalized table coordinates."""

    return (math.cos(float(angle_rad)), math.sin(float(angle_rad)))


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return Euclidean distance between two normalized table points."""

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _distance_to_segment(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Return point distance to one line segment."""

    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    vx, vy = bx - ax, by - ay
    denom = (vx * vx) + (vy * vy)
    if denom <= 1e-9:
        return _distance(point, a)
    t = max(0.0, min(1.0, (((px - ax) * vx) + ((py - ay) * vy)) / denom))
    closest = (ax + (t * vx), ay + (t * vy))
    return _distance(point, closest)


def _ray_circle_intersection(
    *,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> float | None:
    """Return first positive ray/circle intersection distance."""

    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    cx, cy = float(center[0]), float(center[1])
    fx, fy = ox - cx, oy - cy
    a = (dx * dx) + (dy * dy)
    b = 2.0 * ((fx * dx) + (fy * dy))
    c = (fx * fx) + (fy * fy) - (float(radius) * float(radius))
    disc = (b * b) - (4.0 * a * c)
    if disc < 0.0 or a <= 1e-9:
        return None
    root = math.sqrt(disc)
    values = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    candidates = [float(t) for t in values if float(t) >= 1e-5]
    return None if not candidates else min(candidates)


def _ray_rect_intersection(
    *,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    center: Tuple[float, float],
    width: float,
    height: float,
) -> float | None:
    """Return first positive ray/axis-aligned-rectangle intersection distance."""

    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    cx, cy = float(center[0]), float(center[1])
    left = cx - float(width) / 2.0
    right = cx + float(width) / 2.0
    top = cy - float(height) / 2.0
    bottom = cy + float(height) / 2.0
    if abs(dx) <= 1e-9:
        if ox < left or ox > right:
            return None
        tx_min, tx_max = -1.0e9, 1.0e9
    else:
        tx1 = (left - ox) / dx
        tx2 = (right - ox) / dx
        tx_min, tx_max = min(tx1, tx2), max(tx1, tx2)
    if abs(dy) <= 1e-9:
        if oy < top or oy > bottom:
            return None
        ty_min, ty_max = -1.0e9, 1.0e9
    else:
        ty1 = (top - oy) / dy
        ty2 = (bottom - oy) / dy
        ty_min, ty_max = min(ty1, ty2), max(ty1, ty2)
    t_enter = max(float(tx_min), float(ty_min))
    t_exit = min(float(tx_max), float(ty_max))
    if t_exit < max(float(t_enter), 1e-5):
        return None
    return float(t_enter) if float(t_enter) >= 1e-5 else float(t_exit)


def _object_ray_intersection(
    *,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    obj: PinballObject,
) -> float | None:
    """Return first positive ray intersection distance for one pinball object."""

    center = (float(obj.x_norm), float(obj.y_norm))
    if str(obj.kind) in {"bumper", "standup_target"}:
        return _ray_circle_intersection(
            origin=origin,
            direction=direction,
            center=center,
            radius=float(obj.radius_norm),
        )
    return _ray_rect_intersection(
        origin=origin,
        direction=direction,
        center=center,
        width=float(obj.width_norm),
        height=float(obj.height_norm),
    )


def first_hit_object_id(
    *,
    origin: Tuple[float, float],
    angle_rad: float,
    objects: Sequence[PinballObject],
) -> str | None:
    """Return the first object intersected by the launch ray."""

    direction = _unit_from_angle(float(angle_rad))
    hits: list[Tuple[float, str]] = []
    for obj in objects:
        t = _object_ray_intersection(origin=origin, direction=direction, obj=obj)
        if t is not None:
            hits.append((float(t), str(obj.object_id)))
    if not hits:
        return None
    hits.sort(key=lambda item: item[0])
    if len(hits) >= 2 and abs(float(hits[1][0]) - float(hits[0][0])) < 0.035:
        return None
    return str(hits[0][1])


def _safe_object_position(
    *,
    rng: Any,
    kind: str,
    existing: Sequence[PinballObject],
    ball: Tuple[float, float],
    target: Tuple[float, float],
    path: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[float, float] | None:
    """Sample one distractor position away from key points and the first-hit path."""

    for _ in range(180):
        x, y = _sample_zone_position(rng=rng, kind=str(kind))
        point = (x, y)
        if _distance(point, ball) < 0.18 or _distance(point, target) < 0.16:
            continue
        if any(_distance(point, (float(obj.x_norm), float(obj.y_norm))) < 0.14 for obj in existing):
            continue
        if _distance_to_segment(point, path[0], path[1]) < 0.095:
            continue
        return point
    return None


def _kind_for_target_center(*, y_norm: float, rng: Any) -> str:
    """Choose an object kind that fits the playfield zone."""

    y = float(y_norm)
    if y < 0.30:
        return str(rng.choice(("drop_target", "rollover_lane")))
    if y < 0.58:
        return str(rng.choice(("bumper", "bumper", "standup_target")))
    return str(rng.choice(("standup_target", "drop_target")))


def _sample_zone_position(*, rng: Any, kind: str) -> Tuple[float, float]:
    """Sample a plausible normalized playfield zone for one object kind."""

    kind_value = str(kind)
    if kind_value == "bumper":
        return (float(rng.uniform(0.27, 0.73)), float(rng.uniform(0.30, 0.56)))
    if kind_value == "rollover_lane":
        return (float(rng.uniform(0.22, 0.78)), float(rng.uniform(0.14, 0.32)))
    if kind_value == "drop_target":
        if float(rng.random()) < 0.64:
            return (float(rng.uniform(0.20, 0.80)), float(rng.uniform(0.16, 0.36)))
        return (float(rng.choice((rng.uniform(0.14, 0.28), rng.uniform(0.72, 0.86)))), float(rng.uniform(0.42, 0.66)))
    return (float(rng.uniform(0.16, 0.84)), float(rng.uniform(0.34, 0.68)))


def _make_object(
    *,
    index: int,
    label: str,
    kind: str,
    center: Tuple[float, float],
    rng: Any,
    score_value: int | None = None,
    show_label: bool = True,
) -> PinballObject:
    """Create one pinball object with kind-appropriate footprint."""

    kind_value = str(kind)
    radius_norm = BUMPER_RADIUS_NORM
    width_norm = DROP_TARGET_WIDTH_NORM
    height_norm = DROP_TARGET_HEIGHT_NORM
    if kind_value == "bumper":
        radius_norm = float(BUMPER_RADIUS_NORM * float(rng.uniform(0.88, 1.12)))
        width_norm = float(DROP_TARGET_WIDTH_NORM * float(rng.uniform(0.90, 1.12)))
        height_norm = float(DROP_TARGET_HEIGHT_NORM * float(rng.uniform(0.90, 1.12)))
    elif kind_value == "standup_target":
        radius_norm = float(STANDUP_RADIUS_NORM * float(rng.uniform(0.90, 1.10)))
        width_norm = float(DROP_TARGET_WIDTH_NORM * float(rng.uniform(0.90, 1.12)))
        height_norm = float(DROP_TARGET_HEIGHT_NORM * float(rng.uniform(0.90, 1.12)))
    elif kind_value == "rollover_lane":
        radius_norm = float(STANDUP_RADIUS_NORM * float(rng.uniform(0.90, 1.10)))
        width_norm = float(ROLLOVER_WIDTH_NORM * float(rng.uniform(0.90, 1.10)))
        height_norm = float(ROLLOVER_HEIGHT_NORM * float(rng.uniform(0.90, 1.10)))
    else:
        radius_norm = float(STANDUP_RADIUS_NORM * float(rng.uniform(0.90, 1.10)))
        width_norm = float(DROP_TARGET_WIDTH_NORM * float(rng.uniform(0.90, 1.12)))
        height_norm = float(DROP_TARGET_HEIGHT_NORM * float(rng.uniform(0.90, 1.12)))
    return PinballObject(
        object_id=pinball_object_id(index),
        label=str(label),
        kind=kind_value,
        x_norm=float(center[0]),
        y_norm=float(center[1]),
        radius_norm=float(radius_norm),
        width_norm=float(width_norm),
        height_norm=float(height_norm),
        color_index=int(index),
        score_value=None if score_value is None else int(score_value),
        show_label=bool(show_label),
    )


def _objects_with_target_label(
    *,
    rng: Any,
    axes: PinballVisualAxes,
    target_object_label: str,
    ball: Tuple[float, float],
    target: Tuple[float, float],
    path: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[PinballObject, ...] | None:
    """Create labeled objects with a requested label placed on the first-hit object."""

    labels = list(OBJECT_LABELS[: int(axes.object_count)])
    rng.shuffle(labels)
    target_index = labels.index(str(target_object_label))
    target_kind = _kind_for_target_center(y_norm=float(target[1]), rng=rng)
    kind_pool = list(SUPPORTED_PINBALL_OBJECT_KINDS)
    rng.shuffle(kind_pool)
    kind_plan: list[str] = []
    while len(kind_plan) < int(axes.object_count):
        shuffled = list(kind_pool)
        rng.shuffle(shuffled)
        kind_plan.extend(str(kind) for kind in shuffled)
    objects: list[PinballObject] = []
    for index, label in enumerate(labels):
        kind = str(target_kind if int(index) == int(target_index) else kind_plan[int(index)])
        if int(index) == int(target_index):
            center = target
        else:
            maybe = _safe_object_position(
                rng=rng,
                kind=kind,
                existing=objects,
                ball=ball,
                target=target,
                path=path,
            )
            if maybe is None:
                return None
            center = maybe
        objects.append(_make_object(index=index, label=str(label), kind=kind, center=center, rng=rng))
    return tuple(objects)


def sample_unique_first_hit_playfield(
    *,
    rng: Any,
    axes: PinballVisualAxes,
    target_object_label: str,
) -> PinballFirstHitConstruction:
    """Construct a playfield where the launch cue first hits one requested label."""

    for _attempt in range(180):
        ball = (float(rng.uniform(0.38, 0.62)), float(rng.uniform(0.82, 0.90)))
        angle = float(rng.uniform(-2.34, -0.80))
        direction = _unit_from_angle(angle)
        distance = float(rng.uniform(0.42, 0.66))
        target = (ball[0] + (direction[0] * distance), ball[1] + (direction[1] * distance))
        if not (0.16 <= target[0] <= 0.84 and 0.14 <= target[1] <= 0.70):
            continue
        path = (ball, target)
        objects = _objects_with_target_label(
            rng=rng,
            axes=axes,
            target_object_label=str(target_object_label),
            ball=ball,
            target=target,
            path=path,
        )
        if objects is None:
            continue
        target_object = next(obj for obj in objects if str(obj.label) == str(target_object_label))
        first_id = first_hit_object_id(origin=ball, angle_rad=angle, objects=objects)
        if str(first_id) != str(target_object.object_id):
            continue
        scene = PinballSceneState(
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            ball_x_norm=float(ball[0]),
            ball_y_norm=float(ball[1]),
            cue_angle_rad=float(angle),
            cue_visible_fraction=float(rng.uniform(0.34, 0.46)),
            objects=objects,
            construction_mode="unique_straight_launch_first_hit_projected_playfield",
            hidden_path_norm=tuple(path),
        )
        validate_pinball_scene_state(scene)
        return PinballFirstHitConstruction(
            scene=scene,
            target_object_id=str(target_object.object_id),
            target_object_label=str(target_object.label),
        )
    raise ValueError("failed to construct pinball first-hit scene")


def _safe_mixed_object_position(
    *,
    rng: Any,
    kind: str,
    existing: Sequence[PinballObject],
    ball: Tuple[float, float],
) -> Tuple[float, float] | None:
    """Sample one mixed-score object position away from the ball and objects."""

    for _attempt in range(220):
        point = _sample_zone_position(rng=rng, kind=str(kind))
        if _distance(point, ball) < 0.18:
            continue
        if any(_distance(point, (float(obj.x_norm), float(obj.y_norm))) < 0.14 for obj in existing):
            continue
        return point
    return None


def sample_scoreable_object_count_playfield(
    *,
    rng: Any,
    axes: PinballVisualAxes,
    scoreable_count: int,
) -> PinballScoreableObjectCountConstruction:
    """Construct a playfield with mixed numeric-score and blank objects."""

    if int(scoreable_count) < 1:
        raise ValueError("scoreable_count must be positive")
    if int(scoreable_count) >= int(axes.object_count):
        raise ValueError("scoreable_count must leave at least one non-scoreable object")

    for _attempt in range(240):
        ball = (float(rng.uniform(0.38, 0.62)), float(rng.uniform(0.82, 0.90)))
        cue_angle = float(rng.uniform(-2.34, -0.80))
        cue_direction = _unit_from_angle(cue_angle)
        hidden_end = (
            float(ball[0] + (cue_direction[0] * 0.48)),
            float(ball[1] + (cue_direction[1] * 0.48)),
        )
        labels = list(OBJECT_LABELS[: int(axes.object_count)])
        rng.shuffle(labels)
        scoreable_indices = set(rng.sample(range(int(axes.object_count)), int(scoreable_count)))
        kind_cycle = cycle(shuffled_support(rng, SUPPORTED_PINBALL_OBJECT_KINDS))
        objects: list[PinballObject] = []
        score_values: list[int] = []
        annotation_ids: list[str] = []
        for index, label in enumerate(labels):
            kind = str(next(kind_cycle))
            center = _safe_mixed_object_position(
                rng=rng,
                kind=kind,
                existing=objects,
                ball=ball,
            )
            if center is None:
                break
            is_scoreable = int(index) in scoreable_indices
            score_value = int(rng.choice(PATH_SCORE_VALUES)) if is_scoreable else None
            obj = _make_object(
                index=index,
                label=str(label),
                kind=kind,
                center=center,
                rng=rng,
                score_value=score_value,
                show_label=False,
            )
            objects.append(obj)
            if is_scoreable:
                annotation_ids.append(str(obj.object_id))
                score_values.append(int(score_value or 0))
        if len(objects) != int(axes.object_count):
            continue
        if len(annotation_ids) != int(scoreable_count):
            continue
        scene = PinballSceneState(
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            ball_x_norm=float(ball[0]),
            ball_y_norm=float(ball[1]),
            cue_angle_rad=float(cue_angle),
            cue_visible_fraction=float(rng.uniform(0.30, 0.42)),
            objects=tuple(objects),
            construction_mode="mixed_scoreable_object_count",
            hidden_path_norm=(ball, hidden_end),
        )
        validate_pinball_scene_state(scene)
        return PinballScoreableObjectCountConstruction(
            scene=scene,
            annotation_entity_ids=tuple(annotation_ids),
            scoreable_count=int(scoreable_count),
            score_values=tuple(int(value) for value in score_values),
        )
    raise ValueError("failed to construct pinball scoreable-object count scene")


__all__ = [
    "PinballFirstHitConstruction",
    "PinballScoreableObjectCountConstruction",
    "PinballTargetLabelAxis",
    "PinballVisualAxes",
    "first_hit_object_id",
    "resolve_pinball_target_label",
    "resolve_pinball_visual_axes",
    "sample_scoreable_object_count_playfield",
    "sample_unique_first_hit_playfield",
]
