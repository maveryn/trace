"""Sampling primitives for platformer game scenes."""

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
    BONUS_COLLECTIBLE_KINDS,
    DEFAULTS,
    HAZARD_KINDS,
    PLATFORM_LABELS,
    SUPPORTED_PLATFORMER_SCENE_VARIANTS,
    SUPPORTED_PLATFORMER_STYLE_VARIANTS,
)
from .state import (
    PlatformerCollectible,
    PlatformerHazard,
    PlatformerPlatform,
    PlatformerSample,
    collectible_entity_id,
    hazard_entity_id,
    platform_entity_id,
    validate_platformer_sample,
)


@dataclass(frozen=True)
class PlatformerVisualAxes:
    """Resolved visual and count axes for one platformer instance."""

    scene_variant: str
    style_variant: str
    platform_count: int
    hazard_count: int
    distractor_collectible_count: int
    jump_visible_after_peak_min: float
    jump_visible_after_peak_max: float
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    platform_count_probabilities: Dict[str, float]
    hazard_count_probabilities: Dict[str, float]
    distractor_collectible_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PlatformerLabelAxis:
    """Resolved label-valued target support for a platformer task."""

    target_label: str
    target_label_support: Tuple[str, ...]
    target_label_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PlatformerIntegerAxis:
    """Resolved integer-valued target support for a platformer task."""

    target_value: int
    target_value_support: Tuple[int, ...]
    target_value_probabilities: Dict[str, float]


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
    """Resolve one balanced named Platformer axis."""

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


def resolve_platformer_visual_axes(
    instance_seed: int,
    *,
    gen_defaults: Mapping[str, Any],
    namespace: str,
    params: Mapping[str, Any],
) -> PlatformerVisualAxes:
    """Resolve platformer scene-level visual and count axes."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_PLATFORMER_SCENE_VARIANTS,
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
        supported=SUPPORTED_PLATFORMER_STYLE_VARIANTS,
    )
    platform_count, platform_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="platform_count_support",
        explicit_key="platform_count",
        fallback_support=DEFAULTS.platform_count_support,
        namespace=f"{str(namespace)}.platform_count",
        balanced_flag_key="balanced_platform_count_sampling",
        namespace_support_permutation=True,
    )
    hazard_count, hazard_count_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="hazard_count_support",
        explicit_key="hazard_count",
        fallback_support=DEFAULTS.hazard_count_support,
        namespace=f"{str(namespace)}.hazard_count",
        balanced_flag_key="balanced_hazard_count_sampling",
        namespace_support_permutation=True,
    )
    distractor_count, distractor_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="distractor_collectible_count_support",
        explicit_key="distractor_collectible_count",
        fallback_support=DEFAULTS.distractor_collectible_count_support,
        namespace=f"{str(namespace)}.distractor_collectible_count",
        balanced_flag_key="balanced_distractor_collectible_count_sampling",
        namespace_support_permutation=True,
    )
    return PlatformerVisualAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        platform_count=int(platform_count),
        hazard_count=int(hazard_count),
        distractor_collectible_count=int(distractor_count),
        jump_visible_after_peak_min=float(group_default(gen_defaults, "jump_visible_after_peak_min", DEFAULTS.jump_visible_after_peak_min)),
        jump_visible_after_peak_max=float(group_default(gen_defaults, "jump_visible_after_peak_max", DEFAULTS.jump_visible_after_peak_max)),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        platform_count_probabilities=dict(platform_count_probabilities),
        hazard_count_probabilities=dict(hazard_count_probabilities),
        distractor_collectible_count_probabilities=dict(distractor_probabilities),
    )


def _string_support(params: Mapping[str, Any], *, gen_defaults: Mapping[str, Any], key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
    """Resolve a string support list from params/defaults."""

    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    if raw is None:
        raw = tuple(fallback)
    values = (str(raw),) if isinstance(raw, str) else tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one label")
    return values


def resolve_platformer_label_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str] = PLATFORM_LABELS,
    namespace: str,
    balanced_flag_key: str,
) -> PlatformerLabelAxis:
    """Resolve one label-valued support choice for a public objective."""

    support = _string_support(params, gen_defaults=gen_defaults, key=str(support_key), fallback=fallback_support)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{explicit_key}={value!r} is not in {support_key}")
        probabilities = {str(item): (1.0 if str(item) == value else 0.0) for item in support}
        return PlatformerLabelAxis(value, tuple(support), probabilities)
    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    rng = spawn_rng(int(instance_seed), str(namespace))
    value = str(uniform_choice(rng, support))
    return PlatformerLabelAxis(value, tuple(support), probabilities)


def resolve_platformer_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> PlatformerIntegerAxis:
    """Resolve one integer-valued support choice for a public objective."""

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
    support = tuple(int(key) for key in probabilities)
    return PlatformerIntegerAxis(
        target_value=int(value),
        target_value_support=support,
        target_value_probabilities=dict(probabilities),
    )


def integer_support_from_defaults(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve an integer support list used by task-owned scoring objectives."""

    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    values = tuple(int(value) for value in (raw if isinstance(raw, (list, tuple)) else (raw,)))
    if not values:
        raise ValueError(f"{key} must contain at least one integer")
    return values


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Return Euclidean distance in normalized level coordinates."""

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _point_inside_expanded_rect(
    point: Tuple[float, float],
    rect: Tuple[float, float, float, float],
    *,
    margin: float,
) -> bool:
    """Return true when a normalized point is inside one expanded center-size rect."""

    px, py = float(point[0]), float(point[1])
    cx, cy, width, height = (float(value) for value in rect)
    return (
        abs(px - cx) <= (0.5 * float(width)) + float(margin)
        and abs(py - cy) <= (0.5 * float(height)) + float(margin)
    )


def _curve_point(
    *,
    start: Tuple[float, float],
    control: Tuple[float, float],
    end: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """Return one quadratic Bezier point."""

    u = 1.0 - float(t)
    return (
        (u * u * float(start[0])) + (2.0 * u * float(t) * float(control[0])) + (float(t) * float(t) * float(end[0])),
        (u * u * float(start[1])) + (2.0 * u * float(t) * float(control[1])) + (float(t) * float(t) * float(end[1])),
    )


def _jump_arc(
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    rng: Any,
    lift: float | None = None,
    segments: int = 40,
) -> Tuple[Tuple[float, float], ...]:
    """Construct a smooth side-scroller jump arc."""

    control_x = ((float(start[0]) + float(end[0])) / 2.0) + float(rng.uniform(-0.035, 0.035))
    control_y = min(float(start[1]), float(end[1])) - float(lift if lift is not None else rng.uniform(0.24, 0.34))
    control_y = max(0.08, float(control_y))
    return tuple(
        _curve_point(start=start, control=(control_x, control_y), end=end, t=float(index) / float(segments))
        for index in range(int(segments) + 1)
    )


def _point_segment_distance(point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distance from a point to one segment in normalized coordinates."""

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


def _point_path_distance(point: Tuple[float, float], path: Sequence[Tuple[float, float]]) -> float:
    """Distance from a point to a polyline."""

    if len(path) < 2:
        return 1.0
    return min(_point_segment_distance(point, a, b) for a, b in zip(path, path[1:]))


def _safe_center(
    *,
    rng: Any,
    existing: Sequence[Tuple[float, float]],
    avoid_path: Sequence[Tuple[float, float]],
    avoid_points: Sequence[Tuple[float, float]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    min_existing_distance: float,
    min_path_distance: float,
    avoid_rects: Sequence[Tuple[float, float, float, float]] = (),
    avoid_rect_margin: float = 0.0,
) -> Tuple[float, float] | None:
    """Sample a center away from occupied points and the operative path."""

    for _ in range(180):
        point = (float(rng.uniform(float(x_range[0]), float(x_range[1]))), float(rng.uniform(float(y_range[0]), float(y_range[1]))))
        if any(_distance(point, other) < float(min_existing_distance) for other in existing):
            continue
        if any(_distance(point, other) < float(min_existing_distance) for other in avoid_points):
            continue
        if any(_point_inside_expanded_rect(point, rect, margin=float(avoid_rect_margin)) for rect in avoid_rects):
            continue
        if avoid_path and _point_path_distance(point, avoid_path) < float(min_path_distance):
            continue
        return point
    return None


def _make_platform(
    *,
    index: int,
    label: str,
    center: Tuple[float, float],
    rng: Any,
    width_norm: float | None = None,
) -> PlatformerPlatform:
    """Create one platform object."""

    width = float(width_norm if width_norm is not None else rng.uniform(0.15, 0.23))
    return PlatformerPlatform(
        platform_id=platform_entity_id(index),
        label=str(label),
        x_norm=float(center[0]),
        y_norm=float(center[1]),
        width_norm=float(width),
        height_norm=0.085,
        color_index=int(index),
    )


def _make_hazard(
    *,
    index: int,
    label: str,
    center: Tuple[float, float],
    rng: Any,
) -> PlatformerHazard:
    """Create one hazard object."""

    return PlatformerHazard(
        hazard_id=hazard_entity_id(index),
        label=str(label),
        kind=str(uniform_choice(rng, HAZARD_KINDS)),
        x_norm=float(center[0]),
        y_norm=float(center[1]),
        width_norm=float(rng.uniform(0.060, 0.076)),
        height_norm=float(rng.uniform(0.070, 0.088)),
        color_index=int(index),
    )


def _decorative_hazards(
    *,
    rng: Any,
    count: int,
    avoid_path: Sequence[Tuple[float, float]],
    avoid_points: Sequence[Tuple[float, float]],
    avoid_rects: Sequence[Tuple[float, float, float, float]] = (),
) -> Tuple[PlatformerHazard, ...]:
    """Create unlabeled decorative hazards away from the operative path."""

    hazards: list[PlatformerHazard] = []
    centers: list[Tuple[float, float]] = []
    for index in range(int(count)):
        maybe = _safe_center(
            rng=rng,
            existing=centers,
            avoid_path=avoid_path,
            avoid_points=avoid_points,
            x_range=(0.24, 0.86),
            y_range=(0.62, 0.84),
            min_existing_distance=0.13,
            min_path_distance=0.11,
            avoid_rects=avoid_rects,
            avoid_rect_margin=0.075,
        )
        if maybe is None:
            break
        centers.append(maybe)
        hazards.append(_make_hazard(index=index, label="", center=maybe, rng=rng))
    return tuple(hazards)


def _decorative_platforms(
    *,
    rng: Any,
    count: int,
    avoid_path: Sequence[Tuple[float, float]],
    avoid_points: Sequence[Tuple[float, float]],
) -> Tuple[PlatformerPlatform, ...]:
    """Create decorative platforms away from the active arc and important objects."""

    platforms: list[PlatformerPlatform] = []
    centers: list[Tuple[float, float]] = []
    for index in range(int(count)):
        maybe = _safe_center(
            rng=rng,
            existing=centers,
            avoid_path=avoid_path,
            avoid_points=avoid_points,
            x_range=(0.26, 0.90),
            y_range=(0.46, 0.78),
            min_existing_distance=0.15,
            min_path_distance=0.11,
        )
        if maybe is None:
            break
        centers.append(maybe)
        platforms.append(
            _make_platform(
                index=index,
                label="",
                center=maybe,
                rng=rng,
                width_norm=float(rng.uniform(0.14, 0.21)),
            )
        )
    return tuple(platforms)


def _decorative_collectibles(
    *,
    rng: Any,
    start_index: int,
    count: int,
    avoid_path: Sequence[Tuple[float, float]],
    avoid_points: Sequence[Tuple[float, float]],
    min_existing_distance: float = 0.075,
    avoid_rects: Sequence[Tuple[float, float, float, float]] = (),
    avoid_rect_margin: float = 0.0,
) -> Tuple[PlatformerCollectible, ...]:
    """Create decorative collectibles away from the operative path."""

    coins: list[PlatformerCollectible] = []
    centers: list[Tuple[float, float]] = []
    for offset in range(int(count)):
        maybe = _safe_center(
            rng=rng,
            existing=centers,
            avoid_path=avoid_path,
            avoid_points=avoid_points,
            x_range=(0.18, 0.88),
            y_range=(0.22, 0.78),
            min_existing_distance=float(min_existing_distance),
            min_path_distance=0.085,
            avoid_rects=avoid_rects,
            avoid_rect_margin=float(avoid_rect_margin),
        )
        if maybe is None:
            break
        centers.append(maybe)
        index = int(start_index) + int(offset)
        coins.append(
            PlatformerCollectible(
                collectible_id=collectible_entity_id(index),
                x_norm=float(maybe[0]),
                y_norm=float(maybe[1]),
                radius_norm=0.022,
                on_path=False,
                color_index=int(index),
            )
        )
    return tuple(coins)


def sample_landing_scene(
    *,
    rng: Any,
    axes: PlatformerVisualAxes,
    target_platform_label: str,
    mode: str,
) -> PlatformerSample:
    """Construct a scene where the visible jump arc lands on one labeled platform."""

    after_peak_min = max(0.03, min(0.24, float(axes.jump_visible_after_peak_min)))
    after_peak_max = max(after_peak_min, min(0.28, float(axes.jump_visible_after_peak_max)))
    for _attempt in range(160):
        player = (float(rng.uniform(0.12, 0.22)), float(rng.uniform(0.76, 0.83)))
        target_top = (float(rng.uniform(0.60, 0.86)), float(rng.uniform(0.44, 0.66)))
        path = _jump_arc(start=player, end=target_top, rng=rng, lift=float(rng.uniform(0.24, 0.34)))
        peak_index = min(range(len(path)), key=lambda index: float(path[int(index)][1]))
        peak_fraction = float(peak_index) / float(max(1, len(path) - 1))
        target_center = (float(target_top[0]), float(target_top[1] + 0.032))

        labels = [str(target_platform_label)] + [str(label) for label in PLATFORM_LABELS if str(label) != str(target_platform_label)]
        labels = labels[: int(axes.platform_count)]
        rng.shuffle(labels)
        target_index = labels.index(str(target_platform_label))
        platforms: list[PlatformerPlatform] = []
        centers: list[Tuple[float, float]] = []
        for index, label in enumerate(labels):
            if int(index) == int(target_index):
                center = target_center
            else:
                maybe = _safe_center(
                    rng=rng,
                    existing=centers + [target_center],
                    avoid_path=path,
                    avoid_points=(player, target_center),
                    x_range=(0.28, 0.88),
                    y_range=(0.32, 0.74),
                    min_existing_distance=0.15,
                    min_path_distance=0.09,
                )
                if maybe is None:
                    break
                center = maybe
            centers.append(center)
            platforms.append(
                _make_platform(
                    index=index,
                    label=str(label),
                    center=center,
                    rng=rng,
                    width_norm=0.20 if int(index) == int(target_index) else None,
                )
            )
        if len(platforms) != len(labels):
            continue
        target_platform = platforms[target_index]
        platform_centers = tuple((float(platform.x_norm), float(platform.y_norm)) for platform in platforms)
        platform_rects = tuple((float(platform.x_norm), float(platform.y_norm), float(platform.width_norm), float(platform.height_norm)) for platform in platforms)
        avoid_objects = (player, target_center) + tuple(platform_centers)
        hazards = _decorative_hazards(
            rng=rng,
            count=max(2, int(axes.hazard_count) - 3),
            avoid_path=path,
            avoid_points=avoid_objects,
            avoid_rects=platform_rects,
        )
        coins = _decorative_collectibles(
            rng=rng,
            start_index=0,
            count=4,
            avoid_path=path,
            avoid_points=avoid_objects,
            min_existing_distance=0.13,
            avoid_rects=platform_rects,
            avoid_rect_margin=0.055,
        )
        sample = PlatformerSample(
            mode=str(mode),
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=str(target_platform.label),
            player_x_norm=float(player[0]),
            player_y_norm=float(player[1]),
            path_points_norm=tuple(path),
            visible_path_fraction=float(min(0.92, peak_fraction + float(rng.uniform(after_peak_min, after_peak_max)))),
            platforms=tuple(platforms),
            hazards=tuple(hazards),
            collectibles=tuple(coins),
            target_platform_id=str(target_platform.platform_id),
            target_platform_label=str(target_platform.label),
            target_collectible_ids=tuple(),
            annotation_entity_ids=(str(target_platform.platform_id),),
            construction_mode="short_arc_lands_on_labeled_platform",
        )
        validate_platformer_sample(sample)
        return sample
    raise ValueError("failed to construct platformer landing scene")


def sample_collectible_path_scene(
    *,
    rng: Any,
    axes: PlatformerVisualAxes,
    target_collectible_count: int,
    mode: str,
) -> PlatformerSample:
    """Construct a scene with a full shown arc collecting a target coin count."""

    target_count = int(target_collectible_count)
    for _attempt in range(160):
        player = (float(rng.uniform(0.12, 0.20)), float(rng.uniform(0.76, 0.83)))
        end = (float(rng.uniform(0.78, 0.90)), float(rng.uniform(0.54, 0.70)))
        path = _jump_arc(start=player, end=end, rng=rng, lift=float(rng.uniform(0.24, 0.35)))
        t_values = [0.20 + ((0.64 * (idx + 0.5)) / float(target_count)) for idx in range(int(target_count))]
        target_collectibles: list[PlatformerCollectible] = []
        for index, t in enumerate(t_values):
            point = tuple(float(value) for value in path[max(1, min(len(path) - 2, int(round(float(t) * (len(path) - 1)))))])
            target_collectibles.append(
                PlatformerCollectible(
                    collectible_id=collectible_entity_id(index),
                    x_norm=float(point[0]),
                    y_norm=float(point[1]),
                    radius_norm=0.022,
                    on_path=True,
                    color_index=int(index),
                )
            )
        distractors = _decorative_collectibles(
            rng=rng,
            start_index=len(target_collectibles),
            count=int(axes.distractor_collectible_count),
            avoid_path=path,
            avoid_points=(player, end) + tuple((coin.x_norm, coin.y_norm) for coin in target_collectibles),
        )
        coin_points = tuple((float(coin.x_norm), float(coin.y_norm)) for coin in tuple(target_collectibles) + tuple(distractors))
        platform_target = max(3, int(axes.platform_count) - 1)
        platforms = _decorative_platforms(
            rng=rng,
            count=platform_target,
            avoid_path=path,
            avoid_points=(player, end) + tuple(coin_points),
        )
        platform_centers = tuple((float(platform.x_norm), float(platform.y_norm)) for platform in platforms)
        platform_rects = tuple((float(platform.x_norm), float(platform.y_norm), float(platform.width_norm), float(platform.height_norm)) for platform in platforms)
        hazards = _decorative_hazards(
            rng=rng,
            count=max(2, int(axes.hazard_count) - 3),
            avoid_path=path,
            avoid_points=(player, end) + tuple(coin_points) + tuple(platform_centers),
            avoid_rects=platform_rects,
        )
        target_ids = tuple(str(coin.collectible_id) for coin in target_collectibles)
        sample = PlatformerSample(
            mode=str(mode),
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=int(len(target_collectibles)),
            player_x_norm=float(player[0]),
            player_y_norm=float(player[1]),
            path_points_norm=tuple(path),
            visible_path_fraction=1.0,
            platforms=platforms,
            hazards=tuple(hazards),
            collectibles=tuple(target_collectibles) + tuple(distractors),
            target_platform_id=None,
            target_platform_label=None,
            target_collectible_ids=target_ids,
            annotation_entity_ids=target_ids,
            construction_mode="full_arc_collectible_total",
        )
        validate_platformer_sample(sample)
        return sample
    raise ValueError("failed to construct platformer collectible scene")


def sample_scored_collectible_path_scene(
    *,
    rng: Any,
    axes: PlatformerVisualAxes,
    on_arc_coin_count_support: Sequence[int],
    on_arc_bonus_count_support: Sequence[int],
    off_arc_bonus_count_support: Sequence[int],
    bonus_value_support: Sequence[int],
    mode: str,
) -> PlatformerSample:
    """Construct a full-arc scene with scored coins and printed-value bonus items."""

    on_arc_coin_count = int(rng.choice(tuple(int(value) for value in on_arc_coin_count_support)))
    on_arc_bonus_count = int(rng.choice(tuple(int(value) for value in on_arc_bonus_count_support)))
    off_arc_bonus_count = int(rng.choice(tuple(int(value) for value in off_arc_bonus_count_support)))
    bonus_values = tuple(int(value) for value in bonus_value_support)
    target_count = int(on_arc_coin_count + on_arc_bonus_count)
    for _attempt in range(160):
        player = (float(rng.uniform(0.12, 0.20)), float(rng.uniform(0.76, 0.83)))
        end = (float(rng.uniform(0.78, 0.90)), float(rng.uniform(0.54, 0.70)))
        path = _jump_arc(start=player, end=end, rng=rng, lift=float(rng.uniform(0.24, 0.35)))
        target_kinds = (["coin"] * int(on_arc_coin_count)) + (["bonus"] * int(on_arc_bonus_count))
        rng.shuffle(target_kinds)
        t_values = [0.18 + ((0.66 * (idx + 0.5)) / float(target_count)) for idx in range(int(target_count))]
        target_collectibles: list[PlatformerCollectible] = []
        answer = 0
        next_index = 0
        bonus_kind_cycle = cycle(shuffled_support(rng, BONUS_COLLECTIBLE_KINDS))
        for target_kind, t in zip(target_kinds, t_values):
            point = tuple(float(value) for value in path[max(1, min(len(path) - 2, int(round(float(t) * (len(path) - 1)))))])
            if str(target_kind) == "bonus":
                score_value = int(rng.choice(bonus_values))
                kind = str(next(bonus_kind_cycle))
                radius_norm = 0.028
                answer += int(score_value)
            else:
                score_value = None
                kind = "coin"
                radius_norm = 0.022
                answer += 1
            target_collectibles.append(
                PlatformerCollectible(
                    collectible_id=collectible_entity_id(next_index),
                    x_norm=float(point[0]),
                    y_norm=float(point[1]),
                    radius_norm=float(radius_norm),
                    on_path=True,
                    color_index=int(next_index),
                    kind=str(kind),
                    score_value=score_value,
                )
            )
            next_index += 1

        occupied_points = [tuple((coin.x_norm, coin.y_norm)) for coin in target_collectibles]
        off_arc_bonus_collectibles: list[PlatformerCollectible] = []
        for offset in range(int(off_arc_bonus_count)):
            maybe = _safe_center(
                rng=rng,
                existing=occupied_points,
                avoid_path=path,
                avoid_points=(player, end),
                x_range=(0.18, 0.88),
                y_range=(0.22, 0.78),
                min_existing_distance=0.090,
                min_path_distance=0.095,
            )
            if maybe is None:
                break
            occupied_points.append(maybe)
            off_arc_bonus_collectibles.append(
                PlatformerCollectible(
                    collectible_id=collectible_entity_id(next_index),
                    x_norm=float(maybe[0]),
                    y_norm=float(maybe[1]),
                    radius_norm=0.028,
                    on_path=False,
                    color_index=int(next_index),
                    kind=str(next(bonus_kind_cycle)),
                    score_value=int(rng.choice(bonus_values)),
                )
            )
            next_index += 1
        if len(off_arc_bonus_collectibles) < 1:
            continue

        distractors = _decorative_collectibles(
            rng=rng,
            start_index=next_index,
            count=int(axes.distractor_collectible_count),
            avoid_path=path,
            avoid_points=(player, end) + tuple(occupied_points),
            min_existing_distance=0.080,
        )
        all_collectibles = tuple(target_collectibles) + tuple(off_arc_bonus_collectibles) + tuple(distractors)
        collectible_points = tuple((float(coin.x_norm), float(coin.y_norm)) for coin in all_collectibles)
        platform_target = max(3, int(axes.platform_count) - 1)
        platforms = _decorative_platforms(
            rng=rng,
            count=platform_target,
            avoid_path=path,
            avoid_points=(player, end) + tuple(collectible_points),
        )
        platform_centers = tuple((float(platform.x_norm), float(platform.y_norm)) for platform in platforms)
        platform_rects = tuple((float(platform.x_norm), float(platform.y_norm), float(platform.width_norm), float(platform.height_norm)) for platform in platforms)
        hazards = _decorative_hazards(
            rng=rng,
            count=max(2, int(axes.hazard_count) - 3),
            avoid_path=path,
            avoid_points=(player, end) + tuple(collectible_points) + tuple(platform_centers),
            avoid_rects=platform_rects,
        )
        target_ids = tuple(str(coin.collectible_id) for coin in target_collectibles)
        sample = PlatformerSample(
            mode=str(mode),
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            answer=int(answer),
            player_x_norm=float(player[0]),
            player_y_norm=float(player[1]),
            path_points_norm=tuple(path),
            visible_path_fraction=1.0,
            platforms=platforms,
            hazards=tuple(hazards),
            collectibles=all_collectibles,
            target_platform_id=None,
            target_platform_label=None,
            target_collectible_ids=target_ids,
            annotation_entity_ids=target_ids,
            construction_mode="full_arc_collectible_score_sum",
        )
        validate_platformer_sample(sample)
        return sample
    raise ValueError("failed to construct platformer collectible score scene")


__all__ = [
    "PlatformerIntegerAxis",
    "PlatformerLabelAxis",
    "PlatformerVisualAxes",
    "integer_support_from_defaults",
    "resolve_platformer_integer_axis",
    "resolve_platformer_label_axis",
    "resolve_platformer_visual_axes",
    "sample_collectible_path_scene",
    "sample_landing_scene",
    "sample_scored_collectible_path_scene",
]
