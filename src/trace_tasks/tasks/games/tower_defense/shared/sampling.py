"""Sampling and construction helpers for tower-defense maps."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import DEFAULTS, GEN_DEFAULTS
from .rules import (
    MODE_BEST_POSITION,
    MODE_MARKED_ENEMY,
    MODE_NEAREST_EXIT_ENEMY,
    MODE_PATH_NODES,
    ENEMY_OPTION_LABELS,
    OPTION_LABELS,
    candidate_tower_entity_id,
    covered_path_segment_ids,
    covered_tower_ids,
    enemy_entity_id,
    local_distance,
    path_segment_entity_id,
    tower_covers_point,
    tower_entity_id,
    validate_tower_defense_sample,
)
from .state import (
    Point,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
    TowerDefenseAxes,
    TowerDefenseEnemy,
    TowerDefenseRenderParams,
    TowerDefenseSample,
    TowerDefenseTower,
)


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve one balanced named tower-defense axis."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=GEN_DEFAULTS,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        supported_variants=tuple(str(value) for value in supported),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), dict(probabilities)


def _resolve_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> tuple[int, dict[str, float]]:
    """Resolve one integer scene axis from params or config."""

    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return int(value), dict(probabilities)


def resolve_tower_defense_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace_root: str,
    tower_count_support_key: str,
    tower_count_fallback: Sequence[int],
    target_answer_support_key: str,
    target_answer_fallback: Sequence[int],
    path_count_must_cover_target: bool,
    tower_count_must_cover_target: bool,
) -> TowerDefenseAxes:
    """Resolve reusable visual axes plus task-owned count supports."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace_root}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace_root}.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_STYLE_VARIANTS,
    )
    tower_count, tower_count_probabilities = _resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key=str(tower_count_support_key),
        explicit_key="tower_count",
        fallback_support=tuple(int(value) for value in tower_count_fallback),
        namespace=f"{namespace_root}.{tower_count_support_key}",
        balanced_flag_key="balanced_tower_count_sampling",
    )
    path_segment_count, path_segment_count_probabilities = _resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key="path_segment_count_support",
        explicit_key="path_segment_count",
        fallback_support=DEFAULTS.path_segment_count_support,
        namespace=f"{namespace_root}.path_segment_count",
        balanced_flag_key="balanced_path_segment_count_sampling",
    )
    target_answer, target_answer_probabilities = _resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        support_key=str(target_answer_support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in target_answer_fallback),
        namespace=f"{namespace_root}.{target_answer_support_key}",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    target_answer_support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key=str(target_answer_support_key),
        fallback=tuple(int(value) for value in target_answer_fallback),
    )
    tower_support = resolve_integer_support(
        params,
        gen_defaults=GEN_DEFAULTS,
        key=str(tower_count_support_key),
        fallback=tuple(int(value) for value in tower_count_fallback),
    )
    tower_support_min = min(int(value) for value in tower_support)
    tower_support_max = max(int(value) for value in tower_support)
    if bool(path_count_must_cover_target):
        path_segment_count = min(16, max(int(path_segment_count), int(target_answer)))
    if bool(tower_count_must_cover_target):
        tower_count = max(int(tower_count), int(target_answer))
    elif int(target_answer) > int(tower_count):
        tower_count = int(target_answer)
    if not bool(tower_count_must_cover_target) and int(target_answer) > 0:
        tower_count = max(int(tower_count), min(int(tower_support_max), int(target_answer) + 2))
    tower_count = min(int(tower_support_max), max(int(tower_support_min), int(tower_count)))
    return TowerDefenseAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        tower_count=int(tower_count),
        path_segment_count=int(path_segment_count),
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_answer_support),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        tower_count_probabilities=dict(tower_count_probabilities),
        path_segment_count_probabilities=dict(path_segment_count_probabilities),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def axis_support_metadata(axes: TowerDefenseAxes) -> dict[str, Any]:
    """Return JSON-friendly axis metadata for query params."""

    return {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "tower_count": int(axes.tower_count),
        "path_segment_count": int(axes.path_segment_count),
        "target_answer": int(axes.target_answer),
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "tower_count_probabilities": dict(axes.tower_count_probabilities),
        "path_segment_count_probabilities": dict(axes.path_segment_count_probabilities),
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
    }


def _resample_polyline(waypoints: Sequence[Point], *, point_count: int) -> tuple[Point, ...]:
    """Return equally spaced points along one polyline."""

    if len(waypoints) < 2:
        raise ValueError("tower-defense path needs at least two waypoints")
    segment_lengths = [local_distance(start, end) for start, end in zip(waypoints, waypoints[1:])]
    total_length = sum(float(value) for value in segment_lengths)
    if total_length <= 0.0:
        raise ValueError("tower-defense path has zero length")
    count = max(2, int(point_count))
    samples: list[Point] = []
    for sample_index in range(count):
        target = (float(total_length) * float(sample_index)) / float(count - 1)
        cursor = 0.0
        for (start, end), length in zip(zip(waypoints, waypoints[1:]), segment_lengths):
            if target <= cursor + float(length) or (start, end) == (waypoints[-2], waypoints[-1]):
                ratio = 0.0 if float(length) <= 0.0 else (float(target) - float(cursor)) / float(length)
                ratio = max(0.0, min(1.0, float(ratio)))
                samples.append(
                    (
                        round(float(start[0]) + ((float(end[0]) - float(start[0])) * ratio), 3),
                        round(float(start[1]) + ((float(end[1]) - float(start[1])) * ratio), 3),
                    )
                )
                break
            cursor += float(length)
    return tuple(samples)


def sample_path_points(
    *,
    rng,
    scene_variant: str,
    map_width_px: int,
    map_height_px: int,
    path_segment_count: int,
) -> tuple[Point, ...]:
    """Construct one readable winding path in local map coordinates."""

    width = float(map_width_px)
    height = float(map_height_px)
    margin_x = 84.0
    if str(scene_variant) == "switchback_path":
        y_fracs = (0.18, 0.34, 0.52, 0.70, 0.84)
        left_x, right_x = margin_x, width - margin_x
    else:
        y_fracs = (0.20, 0.41, 0.62, 0.80)
        left_x, right_x = margin_x, width - margin_x
    waypoints: list[Point] = []
    for index, y_frac in enumerate(y_fracs):
        y = round(float(height * float(y_frac)), 3)
        if index == 0:
            waypoints.append((left_x, y))
        waypoints.append((right_x if index % 2 == 0 else left_x, y))
        if index < len(y_fracs) - 1:
            next_y = round(float(height * float(y_fracs[index + 1])), 3)
            waypoints.append((right_x if index % 2 == 0 else left_x, next_y))
    if rng.random() < 0.5:
        jittered: list[Point] = []
        for index, (x, y) in enumerate(waypoints):
            if index in {0, len(waypoints) - 1}:
                jittered.append((x, y))
            else:
                jittered.append((x, round(max(72.0, min(height - 72.0, y + rng.uniform(-12.0, 12.0))), 3)))
        waypoints = jittered
    return _resample_polyline(waypoints, point_count=int(path_segment_count))


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return distance from a point to one line segment."""

    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx, dy = ex - sx, ey - sy
    denom = (dx * dx) + (dy * dy)
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    ratio = max(0.0, min(1.0, (((px - sx) * dx) + ((py - sy) * dy)) / denom))
    cx, cy = sx + (ratio * dx), sy + (ratio * dy)
    return math.hypot(px - cx, py - cy)


def _min_path_distance(point: Point, path_points: Sequence[Point]) -> float:
    """Return distance from a point to the visible path polyline."""

    if len(path_points) < 2:
        return float("inf")
    return min(_point_segment_distance(point, start, end) for start, end in zip(path_points, path_points[1:]))


def _tower_center_is_valid(
    center: Point,
    *,
    radius: float,
    edge_radius_px: float | None = None,
    map_width_px: int,
    map_height_px: int,
    path_points: Sequence[Point],
    existing_centers: Sequence[Point],
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
) -> bool:
    """Return whether one tower center is visibly valid."""

    x, y = float(center[0]), float(center[1])
    edge_margin = float(radius if edge_radius_px is None else edge_radius_px) + 26.0
    if x < edge_margin or x > float(map_width_px) - edge_margin:
        return False
    if y < edge_margin or y > float(map_height_px) - edge_margin:
        return False
    if _min_path_distance(center, path_points) < float(tower_path_clearance_px):
        return False
    return all(local_distance(center, other) >= float(tower_min_gap_px) for other in existing_centers)


def _sample_tower_center(
    *,
    rng,
    covers_target: bool,
    target_point: Point,
    radius: float,
    map_width_px: int,
    map_height_px: int,
    path_points: Sequence[Point],
    existing_centers: Sequence[Point],
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
    uncovered_margin_px: float,
) -> Point:
    """Sample one tower center satisfying the target coverage predicate."""

    for _ in range(700):
        if bool(covers_target):
            min_distance = max(float(tower_path_clearance_px) + 12.0, 70.0)
            max_distance = max(min_distance + 4.0, float(radius) - 24.0)
            distance = rng.uniform(min_distance, max_distance)
            angle = rng.uniform(0.0, math.tau)
            center = (
                round(float(target_point[0]) + (math.cos(angle) * distance), 3),
                round(float(target_point[1]) + (math.sin(angle) * distance), 3),
            )
        else:
            center = (
                round(rng.uniform(float(radius) + 34.0, float(map_width_px) - float(radius) - 34.0), 3),
                round(rng.uniform(float(radius) + 34.0, float(map_height_px) - float(radius) - 34.0), 3),
            )
        if not _tower_center_is_valid(
            center,
            radius=float(radius),
            edge_radius_px=20.0,
            map_width_px=int(map_width_px),
            map_height_px=int(map_height_px),
            path_points=path_points,
            existing_centers=existing_centers,
            tower_path_clearance_px=float(tower_path_clearance_px),
            tower_min_gap_px=float(tower_min_gap_px),
        ):
            continue
        distance_to_target = local_distance(center, target_point)
        if bool(covers_target) and distance_to_target <= float(radius) - 18.0:
            return center
        if not bool(covers_target) and distance_to_target >= float(radius) + float(uncovered_margin_px):
            return center
    raise ValueError("failed to sample a valid tower center")


def _select_path_coverage_indices(*, rng, path_count: int, target_answer: int) -> tuple[int, ...]:
    """Select exactly the path indices intended to be covered."""

    if int(target_answer) <= 0:
        raise ValueError("covered path task requires a positive target answer")
    if int(target_answer) > int(path_count):
        raise ValueError("target_answer cannot exceed path segment count")
    return tuple(sorted(int(index) for index in rng.sample(range(int(path_count)), int(target_answer))))


def _path_tangent_at_index(path_points: Sequence[Point], index: int) -> Point:
    """Return local unit tangent near one path point."""

    if len(path_points) < 2:
        return (1.0, 0.0)
    idx = max(0, min(len(path_points) - 1, int(index)))
    if idx == 0:
        start, end = path_points[0], path_points[1]
    elif idx == len(path_points) - 1:
        start, end = path_points[-2], path_points[-1]
    else:
        start, end = path_points[idx - 1], path_points[idx + 1]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _sample_path_chunk_tower(
    *,
    rng,
    tower_id: str,
    chunk_indices: Sequence[int],
    selected_indices: Sequence[int],
    path_points: Sequence[Point],
    map_width_px: int,
    map_height_px: int,
    range_min: int,
    range_max: int,
    existing_centers: Sequence[Point],
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
) -> TowerDefenseTower:
    """Sample one tower that covers its selected path run without extra nodes."""

    if not chunk_indices:
        raise ValueError("path chunk tower needs at least one path index")
    selected_set = {int(index) for index in selected_indices}
    chunk_points = [path_points[int(index)] for index in chunk_indices]
    anchor = (
        round(sum(float(point[0]) for point in chunk_points) / float(len(chunk_points)), 3),
        round(sum(float(point[1]) for point in chunk_points) / float(len(chunk_points)), 3),
    )
    tangent = _path_tangent_at_index(path_points, int(chunk_indices[len(chunk_indices) // 2]))
    perpendicular = (-float(tangent[1]), float(tangent[0]))
    for _ in range(420):
        if len(chunk_points) == 1:
            angle = rng.uniform(0.0, math.tau)
            direction = (math.cos(angle), math.sin(angle))
        else:
            sign = -1.0 if rng.random() < 0.5 else 1.0
            direction = (float(perpendicular[0]) * sign, float(perpendicular[1]) * sign)
        offset = rng.uniform(max(float(tower_path_clearance_px) + 20.0, 68.0), 92.0)
        center = (
            round(float(anchor[0]) + (float(direction[0]) * float(offset)), 3),
            round(float(anchor[1]) + (float(direction[1]) * float(offset)), 3),
        )
        min_radius = max(local_distance(center, point) for point in chunk_points) + rng.uniform(14.0, 23.0)
        radius = float(max(int(range_min), int(math.ceil(min_radius))))
        if float(radius) > float(range_max):
            continue
        if not _tower_center_is_valid(
            center,
            radius=float(radius),
            edge_radius_px=20.0,
            map_width_px=int(map_width_px),
            map_height_px=int(map_height_px),
            path_points=path_points,
            existing_centers=existing_centers,
            tower_path_clearance_px=float(tower_path_clearance_px),
            tower_min_gap_px=float(tower_min_gap_px),
        ):
            continue
        tower = TowerDefenseTower(
            tower_id=str(tower_id),
            center_px=center,
            range_radius_px=float(radius),
            covers_target=False,
        )
        covered_indices = {int(index) for index, point in enumerate(path_points) if tower_covers_point(tower, point)}
        if set(int(index) for index in chunk_indices).issubset(covered_indices) and covered_indices.issubset(selected_set):
            return tower
    raise ValueError("failed to place path chunk tower")


def _sample_nonexpanding_tower(
    *,
    rng,
    tower_id: str,
    selected_indices: Sequence[int],
    path_points: Sequence[Point],
    map_width_px: int,
    map_height_px: int,
    range_min: int,
    range_max: int,
    existing_centers: Sequence[Point],
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
) -> TowerDefenseTower:
    """Sample one decoy tower that covers no unselected path node."""

    selected_set = {int(index) for index in selected_indices}
    for _ in range(700):
        radius = float(rng.randint(int(range_min), int(range_max)))
        center = (
            round(rng.uniform(float(radius) + 34.0, float(map_width_px) - float(radius) - 34.0), 3),
            round(rng.uniform(float(radius) + 34.0, float(map_height_px) - float(radius) - 34.0), 3),
        )
        if not _tower_center_is_valid(
            center,
            radius=float(radius),
            map_width_px=int(map_width_px),
            map_height_px=int(map_height_px),
            path_points=path_points,
            existing_centers=existing_centers,
            tower_path_clearance_px=float(tower_path_clearance_px),
            tower_min_gap_px=float(tower_min_gap_px),
        ):
            continue
        tower = TowerDefenseTower(
            tower_id=str(tower_id),
            center_px=center,
            range_radius_px=float(radius),
            covers_target=False,
        )
        covered_indices = {int(index) for index, point in enumerate(path_points) if tower_covers_point(tower, point)}
        if covered_indices.issubset(selected_set):
            return tower
    raise ValueError("failed to place nonexpanding tower")


def _sample_decorative_tower(
    *,
    rng,
    tower_id: str,
    path_points: Sequence[Point],
    map_width_px: int,
    map_height_px: int,
    range_min: int,
    range_max: int,
    existing_centers: Sequence[Point],
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
) -> TowerDefenseTower:
    """Sample one context tower without imposing an objective-specific relation."""

    for _ in range(700):
        radius = float(rng.randint(int(range_min), int(range_max)))
        center = (
            round(rng.uniform(float(radius) + 34.0, float(map_width_px) - float(radius) - 34.0), 3),
            round(rng.uniform(float(radius) + 34.0, float(map_height_px) - float(radius) - 34.0), 3),
        )
        if _tower_center_is_valid(
            center,
            radius=float(radius),
            map_width_px=int(map_width_px),
            map_height_px=int(map_height_px),
            path_points=path_points,
            existing_centers=existing_centers,
            tower_path_clearance_px=float(tower_path_clearance_px),
            tower_min_gap_px=float(tower_min_gap_px),
        ):
            return TowerDefenseTower(
                tower_id=str(tower_id),
                center_px=center,
                range_radius_px=float(radius),
                covers_target=False,
            )
    raise ValueError("failed to place decorative tower")


def _coverage_count_for_tower(tower: TowerDefenseTower, path_points: Sequence[Point]) -> int:
    """Count path enemies inside one tower range."""

    return sum(1 for point in path_points if tower_covers_point(tower, point))


def _sample_fixed_radius_candidate_center(
    *,
    rng,
    path_points: Sequence[Point],
    map_width_px: int,
    map_height_px: int,
    radius: float,
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
) -> Point:
    """Sample one valid candidate center for a fixed-radius tower ring."""

    radius_value = float(radius)
    min_offset = max(float(tower_path_clearance_px) + 18.0, 58.0)
    max_offset = max(float(min_offset) + 3.0, float(radius_value) - 7.0)
    draw_mode = rng.random()
    if draw_mode < 0.42 and len(path_points) >= 2:
        pair_index = int(rng.randrange(0, len(path_points) - 1))
        start, end = path_points[pair_index], path_points[pair_index + 1]
        anchor = (
            round((float(start[0]) + float(end[0])) * 0.5, 3),
            round((float(start[1]) + float(end[1])) * 0.5, 3),
        )
        tangent = _path_tangent_at_index(path_points, int(pair_index))
    elif draw_mode < 0.90:
        anchor_index = int(rng.randrange(0, len(path_points)))
        anchor = path_points[int(anchor_index)]
        tangent = _path_tangent_at_index(path_points, int(anchor_index))
    else:
        edge_margin = float(radius_value) + 28.0
        return (
            round(rng.uniform(edge_margin, float(map_width_px) - edge_margin), 3),
            round(rng.uniform(edge_margin, float(map_height_px) - edge_margin), 3),
        )
    perpendicular = (-float(tangent[1]), float(tangent[0]))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    tangent_jitter = rng.uniform(-18.0, 18.0)
    offset = rng.uniform(float(min_offset), float(max_offset))
    return (
        round(float(anchor[0]) + (perpendicular[0] * sign * offset) + (tangent[0] * tangent_jitter), 3),
        round(float(anchor[1]) + (perpendicular[1] * sign * offset) + (tangent[1] * tangent_jitter), 3),
    )


def _sample_best_position_candidate_layout(
    *,
    rng,
    path_points: Sequence[Point],
    map_width_px: int,
    map_height_px: int,
    radius: float,
    answer_label: str,
    tower_path_clearance_px: float,
    tower_min_gap_px: float,
) -> tuple[tuple[TowerDefenseTower, ...], dict[str, int]]:
    """Build four same-radius candidates with one count-2 winner and three count-1 decoys."""

    pools: dict[int, list[Point]] = {1: [], 2: []}
    for _ in range(7000):
        center = _sample_fixed_radius_candidate_center(
            rng=rng,
            path_points=path_points,
            map_width_px=int(map_width_px),
            map_height_px=int(map_height_px),
            radius=float(radius),
            tower_path_clearance_px=float(tower_path_clearance_px),
            tower_min_gap_px=float(tower_min_gap_px),
        )
        if not _tower_center_is_valid(
            center,
            radius=float(radius),
            map_width_px=int(map_width_px),
            map_height_px=int(map_height_px),
            path_points=path_points,
            existing_centers=(),
            tower_path_clearance_px=float(tower_path_clearance_px),
            tower_min_gap_px=float(tower_min_gap_px),
        ):
            continue
        probe = TowerDefenseTower(
            tower_id="candidate_probe",
            center_px=center,
            range_radius_px=float(radius),
            covers_target=False,
        )
        coverage_count = _coverage_count_for_tower(probe, path_points)
        if coverage_count in pools:
            pools[int(coverage_count)].append(center)
        if len(pools[1]) >= 18 and len(pools[2]) >= 6:
            break
    if not pools[2] or len(pools[1]) < 3:
        raise ValueError("failed to find enough fixed-radius candidate positions")
    best_candidates = list(pools[2])
    decoy_candidates = list(pools[1])
    rng.shuffle(best_candidates)
    rng.shuffle(decoy_candidates)
    labels = list(OPTION_LABELS)
    answer = str(answer_label)
    remaining_labels = [str(label) for label in labels if str(label) != answer]
    for best_center in best_candidates:
        selected: list[tuple[str, Point, int]] = [(answer, best_center, 2)]
        for decoy_center in decoy_candidates:
            if all(local_distance(decoy_center, existing_center) >= float(tower_min_gap_px) for _, existing_center, _ in selected):
                selected.append((remaining_labels[len(selected) - 1], decoy_center, 1))
                if len(selected) == 4:
                    break
        if len(selected) != 4:
            continue
        towers_by_label = {
            str(label): TowerDefenseTower(
                tower_id=candidate_tower_entity_id(str(label)),
                center_px=center,
                range_radius_px=float(radius),
                covers_target=(str(label) == answer),
            )
            for label, center, _count in selected
        }
        counts_by_label = {str(label): int(count) for label, _center, count in selected}
        return tuple(towers_by_label[str(label)] for label in labels), counts_by_label
    raise ValueError("failed to choose separated same-radius candidate positions")


def sample_best_tower_position_scene(
    *,
    rng,
    axes: TowerDefenseAxes,
    render_params: TowerDefenseRenderParams,
    params: Mapping[str, Any],
) -> TowerDefenseSample:
    """Construct four candidate tower positions with a unique best coverage label."""

    map_width = int(render_params.map_width_px)
    map_height = int(render_params.map_height_px)
    path_points = sample_path_points(
        rng=rng,
        scene_variant=str(axes.scene_variant),
        map_width_px=map_width,
        map_height_px=map_height,
        path_segment_count=int(axes.path_segment_count),
    )
    range_min = int(params.get("best_position_range_radius_min_px", GEN_DEFAULTS.get("best_position_range_radius_min_px", DEFAULTS.best_position_range_radius_min_px)))
    range_max = int(params.get("best_position_range_radius_max_px", GEN_DEFAULTS.get("best_position_range_radius_max_px", DEFAULTS.best_position_range_radius_max_px)))
    tower_path_clearance = float(params.get("tower_path_clearance_px", GEN_DEFAULTS.get("tower_path_clearance_px", DEFAULTS.tower_path_clearance_px)))
    tower_min_gap = float(params.get("tower_min_gap_px", GEN_DEFAULTS.get("tower_min_gap_px", DEFAULTS.tower_min_gap_px)))
    target_count = int(axes.target_answer)
    if target_count <= 0:
        raise ValueError("best-position task needs a positive winning coverage count")
    radius_floor = max(float(range_min), float(tower_path_clearance) + 44.0)
    radius_ceiling = max(float(radius_floor), float(range_max))
    target_radius = float(radius_floor) + (float(target_count - 1) * 8.0) + rng.uniform(0.0, 4.0)
    shared_radius = float(min(float(radius_ceiling), max(float(radius_floor), float(target_radius))))
    raw_answer_option_index = params.get("answer_option_index")
    if raw_answer_option_index is None:
        target_label_index = int(uniform_choice(rng, tuple(range(len(OPTION_LABELS)))))
    else:
        target_label_index = int(raw_answer_option_index)
        if target_label_index < 0 or target_label_index >= len(OPTION_LABELS):
            raise ValueError("answer_option_index must be inside Tower Defense option labels")
    best_label = str(OPTION_LABELS[int(target_label_index)])
    towers, coverage_counts = _sample_best_position_candidate_layout(
        rng=rng,
        path_points=path_points,
        map_width_px=map_width,
        map_height_px=map_height,
        radius=shared_radius,
        answer_label=best_label,
        tower_path_clearance_px=tower_path_clearance,
        tower_min_gap_px=tower_min_gap,
    )
    if coverage_counts[str(best_label)] != int(target_count):
        raise ValueError("best candidate coverage count drifted after construction")
    if max(coverage_counts.values()) != int(target_count) or list(coverage_counts.values()).count(int(target_count)) != 1:
        raise ValueError("candidate tower positions must have one unique maximum")
    sample = TowerDefenseSample(
        mode=MODE_BEST_POSITION,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        map_width_px=int(map_width),
        map_height_px=int(map_height),
        path_points_px=tuple(path_points),
        towers=towers,
        enemy=None,
        answer=str(best_label),
        target_answer=int(target_count),
        annotation_entity_ids=(candidate_tower_entity_id(best_label),),
        construction_mode="construct_four_candidate_positions_by_path_enemy_coverage",
        metadata={
            "candidate_labels": list(OPTION_LABELS),
            "candidate_coverage_counts": {str(label): int(value) for label, value in coverage_counts.items()},
            "candidate_range_radius_px": round(float(shared_radius), 3),
            "answer_option_index": int(target_label_index),
        },
    )
    validate_tower_defense_sample(sample)
    return sample


def sample_nearest_exit_enemy_label_scene(
    *,
    rng,
    axes: TowerDefenseAxes,
    render_params: TowerDefenseRenderParams,
    params: Mapping[str, Any],
) -> TowerDefenseSample:
    """Construct six labeled enemies where one is closest to the path exit."""

    map_width = int(render_params.map_width_px)
    map_height = int(render_params.map_height_px)
    path_points = sample_path_points(
        rng=rng,
        scene_variant=str(axes.scene_variant),
        map_width_px=map_width,
        map_height_px=map_height,
        path_segment_count=int(axes.path_segment_count),
    )
    option_count = int(axes.target_answer)
    labels = tuple(str(label) for label in ENEMY_OPTION_LABELS[:option_count])
    if option_count != 6 or len(labels) != 6:
        raise ValueError("nearest-exit enemy task requires six options")
    raw_answer_option_index = params.get("answer_option_index")
    if raw_answer_option_index is None:
        answer_option_index = int(uniform_choice(rng, tuple(range(len(labels)))))
    else:
        answer_option_index = int(raw_answer_option_index)
        if answer_option_index < 0 or answer_option_index >= len(labels):
            raise ValueError("answer_option_index must be inside Tower Defense enemy labels")
    answer_label = str(labels[int(answer_option_index)])

    candidate_indices = list(range(1, max(1, len(path_points) - 1)))
    target_candidates = [
        int(index)
        for index in candidate_indices
        if sum(1 for other in candidate_indices if int(other) < int(index)) >= option_count - 1
    ]
    if not target_candidates:
        raise ValueError("not enough path points to place six ordered enemy labels")
    target_index = int(rng.choice(target_candidates))
    decoy_pool = [int(index) for index in candidate_indices if int(index) < int(target_index)]
    decoy_indices = [int(index) for index in rng.sample(decoy_pool, option_count - 1)]
    rng.shuffle(decoy_indices)

    labeled_options: list[tuple[int, str]] = []
    decoy_cursor = 0
    for label in labels:
        if str(label) == answer_label:
            labeled_options.append((int(target_index), str(label)))
        else:
            labeled_options.append((int(decoy_indices[decoy_cursor]), str(label)))
            decoy_cursor += 1

    range_min = int(params.get("best_position_range_radius_min_px", GEN_DEFAULTS.get("best_position_range_radius_min_px", DEFAULTS.best_position_range_radius_min_px)))
    range_max = int(params.get("best_position_range_radius_max_px", GEN_DEFAULTS.get("best_position_range_radius_max_px", DEFAULTS.best_position_range_radius_max_px)))
    tower_path_clearance = float(params.get("tower_path_clearance_px", GEN_DEFAULTS.get("tower_path_clearance_px", DEFAULTS.tower_path_clearance_px)))
    tower_min_gap = float(params.get("tower_min_gap_px", GEN_DEFAULTS.get("tower_min_gap_px", DEFAULTS.tower_min_gap_px)))
    towers: list[TowerDefenseTower] = []
    centers: list[Point] = []
    for index in range(int(axes.tower_count)):
        tower = _sample_decorative_tower(
            rng=rng,
            tower_id=tower_entity_id(int(index)),
            path_points=path_points,
            map_width_px=map_width,
            map_height_px=map_height,
            range_min=range_min,
            range_max=range_max,
            existing_centers=centers,
            tower_path_clearance_px=tower_path_clearance,
            tower_min_gap_px=tower_min_gap,
        )
        towers.append(tower)
        centers.append(tower.center_px)

    sample = TowerDefenseSample(
        mode=MODE_NEAREST_EXIT_ENEMY,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        map_width_px=int(map_width),
        map_height_px=int(map_height),
        path_points_px=tuple(path_points),
        towers=tuple(towers),
        enemy=None,
        answer=str(answer_label),
        target_answer=int(option_count),
        annotation_entity_ids=(path_segment_entity_id(int(target_index)),),
        construction_mode="construct_six_labeled_path_enemies_by_exit_order",
        labeled_path_enemy_options=tuple(sorted(labeled_options, key=lambda item: str(item[1]))),
        show_exit_marker=True,
        metadata={
            "enemy_option_labels": list(labels),
            "answer_option_index": int(answer_option_index),
            "answer_label": str(answer_label),
            "answer_path_index": int(target_index),
            "exit_path_index": int(len(path_points) - 1),
            "labeled_enemy_options": [
                {
                    "label": str(label),
                    "path_index": int(index),
                    "entity_id": path_segment_entity_id(int(index)),
                }
                for index, label in sorted(labeled_options, key=lambda item: str(item[1]))
            ],
        },
    )
    validate_tower_defense_sample(sample)
    return sample


def sample_marked_enemy_scene(
    *,
    rng,
    axes: TowerDefenseAxes,
    render_params: TowerDefenseRenderParams,
    params: Mapping[str, Any],
) -> TowerDefenseSample:
    """Construct one tower-defense scene with a marked enemy target."""

    map_width = int(render_params.map_width_px)
    map_height = int(render_params.map_height_px)
    path_points = sample_path_points(
        rng=rng,
        scene_variant=str(axes.scene_variant),
        map_width_px=int(map_width),
        map_height_px=int(map_height),
        path_segment_count=int(axes.path_segment_count),
    )
    low_index = max(2, int(len(path_points) * 0.34))
    high_index = min(len(path_points) - 3, int(len(path_points) * 0.70))
    target_index = int(rng.randint(low_index, max(low_index, high_index)))
    enemy = TowerDefenseEnemy(
        enemy_id=enemy_entity_id(),
        center_px=path_points[int(target_index)],
        path_index=int(target_index),
    )
    range_min = int(params.get("range_radius_min_px", GEN_DEFAULTS.get("range_radius_min_px", DEFAULTS.range_radius_min_px)))
    range_max = int(params.get("range_radius_max_px", GEN_DEFAULTS.get("range_radius_max_px", DEFAULTS.range_radius_max_px)))
    tower_path_clearance = float(params.get("tower_path_clearance_px", GEN_DEFAULTS.get("tower_path_clearance_px", DEFAULTS.tower_path_clearance_px)))
    tower_min_gap = float(params.get("tower_min_gap_px", GEN_DEFAULTS.get("tower_min_gap_px", DEFAULTS.tower_min_gap_px)))
    uncovered_margin = float(params.get("uncovered_margin_px", GEN_DEFAULTS.get("uncovered_margin_px", DEFAULTS.uncovered_margin_px)))
    cover_flags = [True] * int(axes.target_answer) + [False] * (int(axes.tower_count) - int(axes.target_answer))
    rng.shuffle(cover_flags)
    towers: list[TowerDefenseTower] = []
    centers: list[Point] = []
    for index, covers in enumerate(cover_flags):
        radius = float(rng.randint(range_min, range_max))
        center = _sample_tower_center(
            rng=rng,
            covers_target=bool(covers),
            target_point=enemy.center_px,
            radius=radius,
            map_width_px=map_width,
            map_height_px=map_height,
            path_points=path_points,
            existing_centers=centers,
            tower_path_clearance_px=tower_path_clearance,
            tower_min_gap_px=tower_min_gap,
            uncovered_margin_px=uncovered_margin,
        )
        centers.append(center)
        towers.append(
            TowerDefenseTower(
                tower_id=tower_entity_id(index),
                center_px=center,
                range_radius_px=float(radius),
                covers_target=bool(covers),
            )
        )
    annotation_ids = covered_tower_ids(towers, enemy.center_px)
    if len(annotation_ids) != int(axes.target_answer):
        raise ValueError("constructed tower coverage count mismatch")
    sample = TowerDefenseSample(
        mode=MODE_MARKED_ENEMY,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        map_width_px=int(map_width),
        map_height_px=int(map_height),
        path_points_px=tuple(path_points),
        towers=tuple(towers),
        enemy=enemy,
        answer=int(len(annotation_ids)),
        target_answer=int(axes.target_answer),
        annotation_entity_ids=tuple(annotation_ids),
        construction_mode="construct_towers_around_marked_enemy",
    )
    validate_tower_defense_sample(sample)
    return sample


def sample_covered_path_scene(
    *,
    rng,
    axes: TowerDefenseAxes,
    render_params: TowerDefenseRenderParams,
    params: Mapping[str, Any],
) -> TowerDefenseSample:
    """Construct one tower-defense scene with an exact covered-path-node count."""

    map_width = int(render_params.map_width_px)
    map_height = int(render_params.map_height_px)
    path_points = sample_path_points(
        rng=rng,
        scene_variant=str(axes.scene_variant),
        map_width_px=map_width,
        map_height_px=map_height,
        path_segment_count=int(axes.path_segment_count),
    )
    range_min = int(params.get("covered_path_range_radius_min_px", GEN_DEFAULTS.get("covered_path_range_radius_min_px", DEFAULTS.covered_path_range_radius_min_px)))
    range_max = int(params.get("covered_path_range_radius_max_px", GEN_DEFAULTS.get("covered_path_range_radius_max_px", DEFAULTS.covered_path_range_radius_max_px)))
    tower_path_clearance = float(params.get("tower_path_clearance_px", GEN_DEFAULTS.get("tower_path_clearance_px", DEFAULTS.tower_path_clearance_px)))
    tower_min_gap = float(params.get("tower_min_gap_px", GEN_DEFAULTS.get("tower_min_gap_px", DEFAULTS.tower_min_gap_px)))
    selected_indices = _select_path_coverage_indices(
        rng=rng,
        path_count=len(path_points),
        target_answer=int(axes.target_answer),
    )
    selected_set = set(int(index) for index in selected_indices)
    towers: list[TowerDefenseTower] = []
    centers: list[Point] = []
    for selected_index in selected_indices:
        tower = _sample_path_chunk_tower(
            rng=rng,
            tower_id=tower_entity_id(len(towers)),
            chunk_indices=(int(selected_index),),
            selected_indices=selected_indices,
            path_points=path_points,
            map_width_px=map_width,
            map_height_px=map_height,
            range_min=range_min,
            range_max=range_max,
            existing_centers=centers,
            tower_path_clearance_px=tower_path_clearance,
            tower_min_gap_px=tower_min_gap,
        )
        towers.append(tower)
        centers.append(tower.center_px)
    while len(towers) < int(axes.tower_count):
        tower = _sample_nonexpanding_tower(
            rng=rng,
            tower_id=tower_entity_id(len(towers)),
            selected_indices=selected_indices,
            path_points=path_points,
            map_width_px=map_width,
            map_height_px=map_height,
            range_min=range_min,
            range_max=range_max,
            existing_centers=centers,
            tower_path_clearance_px=tower_path_clearance,
            tower_min_gap_px=tower_min_gap,
        )
        towers.append(tower)
        centers.append(tower.center_px)
    annotation_ids = covered_path_segment_ids(towers, path_points)
    expected_ids = tuple(path_segment_entity_id(index) for index in selected_indices)
    if tuple(annotation_ids) != tuple(expected_ids):
        raise ValueError("constructed covered path ids mismatch")
    sample = TowerDefenseSample(
        mode=MODE_PATH_NODES,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        map_width_px=int(map_width),
        map_height_px=int(map_height),
        path_points_px=tuple(path_points),
        towers=tuple(towers),
        enemy=None,
        answer=int(len(annotation_ids)),
        target_answer=int(axes.target_answer),
        annotation_entity_ids=tuple(annotation_ids),
        construction_mode="construct_towers_by_path_node_union_coverage",
    )
    validate_tower_defense_sample(sample)
    return sample


__all__ = [
    "axis_support_metadata",
    "resolve_tower_defense_axes",
    "sample_best_tower_position_scene",
    "sample_covered_path_scene",
    "sample_marked_enemy_scene",
    "sample_nearest_exit_enemy_label_scene",
    "sample_path_points",
]
