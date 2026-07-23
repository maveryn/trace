"""Tower-defense geometry and coverage rules."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .state import Point, TowerDefenseSample, TowerDefenseTower


MODE_MARKED_ENEMY = "enemy_tower_coverage"
MODE_PATH_NODES = "path_node_coverage"
MODE_BEST_POSITION = "candidate_tower_position"
MODE_NEAREST_EXIT_ENEMY = "labeled_path_order"
OPTION_LABELS = ("A", "B", "C", "D")
ENEMY_OPTION_LABELS = ("A", "B", "C", "D", "E", "F")


def tower_entity_id(index: int) -> str:
    """Return stable entity id for one tower."""

    return f"tower_{int(index):02d}"


def candidate_tower_entity_id(label: str) -> str:
    """Return stable entity id for one labeled candidate tower position."""

    normalized = str(label).strip().upper()
    if normalized not in OPTION_LABELS:
        raise ValueError(f"unsupported tower-defense candidate label: {label!r}")
    return f"candidate_{normalized}"


def candidate_tower_label_from_id(tower_id: str) -> str | None:
    """Return the A-D label encoded in one candidate tower id, if any."""

    text = str(tower_id)
    if not text.startswith("candidate_"):
        return None
    label = text.removeprefix("candidate_").strip().upper()
    return label if label in OPTION_LABELS else None


def path_segment_entity_id(index: int) -> str:
    """Return stable entity id for one discrete path witness."""

    return f"path_segment_{int(index):02d}"


def enemy_entity_id() -> str:
    """Return stable entity id for the marked enemy."""

    return "marked_enemy"


def local_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """Return local render-space distance between two points."""

    return math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1]))


def tower_covers_point(tower: TowerDefenseTower, point: Sequence[float], *, epsilon_px: float = 1e-6) -> bool:
    """Return whether one tower covers a point under the visible circle rule."""

    return float(local_distance(tower.center_px, point)) <= float(tower.range_radius_px) + float(epsilon_px)


def covered_tower_ids(towers: Iterable[TowerDefenseTower], point: Sequence[float]) -> tuple[str, ...]:
    """Return ids of towers whose range covers the point."""

    return tuple(str(tower.tower_id) for tower in towers if tower_covers_point(tower, point))


def covered_path_segment_ids(towers: Iterable[TowerDefenseTower], path_points: Sequence[Point]) -> tuple[str, ...]:
    """Return ids of path nodes covered by at least one tower."""

    tower_tuple = tuple(towers)
    return tuple(
        path_segment_entity_id(index)
        for index, point in enumerate(path_points)
        if any(tower_covers_point(tower, point) for tower in tower_tuple)
    )


def validate_tower_defense_sample(sample: TowerDefenseSample) -> None:
    """Validate one generated tower-defense sample contract."""

    if str(sample.mode) not in {MODE_MARKED_ENEMY, MODE_PATH_NODES, MODE_BEST_POSITION, MODE_NEAREST_EXIT_ENEMY}:
        raise ValueError(f"unsupported tower-defense mode: {sample.mode}")
    if int(sample.map_width_px) <= 0 or int(sample.map_height_px) <= 0:
        raise ValueError("tower-defense map dimensions must be positive")
    if len(sample.path_points_px) < 6:
        raise ValueError("tower-defense path needs at least six visible points")
    tower_ids = [str(tower.tower_id) for tower in sample.towers]
    if len(tower_ids) != len(set(tower_ids)):
        raise ValueError("tower ids must be unique")
    if not sample.towers:
        raise ValueError("tower-defense sample must include at least one tower")
    for tower in sample.towers:
        x, y = float(tower.center_px[0]), float(tower.center_px[1])
        if not (0.0 <= x <= float(sample.map_width_px) and 0.0 <= y <= float(sample.map_height_px)):
            raise ValueError(f"tower center outside map: {tower.tower_id}")
        if float(tower.range_radius_px) <= 0.0:
            raise ValueError(f"tower range must be positive: {tower.tower_id}")
    if str(sample.mode) == MODE_MARKED_ENEMY:
        if sample.enemy is None:
            raise ValueError("marked enemy scene requires an enemy")
        if not (0 <= int(sample.enemy.path_index) < len(sample.path_points_px)):
            raise ValueError("marked enemy path_index must reference a visible path point")
        expected_enemy_point = sample.path_points_px[int(sample.enemy.path_index)]
        if local_distance(sample.enemy.center_px, expected_enemy_point) > 1e-6:
            raise ValueError("marked enemy center must lie on its path point")
        expected_annotation = covered_tower_ids(sample.towers, sample.enemy.center_px)
        expected_answer = len(expected_annotation)
    elif str(sample.mode) == MODE_PATH_NODES:
        expected_annotation = covered_path_segment_ids(sample.towers, sample.path_points_px)
        expected_answer = len(expected_annotation)
    elif str(sample.mode) == MODE_BEST_POSITION:
        candidate_counts: dict[str, int] = {}
        candidate_ids: dict[str, str] = {}
        for tower in sample.towers:
            label = candidate_tower_label_from_id(str(tower.tower_id))
            if label is None:
                continue
            candidate_ids[str(label)] = str(tower.tower_id)
            candidate_counts[str(label)] = sum(1 for point in sample.path_points_px if tower_covers_point(tower, point))
        if set(candidate_counts.keys()) != set(OPTION_LABELS):
            raise ValueError("best-position scene requires exactly candidates A-D")
        max_count = max(int(value) for value in candidate_counts.values())
        best_labels = [label for label, value in candidate_counts.items() if int(value) == int(max_count)]
        if len(best_labels) != 1:
            raise ValueError("best-position scene must have one unique best candidate")
        expected_answer = str(best_labels[0])
        expected_annotation = (candidate_ids[str(expected_answer)],)
        if int(sample.target_answer) != int(max_count):
            raise ValueError("best-position target_answer must equal the winning coverage count")
    else:
        if len(sample.labeled_path_enemy_options) != 6:
            raise ValueError("nearest-exit task requires exactly six labeled enemies")
        option_labels = [str(label) for _index, label in sample.labeled_path_enemy_options]
        if set(option_labels) != set(ENEMY_OPTION_LABELS):
            raise ValueError("nearest-exit task requires enemy labels A-F")
        index_by_label = {str(label): int(index) for index, label in sample.labeled_path_enemy_options}
        if len(set(index_by_label.values())) != len(index_by_label):
            raise ValueError("nearest-exit labeled enemies must occupy distinct path points")
        if not all(0 <= int(index) < len(sample.path_points_px) - 1 for index in index_by_label.values()):
            raise ValueError("nearest-exit labeled enemies must be before the exit endpoint")
        expected_answer = max(index_by_label, key=lambda label: int(index_by_label[str(label)]))
        expected_annotation = (path_segment_entity_id(index_by_label[str(expected_answer)]),)
        if int(sample.target_answer) != len(sample.labeled_path_enemy_options):
            raise ValueError("nearest-exit target_answer must equal option count")
    if tuple(str(value) for value in sample.annotation_entity_ids) != tuple(expected_annotation):
        raise ValueError("tower-defense annotation ids must match the task contract")
    if sample.answer != expected_answer:
        raise ValueError("tower-defense answer must match the task contract")
    if str(sample.mode) not in {MODE_BEST_POSITION, MODE_NEAREST_EXIT_ENEMY} and int(sample.answer) != int(sample.target_answer):
        raise ValueError("tower-defense answer must equal target_answer")


def visible_tower_trace(towers: Sequence[TowerDefenseTower]) -> tuple[dict, ...]:
    """Return trace-friendly tower records."""

    return tuple(
        {
            "tower_id": str(tower.tower_id),
            "center_px_local": [round(float(tower.center_px[0]), 3), round(float(tower.center_px[1]), 3)],
            "range_radius_px": round(float(tower.range_radius_px), 3),
            "covers_target": bool(tower.covers_target),
        }
        for tower in towers
    )


__all__ = [
    "MODE_BEST_POSITION",
    "MODE_MARKED_ENEMY",
    "MODE_NEAREST_EXIT_ENEMY",
    "MODE_PATH_NODES",
    "ENEMY_OPTION_LABELS",
    "OPTION_LABELS",
    "candidate_tower_entity_id",
    "candidate_tower_label_from_id",
    "covered_path_segment_ids",
    "covered_tower_ids",
    "enemy_entity_id",
    "local_distance",
    "path_segment_entity_id",
    "tower_covers_point",
    "tower_entity_id",
    "validate_tower_defense_sample",
    "visible_tower_trace",
]
