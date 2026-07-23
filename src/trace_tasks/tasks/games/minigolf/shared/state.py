"""Passive Mini-golf scene state and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from .defaults import FIRST_OBSTACLE_MODE, SHOT_OPTIONS_MODE


@dataclass(frozen=True)
class MinigolfObstacle:
    """One visible mini-golf course obstacle."""

    obstacle_id: str
    label: str
    kind: str
    x_norm: float
    y_norm: float
    radius_norm: float
    color_index: int


@dataclass(frozen=True)
class MinigolfShotOption:
    """One labeled shot-direction option."""

    path_id: str
    label: str
    angle_rad: float
    color_index: int


@dataclass(frozen=True)
class MinigolfSample:
    """Generated mini-golf course state before rendering."""

    mode: str
    scene_variant: str
    style_variant: str
    answer: str
    ball_x_norm: float
    ball_y_norm: float
    hole_x_norm: float
    hole_y_norm: float
    obstacles: Tuple[MinigolfObstacle, ...]
    shot_options: Tuple[MinigolfShotOption, ...]
    target_obstacle_id: str | None
    target_obstacle_label: str | None
    target_path_id: str | None
    target_path_label: str | None
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str
    cue_visible_fraction: float
    hidden_paths_norm: Mapping[str, Tuple[Tuple[float, float], ...]]


def obstacle_entity_id(index: int) -> str:
    """Return stable entity id for one course obstacle."""

    return f"obstacle_{int(index)}"


def path_entity_id(index: int) -> str:
    """Return stable entity id for one shot option."""

    return f"path_{int(index)}"


def path_label(index: int) -> str:
    """Return a prompt-facing numeric shot label."""

    return str(int(index) + 1)


def validate_first_obstacle_sample(sample: MinigolfSample) -> None:
    """Validate a scene where the answer is the first obstacle reached."""

    _validate_unique_ids_and_labels(sample)
    if str(sample.mode) != FIRST_OBSTACLE_MODE:
        raise ValueError("Mini-golf first-obstacle sample has wrong scene mode")
    if sample.target_obstacle_id is None or sample.target_obstacle_label is None:
        raise ValueError("Mini-golf first-obstacle sample requires a target obstacle")
    obstacle_ids = {str(obstacle.obstacle_id) for obstacle in sample.obstacles}
    if str(sample.target_obstacle_id) not in obstacle_ids:
        raise ValueError("Mini-golf target obstacle id must be visible")
    if str(sample.answer) != str(sample.target_obstacle_label):
        raise ValueError("Mini-golf first-obstacle answer does not match target label")
    if tuple(str(entity_id) for entity_id in sample.annotation_entity_ids) != (str(sample.target_obstacle_id),):
        raise ValueError("Mini-golf first-obstacle annotation must mark only the target obstacle")


def validate_shot_options_sample(sample: MinigolfSample) -> None:
    """Validate a scene where the answer is the hole-reaching shot option."""

    _validate_unique_ids_and_labels(sample)
    if str(sample.mode) != SHOT_OPTIONS_MODE:
        raise ValueError("Mini-golf shot-options sample has wrong scene mode")
    if sample.target_path_id is None or sample.target_path_label is None:
        raise ValueError("Mini-golf shot-options sample requires a target path")
    path_ids = {str(path.path_id) for path in sample.shot_options}
    if str(sample.target_path_id) not in path_ids:
        raise ValueError("Mini-golf target path id must be visible")
    if str(sample.answer) != str(sample.target_path_label):
        raise ValueError("Mini-golf shot-options answer does not match target label")
    if tuple(str(entity_id) for entity_id in sample.annotation_entity_ids) != (str(sample.target_path_id),):
        raise ValueError("Mini-golf shot-options annotation must mark only the target path")


def _validate_unique_ids_and_labels(sample: MinigolfSample) -> None:
    """Check scene entity uniqueness and annotation references."""

    obstacle_ids = [str(obstacle.obstacle_id) for obstacle in sample.obstacles]
    obstacle_labels = [str(obstacle.label) for obstacle in sample.obstacles]
    path_ids = [str(path.path_id) for path in sample.shot_options]
    path_labels = [str(path.label) for path in sample.shot_options]
    if len(obstacle_ids) != len(set(obstacle_ids)):
        raise ValueError("Mini-golf obstacle ids must be unique")
    if len(obstacle_labels) != len(set(obstacle_labels)):
        raise ValueError("Mini-golf obstacle labels must be unique")
    if len(path_ids) != len(set(path_ids)):
        raise ValueError("Mini-golf path ids must be unique")
    if len(path_labels) != len(set(path_labels)):
        raise ValueError("Mini-golf path labels must be unique")
    known_entities = set(obstacle_ids) | set(path_ids) | {"ball", "hole"}
    if not set(str(entity_id) for entity_id in sample.annotation_entity_ids) <= known_entities:
        raise ValueError("Mini-golf annotation references unknown entities")


__all__ = [
    "MinigolfObstacle",
    "MinigolfSample",
    "MinigolfShotOption",
    "obstacle_entity_id",
    "path_entity_id",
    "path_label",
    "validate_first_obstacle_sample",
    "validate_shot_options_sample",
]
