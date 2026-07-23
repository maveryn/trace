"""Passive state and constants for visible Snake board tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


DOMAIN = "games"
SCENE_ID = "snake"
SCENE_VARIANTS: Tuple[str, ...] = ("square_grid",)
STYLE_VARIANTS: Tuple[str, ...] = ("classic", "neon", "forest", "paper", "candy")
DIRECTION_NAMES: Tuple[str, ...] = ("up", "down", "left", "right")
DIRECTION_DELTAS: Mapping[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PLANNED_MOVE_OUTCOMES: Tuple[str, ...] = ("point", "game_over")
Coord = Tuple[int, int]


@dataclass(frozen=True)
class SnakeState:
    """One visible Snake state; body is ordered from neck toward tail."""

    board_size: int
    head: Coord
    body: Tuple[Coord, ...]
    food: Coord
    obstacles: Tuple[Coord, ...] = ()


@dataclass(frozen=True)
class SnakeSimulation:
    """Result of simulating one Snake move sequence."""

    outcome: str
    event_step: int
    traversed_coords: Tuple[Coord, ...]
    collision_coord: Coord | None
    final_head: Coord


@dataclass(frozen=True)
class SnakeSample:
    """Generated scene state plus task-owned answer support fields."""

    answer: str | int
    state: SnakeState
    annotation_cell_ids: Tuple[str, ...]
    construction_mode: str
    planned_moves: Tuple[str, ...] = ()
    safe_directions: Tuple[str, ...] = ()
    result_options: Tuple[Mapping[str, Any], ...] = ()
    target_outcome: str | None = None
    observed_event_step: int | None = None


@dataclass(frozen=True)
class SnakeSceneAxes:
    """Resolved scene and rendering axes shared by all Snake objectives."""

    scene_variant: str
    style_variant: str
    board_size: int
    body_length: int
    planned_move_count: int
    obstacle_count: int
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]
    board_size_probabilities: dict[str, float]
    body_length_probabilities: dict[str, float]
    planned_move_count_probabilities: dict[str, float]
    obstacle_count_probabilities: dict[str, float]


__all__ = [
    "Coord",
    "DIRECTION_DELTAS",
    "DIRECTION_NAMES",
    "DOMAIN",
    "PLANNED_MOVE_OUTCOMES",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "STYLE_VARIANTS",
    "SnakeSample",
    "SnakeSceneAxes",
    "SnakeSimulation",
    "SnakeState",
]
