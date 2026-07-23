"""Passive state and constants for Snakes and Ladders game scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


DOMAIN = "games"
SCENE_ID = "snakes_ladders"
BOARD_ROWS = 10
BOARD_COLS = 10
LAST_SQUARE = BOARD_ROWS * BOARD_COLS
SUPPORTED_BOARD_SIDES: Tuple[int, ...] = (5, 6, 7)
SUPPORTED_SNAKES_LADDERS_SCENE_VARIANTS: Tuple[str, ...] = ("standard_board",)
SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "paper",
    "neon",
    "pastel",
    "wood",
)
SUPPORTED_DIE_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class SnakesLaddersJump:
    """One directed snake or ladder jump."""

    jump_id: str
    kind: str
    start_square: int
    end_square: int

    def to_trace(self) -> Dict[str, Any]:
        return {
            "jump_id": str(self.jump_id),
            "kind": str(self.kind),
            "start_square": int(self.start_square),
            "end_square": int(self.end_square),
        }


@dataclass(frozen=True)
class SnakesLaddersMove:
    """Resolved result of one die roll from one square."""

    start_square: int
    die_value: int
    landing_square: int
    final_square: int
    jump_id: str | None

    def to_trace(self) -> Dict[str, Any]:
        return {
            "start_square": int(self.start_square),
            "die_value": int(self.die_value),
            "landing_square": int(self.landing_square),
            "final_square": int(self.final_square),
            "jump_id": None if self.jump_id is None else str(self.jump_id),
        }


@dataclass(frozen=True)
class SnakesLaddersSample:
    """Complete sampled scene and objective trace."""

    mode: str
    scene_variant: str
    style_variant: str
    board_side: int
    answer: int
    start_square: int
    jumps: Tuple[SnakesLaddersJump, ...]
    move: SnakesLaddersMove | None
    horizon_roll_count: int | None
    optimal_route: Tuple[SnakesLaddersMove, ...]
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str


@dataclass(frozen=True)
class SnakesLaddersAxes:
    """Scene-level axes shared by Snakes and Ladders objectives."""

    scene_variant: str
    style_variant: str
    board_side: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    board_side_probabilities: Dict[str, float]


__all__ = [
    "BOARD_COLS",
    "BOARD_ROWS",
    "DOMAIN",
    "LAST_SQUARE",
    "SCENE_ID",
    "SUPPORTED_BOARD_SIDES",
    "SUPPORTED_DIE_VALUES",
    "SUPPORTED_SNAKES_LADDERS_SCENE_VARIANTS",
    "SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS",
    "SnakesLaddersAxes",
    "SnakesLaddersJump",
    "SnakesLaddersMove",
    "SnakesLaddersSample",
]
