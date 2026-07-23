"""Passive state and constants for Reversi scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SCENE_ID = "reversi"
SCENE_NAMESPACE = "games.reversi"

EMPTY = 0
BLACK = 1
WHITE = -1

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("compact_board", "classic_board")
BOARD_SIZE_BY_SCENE_VARIANT = {
    "compact_board": 6,
    "classic_board": 8,
}

Coord = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class ReversiVisualAxes:
    """Resolved visual axes shared by Reversi public tasks."""

    scene_variant: str
    style_variant: str
    board_size: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ReversiTargetAxis:
    """Resolved integer-answer target for one task objective."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SampledReversiScene:
    """One sampled Reversi board and the symbolic witnesses for one objective."""

    board: Board
    current_player: int
    legal_moves: Dict[Coord, Tuple[Coord, ...]]
    annotation_coords: Tuple[Coord, ...]
    annotation_entity_ids: Tuple[str, ...]
    marked_move: Coord | None
    marked_move_flips: Tuple[Coord, ...]
    construction_mode: str

    @property
    def answer(self) -> int:
        """Return the count represented by the symbolic annotation witnesses."""

        return int(len(self.annotation_entity_ids))


__all__ = [
    "BLACK",
    "BOARD_SIZE_BY_SCENE_VARIANT",
    "Board",
    "Coord",
    "EMPTY",
    "ReversiTargetAxis",
    "ReversiVisualAxes",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SampledReversiScene",
    "WHITE",
]
