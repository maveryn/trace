"""State contracts for the Chess games scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.games.shared.piece_board_rules import Board, ChessPiece, Coord, coord_to_cell_id
from trace_tasks.tasks.games.shared.style import SUPPORTED_CHESS_STYLE_VARIANTS as _SUPPORTED_CHESS_STYLE_VARIANTS

SCENE_ID = "chess"
SUPPORTED_CHESS_SCENE_VARIANTS: Tuple[str, ...] = ("sparse_board", "crowded_board")
SUPPORTED_CHESS_STYLE_VARIANTS = tuple(_SUPPORTED_CHESS_STYLE_VARIANTS)
CHESS_OPTION_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + idx) for idx in range(6))
PIECE_COUNT_KIND_SUPPORT: Tuple[str, ...] = ("king", "queen", "rook", "bishop", "knight", "pawn")
PIECE_COUNT_COLOR_SUPPORT: Tuple[str, ...] = ("white", "black")


@dataclass(frozen=True)
class ResolvedChessSceneAxes:
    """Scene/style axes shared by Chess tasks."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant: str
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ChessSceneSample:
    """Generated Chess board plus task-owned semantic witnesses."""

    board: Board
    scene_variant: str
    style_variant: str
    construction_mode: str
    player_color: str
    marked_coord: Coord | None = None
    target_coord: Coord | None = None
    marked_piece: ChessPiece | None = None
    target_piece_kind: str = ""
    target_piece_color: str = ""
    destination_coords: Tuple[Coord, ...] = ()
    capture_coords: Tuple[Coord, ...] = ()
    attacker_coords: Tuple[Coord, ...] = ()
    blocker_coords: Tuple[Coord, ...] = ()
    annotation_coords: Tuple[Coord, ...] = ()
    annotation_entity_ids: Tuple[str, ...] = ()
    annotation_kind: str = "cell"
    occupied_count: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChessMoveOption:
    """One visible candidate chess move option."""

    label: str
    source: Coord
    destination: Coord
    piece: ChessPiece

    @property
    def text(self) -> str:
        from trace_tasks.tasks.games.shared.piece_board_rules import coord_to_square_name

        return f"{self.label}: {self.piece.kind} {coord_to_square_name(self.source)}-{coord_to_square_name(self.destination)}"


@dataclass(frozen=True)
class ChessCheckmateSample:
    """Generated checkmate-option Chess sample."""

    board: Board
    player_color: str
    defender_color: str
    correct_option: ChessMoveOption
    options: Tuple[ChessMoveOption, ...]
    defender_king_coord: Coord
    occupied_count: int
    scene_variant: str
    style_variant: str
    construction_mode: str
    option_count: int
    option_label_support: Tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)


def piece_kind_plural(kind: str) -> str:
    """Return the prompt-facing plural for one chess piece kind."""

    return "kings" if str(kind) == "king" else f"{str(kind)}s"


def cell_entity_ids(coords: Tuple[Coord, ...]) -> Tuple[str, ...]:
    """Return stable cell ids for a coordinate tuple."""

    return tuple(coord_to_cell_id(coord) for coord in coords)


__all__ = [
    "CHESS_OPTION_LABELS",
    "PIECE_COUNT_COLOR_SUPPORT",
    "PIECE_COUNT_KIND_SUPPORT",
    "SCENE_ID",
    "SUPPORTED_CHESS_SCENE_VARIANTS",
    "SUPPORTED_CHESS_STYLE_VARIANTS",
    "ChessCheckmateSample",
    "ChessMoveOption",
    "ChessSceneSample",
    "ResolvedChessSceneAxes",
    "cell_entity_ids",
    "piece_kind_plural",
]
