"""Typed state for circular-chess games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trace_tasks.tasks.games.shared.piece_board_rules import ChessPiece

from .defaults import SCENE_ID


Coord = tuple[int, int]
Board = tuple[tuple[ChessPiece | None, ...], ...]


@dataclass(frozen=True)
class CircularChessSceneAxes:
    """Scene-level axes independent of the public objective."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]


@dataclass(frozen=True)
class MarkedPieceAxes:
    """Visible marked-piece axes used by marked-piece objectives."""

    piece_kind: str
    piece_color: str
    piece_kind_probabilities: dict[str, float]
    piece_color_probabilities: dict[str, float]


@dataclass(frozen=True)
class CircularChessRenderParams:
    """Resolved rendering controls for one circular-chess board."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_size_px: int
    board_frame_width_px: int
    cell_outline_width_px: int
    piece_font_size_px: int
    piece_bbox_fraction: float
    marker_width_px: int
    layout_jitter_meta: dict[str, Any]
    font_family: str = ""


@dataclass(frozen=True)
class CircularChessEvaluation:
    """Evaluation for one semantic circular-chess query."""

    answer: int
    annotation_coords: tuple[Coord, ...]
    annotation_entity_ids: tuple[str, ...]
    annotation_kind: str
    marked_coord: Coord | None = None
    marked_piece: ChessPiece | None = None
    target_coord: Coord | None = None
    target_color: str = ""


@dataclass(frozen=True)
class CircularChessSample:
    """One generated circular-chess sample."""

    board: Board
    evaluation: CircularChessEvaluation
    occupied_count: int
    construction_mode: str
    scene_variant: str
    style_variant: str


__all__ = [
    "Board",
    "CircularChessEvaluation",
    "CircularChessRenderParams",
    "CircularChessSample",
    "CircularChessSceneAxes",
    "Coord",
    "MarkedPieceAxes",
    "SCENE_ID",
]
