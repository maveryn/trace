"""Typed state for chess-variant games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trace_tasks.tasks.games.shared.piece_board_rules import Board, ChessPiece, Coord

from .defaults import SCENE_ID


@dataclass(frozen=True)
class ChessVariantSceneAxes:
    """Scene-level axes independent of the public objective."""

    rule_family: str
    scene_variant: str
    style_variant: str
    range_k: int
    rule_family_probabilities: dict[str, float]
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]
    range_k_probabilities: dict[str, float]


@dataclass(frozen=True)
class ChessVariantRenderParams:
    """Resolved rendering controls for one chess-variant board."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    rule_badge_height_px: int
    rule_badge_width_px: int
    header_gap_px: int
    max_board_size_px: int
    board_corner_radius_px: int
    board_frame_width_px: int
    piece_inset_fraction: float
    marked_square_outline_width_px: int
    rule_badge_font_size_px: int
    piece_font_size_px: int
    layout_jitter_meta: dict[str, Any]
    font_family: str = ""


@dataclass(frozen=True)
class ChessVariantEvaluation:
    """Evaluation for one semantic chess-variant query."""

    answer: int
    legal_destinations: tuple[Coord, ...]
    capture_coords: tuple[Coord, ...]
    annotation_coords: tuple[Coord, ...]
    annotation_entity_ids: tuple[str, ...]
    annotation_kind: str
    marked_coord: Coord
    marked_piece: ChessPiece | None
    marker_role: str = "marked_piece"


@dataclass(frozen=True)
class ChessVariantSample:
    """One generated chess-variant sample."""

    board: Board
    evaluation: ChessVariantEvaluation
    occupied_count: int
    construction_mode: str
    scene_variant: str
    style_variant: str


__all__ = [
    "ChessVariantEvaluation",
    "ChessVariantRenderParams",
    "ChessVariantSample",
    "ChessVariantSceneAxes",
    "SCENE_ID",
]
