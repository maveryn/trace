"""Typed state for Connect Four games tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .rules import Board, Coord
from .defaults import SCENE_ID


@dataclass(frozen=True)
class ConnectFourSceneAxes:
    """Scene-level visual and board-size axes."""

    scene_variant: str
    board_size_variant: str
    board_rows: int
    board_columns: int
    style_variant: str
    scene_variant_probabilities: dict[str, float]
    board_size_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]


@dataclass(frozen=True)
class ConnectFourEvaluation:
    """Evaluation payload derived from one finalized board."""

    answer: int
    annotation_coords: tuple[Coord, ...]
    annotation_entity_ids: tuple[str, ...]
    winning_move_coords: tuple[Coord, ...]
    safe_move_coords: tuple[Coord, ...]


@dataclass(frozen=True)
class ConnectFourCountSample:
    """One sampled Connect Four count scene."""

    board: Board
    current_player: int
    evaluation: ConnectFourEvaluation
    occupied_count: int
    construction_mode: str
    scene_variant: str
    board_size_variant: str
    board_rows: int
    board_columns: int
    style_variant: str


@dataclass(frozen=True)
class ConnectFourLabelSample:
    """One sampled Connect Four winning-column label scene."""

    board: Board
    current_player: int
    evaluation: ConnectFourEvaluation
    occupied_count: int
    construction_mode: str
    scene_variant: str
    board_size_variant: str
    board_rows: int
    board_columns: int
    style_variant: str
    threat_kind: str
    threat_kind_probabilities: dict[str, float]
    column_labels: tuple[str, ...]
    answer_label: str
    answer_column: int
    winning_line_coords: tuple[Coord, ...]


@dataclass(frozen=True)
class ConnectFourColumnProfileSample:
    """One sampled Connect Four column profile label scene."""

    board: Board
    current_player: int
    evaluation: ConnectFourEvaluation
    occupied_count: int
    construction_mode: str
    scene_variant: str
    board_size_variant: str
    board_rows: int
    board_columns: int
    style_variant: str
    column_labels: tuple[str, ...]
    answer_label: str
    answer_column: int
    target_red_count: int
    target_yellow_count: int


__all__ = [
    "ConnectFourColumnProfileSample",
    "ConnectFourCountSample",
    "ConnectFourEvaluation",
    "ConnectFourLabelSample",
    "ConnectFourSceneAxes",
    "SCENE_ID",
]
