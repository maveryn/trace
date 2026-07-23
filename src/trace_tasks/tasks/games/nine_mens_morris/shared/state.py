"""Passive state objects and board constants for Nine Men's Morris."""

from __future__ import annotations

from dataclasses import dataclass


POSITION_LAYOUT: tuple[tuple[str, float, float], ...] = (
    ("a7", 0.10, 0.10),
    ("d7", 0.50, 0.10),
    ("g7", 0.90, 0.10),
    ("b6", 0.23, 0.23),
    ("d6", 0.50, 0.23),
    ("f6", 0.77, 0.23),
    ("c5", 0.36, 0.36),
    ("d5", 0.50, 0.36),
    ("e5", 0.64, 0.36),
    ("a4", 0.10, 0.50),
    ("b4", 0.23, 0.50),
    ("c4", 0.36, 0.50),
    ("e4", 0.64, 0.50),
    ("f4", 0.77, 0.50),
    ("g4", 0.90, 0.50),
    ("c3", 0.36, 0.64),
    ("d3", 0.50, 0.64),
    ("e3", 0.64, 0.64),
    ("b2", 0.23, 0.77),
    ("d2", 0.50, 0.77),
    ("f2", 0.77, 0.77),
    ("a1", 0.10, 0.90),
    ("d1", 0.50, 0.90),
    ("g1", 0.90, 0.90),
)

MILL_POSITION_INDICES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (9, 10, 11),
    (12, 13, 14),
    (15, 16, 17),
    (18, 19, 20),
    (21, 22, 23),
    (0, 9, 21),
    (3, 10, 18),
    (6, 11, 15),
    (1, 4, 7),
    (16, 19, 22),
    (8, 12, 17),
    (5, 13, 20),
    (2, 14, 23),
)


@dataclass(frozen=True)
class NineMensMorrisPieceInstance:
    """One visible piece on the Nine Men's Morris board."""

    piece_id: str
    node_index: int
    node_label: str
    color: str


@dataclass(frozen=True)
class NineMensMorrisBoardState:
    """One generated Morris board and its mill-membership trace."""

    piece_specs: tuple[NineMensMorrisPieceInstance, ...]
    white_piece_ids_in_mill: tuple[str, ...]
    black_piece_ids_in_mill: tuple[str, ...]
    all_piece_ids_in_mill: tuple[str, ...]
    white_mill_ids: tuple[str, ...]
    black_mill_ids: tuple[str, ...]
    white_mill_completion_node_labels: tuple[str, ...]
    black_mill_completion_node_labels: tuple[str, ...]
    overlapping_piece_ids: tuple[str, ...]
    target_answer: int


@dataclass(frozen=True)
class NineMensMorrisOccupancyAnalysis:
    """Derived mill membership information for one board occupancy."""

    white_piece_ids_in_mill: tuple[str, ...]
    black_piece_ids_in_mill: tuple[str, ...]
    all_piece_ids_in_mill: tuple[str, ...]
    white_mill_ids: tuple[str, ...]
    black_mill_ids: tuple[str, ...]
    white_mill_completion_node_labels: tuple[str, ...]
    black_mill_completion_node_labels: tuple[str, ...]
    mill_membership_count_by_piece_id: dict[str, int]


__all__ = [
    "MILL_POSITION_INDICES",
    "POSITION_LAYOUT",
    "NineMensMorrisBoardState",
    "NineMensMorrisOccupancyAnalysis",
    "NineMensMorrisPieceInstance",
]
