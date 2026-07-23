"""Passive dots-and-boxes scene state for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DotsAndBoxesEdgeInstance:
    """One edge in a visible dots-and-boxes board."""

    edge_id: str
    orientation: str
    dot_start: Tuple[int, int]
    dot_end: Tuple[int, int]
    is_drawn: bool
    is_highlighted: bool


@dataclass(frozen=True)
class DotsAndBoxesBoxInstance:
    """One box region in a dots-and-boxes board."""

    box_id: str
    row_index: int
    column_index: int
    edge_ids: Tuple[str, str, str, str]
    owner: str = ""


@dataclass(frozen=True)
class DotsAndBoxesSimulationResult:
    """One forced-turn simulation result from a highlighted starting edge."""

    is_forced: bool
    capture_count: int
    captured_box_ids: Tuple[str, ...]
    move_edge_sequence: Tuple[str, ...]
    branching_edge_ids: Tuple[str, ...]
    initial_completed_box_ids: Tuple[str, ...]


@dataclass(frozen=True)
class DotsAndBoxesBoardState:
    """One generated dots-and-boxes board plus its task trace state."""

    box_rows: int
    box_cols: int
    edges: Tuple[DotsAndBoxesEdgeInstance, ...]
    boxes: Tuple[DotsAndBoxesBoxInstance, ...]
    highlighted_edge_id: str
    drawn_edge_ids: Tuple[str, ...]
    captured_box_ids: Tuple[str, ...]
    move_edge_sequence: Tuple[str, ...]
    branching_edge_ids: Tuple[str, ...]
    path_box_ids: Tuple[str, ...]
    path_turn_count: int
    target_answer: int
    highlighted_edge_ids: Tuple[str, ...] = ()
    counted_box_ids: Tuple[str, ...] = ()
    counted_edge_ids: Tuple[str, ...] = ()
    candidate_edge_ids: Tuple[str, ...] = ()
    option_label_by_box_id: Tuple[Tuple[str, str], ...] = ()
    answer_box_id: str = ""
    answer_label: str = ""


@dataclass(frozen=True)
class DotsAndBoxesSceneAxes:
    """Resolved scene/style axes shared by dots-and-boxes tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class DotsAndBoxesIntegerAxis:
    """Resolved integer sampling axis with trace metadata."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class DotsAndBoxesBoardShapeAxis:
    """Resolved board shape with trace metadata."""

    box_rows: int
    box_cols: int
    probabilities: Dict[str, float]


__all__ = [
    "DotsAndBoxesBoardState",
    "DotsAndBoxesBoardShapeAxis",
    "DotsAndBoxesBoxInstance",
    "DotsAndBoxesEdgeInstance",
    "DotsAndBoxesIntegerAxis",
    "DotsAndBoxesSceneAxes",
    "DotsAndBoxesSimulationResult",
]
