"""State contracts for the Checkers games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.games.shared.style import SUPPORTED_CHECKERS_STYLE_VARIANTS as _SUPPORTED_CHECKERS_STYLE_VARIANTS

from .rules import Board, CheckersCaptureChain, CheckersMove, Coord


SCENE_ID = "checkers"
SUPPORTED_CHECKERS_SCENE_VARIANTS: Tuple[str, ...] = ("midgame_board", "crowded_board")
SUPPORTED_CHECKERS_STYLE_VARIANTS = tuple(_SUPPORTED_CHECKERS_STYLE_VARIANTS)


@dataclass(frozen=True)
class ResolvedCheckersSceneAxes:
    """Scene/style axes shared by Checkers tasks."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant: str
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class TargetAnswerAxis:
    """Resolved task-owned integer answer axis."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CheckersEvaluation:
    """Semantic result and visual witnesses for one finalized board."""

    answer: int
    legal_moves: Tuple[CheckersMove, ...]
    capture_moves: Tuple[CheckersMove, ...]
    annotation_coords: Tuple[Coord, ...]
    annotation_entity_ids: Tuple[str, ...]
    annotation_kind: str = "cell"
    marked_coord: Coord | None = None
    max_capture_chains: Tuple[CheckersCaptureChain, ...] = ()
    selected_capture_chain: CheckersCaptureChain | None = None


@dataclass(frozen=True)
class SampledCheckersScene:
    """Generated Checkers board plus task-owned semantic witnesses."""

    board: Board
    current_player: int
    scene_variant: str
    style_variant: str
    evaluation: CheckersEvaluation
    occupied_count: int
    construction_mode: str
    extra: Mapping[str, Any] | None = None


__all__ = [
    "SCENE_ID",
    "SUPPORTED_CHECKERS_SCENE_VARIANTS",
    "SUPPORTED_CHECKERS_STYLE_VARIANTS",
    "CheckersEvaluation",
    "ResolvedCheckersSceneAxes",
    "SampledCheckersScene",
    "TargetAnswerAxis",
]
