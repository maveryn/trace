"""Passive state records for Sixteen Soldiers games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "sixteen_soldiers"
SCENE_NAMESPACE = "games.sixteen_soldiers"

EMPTY = 0
RED = 1
BLUE = 2

Coord = Tuple[int, int]
PointId = str
Board = Tuple[Tuple[PointId, int], ...]

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "balanced_midgame",
    "center_crossroads_midgame",
    "triangle_wing_midgame",
)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "ground_court",
    "ink_court",
    "cloth_board",
    "slate_court",
    "sand_court",
)


@dataclass(frozen=True)
class JumpSpec:
    """One directed jump line: origin -> adjacent opponent -> empty landing."""

    origin_id: PointId
    middle_id: PointId
    landing_id: PointId


@dataclass(frozen=True)
class SixteenSoldiersVisualAxes:
    """Resolved scene, style, and setup axes shared by scene tasks."""

    scene_variant: str
    style_variant: str
    marked_piece_color: int
    piece_count_per_side: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    marked_piece_color_probabilities: Dict[str, float]
    piece_count_per_side_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SixteenSoldiersTargetAxis:
    """Resolved task-owned target answer axis for one instance."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SixteenSoldiersSample:
    """One symbolic Sixteen Soldiers board plus task witness ids."""

    scene_variant: str
    style_variant: str
    board: Board
    answer: int
    target_answer: int
    annotation_point_ids: Tuple[PointId, ...]
    target_color: int
    marked_point_id: PointId
    construction_mode: str


@dataclass(frozen=True)
class SixteenSoldiersTheme:
    """Scene-local color controls for Sixteen Soldiers boards."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    edge_rgb: Tuple[int, int, int]
    point_fill_rgb: Tuple[int, int, int]
    point_outline_rgb: Tuple[int, int, int]
    red_piece_fill_rgb: Tuple[int, int, int]
    red_piece_outline_rgb: Tuple[int, int, int]
    blue_piece_fill_rgb: Tuple[int, int, int]
    blue_piece_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedSixteenSoldiersScene:
    """Rendered scene plus trace-friendly geometry maps."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


__all__ = [
    "BLUE",
    "Board",
    "Coord",
    "EMPTY",
    "JumpSpec",
    "PointId",
    "RED",
    "RenderedSixteenSoldiersScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "SixteenSoldiersSample",
    "SixteenSoldiersTargetAxis",
    "SixteenSoldiersTheme",
    "SixteenSoldiersVisualAxes",
]
