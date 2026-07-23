"""Passive state records for radial hunt board games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "radial_hunt_board"
SCENE_NAMESPACE = "games.radial_hunt_board"
CENTER: Tuple[int, int] = (0, 0)
Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("open_position", "mixed_position", "crowded_position")
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = ("ink_rings", "carved_wood", "temple_cloth", "night_gold", "chalk_circle")


@dataclass(frozen=True)
class RadialHuntBoardVisualAxes:
    """Resolved scene and style axes shared by radial hunt board tasks."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RadialHuntBoardTargetAxis:
    """Resolved task-owned target answer axis for one radial board instance."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RadialHuntBoardSample:
    """One symbolic radial hunt board sample and witness."""

    scene_variant: str
    style_variant: str
    marked_coord: Coord
    occupied_coords: Tuple[Coord, ...]
    annotation_coords: Tuple[Coord, ...]
    answer: int
    construction_mode: str


@dataclass(frozen=True)
class RadialHuntBoardTheme:
    """Scene-local palette for a radial hunt board render."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    edge_rgb: Tuple[int, int, int]
    point_fill_rgb: Tuple[int, int, int]
    point_outline_rgb: Tuple[int, int, int]
    marked_piece_fill_rgb: Tuple[int, int, int]
    marked_piece_outline_rgb: Tuple[int, int, int]
    opponent_piece_fill_rgb: Tuple[int, int, int]
    opponent_piece_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedRadialHuntBoardScene:
    """Rendered board plus trace-friendly entity maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


__all__ = [
    "CENTER",
    "Coord",
    "Edge",
    "RadialHuntBoardSample",
    "RadialHuntBoardTargetAxis",
    "RadialHuntBoardTheme",
    "RadialHuntBoardVisualAxes",
    "RenderedRadialHuntBoardScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
]
