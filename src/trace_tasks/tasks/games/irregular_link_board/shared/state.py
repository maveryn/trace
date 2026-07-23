"""Passive state types for irregular-link-board scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "irregular_link_board"
SCENE_NAMESPACE = "games.irregular_link_board"
STYLE_VARIANTS: Tuple[str, ...] = ("woodcut", "ink_diagram", "garden_cloth", "night_lines", "parchment")
SCENE_VARIANTS: Tuple[str, ...] = ("sparse_links", "mixed_links", "dense_links")
Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]


@dataclass(frozen=True)
class IrregularLinkBoardAxes:
    """Resolved generation and style axes for one sample."""

    scene_variant: str
    style_variant: str
    board_size: int
    target_answer: int
    target_answer_support: Tuple[int, ...]
    board_size_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class IrregularLinkBoardSample:
    """One symbolic variable-link board and selected count witnesses."""

    board_size: int
    scene_variant: str
    style_variant: str
    marked_coord: Coord
    occupied_coords: Tuple[Coord, ...]
    edges: Tuple[Edge, ...]
    annotation_coords: Tuple[Coord, ...]
    answer: int
    construction_mode: str


@dataclass(frozen=True)
class IrregularLinkBoardTheme:
    """Scene-local palette for board, points, links, and pieces."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    edge_rgb: Tuple[int, int, int]
    point_fill_rgb: Tuple[int, int, int]
    point_outline_rgb: Tuple[int, int, int]
    marked_piece_fill_rgb: Tuple[int, int, int]
    marked_piece_outline_rgb: Tuple[int, int, int]
    blocker_piece_fill_rgb: Tuple[int, int, int]
    blocker_piece_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedIrregularLinkBoardScene:
    """Rendered board plus trace-friendly geometry maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


__all__ = [
    "Coord",
    "Edge",
    "IrregularLinkBoardAxes",
    "IrregularLinkBoardSample",
    "IrregularLinkBoardTheme",
    "RenderedIrregularLinkBoardScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANTS",
    "STYLE_VARIANTS",
]
