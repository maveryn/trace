"""Passive state containers for the Tents puzzle scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image

DOMAIN = "puzzles"
SCENE_ID = "tents"
Cell = Tuple[int, int]

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "tents_classic",
    "tents_card",
    "tents_blueprint",
)
SUPPORTED_PALETTE_VARIANTS: Tuple[str, ...] = (
    "garden",
    "autumn",
    "lake",
    "violet",
    "slate",
)


@dataclass(frozen=True)
class CandidateCellSpec:
    """One labeled candidate cell on a Tents board."""

    label: str
    row: int
    col: int
    is_correct: bool
    is_legal: bool

    @property
    def cell(self) -> Cell:
        return (int(self.row), int(self.col))


@dataclass(frozen=True)
class LabeledTentSpec:
    """One labeled visible tent used as an answer option."""

    label: str
    row: int
    col: int
    is_correct: bool
    violation_type: str = ""

    @property
    def cell(self) -> Cell:
        return (int(self.row), int(self.col))


@dataclass(frozen=True)
class TentsSample:
    """Semantic Tents board state sampled before rendering."""

    rows: int
    cols: int
    grid_rows_range: Tuple[int, int]
    grid_cols_range: Tuple[int, int]
    marked_tree: Cell | None
    candidate_specs: Tuple[CandidateCellSpec, ...]
    labeled_tent_specs: Tuple[LabeledTentSpec, ...]
    visible_tents: Tuple[Cell, ...]
    tree_cells: Tuple[Cell, ...]
    row_clues: Tuple[int, ...]
    col_clues: Tuple[int, ...]
    legal_candidate_cells: Tuple[Cell, ...]
    option_count: int
    target_answer_support: Tuple[Any, ...]
    construction_mode: str
    target_count_range: Tuple[int, int] | None = None


@dataclass(frozen=True)
class TentsRenderParams:
    """Resolved rendering knobs for one Tents grid."""

    canvas_width: int
    canvas_height: int
    cell_size_px: int
    left_clue_width_px: int
    top_clue_height_px: int
    grid_line_width_px: int
    heavy_line_width_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    clue_font_size_px: int
    candidate_font_size_px: int
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    style_overrides: Dict[str, Tuple[int, int, int]]
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedTentsScene:
    """Rendered Tents scene plus traceable geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    cell_bbox_map: Dict[str, List[float]]
    clue_bbox_map: Dict[str, List[float]]
    option_panel_bbox_map: Dict[str, List[float]]
    item_bbox_map: Dict[str, List[float]]


__all__ = [
    "CandidateCellSpec",
    "Cell",
    "DOMAIN",
    "LabeledTentSpec",
    "RenderedTentsScene",
    "SCENE_ID",
    "SUPPORTED_PALETTE_VARIANTS",
    "SUPPORTED_SCENE_VARIANTS",
    "TentsRenderParams",
    "TentsSample",
]
