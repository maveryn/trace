"""Passive state objects for Star Battle puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


DOMAIN = "puzzles"
SCENE_ID = "star_battle"
SCENE_VARIANTS: Tuple[str, ...] = (
    "star_battle_classic",
    "star_battle_pastel",
    "star_battle_blueprint",
)

Cell = Tuple[int, int]
BBox = List[float]


@dataclass(frozen=True)
class StarBattleRenderParams:
    """Resolved rendering knobs for one Star Battle grid."""

    canvas_width: int
    canvas_height: int
    cell_size_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    grid_line_width_px: int
    heavy_line_width_px: int
    clue_size_px: int
    candidate_font_size_px: int
    clue_font_size_px: int
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    style_overrides: Dict[str, Any]
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class CandidateCellSpec:
    """One labeled Star Battle candidate cell."""

    label: str
    row: int
    col: int
    is_correct: bool
    is_legal: bool

    @property
    def cell(self) -> Cell:
        return (int(self.row), int(self.col))


@dataclass(frozen=True)
class StarBattleDataset:
    """Symbolic Star Battle sample before rendering."""

    size: int
    grid_size_range: Tuple[int, int]
    solution_stars: Tuple[Cell, ...]
    visible_stars: Tuple[Cell, ...]
    region_grid: Tuple[Tuple[int, ...], ...]
    regions: Dict[str, Tuple[Cell, ...]]
    legal_cells: Tuple[Cell, ...]
    scope_cells: Tuple[Cell, ...]
    scoped_legal_cells: Tuple[Cell, ...]
    candidate_specs: Tuple[CandidateCellSpec, ...]
    answer_value: str | int
    answer_type: str
    option_count: int
    target_answer_support: Tuple[str | int, ...]
    marked_region_index: int | None = None
    marked_row_index: int | None = None
    marked_col_index: int | None = None
    correct_option_index: int | None = None
    correct_cell: Cell | None = None
    target_count_range: Tuple[int, int] | None = None


@dataclass(frozen=True)
class RenderedStarBattleScene:
    """Rendered Star Battle scene plus traceable geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: BBox
    cell_bbox_map: Dict[str, BBox]
    row_bbox_map: Dict[str, BBox]
    col_bbox_map: Dict[str, BBox]
    region_bbox_map: Dict[str, BBox]
    item_bbox_map: Dict[str, BBox]


__all__ = [
    "BBox",
    "CandidateCellSpec",
    "Cell",
    "DOMAIN",
    "RenderedStarBattleScene",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "StarBattleDataset",
    "StarBattleRenderParams",
]
