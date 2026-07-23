"""Passive state records for counterfactual-board game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image

from trace_tasks.tasks.shared.bbox_projection import BBox

SCENE_ID = "counterfactual_board"

CHESS_CHECKERS_STYLE = "chess_checkers"
SUDOKU_STYLE = "sudoku"
XIANGQI_STYLE = "xiangqi"

CELL_BOARD_KIND = "cell_board"
LINE_BOARD_KIND = "line_board"

ROW_AXIS = "row"
COLUMN_AXIS = "column"
HORIZONTAL_LINE_AXIS = "horizontal_line"
VERTICAL_LINE_AXIS = "vertical_line"

CounterfactualColor = Tuple[int, int, int]


@dataclass(frozen=True)
class BoardStyleSpec:
    """Canonical and sampled dimensions for one board rendering style."""

    board_kind: str
    canonical_rows: int
    canonical_cols: int
    row_support: tuple[int, ...]
    col_support: tuple[int, ...]


STYLE_SPECS: Mapping[str, BoardStyleSpec] = {
    CHESS_CHECKERS_STYLE: BoardStyleSpec(
        board_kind=CELL_BOARD_KIND,
        canonical_rows=8,
        canonical_cols=8,
        row_support=tuple(range(6, 11)),
        col_support=tuple(range(6, 11)),
    ),
    SUDOKU_STYLE: BoardStyleSpec(
        board_kind=CELL_BOARD_KIND,
        canonical_rows=9,
        canonical_cols=9,
        row_support=tuple(range(7, 12)),
        col_support=tuple(range(7, 12)),
    ),
    XIANGQI_STYLE: BoardStyleSpec(
        board_kind=LINE_BOARD_KIND,
        canonical_rows=10,
        canonical_cols=9,
        row_support=tuple(range(8, 13)),
        col_support=tuple(range(7, 12)),
    ),
}


@dataclass(frozen=True)
class BoardLayout:
    """Pixel geometry for one board after size sampling and placement jitter."""

    rows: int
    cols: int
    board_kind: str
    canvas_width_px: int
    canvas_height_px: int
    board_bbox_px: BBox
    unit_size_px: int
    line_annotation_thickness_px: int
    placement_meta: Mapping[str, Any]


@dataclass(frozen=True)
class CountedElement:
    """One visual row, column, or grid-line witness counted by the task."""

    element_id: str
    element_kind: str
    bbox: BBox
    segment: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class CounterfactualBoardCase:
    """Task-owned symbolic case and answer before rendering/projection."""

    style: str
    rows: int
    cols: int
    board_kind: str
    counted_axis: str
    prompt_query_key: str
    answer_value: int
    canonical_bias_answer: int
    case_params: Mapping[str, Any]
    execution_trace: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedCounterfactualBoard:
    """Rendered image plus board projection metadata."""

    image: Image.Image
    layout: BoardLayout
    entities: Sequence[Mapping[str, Any]]
    render_meta: Mapping[str, Any]
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]


__all__ = [
    "BoardLayout",
    "BoardStyleSpec",
    "CELL_BOARD_KIND",
    "CHESS_CHECKERS_STYLE",
    "COLUMN_AXIS",
    "CounterfactualBoardCase",
    "CounterfactualColor",
    "CountedElement",
    "HORIZONTAL_LINE_AXIS",
    "LINE_BOARD_KIND",
    "ROW_AXIS",
    "RenderedCounterfactualBoard",
    "SCENE_ID",
    "STYLE_SPECS",
    "SUDOKU_STYLE",
    "VERTICAL_LINE_AXIS",
    "XIANGQI_STYLE",
]
