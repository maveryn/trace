"""Annotation projection helpers for Bingo games tasks."""

from __future__ import annotations

from typing import Sequence, Tuple

from .state import BINGO_BOARD_SIZE, BingoCardState, cell_id
from .rendering import RenderedBingoCardScene


def completed_line_cell_ids(card_state: BingoCardState, *, line_axis: str) -> Tuple[str, ...]:
    """Return marked cell ids in completed rows or columns."""

    annotation_ids: list[str] = []
    completed_rows = set(int(value) for value in card_state.completed_row_indices)
    completed_columns = set(int(value) for value in card_state.completed_column_indices)
    axis = str(line_axis)
    for cell in card_state.cells:
        if axis == "row" and int(cell.row_index) in completed_rows and bool(cell.is_marked):
            annotation_ids.append(str(cell.cell_id))
        elif axis == "column" and int(cell.column_index) in completed_columns and bool(cell.is_marked):
            annotation_ids.append(str(cell.cell_id))
    if axis not in {"row", "column"}:
        raise ValueError(f"unsupported bingo line axis: {line_axis}")
    return tuple(str(value) for value in annotation_ids)


def line_sum_target_cell_ids(card_state: BingoCardState) -> Tuple[str, ...]:
    """Return the five cell ids in the completed line used for the sum."""

    if not card_state.line_sum_target_cell_ids:
        raise ValueError("line-sum annotation requires target line cell ids")
    return tuple(str(value) for value in card_state.line_sum_target_cell_ids)


def near_complete_gap_cell_ids(card_state: BingoCardState) -> Tuple[str, ...]:
    """Return single unmarked gap cell ids for all near-complete lines."""

    return tuple(str(value) for value in card_state.near_complete_gap_cell_ids)


def called_number_cell_ids(card_state: BingoCardState) -> Tuple[str, ...]:
    """Return card cell ids whose printed number appears in the CALLED list."""

    return tuple(str(value) for value in card_state.called_number_cell_ids)


def row_cell_ids(row_index: int) -> Tuple[str, ...]:
    """Return cell ids for one row index."""

    return tuple(cell_id(row_index=int(row_index), column_index=column_index) for column_index in range(BINGO_BOARD_SIZE))


def column_cell_ids(column_index: int) -> Tuple[str, ...]:
    """Return cell ids for one column index."""

    return tuple(cell_id(row_index=row_index, column_index=int(column_index)) for row_index in range(BINGO_BOARD_SIZE))


def cell_bboxes_for_ids(rendered_scene: RenderedBingoCardScene, cell_ids: Sequence[str]) -> list[list[float]]:
    """Project Bingo cell ids to cell bounding boxes."""

    bbox_map = rendered_scene.render_map["cell_bboxes_px"]
    return [list(bbox_map[str(cell_id)]) for cell_id in cell_ids]


def cell_points_for_ids(rendered_scene: RenderedBingoCardScene, cell_ids: Sequence[str]) -> list[list[float]]:
    """Project Bingo cell ids to cell-center points."""

    point_map = rendered_scene.render_map["cell_mark_centers_px"]
    return [list(point_map[str(cell_id)]) for cell_id in cell_ids]


def cell_point_pairs_for_id_pairs(
    rendered_scene: RenderedBingoCardScene,
    cell_id_pairs: Sequence[Sequence[str]],
) -> list[list[list[float]]]:
    """Project pairs of Bingo cell ids to pairs of cell-center points."""

    point_map = rendered_scene.render_map["cell_mark_centers_px"]
    return [
        [list(point_map[str(pair[0])]), list(point_map[str(pair[1])])]
        for pair in cell_id_pairs
    ]


__all__ = [
    "called_number_cell_ids",
    "cell_bboxes_for_ids",
    "cell_point_pairs_for_id_pairs",
    "cell_points_for_ids",
    "column_cell_ids",
    "completed_line_cell_ids",
    "line_sum_target_cell_ids",
    "near_complete_gap_cell_ids",
    "row_cell_ids",
]
