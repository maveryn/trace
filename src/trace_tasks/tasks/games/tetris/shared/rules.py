"""Pure Tetris board mechanics for scene construction and verification."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .state import Board, Coord, DropCollision, EMPTY, Outcome, Placement, RAW_TETROMINOES


def _normalize_shape(cells: Iterable[Coord]) -> Tuple[Coord, ...]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    min_row = min(rows)
    min_col = min(cols)
    return tuple(sorted((int(r) - min_row, int(c) - min_col) for r, c in cells))


TETROMINOES: Dict[str, Tuple[Tuple[Coord, ...], ...]] = {
    piece: tuple(dict.fromkeys(_normalize_shape(shape) for shape in orientations))
    for piece, orientations in RAW_TETROMINOES.items()
}
PIECE_ORDER: Tuple[str, ...] = tuple(TETROMINOES)


def shape(placement: Placement) -> Tuple[Coord, ...]:
    return TETROMINOES[str(placement.piece)][int(placement.orientation_index)]


def shape_size(shape_cells: Sequence[Coord]) -> Tuple[int, int]:
    return (max(int(r) for r, _c in shape_cells) + 1, max(int(c) for _r, c in shape_cells) + 1)


def board_size(board: Board) -> Tuple[int, int]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    return int(rows), int(cols)


def empty_board(*, rows: int, cols: int) -> Board:
    return tuple(tuple(EMPTY for _ in range(int(cols))) for _ in range(int(rows)))


def freeze(rows: Sequence[Sequence[str]]) -> Board:
    return tuple(tuple(str(cell) for cell in row) for row in rows)


def piece_cells(placement: Placement) -> Tuple[Coord, ...]:
    return tuple(
        sorted((int(placement.top) + int(r), int(placement.col) + int(c)) for r, c in shape(placement))
    )


def valid_cells(board: Board, cells: Sequence[Coord]) -> bool:
    board_rows, board_cols = board_size(board)
    return all(0 <= int(r) < board_rows and 0 <= int(c) < board_cols for r, c in cells)


def can_place(board: Board, placement: Placement) -> bool:
    cells = piece_cells(placement)
    return valid_cells(board, cells) and all(board[int(r)][int(c)] == EMPTY for r, c in cells)


def hard_drop_top(board: Board, *, piece: str, orientation_index: int, col: int) -> int | None:
    board_rows, _board_cols = board_size(board)
    piece_shape = TETROMINOES[str(piece)][int(orientation_index)]
    height, _width = shape_size(piece_shape)
    top_min = -height + 1
    last_valid: int | None = None
    for top in range(top_min, board_rows):
        placement = Placement(str(piece), int(orientation_index), int(col), int(top))
        cells = piece_cells(placement)
        if not valid_cells(board, cells):
            continue
        if all(board[int(r)][int(c)] == EMPTY for r, c in cells):
            last_valid = int(top)
            continue
        if last_valid is not None:
            break
    return last_valid


def lock_piece(board: Board, placement: Placement) -> Board:
    rows = [list(row) for row in board]
    for row, col in piece_cells(placement):
        rows[int(row)][int(col)] = str(placement.piece)
    return freeze(rows)


def clear_full_rows(board: Board) -> Tuple[Board, Tuple[int, ...]]:
    _board_rows, board_cols = board_size(board)
    full_rows = tuple(index for index, row in enumerate(board) if all(cell != EMPTY for cell in row))
    kept = [list(row) for index, row in enumerate(board) if index not in set(full_rows)]
    new_rows = [[EMPTY for _ in range(board_cols)] for _ in range(len(full_rows))] + kept
    return freeze(new_rows), tuple(int(v) for v in full_rows)


def column_heights(board: Board) -> Tuple[int, ...]:
    board_rows, board_cols = board_size(board)
    heights: List[int] = []
    for col in range(board_cols):
        height = 0
        for row in range(board_rows):
            if board[row][col] != EMPTY:
                height = board_rows - row
                break
        heights.append(int(height))
    return tuple(heights)


def hole_count(board: Board) -> int:
    board_rows, board_cols = board_size(board)
    holes = 0
    for col in range(board_cols):
        seen_block = False
        for row in range(board_rows):
            if board[row][col] != EMPTY:
                seen_block = True
            elif seen_block:
                holes += 1
    return int(holes)


def evaluate_outcome(board: Board, placement: Placement) -> Outcome:
    locked = lock_piece(board, placement)
    result, cleared_rows = clear_full_rows(locked)
    heights = column_heights(result)
    return Outcome(
        placement=placement,
        clear_count=len(cleared_rows),
        locked_board=locked,
        result_board=result,
        locked_cells=piece_cells(placement),
        cleared_rows=tuple(int(v) for v in cleared_rows),
        holes_after=hole_count(result),
        max_height_after=max(heights) if heights else 0,
        aggregate_height_after=sum(heights),
    )


def shifted_placement(placement: Placement, *, shift_delta: int) -> Placement:
    return Placement(
        str(placement.piece),
        int(placement.orientation_index),
        int(placement.col) + int(shift_delta),
        int(placement.top),
    )


def drop_collision(board: Board, placement: Placement, *, shift_delta: int) -> DropCollision | None:
    shifted = shifted_placement(placement, shift_delta=int(shift_delta))
    if not can_place(board, shifted):
        return None
    current = shifted
    drop_steps = 0
    while True:
        candidate = Placement(str(current.piece), int(current.orientation_index), int(current.col), int(current.top) + 1)
        if can_place(board, candidate):
            current = candidate
            drop_steps += 1
            continue
        board_rows, _board_cols = board_size(board)
        current_cells = set(piece_cells(current))
        blocker_cells: List[Coord] = []
        bottom_contact_cells: List[Coord] = []
        for row, col in current_cells:
            below = (int(row) + 1, int(col))
            if below in current_cells:
                continue
            if int(row) + 1 >= int(board_rows):
                bottom_contact_cells.append((int(row), int(col)))
            elif board[int(row) + 1][int(col)] != EMPTY:
                blocker_cells.append((int(row) + 1, int(col)))
        if blocker_cells:
            collision_kind = "locked_block"
        elif bottom_contact_cells:
            collision_kind = "bottom_boundary"
        else:
            return None
        return DropCollision(
            start_placement=placement,
            shifted_placement=shifted,
            final_placement=current,
            drop_steps=int(drop_steps),
            blocker_cells=tuple(sorted(set(blocker_cells))),
            bottom_contact_cells=tuple(sorted(set(bottom_contact_cells))),
            collision_kind=str(collision_kind),
        )


def all_placements(board: Board, *, piece: str) -> Tuple[Placement, ...]:
    _board_rows, board_cols = board_size(board)
    placements: List[Placement] = []
    for orientation_index, piece_shape in enumerate(TETROMINOES[str(piece)]):
        _height, width = shape_size(piece_shape)
        for col in range(0, board_cols - int(width) + 1):
            top = hard_drop_top(board, piece=str(piece), orientation_index=int(orientation_index), col=int(col))
            if top is None:
                continue
            placement = Placement(str(piece), int(orientation_index), int(col), int(top))
            if can_place(board, placement):
                placements.append(placement)
    return tuple(placements)


def board_key(board: Board) -> Tuple[Tuple[str, ...], ...]:
    return tuple(tuple(row) for row in board)


def best_clear_outcomes(board: Board, *, piece: str) -> Tuple[int, Tuple[Outcome, ...]]:
    outcomes = tuple(evaluate_outcome(board, placement) for placement in all_placements(board, piece=str(piece)))
    if not outcomes:
        return 0, ()
    best = max(int(outcome.clear_count) for outcome in outcomes)
    return int(best), tuple(outcome for outcome in outcomes if int(outcome.clear_count) == int(best))


def row_empty_count(row: Sequence[str]) -> int:
    return sum(1 for cell in row if str(cell) == EMPTY)


def cell_ids_in_row_matching_status(board: Board, *, row_index: int, status: str, entity_prefix: str) -> Tuple[str, ...]:
    ids: List[str] = []
    for col, cell in enumerate(board[int(row_index)]):
        is_filled = str(cell) != EMPTY
        if (str(status) == "filled" and is_filled) or (str(status) == "empty" and not is_filled):
            ids.append(f"{entity_prefix}_cell_{int(row_index)}_{int(col)}")
    return tuple(ids)


def shift_instruction_text(shift_delta: int) -> str:
    if int(shift_delta) == 0:
        return "do not move it sideways"
    direction = "left" if int(shift_delta) < 0 else "right"
    magnitude = abs(int(shift_delta))
    unit = "column" if int(magnitude) == 1 else "columns"
    return f"move it {int(magnitude)} {unit} {direction}"


def horizontal_sweep_cells(placement: Placement, *, shift_delta: int) -> Tuple[Coord, ...]:
    if int(shift_delta) == 0:
        return piece_cells(placement)
    step = 1 if int(shift_delta) > 0 else -1
    cells: List[Coord] = []
    for delta in range(0, int(shift_delta) + step, step):
        cells.extend(piece_cells(shifted_placement(placement, shift_delta=int(delta))))
    return tuple(sorted(set(cells)))


def bottom_edge_below_cells(placement: Placement) -> Tuple[Coord, ...]:
    cells = set(piece_cells(placement))
    below_cells: List[Coord] = []
    for row, col in cells:
        below = (int(row) + 1, int(col))
        if below not in cells:
            below_cells.append(below)
    return tuple(sorted(set(below_cells)))


def placement_trace(placement: Placement) -> Dict[str, Any]:
    return {
        "piece": str(placement.piece),
        "orientation_index": int(placement.orientation_index),
        "col": int(placement.col),
        "top": int(placement.top),
        "cells": [[int(r), int(c)] for r, c in piece_cells(placement)],
    }


def is_supported_stack_board(board: Board) -> bool:
    board_rows, board_cols = board_size(board)
    for row in range(board_rows - 1):
        for col in range(board_cols):
            if board[row][col] != EMPTY and board[row + 1][col] == EMPTY:
                return False
    return True
