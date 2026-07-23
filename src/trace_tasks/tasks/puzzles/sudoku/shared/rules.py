"""Sudoku rule helpers for units, peers, candidates, and digit summaries."""

from __future__ import annotations

from .state import BOX_SIZE, DIGITS, SIZE, Board, Coord


def coord_to_cell_id(coord: Coord) -> str:
    """Return the canonical visible-cell id for one Sudoku coordinate."""

    row, col = int(coord[0]), int(coord[1])
    return f"cell_r{row}_c{col}"


def unit_coords(unit_type: str, unit_index: int) -> tuple[Coord, ...]:
    """Return coordinates for one row, column, or 3 by 3 box."""

    kind = str(unit_type)
    index = int(unit_index)
    if kind == "row":
        return tuple((index, col) for col in range(SIZE))
    if kind == "column":
        return tuple((row, index) for row in range(SIZE))
    if kind == "box":
        start_row = int(index // BOX_SIZE) * BOX_SIZE
        start_col = int(index % BOX_SIZE) * BOX_SIZE
        return tuple(
            (row, col)
            for row in range(start_row, start_row + BOX_SIZE)
            for col in range(start_col, start_col + BOX_SIZE)
        )
    raise ValueError(f"unsupported Sudoku unit_type: {unit_type}")


def box_index_for_coord(coord: Coord) -> int:
    """Return the 0-based 3 by 3 box index for one coordinate."""

    row, col = int(coord[0]), int(coord[1])
    return int(row // BOX_SIZE) * BOX_SIZE + int(col // BOX_SIZE)


def peer_coords(coord: Coord) -> tuple[Coord, ...]:
    """Return every row, column, or box peer for one coordinate."""

    row, col = int(coord[0]), int(coord[1])
    peers = set(unit_coords("row", row))
    peers.update(unit_coords("column", col))
    peers.update(unit_coords("box", box_index_for_coord((row, col))))
    peers.discard((row, col))
    return tuple(sorted(peers))


def candidate_digits(board: Board, coord: Coord) -> tuple[int, ...]:
    """Return valid candidate digits for one empty cell under Sudoku rules."""

    row, col = int(coord[0]), int(coord[1])
    if int(board[row][col]) != 0:
        return ()
    used = {
        int(board[r][c]) for r, c in peer_coords((row, col)) if int(board[r][c]) != 0
    }
    return tuple(digit for digit in DIGITS if int(digit) not in used)


def repeated_digits_in_unit(
    board: Board,
    *,
    unit_type: str,
    unit_index: int,
) -> tuple[int, ...]:
    """Return sorted digit values that appear more than once in one unit."""

    counts: dict[int, int] = {}
    for row, col in unit_coords(str(unit_type), int(unit_index)):
        value = int(board[row][col])
        if value == 0:
            continue
        counts[value] = int(counts.get(value, 0)) + 1
    return tuple(sorted(digit for digit, count in counts.items() if int(count) > 1))


def missing_digits_in_unit(
    board: Board,
    *,
    unit_type: str,
    unit_index: int,
) -> tuple[int, ...]:
    """Return sorted digit values that are absent from one visible unit."""

    visible = {
        int(board[row][col])
        for row, col in unit_coords(str(unit_type), int(unit_index))
        if int(board[row][col]) != 0
    }
    return tuple(digit for digit in DIGITS if int(digit) not in visible)


__all__ = [
    "box_index_for_coord",
    "candidate_digits",
    "coord_to_cell_id",
    "missing_digits_in_unit",
    "peer_coords",
    "repeated_digits_in_unit",
    "unit_coords",
]
