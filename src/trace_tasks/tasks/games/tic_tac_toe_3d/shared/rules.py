"""Pure 3D Tic-Tac-Toe board rules and coordinate helpers."""

from __future__ import annotations

from itertools import product
from typing import Sequence

from .state import BOARD_SIZE, Board, Coord, LAYERS, Line


def layer_index(layer: str) -> int:
    """Return the numeric z-index for a visible layer name."""

    mapping = {"top": 0, "middle": 1, "bottom": 2}
    if str(layer) not in mapping:
        raise ValueError(f"unsupported 3D Tic-Tac-Toe layer: {layer}")
    return int(mapping[str(layer)])


def coord_id(coord: Coord) -> str:
    """Return the stable entity id for one board coordinate."""

    z, row, col = coord
    return f"cell_z{int(z)}_r{int(row) + 1}_c{int(col) + 1}"


def layer_id(layer_number: int) -> str:
    """Return the stable entity id for one visible layer."""

    return f"layer_{LAYERS[int(layer_number)][0]}"


def all_coords() -> tuple[Coord, ...]:
    """Return every coordinate in z, row, column order."""

    return tuple((z, row, col) for z in range(BOARD_SIZE) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE))


def build_winning_lines() -> tuple[Line, ...]:
    """Enumerate all length-three lines in the 3D board lattice."""

    directions: list[Coord] = []
    for dz, dr, dc in product((-1, 0, 1), repeat=3):
        if (dz, dr, dc) == (0, 0, 0):
            continue
        first_nonzero = next(value for value in (dz, dr, dc) if value != 0)
        if first_nonzero > 0:
            directions.append((int(dz), int(dr), int(dc)))
    lines: list[Line] = []
    for start in all_coords():
        z, row, col = start
        for dz, dr, dc in directions:
            prev = (z - dz, row - dr, col - dc)
            if all(0 <= int(value) < BOARD_SIZE for value in prev):
                continue
            coords: list[Coord] = []
            for step in range(BOARD_SIZE):
                coord = (z + step * dz, row + step * dr, col + step * dc)
                if not all(0 <= int(value) < BOARD_SIZE for value in coord):
                    break
                coords.append(coord)
            if len(coords) == BOARD_SIZE:
                lines.append(tuple(coords))  # type: ignore[arg-type]
    return tuple(lines)


WINNING_LINES: tuple[Line, ...] = build_winning_lines()


def empty_board() -> list[list[list[str]]]:
    """Return a mutable empty 3x3x3 board."""

    return [[["" for _col in range(BOARD_SIZE)] for _row in range(BOARD_SIZE)] for _z in range(BOARD_SIZE)]


def freeze_board(board: Sequence[Sequence[Sequence[str]]]) -> Board:
    """Return an immutable JSON-friendly board value."""

    return tuple(tuple(tuple(str(value) for value in row) for row in layer) for layer in board)


def board_get(board: Board | Sequence[Sequence[Sequence[str]]], coord: Coord) -> str:
    """Read one coordinate from a board-like object."""

    z, row, col = coord
    return str(board[int(z)][int(row)][int(col)])


def board_set(board: list[list[list[str]]], coord: Coord, value: str) -> None:
    """Set one coordinate on a mutable board."""

    z, row, col = coord
    board[int(z)][int(row)][int(col)] = str(value)


def completed_lines(board: Board | Sequence[Sequence[Sequence[str]]], player: str) -> tuple[Line, ...]:
    """Return all currently completed lines for the requested player."""

    completed: list[Line] = []
    for line in WINNING_LINES:
        if all(board_get(board, coord) == str(player) for coord in line):
            completed.append(line)
    return tuple(completed)


def immediate_winning_cells(board: Board | Sequence[Sequence[Sequence[str]]], player: str) -> tuple[Coord, ...]:
    """Return empty cells that immediately complete a winning line."""

    cells: set[Coord] = set()
    for line in WINNING_LINES:
        values = [board_get(board, coord) for coord in line]
        if values.count(str(player)) == BOARD_SIZE - 1 and values.count("") == 1:
            cells.add(line[values.index("")])
    return tuple(sorted(cells))


def opponent(player: str) -> str:
    """Return the opposite mark for the two-player board."""

    return "O" if str(player) == "X" else "X"


def board_trace(board: Board) -> list[list[list[str]]]:
    """Return nested board rows for trace export."""

    return [
        [[str(board[z][row][col]) for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
        for z in range(BOARD_SIZE)
    ]


__all__ = [
    "WINNING_LINES",
    "all_coords",
    "board_get",
    "board_set",
    "board_trace",
    "build_winning_lines",
    "completed_lines",
    "coord_id",
    "empty_board",
    "freeze_board",
    "immediate_winning_cells",
    "layer_id",
    "layer_index",
    "opponent",
]
