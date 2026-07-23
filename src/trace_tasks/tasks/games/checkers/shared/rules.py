"""Identity-free Checkers board rules and entity-id helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


EMPTY = 0
RED = 1
BLACK = -1
BOARD_SIZE = 8

Coord = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class CheckersMove:
    """One single-step legal move for an ordinary non-king checker."""

    origin: Coord
    landing: Coord
    captured: Coord | None


@dataclass(frozen=True)
class CheckersCaptureChain:
    """One multi-jump capture chain for a king checker."""

    origin: Coord
    landings: Tuple[Coord, ...]
    captured: Tuple[Coord, ...]


def player_name(player: int) -> str:
    """Return the prompt-facing player name for one checker color."""

    return "Red" if int(player) == int(RED) else "Black"


def opponent(player: int) -> int:
    """Return the opposite checker color."""

    return int(BLACK if int(player) == int(RED) else RED)


def coord_to_cell_id(coord: Coord) -> str:
    """Return one stable board-cell id for a coordinate."""

    return f"cell_r{int(coord[0])}_c{int(coord[1])}"


def piece_to_entity_id(coord: Coord, *, player: int) -> str:
    """Return one stable piece entity id for a coordinate."""

    color = "red" if int(player) == int(RED) else "black"
    return f"piece_{color}_r{int(coord[0])}_c{int(coord[1])}"


def is_dark_square(row: int, col: int) -> bool:
    """Return whether one board coordinate is playable."""

    return (int(row) + int(col)) % 2 == 1


def playable_coords() -> Tuple[Coord, ...]:
    """Return all playable board coordinates in row-major order."""

    coords: List[Coord] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if is_dark_square(row, col):
                coords.append((int(row), int(col)))
    return tuple(coords)


def empty_board() -> Board:
    """Return one empty 8x8 checkers board."""

    return tuple(tuple(int(EMPTY) for _ in range(BOARD_SIZE)) for _ in range(BOARD_SIZE))


def freeze_board(board: Sequence[Sequence[int]]) -> Board:
    """Freeze one mutable board into the canonical tuple representation."""

    return tuple(tuple(int(cell) for cell in row) for row in board)


def occupied_piece_count(board: Sequence[Sequence[int]]) -> int:
    """Return the total number of visible pieces on one board."""

    return sum(1 for row in board for cell in row if int(cell) != int(EMPTY))


def _in_bounds(row: int, col: int) -> bool:
    """Return whether one coordinate lies on the board."""

    return 0 <= int(row) < BOARD_SIZE and 0 <= int(col) < BOARD_SIZE


def _move_row_delta(player: int) -> int:
    """Return the forward row direction for one ordinary piece."""

    return -1 if int(player) == int(RED) else 1


def allowed_non_king_row(player: int, row: int) -> bool:
    """Return whether an ordinary piece of one color may visibly occupy a row."""

    if int(player) == int(RED):
        return int(row) != 0
    return int(row) != BOARD_SIZE - 1


def enumerate_legal_moves(board: Sequence[Sequence[int]], player: int) -> Tuple[CheckersMove, ...]:
    """Return every single-step move or single jump for one ordinary-piece color."""

    current = int(player)
    other = opponent(int(current))
    row_delta = _move_row_delta(int(current))
    moves: List[CheckersMove] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if int(board[row][col]) != int(current):
                continue
            for col_delta in (-1, 1):
                quiet_row = int(row + row_delta)
                quiet_col = int(col + col_delta)
                if _in_bounds(int(quiet_row), int(quiet_col)) and int(board[quiet_row][quiet_col]) == int(EMPTY):
                    moves.append(
                        CheckersMove(
                            origin=(int(row), int(col)),
                            landing=(int(quiet_row), int(quiet_col)),
                            captured=None,
                        )
                    )
                    continue
                capture_row = int(row + row_delta)
                capture_col = int(col + col_delta)
                landing_row = int(row + (2 * row_delta))
                landing_col = int(col + (2 * col_delta))
                if not _in_bounds(int(capture_row), int(capture_col)) or not _in_bounds(int(landing_row), int(landing_col)):
                    continue
                if int(board[capture_row][capture_col]) != int(other):
                    continue
                if int(board[landing_row][landing_col]) != int(EMPTY):
                    continue
                moves.append(
                    CheckersMove(
                        origin=(int(row), int(col)),
                        landing=(int(landing_row), int(landing_col)),
                        captured=(int(capture_row), int(capture_col)),
                    )
                )
    return tuple(moves)


def enumerate_king_capture_chains(
    board: Sequence[Sequence[int]],
    *,
    player: int,
    origin: Coord,
) -> Tuple[CheckersCaptureChain, ...]:
    """Return all terminal capture chains for one king checker.

    Kings may jump diagonally in any direction. Captured pieces are removed
    before later jumps in the same chain.
    """

    current = int(player)
    other = opponent(int(current))
    start = (int(origin[0]), int(origin[1]))
    if not _in_bounds(*start) or int(board[start[0]][start[1]]) != int(current):
        return ()

    paths: List[CheckersCaptureChain] = []

    def _recurse(
        mutable: list[list[int]],
        position: Coord,
        landings: Tuple[Coord, ...],
        captured: Tuple[Coord, ...],
    ) -> None:
        found = False
        row, col = int(position[0]), int(position[1])
        for row_delta, col_delta in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            capture_row = int(row + row_delta)
            capture_col = int(col + col_delta)
            landing_row = int(row + (2 * row_delta))
            landing_col = int(col + (2 * col_delta))
            if not _in_bounds(capture_row, capture_col) or not _in_bounds(landing_row, landing_col):
                continue
            if int(mutable[capture_row][capture_col]) != int(other):
                continue
            if int(mutable[landing_row][landing_col]) != int(EMPTY):
                continue
            found = True
            next_board = [list(int(cell) for cell in row_values) for row_values in mutable]
            next_board[row][col] = int(EMPTY)
            next_board[capture_row][capture_col] = int(EMPTY)
            next_board[landing_row][landing_col] = int(current)
            _recurse(
                next_board,
                (int(landing_row), int(landing_col)),
                tuple(landings) + ((int(landing_row), int(landing_col)),),
                tuple(captured) + ((int(capture_row), int(capture_col)),),
            )
        if not found:
            paths.append(
                CheckersCaptureChain(
                    origin=start,
                    landings=tuple(landings),
                    captured=tuple(captured),
                )
            )

    _recurse([list(int(cell) for cell in row) for row in board], start, (), ())
    return tuple(paths)


__all__ = [
    "BLACK",
    "BOARD_SIZE",
    "Board",
    "CheckersCaptureChain",
    "CheckersMove",
    "Coord",
    "EMPTY",
    "RED",
    "allowed_non_king_row",
    "coord_to_cell_id",
    "empty_board",
    "enumerate_king_capture_chains",
    "enumerate_legal_moves",
    "freeze_board",
    "is_dark_square",
    "occupied_piece_count",
    "opponent",
    "piece_to_entity_id",
    "playable_coords",
    "player_name",
]
