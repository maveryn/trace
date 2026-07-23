"""Reversi board rules and mechanics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .state import BLACK, EMPTY, WHITE, Board, Coord


_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def player_name(player: int) -> str:
    """Return the canonical prompt-facing player name for one disc value."""

    return "Black" if int(player) == int(BLACK) else "White"


def opponent(player: int) -> int:
    """Return the opposite Reversi player value."""

    return int(WHITE if int(player) == int(BLACK) else BLACK)


def coord_to_cell_id(coord: Coord) -> str:
    """Return one stable scene-entity id for a board coordinate."""

    return f"cell_r{int(coord[0])}_c{int(coord[1])}"


def corner_coords(board_size: int) -> Tuple[Coord, ...]:
    """Return the four corner coordinates for one board size."""

    upper = int(board_size) - 1
    return (
        (0, 0),
        (0, upper),
        (upper, 0),
        (upper, upper),
    )


def _in_bounds(board: Sequence[Sequence[int]], row: int, col: int) -> bool:
    """Return whether a coordinate lies inside the board."""

    size = int(len(board))
    return 0 <= int(row) < size and 0 <= int(col) < size


def adjacent_coords(board: Sequence[Sequence[int]], coord: Coord) -> Tuple[Coord, ...]:
    """Return in-bounds adjacent coordinates around one board coordinate."""

    row, col = int(coord[0]), int(coord[1])
    return tuple(
        (int(row + d_row), int(col + d_col))
        for d_row, d_col in _DIRECTIONS
        if _in_bounds(board, int(row + d_row), int(col + d_col))
    )


def frontier_disc_coords(board: Sequence[Sequence[int]], player: int) -> Tuple[Coord, ...]:
    """Return queried-player discs adjacent to at least one empty square."""

    current = int(player)
    coords: List[Coord] = []
    for row, board_row in enumerate(board):
        for col, value in enumerate(board_row):
            if int(value) != int(current):
                continue
            if any(
                int(board[adj_row][adj_col]) == int(EMPTY)
                for adj_row, adj_col in adjacent_coords(board, (int(row), int(col)))
            ):
                coords.append((int(row), int(col)))
    return tuple(sorted(coords))


def initial_board(board_size: int) -> Board:
    """Return the canonical Reversi opening board for an even board size."""

    size = int(board_size)
    if size < 4 or size % 2 != 0:
        raise ValueError("Reversi boards must be even and at least 4x4")
    board = [[EMPTY for _ in range(size)] for _ in range(size)]
    center = (size // 2) - 1
    board[center][center] = WHITE
    board[center + 1][center + 1] = WHITE
    board[center][center + 1] = BLACK
    board[center + 1][center] = BLACK
    return tuple(tuple(int(cell) for cell in row) for row in board)


def legal_moves_with_flips(board: Sequence[Sequence[int]], player: int) -> Dict[Coord, Tuple[Coord, ...]]:
    """Return every legal destination square plus the discs it would flip."""

    size = int(len(board))
    current = int(player)
    opponent_value = opponent(int(current))
    legal: Dict[Coord, Tuple[Coord, ...]] = {}
    for row in range(size):
        for col in range(size):
            if int(board[row][col]) != int(EMPTY):
                continue
            flipped: List[Coord] = []
            for d_row, d_col in _DIRECTIONS:
                scan_row = int(row + d_row)
                scan_col = int(col + d_col)
                direction_flips: List[Coord] = []
                while _in_bounds(board, scan_row, scan_col) and int(board[scan_row][scan_col]) == int(opponent_value):
                    direction_flips.append((int(scan_row), int(scan_col)))
                    scan_row += int(d_row)
                    scan_col += int(d_col)
                if (
                    direction_flips
                    and _in_bounds(board, scan_row, scan_col)
                    and int(board[scan_row][scan_col]) == int(current)
                ):
                    flipped.extend(direction_flips)
            if flipped:
                legal[(int(row), int(col))] = tuple(flipped)
    return legal


def apply_move(board: Sequence[Sequence[int]], player: int, move: Coord, flips: Iterable[Coord]) -> Board:
    """Return the board after playing one already-validated move."""

    next_board = [list(int(cell) for cell in row) for row in board]
    row, col = int(move[0]), int(move[1])
    next_board[row][col] = int(player)
    for flip_row, flip_col in flips:
        next_board[int(flip_row)][int(flip_col)] = int(player)
    return tuple(tuple(int(cell) for cell in row) for row in next_board)


def simulate_random_state(
    *,
    rng,
    board_size: int,
    min_plies: int,
    max_plies: int,
) -> Board:
    """Simulate one reachable random Reversi position by playing legal moves."""

    board = initial_board(int(board_size))
    player = int(BLACK)
    plies_target = int(rng.randint(int(min_plies), int(max_plies)))
    consecutive_passes = 0
    for _ in range(max(0, int(plies_target))):
        legal = legal_moves_with_flips(board, player)
        if not legal:
            consecutive_passes += 1
            if int(consecutive_passes) >= 2:
                break
            player = opponent(player)
            continue
        consecutive_passes = 0
        ordered_moves = sorted(legal.keys())
        move = ordered_moves[int(rng.randrange(len(ordered_moves)))]
        board = apply_move(board, player, move, legal[move])
        player = opponent(player)
    return board


__all__ = [
    "adjacent_coords",
    "apply_move",
    "coord_to_cell_id",
    "corner_coords",
    "frontier_disc_coords",
    "initial_board",
    "legal_moves_with_flips",
    "opponent",
    "player_name",
    "simulate_random_state",
]
