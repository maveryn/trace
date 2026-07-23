"""Board numbering and movement rules for Snakes and Ladders scenes."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from .state import BOARD_ROWS, SUPPORTED_DIE_VALUES, SnakesLaddersJump, SnakesLaddersMove, SnakesLaddersSample


def board_last_square(board_side: int = BOARD_ROWS) -> int:
    """Return the final square number for a square board side length."""

    side = int(board_side)
    if side < 2:
        raise ValueError(f"board_side must be at least 2: {board_side}")
    return int(side * side)


def square_to_coord(square: int, *, board_side: int = BOARD_ROWS) -> Tuple[int, int]:
    """Return visual ``(row, col)`` for one 1-indexed serpentine square."""

    value = int(square)
    side = int(board_side)
    last_square = board_last_square(side)
    if value < 1 or value > last_square:
        raise ValueError(f"square must be in 1..{last_square}: {square}")
    row_from_bottom = (value - 1) // side
    offset = (value - 1) % side
    row = side - 1 - row_from_bottom
    col = offset if row_from_bottom % 2 == 0 else side - 1 - offset
    return int(row), int(col)


def coord_to_square(row: int, col: int, *, board_side: int = BOARD_ROWS) -> int:
    """Return 1-indexed serpentine square for one visual board coordinate."""

    visual_row = int(row)
    visual_col = int(col)
    side = int(board_side)
    if visual_row < 0 or visual_row >= side or visual_col < 0 or visual_col >= side:
        raise ValueError(f"coord must be inside {side} x {side}: {(row, col)}")
    row_from_bottom = side - 1 - visual_row
    offset = visual_col if row_from_bottom % 2 == 0 else side - 1 - visual_col
    return int((row_from_bottom * side) + offset + 1)


def square_to_cell_id(square: int) -> str:
    """Return the render entity id for one board square."""

    return f"square_{int(square)}"


def jump_lookup(jumps: Sequence[SnakesLaddersJump]) -> Dict[int, SnakesLaddersJump]:
    """Return a mapping from jump start square to jump object."""

    return {int(jump.start_square): jump for jump in jumps}


def apply_die_roll(
    start_square: int,
    die_value: int,
    jumps: Sequence[SnakesLaddersJump],
    *,
    board_side: int = BOARD_ROWS,
) -> SnakesLaddersMove:
    """Move forward by one die value, then resolve one snake/ladder jump."""

    start = int(start_square)
    die = int(die_value)
    if die not in SUPPORTED_DIE_VALUES:
        raise ValueError(f"die_value must be in 1..6: {die_value}")
    last_square = board_last_square(int(board_side))
    landing = int(start + die)
    if landing > last_square:
        landing = int(start)
    jump = jump_lookup(jumps).get(int(landing))
    final = int(landing if jump is None else jump.end_square)
    return SnakesLaddersMove(
        start_square=int(start),
        die_value=int(die),
        landing_square=int(landing),
        final_square=int(final),
        jump_id=None if jump is None else str(jump.jump_id),
    )


def validate_snakes_ladders_sample(sample: SnakesLaddersSample) -> None:
    """Raise if a sampled scene violates the public game-state contract."""

    board_side = int(sample.board_side)
    last_square = board_last_square(board_side)
    starts = [int(jump.start_square) for jump in sample.jumps]
    if len(starts) != len(set(starts)):
        raise ValueError("snakes/ladders jump starts must be unique")
    for jump in sample.jumps:
        if str(jump.kind) == "ladder" and not int(jump.end_square) > int(jump.start_square):
            raise ValueError("ladder endpoint must be higher than start")
        if str(jump.kind) == "snake" and not int(jump.end_square) < int(jump.start_square):
            raise ValueError("snake endpoint must be lower than start")
        if int(jump.start_square) in {1, last_square}:
            raise ValueError("jumps may not start on the first or final square")
        if int(jump.end_square) < 1 or int(jump.end_square) > last_square:
            raise ValueError("jump endpoint out of bounds")
    if int(sample.start_square) < 1 or int(sample.start_square) > last_square:
        raise ValueError("sample start square out of bounds")


__all__ = [
    "apply_die_roll",
    "board_last_square",
    "coord_to_square",
    "jump_lookup",
    "square_to_cell_id",
    "square_to_coord",
    "validate_snakes_ladders_sample",
]
