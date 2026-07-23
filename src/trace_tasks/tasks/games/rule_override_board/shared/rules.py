"""Scene-local rule mechanics for rule-override board tasks."""

from __future__ import annotations

from typing import Sequence

from .state import LOSS_RESULT, WIN_RESULT


def opponent(player: str) -> str:
    """Return the opposing symbol/color for the scene's two board families."""

    if str(player) == "X":
        return "O"
    if str(player) == "O":
        return "X"
    if str(player) == "Black":
        return "White"
    if str(player) == "White":
        return "Black"
    raise ValueError(f"unsupported player: {player}")


def has_full_line(cells: Sequence[Sequence[str]], player: str) -> bool:
    """Return whether one player occupies a complete row, column, or diagonal."""

    size = len(cells)
    target = str(player)
    for row in cells:
        if all(str(value) == target for value in row):
            return True
    for col in range(size):
        if all(str(cells[row][col]) == target for row in range(size)):
            return True
    if all(str(cells[index][index]) == target for index in range(size)):
        return True
    if all(str(cells[index][size - 1 - index]) == target for index in range(size)):
        return True
    return False


def line_result_from_target_line(has_target_line: bool) -> str:
    """Apply the anti-line rule: a full line is a loss for the target player."""

    return LOSS_RESULT if bool(has_target_line) else WIN_RESULT


def piece_result_from_counts(*, target_count: int, opponent_count: int) -> str:
    """Apply the piece-count rule: fewer target pieces means a win."""

    return WIN_RESULT if int(target_count) < int(opponent_count) else LOSS_RESULT


def target_line_needed_for_result(*, target_result: str, counted: bool) -> bool:
    """Return whether a line-board needs a target line for the requested result."""

    if str(target_result) == LOSS_RESULT:
        return bool(counted)
    if str(target_result) == WIN_RESULT:
        return not bool(counted)
    raise ValueError(f"unsupported target result: {target_result}")


def target_fewer_needed_for_result(*, target_result: str, counted: bool) -> bool:
    """Return whether a piece-board needs fewer target pieces for the result."""

    if str(target_result) == WIN_RESULT:
        return bool(counted)
    if str(target_result) == LOSS_RESULT:
        return not bool(counted)
    raise ValueError(f"unsupported target result: {target_result}")


__all__ = [
    "has_full_line",
    "line_result_from_target_line",
    "opponent",
    "piece_result_from_counts",
    "target_fewer_needed_for_result",
    "target_line_needed_for_result",
]
