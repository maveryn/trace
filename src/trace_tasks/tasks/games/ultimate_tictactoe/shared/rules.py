"""Ultimate Tic-Tac-Toe local-board rules and entity ids."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

from .state import (
    LOCAL_LINES,
    MACRO_LABELS,
    PLAYER_O,
    PLAYER_X,
    STATUS_DRAWN,
    STATUS_O_WON,
    STATUS_OPEN,
    STATUS_X_WON,
    UltimateLocalBoard,
)


def board_entity_id(index: int) -> str:
    """Return the stable entity id for one small board."""

    return f"small_board_{MACRO_LABELS[int(index)].lower()}"


def cell_entity_id(board_index: int, cell_index: int) -> str:
    """Return the stable entity id for one cell inside one small board."""

    return f"{board_entity_id(int(board_index))}_cell_{int(cell_index) + 1}"


def opponent_of(player: str) -> str:
    """Return the other standard Tic-Tac-Toe mark."""

    return PLAYER_O if str(player) == PLAYER_X else PLAYER_X


def status_of(cells: Sequence[str]) -> Tuple[str, Tuple[int, int, int] | None]:
    """Evaluate a local 3x3 board under standard Tic-Tac-Toe win rules."""

    for line in LOCAL_LINES:
        values = [str(cells[index]) for index in line]
        if values[0] and values[0] == values[1] == values[2]:
            return f"{values[0]}_won", tuple(int(item) for item in line)
    if all(str(value) for value in cells):
        return STATUS_DRAWN, None
    return STATUS_OPEN, None


def immediate_winning_cells(cells: Sequence[str], player: str) -> Tuple[int, ...]:
    """Return empty local-cell indices where a player immediately completes a line."""

    wins: list[int] = []
    for line in LOCAL_LINES:
        values = [str(cells[index]) for index in line]
        if values.count(str(player)) == 2 and values.count("") == 1:
            wins.append(int(line[values.index("")]))
    return tuple(sorted(set(wins)))


def drawn_cells() -> Tuple[str, ...]:
    """Return a deterministic full board with no local winner."""

    return (PLAYER_X, PLAYER_O, PLAYER_X, PLAYER_X, PLAYER_O, PLAYER_O, PLAYER_O, PLAYER_X, PLAYER_X)


def won_cells(rng, player: str) -> Tuple[Tuple[str, ...], Tuple[int, int, int]]:
    """Construct a local board already won by one player on exactly one line."""

    line = tuple(rng.choice(LOCAL_LINES))
    cells = [""] * 9
    for index in line:
        cells[int(index)] = str(player)
    available = [index for index in range(9) if index not in set(line)]
    rng.shuffle(available)
    fill_count = int(rng.randrange(1, 4))
    opponent = opponent_of(str(player))
    for index in available[:fill_count]:
        cells[int(index)] = str(opponent if rng.random() < 0.7 else player)
        status, winner_line = status_of(cells)
        if str(status) != f"{str(player)}_won":
            cells[int(index)] = opponent
        elif winner_line is not None and set(winner_line) != set(line):
            cells[int(index)] = opponent
    return tuple(cells), tuple(int(item) for item in line)


def open_cells(rng) -> Tuple[str, ...]:
    """Construct an unfinished local board without a current winner."""

    for _attempt in range(200):
        cells = [""] * 9
        indices = list(range(9))
        rng.shuffle(indices)
        fill_count = int(rng.randrange(3, 6))
        for idx, cell_index in enumerate(indices[:fill_count]):
            cells[int(cell_index)] = PLAYER_X if idx % 2 == 0 else PLAYER_O
        status, _line = status_of(cells)
        if status == STATUS_OPEN:
            return tuple(cells)
    return (PLAYER_X, "", PLAYER_O, "", PLAYER_X, "", PLAYER_O, "", "")


def open_cells_without_immediate_win(rng, *, player: str) -> Tuple[str, ...]:
    """Sample an open local board where the requested player has no one-move win."""

    target_player = str(player)
    for _attempt in range(400):
        cells = open_cells(rng)
        if not immediate_winning_cells(cells, target_player):
            return tuple(cells)
    fallback = (PLAYER_X, PLAYER_O, "", "", PLAYER_X, PLAYER_O, PLAYER_O, "", "")
    if target_player == PLAYER_X:
        fallback = (PLAYER_O, PLAYER_X, "", "", PLAYER_O, PLAYER_X, PLAYER_X, "", "")
    if immediate_winning_cells(fallback, target_player):
        raise ValueError("failed to construct open board without immediate win")
    return tuple(fallback)


def local_board_for_status(rng, status: str) -> UltimateLocalBoard:
    """Construct a local board for a requested status label."""

    if str(status) == STATUS_X_WON:
        cells, line = won_cells(rng, PLAYER_X)
        return UltimateLocalBoard(cells=tuple(cells), status=STATUS_X_WON, winning_line=tuple(line))
    if str(status) == STATUS_O_WON:
        cells, line = won_cells(rng, PLAYER_O)
        return UltimateLocalBoard(cells=tuple(cells), status=STATUS_O_WON, winning_line=tuple(line))
    if str(status) == STATUS_DRAWN:
        return UltimateLocalBoard(cells=drawn_cells(), status=STATUS_DRAWN, winning_line=None)
    return UltimateLocalBoard(cells=open_cells(rng), status=STATUS_OPEN, winning_line=None)

