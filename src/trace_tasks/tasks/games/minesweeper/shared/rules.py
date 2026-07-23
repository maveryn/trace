"""Minesweeper rule computations over board coordinates."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

from .state import Coord, all_coords, in_bounds, sorted_coords


def neighbor_coords(coord: Coord, *, size: int) -> Tuple[Coord, ...]:
    """Return all 8-neighborhood coordinates for one cell."""

    row, col = int(coord[0]), int(coord[1])
    coords: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if int(dr) == 0 and int(dc) == 0:
                continue
            candidate = (row + int(dr), col + int(dc))
            if in_bounds(candidate, size=int(size)):
                coords.append(candidate)
    return tuple(coords)


def clue_number(coord: Coord, *, mine_coords: Iterable[Coord], size: int) -> int:
    """Return the adjacent-mine clue number for one coordinate."""

    mines = {(int(row), int(col)) for row, col in mine_coords}
    return sum(1 for item in neighbor_coords(coord, size=int(size)) if item in mines)


def forced_cell_supports(
    *,
    size: int,
    mine_coords: Iterable[Coord],
    revealed_coords: Iterable[Coord],
    flagged_coords: Iterable[Coord],
    hidden_coords: Iterable[Coord],
    force_kind: str,
) -> Dict[Coord, Tuple[Coord, ...]]:
    """Return forced cell -> supporting clue coordinates for basic Minesweeper rules."""

    revealed = {(int(row), int(col)) for row, col in revealed_coords}
    flagged = {(int(row), int(col)) for row, col in flagged_coords}
    hidden = {(int(row), int(col)) for row, col in hidden_coords}
    mines = {(int(row), int(col)) for row, col in mine_coords}
    support: dict[Coord, set[Coord]] = {}
    for clue in sorted(revealed):
        number = clue_number(clue, mine_coords=mines, size=int(size))
        neighbors = set(neighbor_coords(clue, size=int(size)))
        adjacent_flags = neighbors & flagged
        adjacent_hidden = neighbors & hidden
        if not adjacent_hidden:
            continue
        remaining = int(number) - len(adjacent_flags)
        if remaining < 0:
            continue
        if str(force_kind) == "mine":
            if remaining == len(adjacent_hidden) and remaining > 0:
                for coord in adjacent_hidden:
                    support.setdefault(coord, set()).add(clue)
        elif str(force_kind) == "safe":
            if remaining == 0:
                for coord in adjacent_hidden:
                    support.setdefault(coord, set()).add(clue)
        else:
            raise ValueError(f"unsupported Minesweeper force kind: {force_kind}")
    return {coord: tuple(sorted(clues)) for coord, clues in sorted(support.items())}


def forced_mine_supports(
    *,
    size: int,
    mine_coords: Iterable[Coord],
    revealed_coords: Iterable[Coord],
    flagged_coords: Iterable[Coord],
    hidden_coords: Iterable[Coord],
) -> Dict[Coord, Tuple[Coord, ...]]:
    """Return cells forced to be mines by revealed-number constraints."""

    return forced_cell_supports(
        size=int(size),
        mine_coords=mine_coords,
        revealed_coords=revealed_coords,
        flagged_coords=flagged_coords,
        hidden_coords=hidden_coords,
        force_kind="mine",
    )


def forced_safe_supports(
    *,
    size: int,
    mine_coords: Iterable[Coord],
    revealed_coords: Iterable[Coord],
    flagged_coords: Iterable[Coord],
    hidden_coords: Iterable[Coord],
) -> Dict[Coord, Tuple[Coord, ...]]:
    """Return cells forced to be safe by revealed-number constraints."""

    return forced_cell_supports(
        size=int(size),
        mine_coords=mine_coords,
        revealed_coords=revealed_coords,
        flagged_coords=flagged_coords,
        hidden_coords=hidden_coords,
        force_kind="safe",
    )


def adjacent_flag_count(coord: Coord, *, flagged_coords: Iterable[Coord], size: int) -> int:
    """Return the number of flagged neighbors around one cell."""

    flags = {(int(row), int(col)) for row, col in flagged_coords}
    return sum(1 for item in neighbor_coords(coord, size=int(size)) if item in flags)


def validate_board_contract(
    *,
    size: int,
    mine_coords: Sequence[Coord],
    revealed_coords: Sequence[Coord],
    flagged_coords: Sequence[Coord],
    hidden_coords: Sequence[Coord],
) -> None:
    """Validate that one Minesweeper state is internally consistent."""

    all_set = set(all_coords(size=int(size)))
    mines = set(sorted_coords(mine_coords))
    revealed = set(sorted_coords(revealed_coords))
    flagged = set(sorted_coords(flagged_coords))
    hidden = set(sorted_coords(hidden_coords))
    if revealed & flagged or revealed & hidden or flagged & hidden:
        raise ValueError("Minesweeper state partitions overlap")
    if revealed | flagged | hidden != all_set:
        raise ValueError("Minesweeper state partitions do not cover the board")
    if not flagged <= mines:
        raise ValueError("Minesweeper flags must mark true mines")
    if mines & revealed:
        raise ValueError("Revealed Minesweeper cells cannot contain mines")
    if not (mines - flagged) <= hidden:
        raise ValueError("Unflagged Minesweeper mines must remain hidden")


__all__ = [
    "adjacent_flag_count",
    "clue_number",
    "forced_cell_supports",
    "forced_mine_supports",
    "forced_safe_supports",
    "neighbor_coords",
    "validate_board_contract",
]
