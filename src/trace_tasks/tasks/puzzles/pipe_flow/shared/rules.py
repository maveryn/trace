"""Pipe connectivity and option-rotation rules for pipe-flow puzzles."""

from __future__ import annotations

from typing import Iterable, Mapping

from .state import (
    DELTAS,
    DIRECTIONS,
    OPPOSITE,
    ROTATE_CW,
    Cell,
    Openings,
)


def parse_grid_size_variant(value: str) -> tuple[int, int]:
    """Parse a supported grid-size string such as ``8x8``."""

    raw_rows, raw_cols = str(value).split("x", maxsplit=1)
    return int(raw_rows), int(raw_cols)


def parse_gap_size_variant(value: str | int) -> int:
    """Parse the supported pipe-flow gap-size variant."""

    if isinstance(value, int):
        size = int(value)
    else:
        raw = str(value).strip().lower()
        if "x" in raw:
            raw_rows, raw_cols = raw.split("x", maxsplit=1)
            if int(raw_rows) != int(raw_cols):
                raise ValueError(f"gap size must be square: {value!r}")
            size = int(raw_rows)
        else:
            size = int(raw)
    if size != 2:
        raise ValueError(f"unsupported pipe-flow gap size: {value!r}")
    return int(size)


def normalize_openings(openings: Iterable[str]) -> Openings:
    """Return canonical N/E/S/W opening order."""

    present = {str(direction) for direction in openings if str(direction) in set(DIRECTIONS)}
    return tuple(direction for direction in DIRECTIONS if direction in present)


def rotate_openings(openings: Iterable[str], turns: int = 1) -> Openings:
    """Rotate one set of pipe openings clockwise."""

    result = [str(direction) for direction in openings]
    for _ in range(int(turns) % 4):
        result = [ROTATE_CW[direction] for direction in result]
    return normalize_openings(result)


def cell_add(cell: Cell, direction: str) -> Cell:
    """Move one grid cell in the requested cardinal direction."""

    dr, dc = DELTAS[str(direction)]
    return (int(cell[0] + dr), int(cell[1] + dc))


def direction_between(left: Cell, right: Cell) -> str:
    """Return the cardinal direction from one orthogonal neighbor to another."""

    dr = int(right[0] - left[0])
    dc = int(right[1] - left[1])
    for direction, delta in DELTAS.items():
        if tuple(delta) == (dr, dc):
            return str(direction)
    raise ValueError(f"cells are not orthogonal neighbors: {left} -> {right}")


def path_openings(path: tuple[Cell, ...]) -> dict[Cell, Openings]:
    """Convert a cell path into per-cell pipe openings."""

    openings: dict[Cell, set[str]] = {tuple(cell): set() for cell in path}
    for index, cell in enumerate(path):
        cell = tuple(cell)
        if index > 0:
            openings[cell].add(direction_between(cell, tuple(path[index - 1])))
        if index < len(path) - 1:
            openings[cell].add(direction_between(cell, tuple(path[index + 1])))
    return {cell: normalize_openings(value) for cell, value in openings.items()}


def connected_to_destination(
    tile_map: Mapping[Cell, Openings],
    *,
    rows: int,
    cols: int,
    start_cell: Cell,
    destination_cell: Cell,
) -> bool:
    """Return whether start reaches finish through matching pipe openings."""

    seen = {tuple(start_cell)}
    frontier = [tuple(start_cell)]
    while frontier:
        cell = frontier.pop()
        if cell == tuple(destination_cell):
            return True
        for direction in tile_map.get(cell, ()):
            nxt = cell_add(cell, direction)
            if not (0 <= nxt[0] < int(rows) and 0 <= nxt[1] < int(cols)):
                continue
            if OPPOSITE[direction] not in set(tile_map.get(nxt, ())):
                continue
            if nxt in seen:
                continue
            seen.add(nxt)
            frontier.append(nxt)
    return False


def block_cells(origin: Cell, *, gap_size: int = 2) -> tuple[Cell, ...]:
    """Return the global cells covered by a square missing piece."""

    row, col = int(origin[0]), int(origin[1])
    size = parse_gap_size_variant(int(gap_size))
    return tuple(
        (int(row + local_row), int(col + local_col))
        for local_row in range(size)
        for local_col in range(size)
    )


def local_cells(*, gap_size: int = 2) -> tuple[Cell, ...]:
    """Return the local cells in one square option piece."""

    size = parse_gap_size_variant(int(gap_size))
    return tuple((int(row), int(col)) for row in range(size) for col in range(size))


def localize_block(
    global_openings: Mapping[Cell, Openings],
    *,
    origin: Cell,
    gap_size: int = 2,
) -> dict[Cell, Openings]:
    """Convert global missing-region openings into local coordinates."""

    origin_row, origin_col = int(origin[0]), int(origin[1])
    localized: dict[Cell, Openings] = {}
    for row, col in local_cells(gap_size=int(gap_size)):
        global_cell = (int(origin_row + row), int(origin_col + col))
        localized[(int(row), int(col))] = normalize_openings(global_openings.get(global_cell, ()))
    return localized


def globalize_block(
    local_openings: Mapping[Cell, Openings],
    *,
    origin: Cell,
    gap_size: int = 2,
) -> dict[Cell, Openings]:
    """Convert local option openings into global grid coordinates."""

    origin_row, origin_col = int(origin[0]), int(origin[1])
    return {
        (int(origin_row + row), int(origin_col + col)): normalize_openings(
            local_openings.get((row, col), ())
        )
        for row, col in local_cells(gap_size=int(gap_size))
    }


def option_signature(
    local_openings: Mapping[Cell, Openings],
    *,
    gap_size: int = 2,
) -> tuple[tuple[int, int, Openings], ...]:
    """Return a stable signature for one option orientation."""

    return tuple(
        (int(row), int(col), normalize_openings(local_openings.get((row, col), ())))
        for row, col in local_cells(gap_size=int(gap_size))
    )


def rotate_local_option(
    local_openings: Mapping[Cell, Openings],
    *,
    turns: int,
    gap_size: int = 2,
) -> dict[Cell, Openings]:
    """Rotate one square option piece clockwise by quarter-turns."""

    size = parse_gap_size_variant(int(gap_size))
    result = {
        cell: normalize_openings(local_openings.get(cell, ()))
        for cell in local_cells(gap_size=size)
    }
    for _ in range(int(turns) % 4):
        rotated: dict[Cell, Openings] = {cell: tuple() for cell in local_cells(gap_size=size)}
        for (row, col), openings in result.items():
            next_cell = (int(col), int(size - 1 - row))
            rotated[next_cell] = rotate_openings(openings, 1)
        result = rotated
    return result


def rotation_canonical_option_signature(
    local_openings: Mapping[Cell, Openings],
    *,
    gap_size: int = 2,
) -> tuple[tuple[int, int, Openings], ...]:
    """Return a signature treating quarter-turn rotations as equivalent."""

    return min(
        option_signature(
            rotate_local_option(local_openings, turns=turns, gap_size=int(gap_size)),
            gap_size=int(gap_size),
        )
        for turns in range(4)
    )


def option_connects(
    *,
    visible_map: Mapping[Cell, Openings],
    local_openings: Mapping[Cell, Openings],
    origin: Cell,
    rows: int,
    cols: int,
    start_cell: Cell,
    destination_cell: Cell,
    gap_size: int = 2,
) -> bool:
    """Return whether an option would reconnect start to finish."""

    test_map = {
        tuple(cell): normalize_openings(openings)
        for cell, openings in visible_map.items()
    }
    test_map.update(globalize_block(local_openings, origin=origin, gap_size=int(gap_size)))
    return connected_to_destination(
        test_map,
        rows=int(rows),
        cols=int(cols),
        start_cell=tuple(start_cell),
        destination_cell=tuple(destination_cell),
    )


def option_connecting_rotation_turns(
    *,
    visible_map: Mapping[Cell, Openings],
    local_openings: Mapping[Cell, Openings],
    origin: Cell,
    rows: int,
    cols: int,
    start_cell: Cell,
    destination_cell: Cell,
    gap_size: int = 2,
) -> tuple[int, ...]:
    """Return all rotations that make a displayed option solve the path."""

    turns: list[int] = []
    for turn_count in range(4):
        rotated = rotate_local_option(
            local_openings,
            turns=int(turn_count),
            gap_size=int(gap_size),
        )
        if option_connects(
            visible_map=visible_map,
            local_openings=rotated,
            origin=origin,
            rows=int(rows),
            cols=int(cols),
            start_cell=start_cell,
            destination_cell=destination_cell,
            gap_size=int(gap_size),
        ):
            turns.append(int(turn_count))
    return tuple(turns)
