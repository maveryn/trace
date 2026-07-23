"""Tower draughts board coordinates, ownership, and movement rules."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from .state import BLACK, PLAYER_NAMES, RED, Coord, StackSpec


def player_from_name(name: str | int) -> int:
    """Convert a prompt/config player label into the internal player value."""

    if isinstance(name, int):
        return RED if int(name) == RED else BLACK
    normalized = str(name).strip().lower()
    if normalized == "red":
        return RED
    if normalized == "black":
        return BLACK
    raise ValueError(f"unknown tower draughts player {name!r}")


def player_name(player: int) -> str:
    """Return the public player name for an internal player value."""

    return PLAYER_NAMES[RED if int(player) == RED else BLACK]


def opponent(player: int) -> int:
    """Return the opposing player value."""

    return BLACK if int(player) == RED else RED


def playable_coords(board_size: int) -> Tuple[Coord, ...]:
    """Return alternating playable squares for one square board."""

    return tuple(
        (row, col)
        for row in range(int(board_size))
        for col in range(int(board_size))
        if (int(row) + int(col)) % 2 == 1
    )


def cell_id(coord: Coord) -> str:
    """Return the render-map id for one board cell."""

    return f"cell_r{int(coord[0])}_c{int(coord[1])}"


def stack_id(coord: Coord) -> str:
    """Return the render-map id for one unmarked stack."""

    return f"stack_r{int(coord[0])}_c{int(coord[1])}"


def in_bounds(coord: Coord, board_size: int) -> bool:
    """Return whether a row/column coordinate is inside the board."""

    return 0 <= int(coord[0]) < int(board_size) and 0 <= int(coord[1]) < int(board_size)


def movement_directions(player: int, *, crowned: bool) -> Tuple[Coord, ...]:
    """Return legal one-step diagonal directions for a top disk."""

    if bool(crowned):
        return ((-1, -1), (-1, 1), (1, -1), (1, 1))
    row_delta = -1 if int(player) == RED else 1
    return ((row_delta, -1), (row_delta, 1))


def destination_candidates(*, coord: Coord, owner: int, crowned: bool, board_size: int) -> Tuple[Coord, ...]:
    """Return in-bounds playable one-step destinations, ignoring occupancy."""

    candidates: list[Coord] = []
    playable = set(playable_coords(int(board_size)))
    for dr, dc in movement_directions(int(owner), crowned=bool(crowned)):
        candidate = (int(coord[0]) + int(dr), int(coord[1]) + int(dc))
        if in_bounds(candidate, int(board_size)) and candidate in playable:
            candidates.append(candidate)
    return tuple(sorted(candidates))


def capture_paths(*, coord: Coord, owner: int, crowned: bool, board_size: int) -> Tuple[Tuple[Coord, Coord], ...]:
    """Return possible captured/landing coordinate pairs, ignoring occupancy."""

    paths: list[Tuple[Coord, Coord]] = []
    playable = set(playable_coords(int(board_size)))
    for dr, dc in movement_directions(int(owner), crowned=bool(crowned)):
        captured = (int(coord[0]) + int(dr), int(coord[1]) + int(dc))
        landing = (int(coord[0]) + (2 * int(dr)), int(coord[1]) + (2 * int(dc)))
        if (
            in_bounds(captured, int(board_size))
            and in_bounds(landing, int(board_size))
            and captured in playable
            and landing in playable
        ):
            paths.append((captured, landing))
    return tuple(sorted(paths))


def stack_owner_map(stacks: Sequence[StackSpec]) -> Dict[Coord, int]:
    """Return the visible controller for each occupied coordinate."""

    return {tuple(stack.coord): int(stack.owner) for stack in stacks}


def legal_destinations(*, stacks: Sequence[StackSpec], marked_coord: Coord, board_size: int) -> Tuple[Coord, ...]:
    """Return empty one-step destinations for the marked stack."""

    stack_by_coord = {tuple(stack.coord): stack for stack in stacks}
    marked = stack_by_coord[tuple(marked_coord)]
    occupied = set(stack_by_coord)
    return tuple(
        coord
        for coord in destination_candidates(
            coord=tuple(marked_coord),
            owner=int(marked.owner),
            crowned=bool(marked.top_crowned),
            board_size=int(board_size),
        )
        if tuple(coord) not in occupied
    )


def capture_targets(*, stacks: Sequence[StackSpec], marked_coord: Coord, board_size: int) -> Tuple[Coord, ...]:
    """Return adjacent opponent-controlled stacks that can be captured now."""

    stack_by_coord = {tuple(stack.coord): stack for stack in stacks}
    marked = stack_by_coord[tuple(marked_coord)]
    occupied = set(stack_by_coord)
    out: list[Coord] = []
    for captured, landing in capture_paths(
        coord=tuple(marked_coord),
        owner=int(marked.owner),
        crowned=bool(marked.top_crowned),
        board_size=int(board_size),
    ):
        captured_stack = stack_by_coord.get(tuple(captured))
        if captured_stack is None:
            continue
        if int(captured_stack.owner) != opponent(int(marked.owner)):
            continue
        if tuple(landing) in occupied:
            continue
        out.append(tuple(captured))
    return tuple(sorted(out))


def max_controlled_count_for_board(board_size: int) -> int:
    """Return the maximum sampled controlled-stack count for a board size."""

    return min(10, len(playable_coords(int(board_size))))


def max_destination_count_for_board(board_size: int) -> int:
    """Return the maximum one-step destination count for any crowned stack."""

    max_seen = 0
    for player in (RED, BLACK):
        for coord in playable_coords(int(board_size)):
            max_seen = max(
                max_seen,
                len(destination_candidates(coord=coord, owner=int(player), crowned=True, board_size=int(board_size))),
            )
    return min(4, int(max_seen))


def max_capture_count_for_board(board_size: int) -> int:
    """Return the maximum immediate capture count for any crowned stack."""

    max_seen = 0
    for player in (RED, BLACK):
        for coord in playable_coords(int(board_size)):
            max_seen = max(
                max_seen,
                len(capture_paths(coord=coord, owner=int(player), crowned=True, board_size=int(board_size))),
            )
    return min(4, int(max_seen))


__all__ = [
    "capture_paths",
    "capture_targets",
    "cell_id",
    "destination_candidates",
    "in_bounds",
    "legal_destinations",
    "max_capture_count_for_board",
    "max_controlled_count_for_board",
    "max_destination_count_for_board",
    "movement_directions",
    "opponent",
    "playable_coords",
    "player_from_name",
    "player_name",
    "stack_id",
    "stack_owner_map",
]
