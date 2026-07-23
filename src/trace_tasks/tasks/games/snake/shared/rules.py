"""Snake movement and validation rules."""

from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence, Tuple

from .state import Coord, DIRECTION_DELTAS, DIRECTION_NAMES, SnakeSimulation, SnakeState


def coord_to_cell_id(coord: Coord) -> str:
    """Return a stable visible-cell id."""

    return f"cell_r{int(coord[0])}_c{int(coord[1])}"


def all_coords(size: int) -> Tuple[Coord, ...]:
    """Return all grid coordinates in reading order."""

    return tuple((row, col) for row in range(int(size)) for col in range(int(size)))


def in_bounds(coord: Coord, *, size: int) -> bool:
    """Return true when `coord` lies inside the square board."""

    return 0 <= int(coord[0]) < int(size) and 0 <= int(coord[1]) < int(size)


def step_coord(coord: Coord, direction: str) -> Coord:
    """Move one coordinate by one named cardinal direction."""

    delta = DIRECTION_DELTAS[str(direction)]
    return (int(coord[0]) + int(delta[0]), int(coord[1]) + int(delta[1]))


def neighbor_coords(coord: Coord, *, size: int) -> Tuple[Coord, ...]:
    """Return in-board cardinal neighbors."""

    return tuple(
        candidate
        for direction in DIRECTION_NAMES
        for candidate in (step_coord(coord, direction),)
        if in_bounds(candidate, size=int(size))
    )


def direction_text(direction: str) -> str:
    """Return display text for one move direction."""

    return str(direction).upper()


def move_sequence_text(moves: Sequence[str]) -> str:
    """Return compact prompt text for a planned move sequence."""

    return ", ".join(f"{index + 1}. {direction_text(move)}" for index, move in enumerate(moves))


def simulate_snake_moves(state: SnakeState, moves: Sequence[str]) -> SnakeSimulation:
    """Simulate normal Snake movement until the first wall/body/food event."""

    size = int(state.board_size)
    head = (int(state.head[0]), int(state.head[1]))
    body = tuple((int(row), int(col)) for row, col in state.body)
    food = (int(state.food[0]), int(state.food[1]))
    obstacles = set((int(row), int(col)) for row, col in state.obstacles)
    traversed: list[Coord] = []

    for step_index, direction in enumerate(moves, start=1):
        new_head = step_coord(head, str(direction))
        if not in_bounds(new_head, size=size):
            return SnakeSimulation("wall", int(step_index), tuple(traversed), None, head)
        if new_head in obstacles:
            traversed.append(new_head)
            return SnakeSimulation("wall", int(step_index), tuple(traversed), new_head, head)

        traversed.append(new_head)
        tail_to_vacate = body[-1] if body else None
        if new_head in set(body) and new_head != tail_to_vacate:
            return SnakeSimulation("body", int(step_index), tuple(traversed), new_head, new_head)
        if new_head == food:
            return SnakeSimulation("food", int(step_index), tuple(traversed), new_head, new_head)

        body = (head,) + body[:-1]
        head = new_head

    return SnakeSimulation(
        "safe" if len(tuple(moves)) == 1 else "survives",
        0,
        tuple(traversed),
        None,
        head,
    )


def immediate_outcome(state: SnakeState, direction: str) -> SnakeSimulation:
    """Return the one-step outcome for one direction."""

    return simulate_snake_moves(state, (str(direction),))


def safe_next_directions(state: SnakeState) -> Tuple[str, ...]:
    """Return directions whose next move does not hit wall, body, or obstacle."""

    safe: list[str] = []
    for direction in DIRECTION_NAMES:
        outcome = immediate_outcome(state, direction).outcome
        if outcome in {"safe", "food"}:
            safe.append(str(direction))
    return tuple(safe)


def candidate_move_sequences(length: int) -> Tuple[Tuple[str, ...], ...]:
    """Return all cardinal move sequences of one length."""

    return tuple(tuple(str(move) for move in sequence) for sequence in product(DIRECTION_NAMES, repeat=int(length)))


def validate_snake_state(state: SnakeState) -> None:
    """Validate board bounds, occupancy uniqueness, and visible blockers."""

    size = int(state.board_size)
    head = (int(state.head[0]), int(state.head[1]))
    body = tuple((int(row), int(col)) for row, col in state.body)
    food = (int(state.food[0]), int(state.food[1]))
    obstacles = tuple((int(row), int(col)) for row, col in state.obstacles)
    if size < 4:
        raise ValueError("snake board_size must be at least 4")
    for coord in (head, food, *body, *obstacles):
        if not in_bounds(coord, size=size):
            raise ValueError("snake coordinate outside board")
    occupied = (head,) + body + obstacles
    if len(set(occupied)) != len(occupied):
        raise ValueError("snake state has overlapping head/body/obstacles")
    if food in set(occupied):
        raise ValueError("snake food overlaps occupied cell")
    prev = head
    for coord in body:
        if abs(int(coord[0]) - int(prev[0])) + abs(int(coord[1]) - int(prev[1])) != 1:
            raise ValueError("snake body is not connected")
        prev = coord


def visible_snake_trace(state: SnakeState) -> Mapping[str, object]:
    """Return trace-friendly Snake state coordinates."""

    return {
        "board_size": int(state.board_size),
        "head": [int(state.head[0]), int(state.head[1])],
        "body": [[int(row), int(col)] for row, col in state.body],
        "food": [int(state.food[0]), int(state.food[1])],
        "obstacles": [[int(row), int(col)] for row, col in state.obstacles],
    }


__all__ = [
    "all_coords",
    "candidate_move_sequences",
    "coord_to_cell_id",
    "direction_text",
    "immediate_outcome",
    "in_bounds",
    "move_sequence_text",
    "neighbor_coords",
    "safe_next_directions",
    "simulate_snake_moves",
    "step_coord",
    "validate_snake_state",
    "visible_snake_trace",
]
