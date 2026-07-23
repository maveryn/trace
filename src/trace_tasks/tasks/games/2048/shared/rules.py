"""Pure 2048 board mechanics for the games scene."""

from __future__ import annotations

from typing import Tuple

from .state import Board, Coord, EMPTY, SIZE, SUPPORTED_2048_DIRECTIONS, Move2048Result, validate_board


def line_coords(index: int, direction: str) -> Tuple[Coord, ...]:
    """Return board coordinates in move-compression order for one line."""

    if str(direction) == "left":
        return tuple((int(index), col) for col in range(SIZE))
    if str(direction) == "right":
        return tuple((int(index), col) for col in range(SIZE - 1, -1, -1))
    if str(direction) == "up":
        return tuple((row, int(index)) for row in range(SIZE))
    if str(direction) == "down":
        return tuple((row, int(index)) for row in range(SIZE - 1, -1, -1))
    raise ValueError(f"unsupported 2048 move direction: {direction!r}")


def simulate_2048_move(board: Board, direction: str) -> Move2048Result:
    """Apply standard 2048 slide-and-merge rules for one move."""

    validate_board(board)
    if str(direction) not in SUPPORTED_2048_DIRECTIONS:
        raise ValueError(f"unsupported 2048 move direction: {direction!r}")

    result = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    merge_pairs: list[Tuple[Coord, Coord]] = []
    result_sources: dict[Coord, Tuple[Coord, ...]] = {}
    score = 0

    for line_index in range(SIZE):
        coords = line_coords(line_index, str(direction))
        tiles: list[tuple[int, Tuple[Coord, ...]]] = []
        for coord in coords:
            value = int(board[int(coord[0])][int(coord[1])])
            if value != EMPTY:
                tiles.append((int(value), (coord,)))

        compressed: list[tuple[int, Tuple[Coord, ...]]] = []
        cursor = 0
        while cursor < len(tiles):
            value, sources = tiles[cursor]
            if cursor + 1 < len(tiles) and int(tiles[cursor + 1][0]) == int(value):
                merged_sources = tuple(sources + tiles[cursor + 1][1])
                merged_value = int(value) * 2
                compressed.append((int(merged_value), merged_sources))
                merge_pairs.append((merged_sources[0], merged_sources[1]))
                score += int(merged_value)
                cursor += 2
            else:
                compressed.append((int(value), tuple(sources)))
                cursor += 1

        for output_index, (value, sources) in enumerate(compressed):
            dest = coords[int(output_index)]
            result[int(dest[0])][int(dest[1])] = int(value)
            result_sources[dest] = tuple(sources)

    after = tuple(tuple(int(value) for value in row) for row in result)
    return Move2048Result(
        direction=str(direction),
        before=tuple(tuple(int(value) for value in row) for row in board),
        after=after,
        merge_pairs=tuple(tuple(pair) for pair in merge_pairs),
        score=int(score),
        moved=after != tuple(tuple(int(value) for value in row) for row in board),
        result_sources=dict(result_sources),
    )


def board_empty_count(board: Board) -> int:
    """Return the number of empty cells on one 2048 board."""

    return sum(1 for row in board for value in row if int(value) == EMPTY)


def board_max_tile(board: Board) -> int:
    """Return the largest visible tile value on one 2048 board."""

    values = [int(value) for row in board for value in row]
    return max(values) if values else EMPTY


def board_key(board: Board) -> Tuple[Tuple[int, ...], ...]:
    """Return a hashable normalized board key."""

    return tuple(tuple(int(value) for value in row) for row in board)


def unique_max_result(result: Move2048Result) -> bool:
    """Return true when the post-move board has exactly one max-valued tile."""

    max_value = int(board_max_tile(result.after))
    return sum(1 for row in result.after for value in row if int(value) == int(max_value)) == 1


__all__ = [
    "board_empty_count",
    "board_key",
    "board_max_tile",
    "line_coords",
    "simulate_2048_move",
    "unique_max_result",
]
