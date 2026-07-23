"""Domino tile mechanics shared by dominoes scene tasks."""

from __future__ import annotations

from typing import Tuple


OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEFGHIJKL")
PIP_VALUES: Tuple[int, ...] = tuple(range(7))
CANONICAL_DOMINOES: Tuple[Tuple[int, int], ...] = tuple(
    (int(left_value), int(right_value))
    for left_value in range(7)
    for right_value in range(left_value, 7)
)


def canonical_tile(left_value: int, right_value: int) -> Tuple[int, int]:
    """Return the canonical unordered representation for one domino tile."""

    lower = min(int(left_value), int(right_value))
    upper = max(int(left_value), int(right_value))
    return (int(lower), int(upper))


def tile_sum(tile: Tuple[int, int]) -> int:
    """Return the total pip count for one canonical domino tile."""

    return int(tile[0] + tile[1])


def can_connect(tile: Tuple[int, int], open_end: int) -> bool:
    """Return whether one canonical tile can connect to the given open end."""

    return int(open_end) in {int(tile[0]), int(tile[1])}


def chain_open_end_after_play(tile: Tuple[int, int], open_end: int) -> int:
    """Return the open value left after playing one tile on a matching end."""

    if not can_connect(tile, int(open_end)):
        raise ValueError("tile does not connect to open end")
    if int(tile[0]) == int(tile[1]):
        return int(open_end)
    return int(tile[1]) if int(tile[0]) == int(open_end) else int(tile[0])


__all__ = [
    "CANONICAL_DOMINOES",
    "OPTION_LABELS",
    "PIP_VALUES",
    "can_connect",
    "chain_open_end_after_play",
    "canonical_tile",
    "tile_sum",
]
