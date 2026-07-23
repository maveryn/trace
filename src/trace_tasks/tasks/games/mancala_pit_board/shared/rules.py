"""Pure sowing rules for Mancala-style pit-board tasks."""

from __future__ import annotations

from typing import Sequence, Tuple

from .state import LABELS, PIT_COUNT, PITS_PER_ROW


def pit_label(index: int) -> str:
    """Return the visible pit label for a circular board index."""

    return str(LABELS[int(index) % len(LABELS)])


def pit_index(label: str) -> int:
    """Return the circular board index for a visible pit label."""

    normalized = str(label).strip().upper()
    if normalized not in LABELS:
        raise ValueError(f"unknown pit label: {label!r}")
    return int(LABELS.index(normalized))


def visual_row_col(index: int) -> Tuple[int, int]:
    """Map circular pit order to the two-row visual layout."""

    resolved = int(index) % PIT_COUNT
    if resolved < PITS_PER_ROW:
        return 0, resolved
    return 1, (PIT_COUNT - 1) - resolved


def sowing_path(source_index: int, seed_count: int) -> Tuple[int, ...]:
    """Return the ordered pits receiving seeds after one sowing move."""

    return tuple((int(source_index) + step) % PIT_COUNT for step in range(1, int(seed_count) + 1))


def sow_counts(initial_counts: Sequence[int], source_index: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Apply one sowing move and return final counts plus the receiving path."""

    counts = [int(value) for value in initial_counts]
    source = int(source_index) % PIT_COUNT
    seed_count = int(counts[source])
    counts[source] = 0
    path = sowing_path(source, seed_count)
    for pit in path:
        counts[int(pit)] += 1
    return tuple(int(value) for value in counts), tuple(int(value) for value in path)


__all__ = [
    "pit_index",
    "pit_label",
    "sow_counts",
    "sowing_path",
    "visual_row_col",
]
