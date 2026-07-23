"""Shared deterministic helpers for sequence transformations."""

from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def rotate_sequence(values: Sequence[T], *, shift: int) -> list[T]:
    """Rotate one sequence left by `shift` positions."""
    if not values:
        return []
    n = int(len(values))
    s = int(shift) % int(n)
    seq = list(values)
    return [*seq[s:], *seq[:s]]
