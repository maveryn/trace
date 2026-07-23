"""Pure cyclic-order equivalence rules."""

from __future__ import annotations

from typing import Sequence, Tuple


def rotate_token_sequence(tokens: Sequence[str], offset: int) -> Tuple[str, ...]:
    """Return one cyclic rotation of the token sequence."""

    items = [str(token) for token in tokens]
    if not items:
        return tuple()
    shift = int(offset) % int(len(items))
    return tuple(items[shift:] + items[:shift])


def token_sequences_are_rotation_equivalent(a: Sequence[str], b: Sequence[str]) -> bool:
    """Return whether two token sequences match up to rotation only."""

    left = tuple(str(item) for item in a)
    right = tuple(str(item) for item in b)
    if len(left) != len(right):
        return False
    if not left:
        return True
    doubled = list(left) + list(left)
    width = len(left)
    return any(tuple(doubled[index : index + width]) == right for index in range(width))


def invalid_candidate_sequences(reference_tokens: Sequence[str]) -> list[Tuple[str, ...]]:
    """Return deterministic non-equivalent distractor sequences for one loop."""

    tokens = [str(token) for token in reference_tokens]
    size = len(tokens)
    if size < 2:
        return []

    candidates: list[Tuple[str, ...]] = []
    candidates.append(tuple(reversed(tokens)))
    for index in range(size - 1):
        swapped = list(tokens)
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        candidates.append(tuple(swapped))

    swapped = list(tokens)
    swapped[0], swapped[-1] = swapped[-1], swapped[0]
    candidates.append(tuple(swapped))

    for insertion_index in range(2, size):
        moved = list(tokens)
        token = moved.pop(0)
        moved.insert(int(insertion_index), token)
        candidates.append(tuple(moved))

    unique: list[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if token_sequences_are_rotation_equivalent(tokens, candidate):
            continue
        unique.append(candidate)
    return unique


__all__ = [
    "invalid_candidate_sequences",
    "rotate_token_sequence",
    "token_sequences_are_rotation_equivalent",
]
