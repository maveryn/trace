"""Vector math helpers for free-body-force diagrams."""

from __future__ import annotations

import math

from .state import DIRECTION_VECTORS, VECTOR_TO_DIRECTION


def unit_from_direction(direction: str) -> tuple[float, float]:
    """Return image-space unit vector for a named physics direction."""

    x, y_up = DIRECTION_VECTORS[str(direction)]
    length = math.hypot(float(x), float(y_up))
    return (float(x) / float(length), -float(y_up) / float(length))


def vector_direction(vector: tuple[int, int]) -> str:
    """Return the named compass direction of a nonzero resultant vector."""

    x, y = int(vector[0]), int(vector[1])
    sx = 0 if x == 0 else (1 if x > 0 else -1)
    sy = 0 if y == 0 else (1 if y > 0 else -1)
    if (sx, sy) == (0, 0):
        raise ValueError("zero resultant force has no direction")
    if sx != 0 and sy != 0 and abs(int(x)) != abs(int(y)):
        raise ValueError(f"diagonal resultant must use equal components, got {(x, y)}")
    return str(VECTOR_TO_DIRECTION[(sx, sy)])


def sum_force_vectors(vectors: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Sum symbolic applied-force vectors."""

    return (
        sum(int(vector[0]) for vector in vectors),
        sum(int(vector[1]) for vector in vectors),
    )
