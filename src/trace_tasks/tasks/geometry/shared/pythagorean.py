"""Scene-neutral integer right-triangle helpers for geometry tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class IntegerRightTriangle:
    """One oriented integer right triangle."""

    leg_a: int
    leg_b: int
    hypotenuse: int

    @property
    def key(self) -> str:
        return f"{int(self.leg_a)}-{int(self.leg_b)}-{int(self.hypotenuse)}"

    @property
    def sorted_leg_key(self) -> str:
        a, b = sorted((int(self.leg_a), int(self.leg_b)))
        return f"{a}-{b}-{int(self.hypotenuse)}"


def validate_integer_right_triangle(triangle: IntegerRightTriangle) -> None:
    """Reject non-positive or non-Pythagorean integer triangles."""

    if int(triangle.leg_a) <= 0 or int(triangle.leg_b) <= 0 or int(triangle.hypotenuse) <= 0:
        raise ValueError("right-triangle dimensions must be positive")
    if int(triangle.leg_a) ** 2 + int(triangle.leg_b) ** 2 != int(triangle.hypotenuse) ** 2:
        raise ValueError("right-triangle dimensions must satisfy a^2 + b^2 = c^2")


@lru_cache(maxsize=32)
def integer_right_triangles(
    *,
    min_leg: int = 1,
    max_leg: int = 100,
    max_hypotenuse: int = 150,
    include_swapped: bool = False,
) -> tuple[IntegerRightTriangle, ...]:
    """Return integer right triangles inside the requested bounds.

    The helper intentionally uses brute-force enumeration. Geometry task pools
    are small, and the resulting behavior is straightforward to audit.
    """

    min_leg = max(1, int(min_leg))
    max_leg = int(max_leg)
    max_hypotenuse = int(max_hypotenuse)
    if max_leg < min_leg:
        raise ValueError("max_leg must be >= min_leg")
    if max_hypotenuse <= 0:
        raise ValueError("max_hypotenuse must be positive")

    triangles: list[IntegerRightTriangle] = []
    for leg_a in range(min_leg, max_leg + 1):
        for leg_b in range(leg_a, max_leg + 1):
            hypotenuse_sq = int(leg_a * leg_a + leg_b * leg_b)
            hypotenuse = int(math.isqrt(hypotenuse_sq))
            if int(hypotenuse * hypotenuse) != hypotenuse_sq:
                continue
            if int(hypotenuse) > max_hypotenuse:
                continue
            triangles.append(
                IntegerRightTriangle(
                    leg_a=int(leg_a),
                    leg_b=int(leg_b),
                    hypotenuse=int(hypotenuse),
                )
            )
            if bool(include_swapped) and int(leg_a) != int(leg_b):
                triangles.append(
                    IntegerRightTriangle(
                        leg_a=int(leg_b),
                        leg_b=int(leg_a),
                        hypotenuse=int(hypotenuse),
                    )
                )
    if not triangles:
        raise ValueError("integer_right_triangles resolved empty")
    return tuple(
        sorted(
            triangles,
            key=lambda triangle: (
                int(triangle.hypotenuse),
                int(triangle.leg_a),
                int(triangle.leg_b),
            ),
        )
    )


__all__ = [
    "IntegerRightTriangle",
    "integer_right_triangles",
    "validate_integer_right_triangle",
]
