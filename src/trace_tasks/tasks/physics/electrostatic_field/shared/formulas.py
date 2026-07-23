"""Formula helpers for electrostatic field-map scenes."""

from __future__ import annotations

from .state import DIRECTION_MODE_NEGATIVE_FORCE, DIRECTION_VECTORS


def opposite_direction(direction: str) -> str:
    """Return the opposite named compass direction."""

    dx, dy = DIRECTION_VECTORS[str(direction)]
    for name, vector in DIRECTION_VECTORS.items():
        if vector == (-int(dx), -int(dy)):
            return str(name)
    raise ValueError(f"unsupported direction: {direction}")


def requested_field_direction(*, requested_direction: str, direction_mode: str) -> str:
    """Return the actual field direction needed for the requested force wording."""

    if str(direction_mode) == DIRECTION_MODE_NEGATIVE_FORCE:
        return opposite_direction(str(requested_direction))
    return str(requested_direction)
