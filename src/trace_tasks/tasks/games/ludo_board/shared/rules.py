"""Path and roll rules for Ludo board scene tasks."""

from __future__ import annotations

from typing import Tuple

from .state import Coord, HOME_ENTRY_COORDS, HOME_LANES, MAIN_PATH, START_COORDS


def path_index(coord: Coord) -> int:
    """Return the index of a playable main-track cell."""

    return int(MAIN_PATH.index(tuple(coord)))


def color_rgb(color: str) -> Tuple[int, int, int]:
    """Return the canonical visible token color for a Ludo player name."""

    colors = {
        "red": (221, 55, 61),
        "green": (45, 166, 85),
        "blue": (47, 98, 214),
        "yellow": (241, 199, 49),
    }
    return colors[str(color)]


def soft_color(color: str, *, amount: float = 0.56) -> Tuple[int, int, int]:
    """Mix a semantic player color with white for board fills."""

    base = color_rgb(str(color))
    return tuple(int(round((float(channel) * float(amount)) + (255.0 * (1.0 - float(amount))))) for channel in base)


def route_for_color(color: str) -> Tuple[Coord, ...]:
    """Return the full clockwise route from a player's start into its home lane."""

    start_index = path_index(START_COORDS[str(color)])
    entry_index = path_index(HOME_ENTRY_COORDS[str(color)])
    if int(entry_index) >= int(start_index):
        main_segment = MAIN_PATH[start_index : entry_index + 1]
    else:
        main_segment = MAIN_PATH[start_index:] + MAIN_PATH[: entry_index + 1]
    return tuple(main_segment) + tuple(HOME_LANES[str(color)])


def roll_sequence_for_total(total: int) -> Tuple[int, ...]:
    """Encode a total move as the visible Ludo die sequence used by this scene."""

    total = int(total)
    if 1 <= total <= 6:
        return (int(total),)
    if 7 <= total <= 11:
        return (6, int(total - 6))
    if 13 <= total <= 17:
        return (6, 6, int(total - 12))
    raise ValueError("Ludo move total must be in 1..11 or 13..17")


def roll_option_text(distance: int) -> str:
    """Render a capture option distance as a single roll or `6 then k`."""

    distance = int(distance)
    if 1 <= distance <= 6:
        return str(distance)
    if 7 <= distance <= 11:
        return f"6 then {distance - 6}"
    raise ValueError("Ludo roll option distance must be in 1..11")


__all__ = [
    "color_rgb",
    "path_index",
    "roll_option_text",
    "roll_sequence_for_total",
    "route_for_color",
    "soft_color",
]
