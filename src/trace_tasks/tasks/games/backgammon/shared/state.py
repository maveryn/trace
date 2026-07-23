"""State contracts for the Backgammon games scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple


SCENE_ID = "backgammon"

SUPPORTED_BACKGAMMON_SCENE_VARIANTS: Tuple[str, ...] = ("standard_board",)
SUPPORTED_BACKGAMMON_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "navy",
    "parchment",
    "slate",
    "tournament",
)
PLAYER_BLACK = "black"
PLAYER_WHITE = "white"
ACTIVE_PLAYERS: Tuple[str, ...] = (PLAYER_BLACK, PLAYER_WHITE)
POINT_IDS: Tuple[int, ...] = tuple(range(1, 25))
DESTINATION_STATUS_LEGAL = "legal"
DESTINATION_STATUS_HIT = "hit"
DESTINATION_STATUS_BLOCKED = "blocked"
DESTINATION_STATUSES: Tuple[str, ...] = (
    DESTINATION_STATUS_LEGAL,
    DESTINATION_STATUS_HIT,
    DESTINATION_STATUS_BLOCKED,
)
STACK_STATE_SINGLE = "single"
STACK_STATE_TWO_OR_MORE = "two_or_more"
STACK_STATES: Tuple[str, ...] = (STACK_STATE_SINGLE, STACK_STATE_TWO_OR_MORE)


@dataclass(frozen=True)
class BackgammonPoint:
    """Visible checker stack on one numbered Backgammon point."""

    owner: str | None
    count: int


@dataclass(frozen=True)
class BackgammonOutcome:
    """Computed single-die destination sets for one active player."""

    legal_destinations: Tuple[int, ...]
    hit_destinations: Tuple[int, ...]
    blocked_destinations: Tuple[int, ...]


@dataclass(frozen=True)
class BackgammonSample:
    """One generated Backgammon position plus answer/annotation contract."""

    points: Mapping[int, BackgammonPoint]
    dice: Tuple[int, int]
    active_player: str
    answer: int
    target_destinations: Tuple[int, ...]
    outcome: BackgammonOutcome
    style_variant: str
    target_answer: int
    target_points: Tuple[int, ...] = ()
    destination_status: str = ""
    checker_color: str = ""
    stack_state: str = ""
    pip_count_contributions: Mapping[int, int] = field(default_factory=dict)
    use_dice_for_moves: bool = True


def point_entity_id(point_id: int) -> str:
    """Return the stable entity id for one numbered board point."""

    return f"point_{int(point_id)}"


def checker_entity_id(point_id: int, stack_index: int) -> str:
    """Return the stable entity id for one visible checker."""

    return f"checker_p{int(point_id)}_{int(stack_index)}"


def die_entity_id(index: int) -> str:
    """Return the stable entity id for one visible die."""

    return f"die_{int(index)}"


def empty_points() -> dict[int, BackgammonPoint]:
    """Return an empty 24-point Backgammon board."""

    return {point: BackgammonPoint(owner=None, count=0) for point in POINT_IDS}


def stack_at(points: Mapping[int, BackgammonPoint], point_id: int) -> BackgammonPoint:
    """Return the stack at one point, treating missing points as empty."""

    stack = points.get(int(point_id))
    if stack is None:
        return BackgammonPoint(owner=None, count=0)
    return stack


__all__ = [
    "ACTIVE_PLAYERS",
    "DESTINATION_STATUSES",
    "DESTINATION_STATUS_BLOCKED",
    "DESTINATION_STATUS_HIT",
    "DESTINATION_STATUS_LEGAL",
    "POINT_IDS",
    "PLAYER_BLACK",
    "PLAYER_WHITE",
    "SCENE_ID",
    "STACK_STATES",
    "STACK_STATE_SINGLE",
    "STACK_STATE_TWO_OR_MORE",
    "SUPPORTED_BACKGAMMON_SCENE_VARIANTS",
    "SUPPORTED_BACKGAMMON_STYLE_VARIANTS",
    "BackgammonOutcome",
    "BackgammonPoint",
    "BackgammonSample",
    "checker_entity_id",
    "die_entity_id",
    "empty_points",
    "point_entity_id",
    "stack_at",
]
