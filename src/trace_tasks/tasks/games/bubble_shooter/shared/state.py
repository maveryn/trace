"""Shared Bubble-shooter passive state for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

SUPPORTED_BUBBLE_SHOOTER_SCENE_VARIANTS: Tuple[str, ...] = (
    "open_pack",
    "dense_pack",
)
SUPPORTED_BUBBLE_SHOOTER_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "pastel",
    "neon",
    "paper",
    "arcade",
)
BUBBLE_COLOR_KEYS: Tuple[str, ...] = (
    "red",
    "yellow",
    "blue",
    "green",
    "purple",
    "orange",
)
BUBBLE_OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")

Coord = Tuple[int, int]
Board = Tuple[Tuple[str | None, ...], ...]


@dataclass(frozen=True)
class BubbleShooterOption:
    """One labeled next-bubble color option."""

    label: str
    color_key: str
    is_answer: bool


@dataclass(frozen=True)
class BubbleShooterLandingOption:
    """One labeled landing target option."""

    label: str
    landing_coord: Coord
    is_answer: bool


@dataclass(frozen=True)
class BubbleShotOutcome:
    """Computed effect of shooting one bubble to one landing slot."""

    landing_coord: Coord
    color_key: str
    connected_same_color_coords: Tuple[Coord, ...]
    popped_coords: Tuple[Coord, ...]
    dropped_coords: Tuple[Coord, ...]


@dataclass(frozen=True)
class BubbleShooterState:
    """Generated Bubble-shooter board state before task-specific binding."""

    row_count: int
    col_count: int
    scene_variant: str
    style_variant: str
    board: Board
    landing_coord: Coord
    shooter_color_key: str | None
    option_specs: Tuple[BubbleShooterOption, ...]
    outcome: BubbleShotOutcome
    construction_mode: str
    landing_option_specs: Tuple[BubbleShooterLandingOption, ...] = tuple()


def bubble_entity_id(coord: Coord) -> str:
    """Return the stable entity id for one visible board bubble."""

    row, col = coord
    return f"bubble_r{int(row)}_c{int(col)}"


def landing_slot_entity_id() -> str:
    """Return the stable entity id for the marked landing slot."""

    return "landing_slot"


def shooter_bubble_entity_id() -> str:
    """Return the stable entity id for the visible shooter bubble."""

    return "shooter_bubble"


def option_entity_id(label: str) -> str:
    """Return the stable entity id for one labeled color option."""

    return f"option_{str(label)}"


def landing_option_entity_id(label: str) -> str:
    """Return the stable entity id for one labeled landing target option."""

    return f"landing_option_{str(label)}"


__all__ = [
    "BUBBLE_COLOR_KEYS",
    "BUBBLE_OPTION_LABELS",
    "SUPPORTED_BUBBLE_SHOOTER_SCENE_VARIANTS",
    "SUPPORTED_BUBBLE_SHOOTER_STYLE_VARIANTS",
    "Board",
    "BubbleShooterLandingOption",
    "BubbleShooterOption",
    "BubbleShooterState",
    "BubbleShotOutcome",
    "Coord",
    "bubble_entity_id",
    "landing_slot_entity_id",
    "landing_option_entity_id",
    "option_entity_id",
    "shooter_bubble_entity_id",
]
