"""Passive state and constants for Ludo board scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


SCENE_ID = "ludo_board"
SCENE_NAMESPACE = "games.ludo_board"
PLAYER_COLORS: Tuple[str, ...] = ("red", "green", "blue", "yellow")
STYLE_VARIANTS: Tuple[str, ...] = ("classic_bright", "ivory_board", "slate_table", "soft_plastic", "arcade_gloss")
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
Coord = Tuple[int, int]
BBox = Tuple[float, float, float, float]


MAIN_PATH: Tuple[Coord, ...] = (
    (6, 1), (6, 2), (6, 3), (6, 4), (6, 5),
    (5, 6), (4, 6), (3, 6), (2, 6), (1, 6), (0, 6),
    (0, 7),
    (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8),
    (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
    (7, 14),
    (8, 14), (8, 13), (8, 12), (8, 11), (8, 10), (8, 9),
    (9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8),
    (14, 7),
    (14, 6), (13, 6), (12, 6), (11, 6), (10, 6), (9, 6),
    (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (8, 0),
    (7, 0), (6, 0),
)
START_COORDS: dict[str, Coord] = {
    "red": (6, 1),
    "green": (1, 8),
    "yellow": (8, 13),
    "blue": (13, 6),
}
HOME_LANES: dict[str, Tuple[Coord, ...]] = {
    "red": ((7, 1), (7, 2), (7, 3), (7, 4), (7, 5)),
    "green": ((1, 7), (2, 7), (3, 7), (4, 7), (5, 7)),
    "yellow": ((7, 13), (7, 12), (7, 11), (7, 10), (7, 9)),
    "blue": ((13, 7), (12, 7), (11, 7), (10, 7), (9, 7)),
}
HOME_ENTRY_COORDS: dict[str, Coord] = {
    "red": (7, 0),
    "green": (0, 7),
    "yellow": (7, 14),
    "blue": (14, 7),
}
FLOW_ARROW_SPECS: Tuple[Tuple[Coord, Coord, str], ...] = (
    ((6, 2), (6, 3), "start_forward_red"),
    ((2, 8), (3, 8), "start_forward_green"),
    ((8, 12), (8, 11), "start_forward_yellow"),
    ((12, 6), (11, 6), "start_forward_blue"),
    ((6, 5), (5, 6), "corner_turn_top_left"),
    ((5, 8), (6, 9), "corner_turn_top_right"),
    ((8, 9), (9, 8), "corner_turn_bottom_right"),
    ((9, 6), (8, 5), "corner_turn_bottom_left"),
    ((7, 0), (7, 1), "home_entry_red"),
    ((0, 7), (1, 7), "home_entry_green"),
    ((7, 14), (7, 13), "home_entry_yellow"),
    ((14, 7), (13, 7), "home_entry_blue"),
)
FLOW_ARROW_CELLS: frozenset[Coord] = frozenset(coord for start, end, _role in FLOW_ARROW_SPECS for coord in (start, end))
YARD_BBOX_CELLS: dict[str, Tuple[int, int, int, int]] = {
    "red": (0, 0, 6, 6),
    "green": (0, 9, 6, 15),
    "blue": (9, 0, 15, 6),
    "yellow": (9, 9, 15, 15),
}


@dataclass(frozen=True)
class LudoDefaults:
    """Stable fallback defaults for Ludo board scenes."""

    winning_roll_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    capture_distance_support: Tuple[int, ...] = tuple(range(1, 12))
    move_roll_total_support: Tuple[int, ...] = tuple(range(1, 12))
    option_label_support: Tuple[str, ...] = OPTION_LABELS
    capture_option_count_support: Tuple[int, ...] = (4, 6)
    move_result_option_count_support: Tuple[int, ...] = (4, 5, 6)
    cell_size_min_px: int = 36
    cell_size_max_px: int = 48
    canvas_width: int = 920
    canvas_height: int = 980
    canvas_side_padding_px: int = 180
    canvas_vertical_padding_px: int = 210
    board_padding_px: int = 24
    grid_width_px: int = 2
    token_radius_fraction: float = 0.36
    flow_arrow_enabled: bool = True
    flow_arrow_width_px: int = 2
    option_card_width_px: int = 112
    option_card_height_px: int = 54
    option_card_gap_px: int = 10


@dataclass(frozen=True)
class LudoSceneAxes:
    """Resolved shared visual/player axes for one Ludo board instance."""

    style_variant: str
    query_color: str
    target_color: str
    style_variant_probabilities: Mapping[str, float]
    query_color_probabilities: Mapping[str, float]
    target_color_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class LudoRollOption:
    """One image-drawn roll option."""

    label: str
    distance: int
    text: str


@dataclass(frozen=True)
class LudoDestinationOption:
    """One board-drawn destination option."""

    label: str
    coord: Coord


@dataclass(frozen=True)
class LudoRenderState:
    """Task-owned symbolic Ludo state consumed by the shared renderer."""

    style_variant: str
    token_coords: Mapping[str, Coord]
    query_color: str
    target_color: str | None
    roll_sequence: Tuple[int, ...]
    roll_options: Tuple[LudoRollOption, ...]
    destination_options: Tuple[LudoDestinationOption, ...]


DEFAULTS = LudoDefaults()


__all__ = [
    "BBox",
    "Coord",
    "DEFAULTS",
    "FLOW_ARROW_CELLS",
    "FLOW_ARROW_SPECS",
    "HOME_ENTRY_COORDS",
    "HOME_LANES",
    "LudoDefaults",
    "LudoDestinationOption",
    "LudoRenderState",
    "LudoRollOption",
    "LudoSceneAxes",
    "MAIN_PATH",
    "OPTION_LABELS",
    "PLAYER_COLORS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "START_COORDS",
    "STYLE_VARIANTS",
    "YARD_BBOX_CELLS",
]
