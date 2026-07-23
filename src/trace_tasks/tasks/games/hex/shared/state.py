"""Scene-local records and defaults for Hex tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .rules import HEX_CANDIDATE_LABELS


SCENE_ID = "hex"
HEX_NAMESPACE = "games.hex"


@dataclass(frozen=True)
class HexDefaults:
    """Stable scene-level fallback defaults for visible Hex-board tasks."""

    board_size_support: Tuple[int, ...] = (5, 6, 7, 8)
    candidate_count_support: Tuple[int, ...] = (4, 5, 6)
    winning_move_label_support: Tuple[str, ...] = HEX_CANDIDATE_LABELS[:6]
    min_extra_own_stones: int = 2
    max_extra_own_stones: int = 7
    min_extra_opponent_stones: int = 5
    max_extra_opponent_stones: int = 14
    canvas_width: int = 980
    canvas_height: int = 900
    panel_margin_px: int = 54
    max_board_width_px: int = 820
    max_board_height_px: int = 760
    hex_border_width_px: int = 3
    stone_radius_fraction: float = 0.46
    candidate_label_font_size_px: int = 30
    side_band_width_px: int = 8
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 600
    canvas_min_height_px: int = 560
    canvas_side_padding_px: int = 110
    canvas_vertical_padding_px: int = 110


DEFAULTS = HexDefaults()


@dataclass(frozen=True)
class HexIntegerAxis:
    """One resolved integer axis plus trace probabilities."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class HexStringAxis:
    """One resolved string axis plus trace probabilities."""

    value: str
    support: Tuple[str, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class HexSceneAxes:
    """Scene-level visual and board axes shared by all Hex tasks."""

    scene_variant: str
    style_variant: str
    player_color: str
    board_size: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    player_color_probabilities: Dict[str, float]
    board_size_probabilities: Dict[str, float]


__all__ = [
    "DEFAULTS",
    "HEX_NAMESPACE",
    "SCENE_ID",
    "HexDefaults",
    "HexIntegerAxis",
    "HexSceneAxes",
    "HexStringAxis",
]
