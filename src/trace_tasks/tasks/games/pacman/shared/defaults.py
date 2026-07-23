"""Scene constants and fallback defaults for Pac-Man tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "pacman"

SUPPORTED_PACMAN_SCENE_VARIANTS: Tuple[str, ...] = (
    "compact_maze",
    "wide_maze",
)
SUPPORTED_PACMAN_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "neon",
    "paper",
    "terminal",
    "pastel",
)
PACMAN_ITEM_LABELS: Tuple[str, ...] = tuple("ABCDEF")
PACMAN_ITEM_KINDS: Tuple[str, ...] = (
    "cherry",
    "strawberry",
    "orange",
    "bell",
    "melon",
    "key",
)
PACMAN_GHOST_COLOR_KEYS: Tuple[str, ...] = (
    "red",
    "pink",
    "cyan",
    "orange",
    "purple",
)
ROUTE_SCORE_BONUS_VALUE_SUPPORT: Tuple[int, ...] = (2, 3, 4)


@dataclass(frozen=True)
class PacmanDefaults:
    """Stable fallback defaults for visible Pac-Man maze scenes."""

    row_count_support: Tuple[int, ...] = (7, 8, 9)
    col_count_support: Tuple[int, ...] = (9, 11, 13)
    pellet_count_before_ghost_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    route_score_on_route_pellet_count_support: Tuple[int, ...] = (1, 2, 3, 4)
    route_score_on_route_bonus_count_support: Tuple[int, ...] = (1, 2)
    route_score_off_route_bonus_count_support: Tuple[int, ...] = (1, 2, 3)
    route_score_bonus_value_support: Tuple[int, ...] = ROUTE_SCORE_BONUS_VALUE_SUPPORT
    next_item_label_support: Tuple[str, ...] = PACMAN_ITEM_LABELS
    item_count_support: Tuple[int, ...] = (4, 5, 6)
    canvas_width: int = 980
    canvas_height: int = 760
    panel_margin_px: int = 38
    maze_width_px: int = 840
    maze_height_px: int = 620
    wall_gap_px: int = 2
    wall_outline_width_px: int = 1
    pellet_radius_px: int = 13
    item_radius_px: int = 19
    ghost_radius_px: int = 18
    route_width_px: int = 8
    item_label_font_size_px: int = 23


DEFAULTS = PacmanDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "DEFAULTS",
    "PACMAN_GHOST_COLOR_KEYS",
    "PACMAN_ITEM_KINDS",
    "PACMAN_ITEM_LABELS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "ROUTE_SCORE_BONUS_VALUE_SUPPORT",
    "SCENE_ID",
    "SUPPORTED_PACMAN_SCENE_VARIANTS",
    "SUPPORTED_PACMAN_STYLE_VARIANTS",
    "PacmanDefaults",
]
