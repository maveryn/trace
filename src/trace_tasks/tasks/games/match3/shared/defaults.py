"""Scene constants and fallback defaults for match-3 game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "match3"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("square_board", "wide_board", "tall_board")
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "faceted_jewels",
    "round_candies",
    "beveled_tiles",
    "diamond_gems",
    "orb_tokens",
)
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
GEM_KEYS: Tuple[str, ...] = ("red", "blue", "green", "yellow", "purple", "cyan")
GEM_RGB: Dict[str, Tuple[int, int, int]] = {
    str(name): tuple(int(value) for value in named_color(str(name)))
    for name in GEM_KEYS
}
MATCH3_STYLE_RGB: Dict[str, Dict[str, Any]] = {
    "faceted_jewels": {
        "gem_shape": "faceted",
        "cell_alpha": 0.78,
        "grid_width": 1,
        "gem_outline_width": 2,
        "shadow": True,
    },
    "round_candies": {
        "gem_shape": "circle",
        "cell_alpha": 0.70,
        "grid_width": 2,
        "gem_outline_width": 3,
        "shadow": False,
    },
    "beveled_tiles": {
        "gem_shape": "rounded_square",
        "cell_alpha": 0.82,
        "grid_width": 2,
        "gem_outline_width": 2,
        "shadow": True,
    },
    "diamond_gems": {
        "gem_shape": "diamond",
        "cell_alpha": 0.74,
        "grid_width": 1,
        "gem_outline_width": 3,
        "shadow": True,
    },
    "orb_tokens": {
        "gem_shape": "orb",
        "cell_alpha": 0.66,
        "grid_width": 2,
        "gem_outline_width": 3,
        "shadow": False,
    },
}


@dataclass(frozen=True)
class Match3Defaults:
    """Stable fallback defaults used when scene config omits a knob."""

    row_count_support: Tuple[int, ...] = (5, 6, 7)
    col_count_support: Tuple[int, ...] = (5, 6, 7)
    gem_type_count_support: Tuple[int, ...] = (5, 6)
    option_count_support: Tuple[int, ...] = (4, 5, 6)
    gem_count_answer_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    canvas_width: int = 760
    canvas_height: int = 720
    panel_margin_px: int = 42
    board_inner_margin_px: int = 38
    index_margin_px: int = 34
    cell_size_px: int = 78
    cell_gap_px: int = 8
    gem_inset_px: int = 10
    label_font_size_px: int = 21
    index_font_size_px: int = 18
    arrow_width_px: int = 7


DEFAULTS = Match3Defaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "DEFAULTS",
    "GEM_KEYS",
    "GEM_RGB",
    "MATCH3_STYLE_RGB",
    "OPTION_LABELS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "Match3Defaults",
]
