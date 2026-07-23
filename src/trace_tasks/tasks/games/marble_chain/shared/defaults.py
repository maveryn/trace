"""Scene-level defaults for marble-chain game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "marble_chain"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "semicircle_track",
    "spiral_track",
    "double_arc_track",
)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic_track",
    "arcade_track",
    "neon_track",
    "chalk_track",
    "copper_track",
)
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
COLOR_KEYS: Tuple[str, ...] = ("red", "blue", "green", "yellow", "purple", "orange")


@dataclass(frozen=True)
class MarbleDefaults:
    """Stable fallback defaults for visible marble-chain scenes."""

    chain_length_support: Tuple[int, ...] = (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
    color_count_support: Tuple[int, ...] = (4, 5, 6)
    option_count_support: Tuple[int, ...] = (4, 5, 6)
    target_pop_count_support: Tuple[int, ...] = (0, 2, 3, 4, 5)
    canvas_width: int = 900
    canvas_height: int = 760
    panel_margin_px: int = 36
    track_panel_top_px: int = 36
    track_panel_height_px: int = 688
    marble_radius_px: int = 28
    track_width_px: int = 26
    label_font_size_px: int = 19


DEFAULTS = MarbleDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "COLOR_KEYS",
    "DEFAULTS",
    "OPTION_LABELS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "MarbleDefaults",
]
