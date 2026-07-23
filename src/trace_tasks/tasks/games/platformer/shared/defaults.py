"""Scene-local constants and fallback defaults for platformer tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "platformer"
PLATFORM_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(8))
HAZARD_KINDS: Tuple[str, ...] = ("spikes", "patrol")
BONUS_COLLECTIBLE_KINDS: Tuple[str, ...] = ("gem", "star")
SUPPORTED_PLATFORMER_SCENE_VARIANTS: Tuple[str, ...] = ("side_scroller",)
SUPPORTED_PLATFORMER_STYLE_VARIANTS: Tuple[str, ...] = (
    "day",
    "cave",
    "neon",
    "snow",
    "sunset",
)


@dataclass(frozen=True)
class PlatformerDefaults:
    """Stable fallback defaults for visible platformer scenes."""

    platform_count_support: Tuple[int, ...] = (4, 5, 6, 7)
    hazard_count_support: Tuple[int, ...] = (4, 5, 6, 7, 8)
    distractor_collectible_count_support: Tuple[int, ...] = (4, 5, 6, 7, 8)
    target_platform_label_support: Tuple[str, ...] = PLATFORM_LABELS
    target_collectible_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    score_on_arc_coin_count_support: Tuple[int, ...] = (1, 2, 3, 4)
    score_on_arc_bonus_count_support: Tuple[int, ...] = (1, 2)
    score_off_arc_bonus_count_support: Tuple[int, ...] = (1, 2, 3)
    score_bonus_value_support: Tuple[int, ...] = (2, 3, 4)
    jump_visible_after_peak_min: float = 0.08
    jump_visible_after_peak_max: float = 0.14
    canvas_width: int = 1000
    canvas_height: int = 740
    level_width_px: int = 860
    level_height_px: int = 610
    level_border_width_px: int = 5
    platform_height_px: int = 34
    player_width_px: int = 38
    player_height_px: int = 58
    hazard_width_px: int = 54
    hazard_height_px: int = 54
    collectible_radius_px: int = 18
    path_width_px: int = 6
    label_font_size_px: int = 24


DEFAULTS = PlatformerDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "BONUS_COLLECTIBLE_KINDS",
    "DEFAULTS",
    "HAZARD_KINDS",
    "PLATFORM_LABELS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SUPPORTED_PLATFORMER_SCENE_VARIANTS",
    "SUPPORTED_PLATFORMER_STYLE_VARIANTS",
    "PlatformerDefaults",
]
