"""Scene defaults for space-shooter game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import SCENE_ID


@dataclass(frozen=True)
class SpaceShooterDefaults:
    """Stable fallback defaults for visible space-shooter scenes."""

    enemy_ship_hit_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    hit_enemy_ship_label_option_support: Tuple[int, ...] = (0, 1, 2, 3)
    first_hit_enemy_ship_label_option_support: Tuple[int, ...] = (0, 1, 2, 3)
    safe_lane_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    lane_count_support: Tuple[int, ...] = (4, 5, 6, 7, 8)
    enemy_count_support: Tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9, 10)
    canvas_width: int = 1060
    canvas_height: int = 820
    panel_margin_px: int = 42
    playfield_width_px: int = 940
    playfield_height_px: int = 720
    playfield_border_width_px: int = 5
    lane_pad_height_px: int = 38
    lane_pad_gap_px: int = 10
    enemy_width_px: int = 62
    enemy_height_px: int = 48
    projectile_width_px: int = 24
    projectile_height_px: int = 36
    enemy_projectile_per_lane_support: Tuple[int, ...] = (1, 2, 3)
    player_ship_width_px: int = 72
    player_ship_height_px: int = 58
    label_font_size_px: int = 24


DEFAULTS = SpaceShooterDefaults()
SCENE_DEFAULTS = get_scene_defaults("games", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)
