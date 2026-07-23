"""Scene constants and fallback defaults for pool-table tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "pool"


@dataclass(frozen=True)
class PoolDefaults:
    """Stable fallback defaults for visible pool-table scenes."""

    current_group_ball_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    blocking_ball_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    object_ball_count_support: Tuple[int, ...] = (7, 8, 9, 10)
    line_clearance: float = 0.055
    min_ball_distance: float = 0.075
    canvas_width: int = 1120
    canvas_height: int = 760
    panel_margin_px: int = 42
    table_width_px: int = 940
    table_height_px: int = 520
    rail_width_px: int = 44
    pocket_radius_px: int = 24
    ball_radius_px: int = 18
    ball_number_font_size_px: int = 15
    badge_font_size_px: int = 22


DEFAULTS = PoolDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


__all__ = [
    "DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PoolDefaults",
    "SCENE_ID",
]
