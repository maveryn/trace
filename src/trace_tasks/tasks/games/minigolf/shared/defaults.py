"""Scene constants and fallback defaults for Mini-golf games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "minigolf"
DEFAULT_BRANCH_ID = SINGLE_QUERY_ID
FIRST_OBSTACLE_MODE = "first_obstacle"
SHOT_OPTIONS_MODE = "shot_options"
SCENE_VARIANTS: Tuple[str, ...] = ("putting_course",)
STYLE_VARIANTS: Tuple[str, ...] = ("classic", "desert", "neon", "garden", "blueprint")
OBSTACLE_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(6))
OBSTACLE_KINDS: Tuple[str, ...] = ("rock", "sand", "water", "block")
SHOT_MODES: Tuple[str, ...] = ("direct", "bank_left", "bank_right", "bank_top")
OBSTACLE_RADIUS_NORM = 0.045
HOLE_RADIUS_NORM = 0.038
BALL_RADIUS_NORM = 0.026


@dataclass(frozen=True)
class MinigolfDefaults:
    """Stable scene/render fallback defaults for visible Mini-golf scenes."""

    obstacle_count_support: Tuple[int, ...] = (4, 6)
    canvas_width: int = 1000
    canvas_height: int = 740
    panel_margin_px: int = 34
    course_width_px: int = 820
    course_height_px: int = 640
    course_border_width_px: int = 7
    ball_radius_px: int = 18
    hole_radius_px: int = 20
    obstacle_radius_px: int = 34
    path_width_px: int = 6
    label_font_size_px: int = 24


DEFAULTS = MinigolfDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "BALL_RADIUS_NORM",
    "DEFAULTS",
    "DEFAULT_BRANCH_ID",
    "FIRST_OBSTACLE_MODE",
    "HOLE_RADIUS_NORM",
    "MinigolfDefaults",
    "OBSTACLE_KINDS",
    "OBSTACLE_LABELS",
    "OBSTACLE_RADIUS_NORM",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SHOT_MODES",
    "SHOT_OPTIONS_MODE",
    "STYLE_VARIANTS",
]
