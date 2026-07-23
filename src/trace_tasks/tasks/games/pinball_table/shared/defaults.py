"""Scene-level defaults for pinball-table games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "pinball_table"
OBJECT_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(8))
PATH_SCORE_VALUES: Tuple[int, ...] = (10, 20, 30, 50)

BUMPER_RADIUS_NORM = 0.045
STANDUP_RADIUS_NORM = 0.034
DROP_TARGET_WIDTH_NORM = 0.092
DROP_TARGET_HEIGHT_NORM = 0.060
ROLLOVER_WIDTH_NORM = 0.135
ROLLOVER_HEIGHT_NORM = 0.052


@dataclass(frozen=True)
class PinballDefaults:
    """Stable fallback defaults for pinball-table scenes."""

    object_count_support: Tuple[int, ...] = (5, 6, 7, 8)
    canvas_width: int = 900
    canvas_height: int = 760
    panel_margin_px: int = 28
    table_width_px: int = 700
    table_height_px: int = 680
    table_border_width_px: int = 7
    ball_radius_px: int = 17
    bumper_radius_px: int = 32
    target_width_px: int = 72
    target_height_px: int = 34
    cue_width_px: int = 6
    label_font_size_px: int = 26


DEFAULTS = PinballDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "BUMPER_RADIUS_NORM",
    "DEFAULTS",
    "DROP_TARGET_HEIGHT_NORM",
    "DROP_TARGET_WIDTH_NORM",
    "OBJECT_LABELS",
    "PATH_SCORE_VALUES",
    "POST_IMAGE_NOISE_DEFAULTS",
    "ROLLOVER_HEIGHT_NORM",
    "ROLLOVER_WIDTH_NORM",
    "SCENE_ID",
    "STANDUP_RADIUS_NORM",
    "PinballDefaults",
]
