"""Darts scene defaults and prompt wiring constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "darts"
DARTS_NAMESPACE = "games.darts"
PROMPT_WIRING_KEYS: Tuple[str, ...] = ("bundle_id", "scene_key", "task_key")
SUPPORTED_DARTS_SCENE_VARIANTS: Tuple[str, ...] = ("single_board",)
BULLSEYE_SCORE = 50
STANDARD_DART_SECTORS: Tuple[int, ...] = (
    10,
    1,
    9,
    2,
    8,
    3,
    7,
    4,
    6,
    5,
)


@dataclass(frozen=True)
class DartsSceneDefaults:
    """Stable scene fallback defaults for simplified dartboard tasks."""

    score_value_support: Tuple[int, ...] = STANDARD_DART_SECTORS + (BULLSEYE_SCORE,)
    count_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    count_query_dart_count_support: Tuple[int, ...] = (4, 5, 6, 7)
    canvas_width: int = 1040
    canvas_height: int = 900
    board_center_x_px: int = 520
    board_center_y_px: int = 430
    board_radius_px: int = 330
    marker_radius_px: int = 14
    number_font_size_px: int = 36


DEFAULTS = DartsSceneDefaults()


__all__ = [
    "BULLSEYE_SCORE",
    "DARTS_NAMESPACE",
    "DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "SCENE_ID",
    "STANDARD_DART_SECTORS",
    "SUPPORTED_DARTS_SCENE_VARIANTS",
    "DartsSceneDefaults",
]
