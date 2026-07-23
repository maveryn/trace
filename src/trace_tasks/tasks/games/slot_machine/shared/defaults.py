"""Scene-level defaults for slot-machine games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "slot_machine"
SCENE_NAMESPACE = "games.slot_machine"
REEL_COUNT = 3
ROW_COUNT = 3
PAYLINE_IDS: Tuple[str, ...] = ("row_0", "row_1", "row_2", "diag_down", "diag_up")
PAYLINE_COORDS: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2)),
    ((1, 0), (1, 1), (1, 2)),
    ((2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 1), (2, 2)),
    ((2, 0), (1, 1), (0, 2)),
)
SYMBOL_KEYS: Tuple[str, ...] = ("seven", "bar", "gem", "star", "cherry", "coin")
PAYTABLE_SCORE_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("front_cabinet",)
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic_red",
    "chrome_blue",
    "neon_night",
    "candy_arcade",
    "paper_ticket",
)
WINNING_PAYLINE_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
SCORE_TASK_WINNING_PAYLINE_COUNT_SUPPORT: Tuple[int, ...] = (1, 2)


@dataclass(frozen=True)
class SlotMachineDefaults:
    """Stable fallback defaults for slot-machine scenes."""

    canvas_width: int = 760
    canvas_height: int = 680
    cabinet_width_px: int = 540
    cabinet_height_px: int = 540
    reel_cell_width_px: int = 104
    reel_cell_height_px: int = 106
    reel_gap_px: int = 10
    row_gap_px: int = 10
    cabinet_pad_px: int = 34
    label_font_size_px: int = 24
    symbol_font_size_px: int = 28


DEFAULTS = SlotMachineDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.45)


__all__ = [
    "DEFAULTS",
    "PAYLINE_COORDS",
    "PAYLINE_IDS",
    "PAYTABLE_SCORE_VALUES",
    "POST_IMAGE_NOISE_DEFAULTS",
    "REEL_COUNT",
    "ROW_COUNT",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCORE_TASK_WINNING_PAYLINE_COUNT_SUPPORT",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "SYMBOL_KEYS",
    "SlotMachineDefaults",
    "WINNING_PAYLINE_COUNT_SUPPORT",
]
