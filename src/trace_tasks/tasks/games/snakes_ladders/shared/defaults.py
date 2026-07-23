"""Scene defaults for Snakes and Ladders tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults

from .state import SCENE_ID, SUPPORTED_BOARD_SIDES, SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS


@dataclass(frozen=True)
class SnakesLaddersDefaults:
    """Stable fallback defaults for visible Snakes and Ladders scenes."""

    board_side_support: Tuple[int, ...] = SUPPORTED_BOARD_SIDES
    style_variant_support: Tuple[str, ...] = SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS
    scene_variant_support: Tuple[str, ...] = ("standard_board",)
    canvas_width: int = 1000
    canvas_height: int = 760
    board_left_px: int = 48
    board_top_px: int = 56
    board_size_px: int = 660
    side_panel_width_px: int = 200
    cell_gap_px: int = 4
    cell_radius_px: int = 6
    number_font_size_px: int = 48
    token_radius_px: int = 19
    die_size_px: int = 88
    jump_width_px: int = 6
    move_outcome_jump_probability: float = 0.30


DEFAULTS = SnakesLaddersDefaults()
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "DEFAULTS",
    "GEN_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "RENDER_DEFAULTS",
    "SnakesLaddersDefaults",
]
