"""Scene constants and fallback defaults for Minesweeper games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults


SCENE_ID = "minesweeper"
DEFAULT_BRANCH_ID = SINGLE_QUERY_ID
SCENE_VARIANTS: Tuple[str, ...] = ("open_grid", "mixed_grid")


@dataclass(frozen=True)
class MinesweeperDefaults:
    """Stable fallback defaults for visible Minesweeper board rendering."""

    open_min_distractor_hidden_count: int = 2
    open_max_distractor_hidden_count: int = 4
    mixed_min_distractor_hidden_count: int = 4
    mixed_max_distractor_hidden_count: int = 8
    canvas_width: int = 900
    canvas_height: int = 900
    panel_margin_px: int = 56
    max_board_size_px: int = 720
    board_border_width_px: int = 5
    grid_line_width_px: int = 2
    cell_padding_px: int = 6
    number_font_size_px: int = 48


DEFAULTS = MinesweeperDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = [
    "DEFAULTS",
    "DEFAULT_BRANCH_ID",
    "MinesweeperDefaults",
    "POST_IMAGE_NOISE_DEFAULTS",
    "SCENE_ID",
    "SCENE_VARIANTS",
]
