"""Scene-level defaults for dots-and-boxes games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "dots_and_boxes"
DOTS_AND_BOXES_NAMESPACE = "games.dots_and_boxes"
SUPPORTED_DOTS_AND_BOXES_SCENE_VARIANTS: Tuple[str, ...] = ("single_board",)


@dataclass(frozen=True)
class DotsAndBoxesSceneDefaults:
    """Stable fallback defaults for visible dots-and-boxes boards."""

    three_sided_box_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    option_label_support: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    owned_box_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    box_rows_support: Tuple[int, ...] = (3, 4)
    box_cols_support: Tuple[int, ...] = (3, 4)
    canvas_width: int = 1180
    canvas_height: int = 820
    board_width_px: int = 880
    board_height_px: int = 640
    board_corner_radius_px: int = 24
    panel_margin_px: int = 56
    title_font_size_px: int = 34
    title_band_height_px: int = 62
    board_padding_px: int = 62
    dot_radius_px: int = 7
    dash_length_px: int = 30
    dash_gap_px: int = 18
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 620
    canvas_min_height_px: int = 520
    canvas_side_padding_px: int = 150
    canvas_vertical_padding_px: int = 110


DEFAULTS = DotsAndBoxesSceneDefaults()


__all__ = [
    "DEFAULTS",
    "DOTS_AND_BOXES_NAMESPACE",
    "SCENE_ID",
    "SUPPORTED_DOTS_AND_BOXES_SCENE_VARIANTS",
    "DotsAndBoxesSceneDefaults",
]
