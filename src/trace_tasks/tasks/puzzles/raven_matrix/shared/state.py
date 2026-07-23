"""Scene constants and passive state for Raven-matrix puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DOMAIN = "puzzles"
SCENE_ID = "raven_matrix"
PROMPT_BUNDLE_ID = "puzzles_raven_matrix_v1"
PROMPT_SCENE_KEY = "raven_matrix"

SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    "raven_strip",
    "raven_card",
    "raven_outline",
)


@dataclass(frozen=True)
class RavenAxes:
    """Resolved presentation axes shared by one Raven-matrix sample."""

    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    option_count: int
    answer_option_index: int
    answer_option_probabilities: dict[str, float]


@dataclass(frozen=True)
class RavenRenderParams:
    """Resolved render parameters for one Raven-matrix scene."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    cell_size_px: int
    cell_gap_px: int
    board_panel_padding_px: int
    board_to_options_gap_px: int
    option_panel_width_px: int
    option_panel_height_px: int
    option_gap_px: int
    option_symbol_box_size_px: int
    option_label_gap_px: int
    slot_corner_radius_px: int
    border_width_px: int
    panel_corner_radius_px: int
    value_font_size_px: int
    option_label_font_size_px: int
    panel_fill_rgb: tuple[int, int, int]
    cell_fill_rgb: tuple[int, int, int]
    unknown_cell_fill_rgb: tuple[int, int, int]
    option_panel_fill_rgb: tuple[int, int, int]
    option_symbol_fill_rgb: tuple[int, int, int]
    border_color_rgb: tuple[int, int, int]
    text_color_rgb: tuple[int, int, int]
    text_stroke_rgb: tuple[int, int, int]
    accent_color_rgb: tuple[int, int, int]
    unit_size_jitter: dict[str, Any]


@dataclass(frozen=True)
class RenderedRavenScene:
    """Rendered Raven matrix image plus traced matrix and option geometry."""

    image: Any
    entities: list[dict[str, Any]]
    scene_bbox_px: list[float]
    matrix_cell_bbox_map: dict[str, list[float]]
    option_panel_bbox_map: dict[str, list[float]]
    option_cell_bbox_map: dict[str, list[float]]


__all__ = [
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "PROMPT_SCENE_KEY",
    "RavenAxes",
    "RavenRenderParams",
    "RenderedRavenScene",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
]
