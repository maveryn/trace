"""Passive constants and dataclasses for cyclic-order puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


DOMAIN = "puzzles"
SCENE_ID = "cyclic_order"

SCENE_VARIANTS: Tuple[str, ...] = (
    "necklace_board",
    "charm_card_grid",
    "route_loop_diagram",
    "token_ring_outline",
)
SOURCE_SCENE_VARIANT_MAP = {
    "loop_strip": "necklace_board",
    "loop_card": "charm_card_grid",
    "loop_outline": "token_ring_outline",
}

TOKEN_RENDER_STYLES: Tuple[str, ...] = (
    "colored_beads",
    "shape_tokens",
    "colored_shape_tokens",
    "outline_shape_tokens",
    "symbol_badges",
)
SOURCE_TOKEN_MODE_MAP = {
    "color": "colored_beads",
    "shape": "shape_tokens",
    "mixed": "colored_shape_tokens",
}
TOKEN_STYLE_SOURCE_MODE = {
    "colored_beads": "color",
    "shape_tokens": "shape",
    "colored_shape_tokens": "mixed",
    "outline_shape_tokens": "shape",
    "symbol_badges": "mixed",
}
TOKEN_STYLE_PROMPT_INSTRUCTION = {
    "colored_beads": "Use the token colors when comparing cyclic order.",
    "shape_tokens": "Use the token shapes when comparing cyclic order.",
    "colored_shape_tokens": "Use both token color and token shape when comparing cyclic order.",
    "outline_shape_tokens": "Use the outlined token shapes when comparing cyclic order.",
    "symbol_badges": "Use both the badge symbol and badge color when comparing cyclic order.",
}

LOOP_PATH_STYLES: Tuple[str, ...] = (
    "ellipse",
    "rounded_rect",
    "polygon_loop",
    "wavy_loop",
    "beaded_string",
)
LOOP_SHAPE_VARIANTS: Tuple[str, ...] = (
    "circle",
    "wide",
    "tall",
)
LOOP_START_ANGLES_DEG: Tuple[int, ...] = (-90, -45, 0, 45, 90, 135, 180)

TOKEN_COLOR_SPECS: Tuple[Tuple[str, Tuple[int, int, int]], ...] = (
    ("sky", (64, 175, 225)),
    ("forest", (46, 95, 60)),
    ("violet", (83, 54, 154)),
    ("lime", (185, 225, 30)),
    ("sand", (208, 144, 98)),
    ("magenta", (205, 85, 138)),
    ("mint", (86, 225, 142)),
    ("crimson", (222, 42, 33)),
)


@dataclass(frozen=True)
class CyclicOrderDefaults:
    """Stable code fallbacks for cyclic-order generation bounds."""

    option_count_min: int = 4
    option_count_max: int = 4
    bead_count_min: int = 4
    bead_count_max: int = 5
    shape_bead_count_max: int = 5
    min_color_distance: float = 50.0
    color_distance_space: str = "lab"


@dataclass(frozen=True)
class CyclicOrderRenderParams:
    """Resolved rendering parameters for cyclic-order puzzle scenes."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    reference_panel_height_px: int
    reference_panel_padding_px: int
    reference_loop_width_px: int
    reference_loop_height_px: int
    reference_label_font_size_px: int
    reference_to_options_gap_px: int
    option_image_width_px: int
    option_image_height_px: int
    option_gap_px: int
    option_row_gap_px: int
    option_label_gap_px: int
    option_label_font_size_px: int
    panel_corner_radius_px: int
    border_width_px: int
    loop_stroke_width_px: int
    bead_size_px: int
    shape_bead_inset_px: int
    panel_fill_rgb: Tuple[int, int, int]
    instruction_fill_rgb: Tuple[int, int, int]
    border_color_rgb: Tuple[int, int, int]
    loop_color_rgb: Tuple[int, int, int]
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    shape_fill_rgb: Tuple[int, int, int]


DEFAULTS = CyclicOrderDefaults()


__all__ = [
    "DEFAULTS",
    "DOMAIN",
    "LOOP_PATH_STYLES",
    "LOOP_SHAPE_VARIANTS",
    "LOOP_START_ANGLES_DEG",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SOURCE_SCENE_VARIANT_MAP",
    "SOURCE_TOKEN_MODE_MAP",
    "TOKEN_COLOR_SPECS",
    "TOKEN_RENDER_STYLES",
    "TOKEN_STYLE_PROMPT_INSTRUCTION",
    "TOKEN_STYLE_SOURCE_MODE",
    "CyclicOrderDefaults",
    "CyclicOrderRenderParams",
]
