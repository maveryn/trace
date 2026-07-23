"""Passive state and constants for color-gradient puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

DOMAIN = "puzzles"
SCENE_ID = "color_gradient"

GRID_SIZE_VARIANTS: Tuple[str, ...] = ("3x3", "4x4")
RULE_VARIANTS: Tuple[str, ...] = (
    "column_hue_row_lightness",
    "row_hue_column_lightness",
    "column_hue_row_saturation",
)
COMPLETION_LENGTH_VARIANTS: Tuple[str, ...] = ("5_cell", "6_cell", "7_cell")
COMPLETION_OPTION_COUNT_VARIANTS: Tuple[str, ...] = (
    "4_options",
    "5_options",
    "6_options",
)
COMPLETION_RULE_VARIANTS: Tuple[str, ...] = (
    "hue_gradient",
    "lightness_gradient",
    "hue_lightness_gradient",
)
SCENE_VARIANTS: Tuple[str, ...] = (
    "swatch_clean",
    "swatch_card",
    "swatch_notebook",
)
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class RenderParams:
    """Resolved rendering parameters for one color-gradient image."""

    canvas_width: int
    canvas_height: int
    swatch_size_px: int
    swatch_gap_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    swatch_corner_radius_px: int
    swatch_border_width_px: int
    label_chip_size_px: int
    label_margin_px: int
    label_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    swatch_border_rgb: Tuple[int, int, int]
    notebook_line_rgb: Tuple[int, int, int]
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class CellSpec:
    """One swatch-grid cell before rendering."""

    cell_id: str
    label: str
    row: int
    col: int
    expected_hsl: Tuple[float, float, float]
    observed_hsl: Tuple[float, float, float]
    expected_rgb: Tuple[int, int, int]
    observed_rgb: Tuple[int, int, int]
    is_violation: bool


@dataclass(frozen=True)
class ViolationDataset:
    """Sampled color-gradient violation puzzle dataset."""

    rows: int
    cols: int
    grid_size_variant: str
    grid_size_variant_probabilities: Dict[str, float]
    rule_variant: str
    rule_variant_probabilities: Dict[str, float]
    rule_params: Dict[str, Any]
    cells: Tuple[CellSpec, ...]
    answer_label: str
    answer_label_probabilities: Dict[str, float]
    violation_cell_id: str
    violation_index: int
    borrowed_from_label: str


@dataclass(frozen=True)
class CompletionCellSpec:
    """One cell in the visible linear-gradient completion strip."""

    cell_id: str
    index: int
    expected_hsl: Tuple[float, float, float]
    expected_rgb: Tuple[int, int, int]
    is_missing: bool


@dataclass(frozen=True)
class CompletionOptionSpec:
    """One labeled answer option for the linear-gradient completion task."""

    option_id: str
    label: str
    hsl: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    is_correct: bool


@dataclass(frozen=True)
class CompletionDataset:
    """Sampled linear-gradient completion puzzle dataset."""

    sequence_length: int
    sequence_length_variant: str
    sequence_length_variant_probabilities: Dict[str, float]
    option_count: int
    option_count_variant: str
    option_count_variant_probabilities: Dict[str, float]
    rule_variant: str
    rule_variant_probabilities: Dict[str, float]
    rule_params: Dict[str, Any]
    cells: Tuple[CompletionCellSpec, ...]
    options: Tuple[CompletionOptionSpec, ...]
    missing_index: int
    missing_index_probabilities: Dict[str, float]
    missing_cell_id: str
    answer_label: str
    answer_label_probabilities: Dict[str, float]
    correct_option_id: str


@dataclass(frozen=True)
class RenderedScene:
    """Rendered color-gradient image and trace maps."""

    image: Image.Image
    scene_bbox_px: Tuple[int, int, int, int]
    cell_bbox_map: Dict[str, Tuple[int, int, int, int]]
    item_bbox_map: Dict[str, Tuple[int, int, int, int]]
    entities: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class Defaults:
    """Stable fallback defaults for color-gradient rendering and sampling."""

    canvas_width: int = 900
    canvas_height: int = 760
    swatch_size_px: int = 112
    swatch_gap_px: int = 14
    panel_padding_px: int = 34
    panel_corner_radius_px: int = 26
    panel_border_width_px: int = 3
    swatch_corner_radius_px: int = 12
    swatch_border_width_px: int = 3
    label_chip_size_px: int = 32
    label_margin_px: int = 9
    label_font_size_px: int = 18
    panel_fill_rgb: Tuple[int, int, int] = (250, 251, 253)
    panel_border_rgb: Tuple[int, int, int] = (88, 96, 110)
    swatch_border_rgb: Tuple[int, int, int] = (40, 46, 58)
    notebook_line_rgb: Tuple[int, int, int] = (218, 226, 236)
    hue_step_min: float = 34.0
    hue_step_max: float = 58.0
    lightness_step_min: float = 0.060
    lightness_step_max: float = 0.090
    saturation_step_min: float = 0.070
    saturation_step_max: float = 0.110


DEFAULTS = Defaults()


__all__ = [
    "COMPLETION_LENGTH_VARIANTS",
    "COMPLETION_OPTION_COUNT_VARIANTS",
    "COMPLETION_RULE_VARIANTS",
    "DEFAULTS",
    "DOMAIN",
    "GRID_SIZE_VARIANTS",
    "LABELS",
    "RULE_VARIANTS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "CellSpec",
    "CompletionCellSpec",
    "CompletionDataset",
    "CompletionOptionSpec",
    "Defaults",
    "RenderedScene",
    "RenderParams",
    "ViolationDataset",
]
