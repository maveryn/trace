"""Passive state for symbolic radial code-wheel scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


CODE_SYMBOLS: tuple[str, ...] = ("A", "B", "C", "D")
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SUPPORTED_RADIAL_SCENE_VARIANTS: tuple[str, ...] = ("clean_wheel", "notebook_wheel", "exam_scan")


@dataclass(frozen=True)
class RadialReferenceSpec:
    """A source/target card displayed next to the wheel."""

    item_id: str
    title: str
    value: str
    role: str


@dataclass(frozen=True)
class RadialOptionSpec:
    """One visible multiple-choice card."""

    item_id: str
    label: str
    value: str
    role: str


@dataclass(frozen=True)
class RadialTerminalSpec:
    """One terminal sector label on the wheel."""

    item_id: str
    code: str
    output_label: str
    terminal_index: int


@dataclass(frozen=True)
class RadialChoiceDataset:
    """Resolved radial code-wheel data for one option task."""

    scene_variant: str
    answer_value: str
    target_answer_support: tuple[str, ...]
    reference: RadialReferenceSpec
    options: tuple[RadialOptionSpec, ...]
    terminal_specs: tuple[RadialTerminalSpec, ...]
    target_code: str
    target_output_label: str
    target_terminal_index: int
    annotation_item_ids: tuple[str, str, str]
    metadata: dict[str, Any]
    scene_variant_probabilities: dict[str, float]


@dataclass(frozen=True)
class RadialMissingSymbolDataset:
    """Resolved radial code-wheel data for one missing-symbol task."""

    scene_variant: str
    answer_value: str
    target_answer_support: tuple[str, ...]
    target_output_label: str
    target_code: str
    partial_code: str
    missing_position_index: int
    missing_ring_role: str
    terminal_specs: tuple[RadialTerminalSpec, ...]
    annotation_item_id: str
    metadata: dict[str, Any]
    scene_variant_probabilities: dict[str, float]


@dataclass(frozen=True)
class RadialRenderParams:
    """Concrete render parameters for the radial code-wheel scene."""

    canvas_width: int = 1080
    canvas_height: int = 860
    wheel_center_x_px: int = 380
    wheel_center_y_px: int = 430
    wheel_inner_radius_px: int = 44
    ring_width_px: int = 80
    terminal_label_radius_px: int = 326
    source_card_left_px: int = 760
    source_card_top_px: int = 58
    source_card_width_px: int = 260
    source_card_height_px: int = 108
    missing_code_card_top_px: int = 184
    option_card_width_px: int = 126
    option_card_height_px: int = 82
    option_grid_left_px: int = 742
    option_grid_top_px: int = 248
    option_grid_col_gap_px: int = 28
    option_grid_row_gap_px: int = 42
    option_label_font_size_px: int = 24
    option_value_font_size_px: int = 28
    source_title_font_size_px: int = 20
    source_value_font_size_px: int = 34
    ring_symbol_font_size_px: int = 16
    inner_ring_symbol_font_size_px: int = 23
    outer_ring_symbol_font_size_px: int = 10
    terminal_label_font_size_px: int = 13
    ring_line_width_px: int = 2
    wheel_outline_width_px: int = 3
    card_corner_radius_px: int = 16
    card_border_width_px: int = 2


@dataclass(frozen=True)
class RenderedRadialScene:
    """Rendered radial code-wheel image and projection maps."""

    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    item_points: dict[str, list[float]]
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


__all__ = [
    "CODE_SYMBOLS",
    "OPTION_LABELS",
    "RadialMissingSymbolDataset",
    "SUPPORTED_RADIAL_SCENE_VARIANTS",
    "RadialChoiceDataset",
    "RadialOptionSpec",
    "RadialReferenceSpec",
    "RadialRenderParams",
    "RadialTerminalSpec",
    "RenderedRadialScene",
]
