"""Passive state and ids for sheet-transform puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

DOMAIN = "puzzles"
SCENE_ID = "sheet_transform"

FOLD_SCENE_VARIANTS: tuple[str, ...] = (
    "fold_strip",
    "fold_card",
    "fold_outline",
)
OVERLAY_SCENE_VARIANTS: tuple[str, ...] = (
    "overlay_strip",
    "overlay_card",
    "overlay_outline",
)
SUPPORTED_FOLD_AXES: tuple[str, ...] = ("vertical", "horizontal")
SUPPORTED_FOLD_COUNTS: tuple[int, ...] = (1, 2)
SUPPORTED_CUT_HOLE_SHAPES: tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "rounded_square",
)
FOLD_MARK_TYPES: tuple[str, ...] = (
    "circle",
    "diamond",
    "square",
    "hexagon",
    "star",
)
OVERLAY_MARK_SHAPES: tuple[str, ...] = (
    "circle",
    "square",
    "diamond",
    "rounded_square",
)

Cells = Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class PaperFoldDataset:
    """Sampled fold-projection puzzle data before rendering."""

    grid_size: int
    result_grid_cols: int
    result_grid_rows: int
    option_count: int
    option_count_range: tuple[int, int]
    mark_count: int
    mark_count_range: tuple[int, int]
    fold_axis: str
    fold_direction: str
    original_mark_specs: tuple[Dict[str, Any], ...]
    folded_result_mark_specs: tuple[Dict[str, Any], ...]
    option_specs: tuple[Dict[str, Any], ...]
    answer_option_label: str
    correct_option_index: int
    correct_option_choice_id: str
    folded_mark_count: int
    kept_mark_count: int
    option_count_probabilities: Dict[str, float]
    correct_option_index_probabilities: Dict[str, float]
    correct_option_index_sampling_mode: str


@dataclass(frozen=True)
class PaperFoldCutDataset:
    """Sampled fold-cut puzzle data before rendering."""

    internal_grammar_id: str
    grid_size: int
    folded_grid_cols: int
    folded_grid_rows: int
    fold_sequence: tuple[Dict[str, Any], ...]
    fold_count: int
    folded_dimensions_by_step: tuple[tuple[int, int], ...]
    cut_count: int
    cut_count_range: tuple[int, int]
    cut_cells: Cells
    cut_specs: tuple[Dict[str, Any], ...]
    unfolded_hole_cells: Cells
    unfolded_hole_specs: tuple[Dict[str, Any], ...]
    unfolded_hole_count: int
    option_count: int
    option_count_range: tuple[int, int]
    option_specs: tuple[Dict[str, Any], ...]
    answer_option_label: str
    correct_option_index: int
    correct_option_choice_id: str
    valid_option_choice_ids: tuple[str, ...]
    option_count_probabilities: Dict[str, float]
    fold_count_probabilities: Dict[str, float]
    fold_axis_probabilities: Dict[str, float]
    cut_count_probabilities: Dict[str, float]
    correct_option_index_probabilities: Dict[str, float]
    correct_option_index_sampling_mode: str


@dataclass(frozen=True)
class OverlayDataset:
    """Sampled transparent-sheet overlay data before rendering."""

    grid_size: int
    grid_size_range: tuple[int, int]
    option_count: int
    option_count_range: tuple[int, int]
    sheet_mark_count_range: tuple[int, int]
    overlap_count_range: tuple[int, int]
    left_cells: Cells
    right_cells: Cells
    overlap_cells: Cells
    union_cells: Cells
    left_mark_specs: tuple[Dict[str, Any], ...]
    right_mark_specs: tuple[Dict[str, Any], ...]
    left_mark_count: int
    right_mark_count: int
    overlap_count: int
    union_mark_count: int
    option_specs: tuple[Dict[str, Any], ...]
    answer_option_label: str
    correct_option_index: int
    correct_option_choice_id: str
    option_count_probabilities: Dict[str, float]
    grid_size_probabilities: Dict[str, float]
    correct_option_index_probabilities: Dict[str, float]
    correct_option_index_sampling_mode: str


__all__ = [
    "Cells",
    "DOMAIN",
    "FOLD_MARK_TYPES",
    "FOLD_SCENE_VARIANTS",
    "OVERLAY_MARK_SHAPES",
    "OVERLAY_SCENE_VARIANTS",
    "OverlayDataset",
    "PaperFoldCutDataset",
    "PaperFoldDataset",
    "SCENE_ID",
    "SUPPORTED_CUT_HOLE_SHAPES",
    "SUPPORTED_FOLD_AXES",
    "SUPPORTED_FOLD_COUNTS",
]
