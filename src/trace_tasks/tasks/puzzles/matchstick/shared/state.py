"""Passive state, constants, and dataclasses for matchstick puzzles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


DOMAIN = "puzzles"
SCENE_ID = "matchstick"
SCENE_VARIANTS: Tuple[str, ...] = (
    "wooden_matches",
    "colored_rods",
    "chalk_sticks",
    "neon_rods",
    "metal_rods",
)
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")

Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Segment = Tuple[Point, Point]


@dataclass(frozen=True)
class RenderParams:
    """Resolved render dimensions for one matchstick scene."""

    canvas_width: int
    canvas_height: int
    margin_px: int
    source_panel_height_px: int
    option_panel_width_px: int
    option_panel_height_px: int
    option_gap_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    stick_width_px: int
    option_label_font_size_px: int
    caption_font_size_px: int
    source_caption_font_size_px: int


@dataclass(frozen=True)
class OptionSpec:
    """One labeled candidate panel."""

    label: str
    is_correct: bool
    value: Any
    metric_value: int | None = None


@dataclass(frozen=True)
class NumberDataset:
    """Concrete number-transform instance before rendering."""

    scene_variant: str
    source_number: int
    answer_number: int
    answer_label: str
    option_count: int
    option_specs: Tuple[OptionSpec, ...]
    changed_digit_index: int
    removed_segment_keys: Tuple[str, ...]
    added_segment_keys: Tuple[str, ...]


@dataclass(frozen=True)
class EquationRepairDataset:
    """Concrete remove-one-stick equation instance before rendering."""

    scene_variant: str
    operator: str
    source_digits: Tuple[int, int, int]
    repaired_digits: Tuple[int, int, int]
    answer_label: str
    option_count: int
    option_specs: Tuple[OptionSpec, ...]
    repair_stick_id: str
    repair_digit_index: int
    repair_segment_key: str
    all_removal_outcomes: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class SquareCompletionDataset:
    """Concrete square-lattice instance before rendering."""

    scene_variant: str
    rows: int
    cols: int
    add_count: int
    answer_value: int
    answer_label: str
    option_count: int
    present_edges: Tuple[str, ...]
    missing_edges: Tuple[str, ...]
    initial_completed_square_ids: Tuple[str, ...]
    completed_square_ids: Tuple[str, ...]
    optimal_added_edges: Tuple[str, ...]
    optimal_added_edge_sets: Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class RenderedScene:
    """Rendered matchstick image plus item projections."""

    image: Any
    scene_bbox_px: BBox
    item_bbox_map: Dict[str, BBox]
    entities: Tuple[Dict[str, Any], ...]
    item_segment_map: Dict[str, Segment] = field(default_factory=dict)


__all__ = [
    "BBox",
    "Color",
    "DOMAIN",
    "EquationRepairDataset",
    "NumberDataset",
    "OPTION_LABELS",
    "OptionSpec",
    "Point",
    "RenderParams",
    "RenderedScene",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "Segment",
    "SquareCompletionDataset",
]
