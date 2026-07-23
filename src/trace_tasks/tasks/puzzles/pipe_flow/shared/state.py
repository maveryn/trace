"""Passive state and constants for pipe-flow repair puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "pipe_flow"
PROMPT_BUNDLE_ID = "puzzles_pipe_flow_v1"
PROMPT_SCENE_KEY = "pipe_flow"
PROMPT_TASK_KEY = "pipe_flow_repair_tile_label_query"

GRID_SIZE_VARIANTS: Tuple[str, ...] = ("5x5", "6x6", "7x7")
GAP_SIZE_VARIANTS: Tuple[str, ...] = ("2x2",)
SCENE_VARIANTS: Tuple[str, ...] = (
    "water_pipe",
    "circuit_trace",
    "industrial_conduit",
)
LABEL_POOL = tuple("ABCDEF")
DIRECTIONS: Tuple[str, ...] = ("N", "E", "S", "W")
DELTAS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
ROTATE_CW = {"N": "E", "E": "S", "S": "W", "W": "N"}

Cell = Tuple[int, int]
Openings = Tuple[str, ...]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class RenderParams:
    """Resolved pipe-grid render parameters."""

    canvas_width: int
    canvas_height: int
    scene_margin_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    cell_gap_px: int
    cell_size_min_px: int
    cell_size_max_px: int
    cell_border_width_px: int
    pipe_width_px: int
    source_dest_font_size_px: int
    tile_label_font_size_px: int
    panel_fill_rgb: Color
    cell_fill_rgb: Color
    grid_line_rgb: Color
    pipe_rgb: Color
    pipe_shadow_rgb: Color
    source_fill_rgb: Color
    source_outline_rgb: Color
    dest_fill_rgb: Color
    dest_outline_rgb: Color
    label_fill_rgb: Color
    label_text_rgb: Color
    text_stroke_rgb: Color
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class TileSpec:
    """One visible pipe-grid tile."""

    tile_id: str
    row: int
    col: int
    required_openings: Openings
    current_openings: Openings
    is_path: bool
    is_branch: bool
    label: str


@dataclass(frozen=True)
class OptionSpec:
    """One labeled replacement-piece option."""

    option_id: str
    label: str
    local_openings: Tuple[Tuple[int, int, Openings], ...]
    is_correct: bool
    connects_in_place: bool
    connects_after_rotation_turns: Tuple[int, ...]
    display_rotation_turns: int


@dataclass(frozen=True)
class RotatedTileCandidateSpec:
    """One labeled tile candidate in the misrotated-tile task."""

    candidate_id: str
    label: str
    tile_id: str
    row: int
    col: int
    required_openings: Openings
    current_openings: Openings
    repair_rotation_turns: Tuple[int, ...]
    is_correct: bool
    connects_after_rotation: bool


@dataclass(frozen=True)
class PipeFlowDataset:
    """One sampled pipe-flow repair puzzle."""

    rows: int
    cols: int
    grid_size_variant: str
    gap_size_variant: str
    gap_size: int
    scene_variant: str
    path_cells: Tuple[Cell, ...]
    branch_cells: Tuple[Cell, ...]
    branch_terminal_cells: Tuple[Cell, ...]
    start_cell: Cell
    destination_cell: Cell
    missing_origin: Cell
    missing_cells: Tuple[Cell, ...]
    missing_region_id: str
    correct_option_panel_id: str
    answer_label: str
    candidate_count: int
    tiles: Tuple[TileSpec, ...]
    options: Tuple[OptionSpec, ...]


@dataclass(frozen=True)
class PipeFlowMisrotatedDataset:
    """One sampled pipe-flow misrotated-tile puzzle."""

    rows: int
    cols: int
    grid_size_variant: str
    scene_variant: str
    path_cells: Tuple[Cell, ...]
    branch_cells: Tuple[Cell, ...]
    branch_terminal_cells: Tuple[Cell, ...]
    start_cell: Cell
    destination_cell: Cell
    answer_label: str
    candidate_count: int
    misrotated_tile_id: str
    tiles: Tuple[TileSpec, ...]
    candidates: Tuple[RotatedTileCandidateSpec, ...]


@dataclass(frozen=True)
class RenderedPipeFlowScene:
    """Rendered pipe-flow scene plus projection maps."""

    image: Image.Image
    scene_bbox_px: BBox
    tile_bbox_map: Dict[str, BBox]
    item_bbox_map: Dict[str, BBox]
    entities: Tuple[Dict[str, Any], ...]
