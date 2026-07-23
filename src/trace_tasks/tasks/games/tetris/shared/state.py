"""State objects for variable-size Tetris board scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

DOMAIN = "games"
SCENE_ID = "tetris"
EMPTY = "."
Coord = Tuple[int, int]
Board = Tuple[Tuple[str, ...], ...]
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("low_stack", "notched_stack", "high_stack")
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic_blocks",
    "beveled_blocks",
    "paper_tiles",
    "neon_blocks",
)

RAW_TETROMINOES: Dict[str, Tuple[Tuple[Coord, ...], ...]] = {
    "I": (((0, 0), (0, 1), (0, 2), (0, 3)), ((0, 0), (1, 0), (2, 0), (3, 0))),
    "O": (((0, 0), (0, 1), (1, 0), (1, 1)),),
    "T": (
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (1, 1), (2, 0)),
    ),
    "L": (
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 2), (1, 0), (1, 1), (1, 2)),
    ),
    "J": (
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
    ),
    "S": (((0, 1), (0, 2), (1, 0), (1, 1)), ((0, 0), (1, 0), (1, 1), (2, 1))),
    "Z": (((0, 0), (0, 1), (1, 1), (1, 2)), ((0, 1), (1, 0), (1, 1), (2, 0))),
}
PIECE_ORDER: Tuple[str, ...] = tuple(RAW_TETROMINOES)


@dataclass(frozen=True)
class TetrisDefaults:
    """Stable fallback defaults for Tetris generation and rendering."""

    line_clear_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    drop_result_clear_count_support: Tuple[int, ...] = (0, 1, 2)
    row_occupancy_status_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    drop_collision_time_support: Tuple[int, ...] = tuple(range(0, 9))
    shift_magnitude_support: Tuple[int, ...] = (1, 2, 3)
    option_count_support: Tuple[int, ...] = (4,)
    board_row_count_support: Tuple[int, ...] = tuple(range(10, 16))
    board_col_count_support: Tuple[int, ...] = tuple(range(7, 12))
    canvas_width: int = 1100
    canvas_height: int = 990
    panel_margin_px: int = 42
    board_gap_px: int = 34
    cell_size_px: int = 24
    line_cell_size_px: int = 34
    cell_gap_px: int = 2
    panel_pad_px: int = 12
    label_band_height_px: int = 28
    cell_outline_width_px: int = 1
    ghost_outline_width_px: int = 4
    label_font_size_px: int = 20
    small_label_font_size_px: int = 15


@dataclass(frozen=True)
class Placement:
    """One positioned tetromino placement in board coordinates."""

    piece: str
    orientation_index: int
    col: int
    top: int

    @property
    def entity_id(self) -> str:
        return f"placement_{self.piece.lower()}_{self.orientation_index}_{self.col}_{self.top}"


@dataclass(frozen=True)
class Outcome:
    """Result of locking one placement and clearing full rows."""

    placement: Placement
    clear_count: int
    locked_board: Board
    result_board: Board
    locked_cells: Tuple[Coord, ...]
    cleared_rows: Tuple[int, ...]
    holes_after: int
    max_height_after: int
    aggregate_height_after: int


@dataclass(frozen=True)
class DropCollision:
    """Result of shifting a falling piece and dropping until it collides."""

    start_placement: Placement
    shifted_placement: Placement
    final_placement: Placement
    drop_steps: int
    blocker_cells: Tuple[Coord, ...]
    bottom_contact_cells: Tuple[Coord, ...]
    collision_kind: str


@dataclass(frozen=True)
class Option:
    """One labeled result-board option panel."""

    label: str
    board: Board
    placement: Placement | None
    outcome: Outcome | None
    is_answer: bool
    metric_value: int | None = None

    @property
    def entity_id(self) -> str:
        return f"option_{self.label.lower()}"


@dataclass(frozen=True)
class SceneAxes:
    """Scene-level sampled axes shared by all Tetris objectives."""

    scene_variant: str
    style_variant: str
    option_count: int
    board_rows: int
    board_cols: int
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    option_count_probabilities: Dict[str, float]
    board_row_probabilities: Dict[str, float]
    board_col_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderParams:
    """Resolved canvas, panel, cell, font, and jitter controls."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    board_gap_px: int
    cell_size_px: int
    line_cell_size_px: int
    cell_gap_px: int
    panel_pad_px: int
    label_band_height_px: int
    cell_outline_width_px: int
    ghost_outline_width_px: int
    label_font_size_px: int
    small_label_font_size_px: int
    font_family: str
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class TetrisSample:
    """Task-owned generated scene state before rendering."""

    answer: int | str
    answer_type: str
    board: Board
    piece: str
    preview_orientation_index: int
    placement: Placement | None
    falling_placement: Placement | None
    outcome: Outcome | None
    options: Tuple[Option, ...]
    annotation_entity_ids: Tuple[str, ...]
    annotation_kind: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RenderedTetrisScene:
    """Rendered Tetris image plus entity and projection maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
