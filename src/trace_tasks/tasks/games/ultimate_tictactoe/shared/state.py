"""Passive state containers for Ultimate Tic-Tac-Toe scene generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from PIL import Image


DOMAIN = "games"
SCENE_ID = "ultimate_tictactoe"
SCENE_NAMESPACE = "games.ultimate_tictactoe"

PLAYER_X = "X"
PLAYER_O = "O"
STATUS_X_WON = "X_won"
STATUS_O_WON = "O_won"
STATUS_DRAWN = "drawn"
STATUS_OPEN = "open"
STATUS_NEITHER_WON = "neither_won"

SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic_grid",
    "soft_marker",
    "paper_grid",
    "neon_board",
    "tournament_board",
)
LOCAL_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
MACRO_LABELS: Tuple[str, ...] = tuple("ABCDEFGHI")
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEFGH")
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class UltimateDefaults:
    """Stable fallback defaults for visible Ultimate Tic-Tac-Toe boards."""

    won_board_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    drawn_board_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    neither_won_board_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    macro_threat_board_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    option_count_support: Tuple[int, ...] = (5,)
    canvas_width: int = 760
    canvas_height: int = 760
    panel_margin_px: int = 36
    board_inner_margin_px: int = 48
    local_cell_size_px: int = 58
    local_gap_px: int = 3
    macro_gap_px: int = 10
    symbol_font_size_px: int = 33
    option_font_size_px: int = 22
    label_font_size_px: int = 18
    small_board_border_width_px: int = 8
    highlight_width_px: int = 6


@dataclass(frozen=True)
class UltimateLocalBoard:
    """One local 3x3 board inside the Ultimate board."""

    cells: Tuple[str, ...]
    status: str
    winning_line: Tuple[int, int, int] | None = None


@dataclass(frozen=True)
class UltimateSample:
    """Constructed board plus task-owned answer and annotation witnesses."""

    board: Tuple[UltimateLocalBoard, ...]
    answer: int | str
    answer_type: str
    target_answer: int | None
    highlighted_board_index: int | None
    option_cells: Tuple[int, ...]
    answer_cell: int | None
    support_cells: Tuple[int, ...]
    annotation_entity_ids: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedUltimateScene:
    """Rendered image plus entity maps used for annotation projection."""

    image: Image.Image
    entities: Tuple[dict[str, Any], ...]
    render_map: dict[str, Any]
    style_meta: dict[str, Any]
    background_meta: dict[str, Any]


@dataclass(frozen=True)
class UltimateBoardVisualStyle:
    """Resolved scene-local board palette and marker colors."""

    cell_fill_rgb: Tuple[int, int, int]
    board_fill_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    border_rgb: Tuple[int, int, int]
    highlight_rgb: Tuple[int, int, int]
    x_rgb: Tuple[int, int, int]
    o_rgb: Tuple[int, int, int]
    option_fill_rgb: Tuple[int, int, int]
    option_outline_rgb: Tuple[int, int, int]
    option_text_rgb: Tuple[int, int, int]
