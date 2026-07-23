"""State records for word-search puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trace_tasks.tasks.puzzles.shared.word_grid import Cell, WordPlacement

DOMAIN = "puzzles"
SCENE_ID = "word_search"

SCENE_VARIANTS: tuple[str, ...] = (
    "word_search_classic",
    "word_search_notebook",
    "word_search_card",
)
OPTION_LABELS: tuple[str, ...] = tuple("ABCDEFGH")


@dataclass(frozen=True)
class WordSearchOption:
    """One visible answer option for a word-search task."""

    label: str
    row_1based: int | None = None
    col_1based: int | None = None
    direction: str | None = None
    display_text: str = ""
    word: str = ""
    is_correct: bool = False


@dataclass(frozen=True)
class WordSearchRenderParams:
    """Resolved rendering knobs for one word-search instance."""

    canvas_width: int
    canvas_height: int
    cell_size_px: int
    header_size_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    grid_line_width_px: int
    option_panel_width_px: int
    option_panel_height_px: int
    option_gap_px: int
    option_font_size_px: int
    letter_font_size_px: int
    index_font_size_px: int
    panel_fill_rgb: tuple[int, int, int]
    grid_fill_rgb: tuple[int, int, int]
    header_fill_rgb: tuple[int, int, int]
    grid_line_rgb: tuple[int, int, int]
    text_rgb: tuple[int, int, int]
    text_stroke_rgb: tuple[int, int, int]
    option_fill_rgb: tuple[int, int, int]
    option_border_rgb: tuple[int, int, int]
    option_text_rgb: tuple[int, int, int]
    unit_size_jitter: dict[str, Any]


@dataclass(frozen=True)
class WordSearchDataset:
    """Generated word-search state before rendering."""

    rows: int
    cols: int
    grid_size_range: tuple[int, int]
    grid: tuple[tuple[str, ...], ...]
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    target_word: str
    target_letter: str
    answer_value: int | str
    answer_support: tuple[int | str, ...]
    option_specs: tuple[WordSearchOption, ...]
    word_bank: tuple[str, ...]
    present_words: tuple[str, ...]
    placements: tuple[WordPlacement, ...]
    target_cells: tuple[Cell, ...]


@dataclass(frozen=True)
class RenderedWordSearch:
    """Rendered word-search image and pixel maps."""

    image: Any
    entities: tuple[dict[str, Any], ...]
    scene_bbox_px: list[float]
    item_bbox_map: dict[str, list[float]]
    cell_bbox_map: dict[str, list[float]]
    cell_centers_px: dict[str, tuple[float, float]]
    layout_jitter: dict[str, Any]


__all__ = [
    "DOMAIN",
    "OPTION_LABELS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "RenderedWordSearch",
    "WordSearchDataset",
    "WordSearchOption",
    "WordSearchRenderParams",
]
