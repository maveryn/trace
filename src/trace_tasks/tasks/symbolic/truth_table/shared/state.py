"""Passive state objects for symbolic truth-table scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
EXPRESSION_OPTION_LABELS: tuple[str, ...] = ("W", "X", "Y", "Z")
TRUTH_VARIABLES: tuple[str, ...] = ("A", "B", "C")
SUPPORTED_TRUTH_TABLE_SCENE_VARIANTS: tuple[str, ...] = (
    "clean_table",
    "notebook_table",
    "exam_scan",
)


@dataclass(frozen=True)
class TruthRowSpec:
    row_index: int
    row_label: str
    values: dict[str, int]


@dataclass(frozen=True)
class TruthExpressionSpec:
    expression_id: str
    display: str
    ast: Any
    pattern: tuple[int, ...]

    @property
    def true_count(self) -> int:
        return sum(1 for value in self.pattern if int(value) == 1)

    @property
    def pattern_string(self) -> str:
        return "".join(str(int(value)) for value in self.pattern)


@dataclass(frozen=True)
class TruthTableRenderParams:
    canvas_width: int = 1120
    canvas_height: int = 780
    table_left_px: int = 88
    table_top_px: int = 102
    row_label_width_px: int = 58
    variable_cell_width_px: int = 66
    output_cell_width_px: int = 210
    compact_output_cell_width_px: int = 82
    row_height_px: int = 52
    header_height_px: int = 72
    card_corner_radius_px: int = 14
    card_border_width_px: int = 2
    grid_line_width_px: int = 2
    expression_font_size_px: int = 24
    header_font_size_px: int = 22
    cell_font_size_px: int = 24
    option_label_font_size_px: int = 25
    pattern_font_size_px: int = 28
    title_font_size_px: int = 22


@dataclass(frozen=True)
class RenderedTruthTableScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    cell_bboxes: dict[str, list[float]]
    column_bboxes: dict[str, list[float]]
    row_bboxes: dict[str, list[float]]
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


__all__ = [
    "OPTION_LABELS",
    "EXPRESSION_OPTION_LABELS",
    "SUPPORTED_TRUTH_TABLE_SCENE_VARIANTS",
    "TRUTH_VARIABLES",
    "RenderedTruthTableScene",
    "TruthExpressionSpec",
    "TruthRowSpec",
    "TruthTableRenderParams",
]
