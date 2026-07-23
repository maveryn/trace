"""Passive state records for rectangular cell-board puzzles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image

from trace_tasks.tasks.shared.bbox_projection import BBox

SCENE_ID = "cell_board"
Coord = Tuple[int, int]
Color = Tuple[int, int, int]
NamedColor = Tuple[str, Color]

OPEN_COLOR: NamedColor = ("open", (246, 248, 244))
WALL_COLOR: NamedColor = ("wall", (55, 58, 64))
START_COLOR: NamedColor = ("green", (55, 185, 75))
GOAL_COLOR: NamedColor = ("red", (230, 50, 50))
TARGET_COLOR: NamedColor = ("red", (230, 50, 50))


@dataclass(frozen=True)
class BoardLayout:
    """Pixel geometry for one rendered board after jitter and gutters."""

    rows: int
    cols: int
    canvas_width_px: int
    canvas_height_px: int
    tile_width_px: int
    tile_height_px: int
    board_origin_x_px: int
    board_origin_y_px: int
    board_width_px: int
    board_height_px: int
    coordinate_labels: bool
    label_font_size_px: int


@dataclass(frozen=True)
class RenderedCellBoard:
    """Rendered image plus projection metadata for one cell-board sample."""

    image: Image.Image
    bbox_map: Mapping[str, BBox]
    layout: BoardLayout
    entities: Sequence[Mapping[str, Any]]
    render_meta: Mapping[str, Any]
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]


@dataclass(frozen=True)
class CellBoardCase:
    """Task-owned symbolic board, answer, and annotation contract."""

    rows: int
    cols: int
    board_colors: Mapping[Coord, NamedColor]
    answer_value: int
    annotation_kind: str
    annotation_coords: Sequence[Coord] = field(default_factory=tuple)
    annotation_path: Sequence[Coord] = field(default_factory=tuple)
    annotation_coord_pairs: Sequence[tuple[Coord, Coord]] = field(default_factory=tuple)
    cell_text: Mapping[Coord, str] = field(default_factory=dict)
    coordinate_labels: bool = False
    prompt_task_key: str = "cell_board_topology_query"
    prompt_query_key: str = "single"
    prompt_slots: Mapping[str, Any] = field(default_factory=dict)
    execution_trace: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "BoardLayout",
    "CellBoardCase",
    "Color",
    "Coord",
    "GOAL_COLOR",
    "NamedColor",
    "OPEN_COLOR",
    "RenderedCellBoard",
    "SCENE_ID",
    "START_COLOR",
    "TARGET_COLOR",
    "WALL_COLOR",
]
