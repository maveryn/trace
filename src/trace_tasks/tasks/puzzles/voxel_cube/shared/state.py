"""Passive state types for voxel-cube puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

DOMAIN = "puzzles"
SCENE_ID = "voxel_cube"

GridCell = Tuple[int, int]
BBox = Tuple[float, float, float, float]
HeightGrid = Tuple[Tuple[int, ...], ...]
Direction = str
Color = Tuple[int, int, int]

VIEW_DIRECTIONS: Tuple[Direction, ...] = ("top", "front", "right")
CHANGE_TYPES: Tuple[str, ...] = ("missing_to_complete", "removed")


@dataclass(frozen=True)
class VoxelPalette:
    """One fixed non-semantic color palette for a rendered voxel question."""

    palette_id: str
    cube_top_rgb: Color
    cube_left_rgb: Color
    cube_right_rgb: Color
    cube_edge_rgb: Color
    projection_fill_rgb: Color
    projection_empty_rgb: Color


@dataclass(frozen=True)
class CubeStack:
    """One voxel structure encoded as a row-by-column height grid."""

    heights: HeightGrid

    @property
    def rows(self) -> int:
        """Return the number of footprint rows."""

        return len(self.heights)

    @property
    def cols(self) -> int:
        """Return the number of footprint columns."""

        return len(self.heights[0]) if self.heights else 0


@dataclass(frozen=True)
class ProjectionGrid:
    """One orthographic projection grid."""

    direction: Direction
    rows: int
    cols: int
    filled_cells: Tuple[GridCell, ...]


@dataclass(frozen=True)
class ProjectionOption:
    """One labeled projection option panel."""

    label: str
    projection: ProjectionGrid
    is_correct: bool


@dataclass(frozen=True)
class VoxelDataset:
    """Task-neutral generated state passed from sampler to renderer."""

    stack: CubeStack
    semantic_params: dict[str, object]
    answer_support: Tuple[object, ...]


@dataclass(frozen=True)
class CountDataset(VoxelDataset):
    """Generated state for a single-structure integer count task."""

    answer_value: int


@dataclass(frozen=True)
class ChangeDataset(VoxelDataset):
    """Generated state for comparing two voxel structures."""

    reference_stack: CubeStack
    changed_stack: CubeStack
    answer_value: int


@dataclass(frozen=True)
class ProjectionCountDataset(VoxelDataset):
    """Generated state for a projection-cell count task."""

    projection: ProjectionGrid
    answer_value: int


@dataclass(frozen=True)
class ProjectionMatchDataset(VoxelDataset):
    """Generated state for a projection option-label task."""

    query_projection: ProjectionGrid
    options: Tuple[ProjectionOption, ...]
    answer_label: str


@dataclass(frozen=True)
class VoxelRenderParams:
    """Render dimensions and typography for one voxel-cube scene."""

    canvas_width: int
    canvas_height: int
    cube_size_px: int
    projection_cell_size_px: int
    panel_gap_px: int
    label_font_size_px: int
    palette: VoxelPalette


@dataclass(frozen=True)
class RenderedVoxelScene:
    """Rendered image plus pixel projections for answer witnesses."""

    image: object
    scene_bbox_px: BBox
    stack_bbox_px: BBox | None
    reference_stack_bbox_px: BBox | None
    changed_stack_bbox_px: BBox | None
    projection_cell_bbox_map: dict[str, BBox]
    option_panel_bbox_map: dict[str, BBox]
