"""Passive state records for the named-grid icons scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple

from PIL import Image


class NamedGridRenderSpec(Protocol):
    """Minimum symbolic state needed by the neutral named-grid renderer."""

    target_shape_id: str
    grid_rows: int
    grid_cols: int
    shape_ids_by_cell: Tuple[Tuple[str, ...], ...]
    counted_cells: Tuple[Tuple[int, int], ...]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedGridIcon:
    """Rendered named-grid icon metadata."""

    instance_id: str
    row_index: int
    col_index: int
    row_number: int
    column_number: int
    shape_id: str
    shape_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    cell_bbox_xyxy: Tuple[int, int, int, int]
    nominal_size_px: int
    rotation_degrees: int
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None
    is_target_shape: bool
    is_counted: bool


@dataclass(frozen=True)
class NamedGridScenePayload:
    """Trace-ready rendered named-grid scene payload."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    grid_bbox_xyxy: Tuple[int, int, int, int]
    cell_bboxes_xyxy: Tuple[Tuple[Tuple[int, int, int, int], ...], ...]
    icons: Tuple[RenderedGridIcon, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    cell_size_px: int
