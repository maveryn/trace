"""Passive state containers for named-strip icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class NamedStripIconPlan:
    """One planned procedural named icon in the row."""

    cell_index: int
    shape_id: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    nominal_size_px: int
    rotation_degrees: int
    noise_edits: Tuple[Any, ...]
    noise_seed: int | None


@dataclass(frozen=True)
class RenderedNamedStripIcon:
    """Rendered named-strip icon metadata."""

    instance_id: str
    cell_index: int
    shape_id: str
    shape_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    cell_bbox_xyxy: Tuple[int, int, int, int]
    nominal_size_px: int
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    rotation_degrees: int
    is_target_shape: bool
    is_selected_run_member: bool
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None


@dataclass(frozen=True)
class NamedStripScenePayload:
    """Trace-ready rendered scene payload."""

    image: Image.Image
    icons: Tuple[RenderedNamedStripIcon, ...]
    cells: Tuple[Dict[str, Any], ...]
    panel_geometry: Dict[str, Any]
    cell_box_width_px: int
    cell_box_height_px: int
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]


__all__ = ["NamedStripIconPlan", "NamedStripScenePayload", "RenderedNamedStripIcon"]
