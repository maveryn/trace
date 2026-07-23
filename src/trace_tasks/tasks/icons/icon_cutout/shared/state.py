"""Passive state objects for icon-cutout scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class FragmentPayload:
    """One sampled partial-fragment crop and metadata."""

    image: Image.Image
    crop_xyxy: Tuple[int, int, int, int]
    window_style: str
    visible_alpha_ratio: float
    alpha_density: float


@dataclass(frozen=True)
class IconCutoutScenePayload:
    """Trace-ready payload for one icon-cutout option instance."""

    object_count: int
    cell_labels: Tuple[str, ...]
    answer_label: str
    correct_icon_id: str
    option_icon_ids: Tuple[str, ...]
    tint_rgb: Tuple[int, int, int]
    rotation_degrees: int
    fragment_window_style: str
    fragment_visible_alpha_ratio: float
    fragment_alpha_density: float
    fragment_crop_xyxy: Tuple[int, int, int, int]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    reference_cell: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


__all__ = ["FragmentPayload", "IconCutoutScenePayload"]
