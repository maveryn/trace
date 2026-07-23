"""Passive state objects for the pair-grid icons scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

from ...shared.icon_noise import NoiseEdit
from ...shared.icon_scene import IconPanelLayout


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class IconPairSpec:
    """One before/after icon pair to render in a labeled scene cell."""

    icon_id: str
    transform_id: str
    tint_rgb: Tuple[int, int, int]
    left_tint_rgb: Tuple[int, int, int] | None = None
    right_tint_rgb: Tuple[int, int, int] | None = None
    left_size_scale: float = 1.0
    right_size_scale: float = 1.0
    left_noise_edits: Tuple[NoiseEdit, ...] = ()
    left_noise_seed: int | None = None
    right_noise_edits: Tuple[NoiseEdit, ...] = ()
    right_noise_seed: int | None = None


@dataclass(frozen=True)
class RenderedReferencePair:
    """Rendered metadata for the left-panel reference pair."""

    icon_id: str
    transform_id: str
    tint_rgb: Tuple[int, int, int]
    left_tint_rgb: Tuple[int, int, int]
    right_tint_rgb: Tuple[int, int, int]
    left_size_scale: float
    right_size_scale: float
    left_bbox_xyxy: BBox
    right_bbox_xyxy: BBox
    left_noise_edits: Tuple[Dict[str, Any], ...]
    left_noise_seed: int | None
    right_noise_edits: Tuple[Dict[str, Any], ...]
    right_noise_seed: int | None


@dataclass(frozen=True)
class RenderedScenePairCell:
    """Rendered metadata for one labeled scene cell."""

    label: str
    icon_id: str
    transform_id: str
    tint_rgb: Tuple[int, int, int]
    left_tint_rgb: Tuple[int, int, int]
    right_tint_rgb: Tuple[int, int, int]
    left_size_scale: float
    right_size_scale: float
    cell_bbox_xyxy: BBox
    left_bbox_xyxy: BBox
    right_bbox_xyxy: BBox
    left_noise_edits: Tuple[Dict[str, Any], ...]
    left_noise_seed: int | None
    right_noise_edits: Tuple[Dict[str, Any], ...]
    right_noise_seed: int | None


@dataclass(frozen=True)
class RenderedIconPairGridScene:
    """Complete rendered output for one reference-pair grid scene."""

    image: Image.Image
    layout: IconPanelLayout
    reference_pair: RenderedReferencePair
    scene_cells: Tuple[RenderedScenePairCell, ...]
