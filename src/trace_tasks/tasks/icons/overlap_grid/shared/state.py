"""Passive state containers for overlap-grid icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class OverlapGridScenePayload:
    """Trace-ready payload for one occlusion-order overlap-grid scene."""

    object_count: int
    target_count: int
    distractor_count: int
    reference_order_id: str
    icon_a_id: str
    icon_b_id: str
    cell_labels: Tuple[str, ...]
    matching_labels: Tuple[str, ...]
    cell_order_ids: Tuple[str, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    reference_pair: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


__all__ = ["OverlapGridScenePayload"]
