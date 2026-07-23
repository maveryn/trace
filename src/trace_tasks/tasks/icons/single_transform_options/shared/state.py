"""Passive state objects for single-transform option icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class SingleTransformOptionsScenePayload:
    """Trace-ready payload for one rendered transform-result option scene."""

    object_count: int
    cell_labels: Tuple[str, ...]
    target_transform_id: str
    operation_cue: str
    answer_label: str
    icon_id: str
    option_transform_ids: Tuple[str, ...]
    tint_rgb: Tuple[int, int, int]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    reference_cell: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


__all__ = ["SingleTransformOptionsScenePayload"]
