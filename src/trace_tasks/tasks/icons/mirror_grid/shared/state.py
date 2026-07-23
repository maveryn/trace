"""Passive state objects for mirror-grid icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class MirrorGridScenePayload:
    """Trace-ready payload for one rendered mirror-grid scene."""

    object_count: int
    target_count: int
    distractor_count: int
    reference_symmetry_kind: str
    cell_labels: Tuple[str, ...]
    matching_labels: Tuple[str, ...]
    scene_cell_symmetry_kinds: Tuple[str, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    reference_cell: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class MirrorGridCompletionScenePayload:
    """Trace-ready payload for one rendered missing-cell mirror-grid scene."""

    object_count: int
    option_count: int
    mirror_axis: str
    missing_row: int
    missing_col: int
    counterpart_row: int
    counterpart_col: int
    answer_label: str
    cell_labels: Tuple[str, ...]
    option_labels: Tuple[str, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    grid_panel: Dict[str, Any]
    grid_cells: Tuple[Dict[str, Any], ...]
    option_cells: Tuple[Dict[str, Any], ...]


__all__ = ["MirrorGridCompletionScenePayload", "MirrorGridScenePayload"]
