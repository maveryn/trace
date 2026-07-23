"""Passive state objects for icon-grid scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class IconGridFrequencySpec:
    """Resolved category-frequency structure for one visible icon grid."""

    object_count: int
    singleton_count: int
    repeated_type_multiplicities: Tuple[int, ...]
    object_count_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    distinct_color_count: int | None = None

    @property
    def repeated_type_count(self) -> int:
        return int(len(self.repeated_type_multiplicities))

    @property
    def distinct_type_count(self) -> int:
        return int(self.singleton_count) + int(self.repeated_type_count)


@dataclass(frozen=True)
class IconGridScenePayload:
    """Trace-ready payload for one icon-grid instance."""

    object_count: int
    distinct_type_count: int
    distinct_color_count: int
    grid_rows: int
    grid_cols: int
    singleton_icon_ids: Tuple[str, ...]
    repeated_icon_ids: Tuple[str, ...]
    scene_icon_ids: Tuple[str, ...]
    scene_rotations_degrees: Tuple[int, ...]
    scene_tint_rgbs: Tuple[Tuple[int, int, int], ...]
    scene_color_keys: Tuple[str, ...]
    color_group_indices: Tuple[int, ...]
    type_frequencies: Dict[str, int]
    color_frequencies: Dict[str, int]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    grid_bbox_xyxy: Tuple[int, int, int, int]
    cell_bboxes_xyxy: Tuple[Tuple[Tuple[int, int, int, int], ...], ...]
    panel_geometry: Dict[str, Any]
    scene_instances: Tuple[Dict[str, Any], ...]


__all__ = ["IconGridFrequencySpec", "IconGridScenePayload"]
