"""Passive state objects for icon-field scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class TypeFrequencySpec:
    """Resolved icon-type frequency structure for one field."""

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
class IconFieldScenePayload:
    """Trace-ready payload for one icon-field instance."""

    object_count: int
    singleton_count: int
    repeated_type_count: int
    repeated_type_multiplicities: Tuple[int, ...]
    distinct_type_count: int
    distinct_color_count: int
    singleton_icon_ids: Tuple[str, ...]
    repeated_icon_ids: Tuple[str, ...]
    scene_icon_ids: Tuple[str, ...]
    scene_rotations_degrees: Tuple[int, ...]
    scene_tint_rgbs: Tuple[Tuple[int, int, int], ...]
    scene_color_keys: Tuple[str, ...]
    color_group_indices: Tuple[int, ...]
    singleton_indices: Tuple[int, ...]
    singleton_bboxes: Tuple[Tuple[int, int, int, int], ...]
    repeated_indices: Tuple[int, ...]
    repeated_bboxes: Tuple[Tuple[int, int, int, int], ...]
    type_frequencies: Dict[str, int]
    color_frequencies: Dict[str, int]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    placement_mode: str
    panel_geometry: Dict[str, Any]
    scene_instances: Tuple[Dict[str, Any], ...]


__all__ = ["IconFieldScenePayload", "TypeFrequencySpec"]
