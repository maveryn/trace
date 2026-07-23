"""Passive state records for the named-path icons scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class IconPlan:
    """Semantic plan for one path-stop icon."""

    position_index: int
    role: str
    label: str
    shape_id: str
    color_name: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    nominal_size_px: int
    rotation_degrees: int
    noise_edits: Tuple[Any, ...]
    noise_seed: int | None


@dataclass(frozen=True)
class RenderedPathIcon:
    """Rendered path-stop icon metadata."""

    instance_id: str
    position_index: int
    role: str
    label: str
    shape_id: str
    shape_name: str
    color_name: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    nominal_size_px: int
    rotation_degrees: int
    target_occurrence_rank: int | None
    is_query_occurrence: bool
    is_answer_neighbor: bool
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None


@dataclass(frozen=True)
class PathScenePayload:
    """Trace-ready payload for one rendered named-path scene."""

    image: Image.Image
    answer_label: str
    target_shape_id: str
    target_shape_name: str
    target_occurrence_count: int
    stop_count: int
    distractor_count: int
    query_position_index: int
    answer_position_index: int
    neighbor_direction: str
    target_positions: Tuple[int, ...]
    option_positions: Tuple[int, ...]
    labels_by_position: Dict[int, str]
    path_points_xy: Tuple[Tuple[float, float], ...]
    icons: Tuple[RenderedPathIcon, ...]
    panel_geometry: Dict[str, Any]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
