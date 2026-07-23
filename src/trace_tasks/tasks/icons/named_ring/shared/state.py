"""Passive state objects for named-ring icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class RingArcPlan:
    """Symbolic ring arc and target placement before rendering."""

    direction: str
    target_shape_id: str
    target_shape_name: str
    answer_count: int
    ring_icon_count: int
    arc_span_count: int
    start_index: int
    end_index: int
    arc_indices: Tuple[int, ...]
    counted_indices: Tuple[int, ...]
    off_arc_target_indices: Tuple[int, ...]
    shape_ids_by_index: Tuple[str, ...]
    answer_probabilities: Dict[str, float]
    ring_icon_count_probabilities: Dict[str, float]
    arc_span_probabilities: Dict[str, float]
    off_arc_target_count_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedRingIcon:
    """Rendered named-ring icon metadata."""

    instance_id: str
    ring_index: int
    clockwise_position_number: int
    role: str
    marker_label: str
    shape_id: str
    shape_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    nominal_size_px: int
    rotation_degrees: int
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None
    is_target_shape: bool
    is_arc_member: bool
    is_counted: bool


@dataclass(frozen=True)
class RingScenePayload:
    """Trace-ready rendered named-ring scene payload."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    ring_bbox_xyxy: Tuple[int, int, int, int]
    ring_center_xy: Tuple[float, float]
    ring_radius_xy: Tuple[float, float]
    icons: Tuple[RenderedRingIcon, ...]
    marker_label_bboxes: Dict[str, Tuple[int, int, int, int]]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
