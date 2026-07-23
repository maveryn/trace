"""State containers for pages cycle diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class CycleDefaults:
    """Stable fallback defaults for directed cycle diagrams."""

    stage_count_min: int = 5
    stage_count_max: int = 12
    step_count_min: int = 2


@dataclass(frozen=True)
class CycleRenderParams:
    """Pixel-space rendering knobs for one directed cycle diagram."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    title_font_size_px: int
    title_band_height_px: int
    node_width_px: int
    node_height_px: int
    node_corner_radius_px: int
    node_border_width_px: int
    ring_radius_x_px: int
    ring_radius_y_px: int
    edge_width_px: int
    arrow_head_length_px: int
    arrow_head_width_px: int
    label_font_size_px: int
    panel_fill_rgb: Color
    panel_border_rgb: Color
    title_color_rgb: Color
    node_fill_rgb: Color
    node_border_rgb: Color
    label_color_rgb: Color
    label_stroke_rgb: Color
    edge_color_rgb: Color
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class CycleCase:
    """Sampled semantic state for a directed cycle offset question."""

    scene_title: str
    scene_variant: str
    query_relationship: str
    cycle_direction: str
    stage_count: int
    step_count: int
    query_stage_index: int
    query_stage_id: str
    query_stage_label: str
    answer_stage_index: int
    answer_stage_id: str
    answer_stage_label: str
    answer_stage_bbox_id: str
    stage_specs: Tuple[Dict[str, Any], ...]
    edge_specs: Tuple[Dict[str, Any], ...]
    prompt_slots: Dict[str, Any]
    scene_variant_probabilities: Dict[str, float]
    query_relationship_probabilities: Dict[str, float]
    cycle_direction_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedCycleScene:
    """Rendered image plus traced geometry for one directed cycle."""

    image: Image.Image
    render_params: CycleRenderParams
    entities: List[Dict[str, object]]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    stage_bbox_map: Dict[str, List[float]]
    stage_label_bbox_map: Dict[str, List[float]]
    edge_bbox_map: Dict[str, List[float]]
    layout_jitter_meta: Dict[str, object]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
