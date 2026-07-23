"""Passive state and defaults for hydraulic piston diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "hydraulic"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "wide_bench",
    "compact_frame",
    "tall_columns",
)
HYDRAULIC_SEMANTIC_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 231, 231),
    (187, 56, 56),
    (167, 38, 38),
)


@dataclass(frozen=True)
class HydraulicTaskDefaults:
    """Stable fallback defaults for hydraulic piston scenes."""

    canvas_width: int = 1120
    canvas_height: int = 660
    chamber_top_px: int = 178
    chamber_height_px: int = 318
    chamber_min_width_px: int = 92
    chamber_area_scale_px: int = 3
    chamber_corner_radius_px: int = 18
    chamber_outline_width_px: int = 5
    piston_height_px: int = 34
    piston_outline_width_px: int = 4
    fluid_top_gap_px: int = 96
    pipe_height_px: int = 58
    pipe_outline_width_px: int = 5
    force_arrow_length_px: int = 104
    force_arrow_width_px: int = 8
    label_font_size_px: int = 25
    small_label_font_size_px: int = 22
    label_padding_px: int = 8
    label_stroke_width_px: int = 3
    texture_line_width_px: int = 2
    texture_spacing_px: int = 18
    input_force_support: Tuple[int, ...] = tuple(range(4, 13))
    input_area_support: Tuple[int, ...] = tuple(range(2, 10))
    mechanical_advantage_support: Tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    output_force_support: Tuple[int, ...] = (
        12,
        15,
        16,
        18,
        20,
        21,
        22,
        24,
        25,
        27,
        28,
        30,
        32,
        33,
        35,
        36,
        40,
        42,
        44,
        45,
        48,
        49,
        50,
        54,
        55,
        56,
        60,
        63,
        64,
        66,
        70,
        72,
        77,
        80,
        84,
        88,
        96,
    )
    output_area_support: Tuple[int, ...] = (
        6,
        8,
        9,
        10,
        12,
        14,
        15,
        16,
        18,
        20,
        21,
        24,
        25,
        27,
        28,
        30,
        32,
        35,
        36,
        40,
        42,
        45,
        48,
        49,
        54,
        56,
        63,
        64,
        72,
    )


@dataclass(frozen=True)
class RenderedHydraulicScene:
    """Rendered hydraulic scene plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bboxes: List[List[float]]
    annotation_bbox_map: Dict[str, List[float]]
    annotation_entity_ids: List[str]
    annotation_key_by_entity_id: Dict[str, str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


__all__ = [
    "HYDRAULIC_SEMANTIC_COLORS",
    "HydraulicTaskDefaults",
    "RenderedHydraulicScene",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
]
