"""Passive state records for pulley mechanical-advantage diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "pulley"
SCENE_NAMESPACE = "physics_pulley"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    "open_block",
    "compact_block",
    "tall_block",
)
SUPPORTED_SOLVE_FOR_TARGETS: tuple[str, ...] = ("effort_force", "load_force")
LOAD_FORCE_SUPPORT: tuple[int, ...] = (
    8,
    10,
    12,
    14,
    15,
    16,
    18,
    20,
    21,
    22,
    24,
    25,
    26,
    27,
    28,
    30,
    32,
    33,
    34,
    35,
    36,
    39,
    40,
    42,
    44,
    45,
    48,
    50,
    51,
    52,
    54,
    55,
    56,
    60,
    64,
    65,
    66,
    68,
    70,
    72,
    75,
    78,
    80,
    84,
    85,
    90,
    96,
    102,
    108,
)
PULLEY_SEMANTIC_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 231, 231),
    (187, 56, 56),
    (167, 38, 38),
)


@dataclass(frozen=True)
class PulleyTaskDefaults:
    """Stable fallback defaults for pulley-system scenes."""

    canvas_width: int = 1280
    canvas_height: int = 760
    top_block_x_px: int = 250
    top_block_y_px: int = 94
    top_block_width_px: int = 780
    top_block_height_px: int = 72
    lower_block_y_px: int = 430
    compact_lower_block_y_px: int = 390
    tall_lower_block_y_px: int = 470
    lower_block_height_px: int = 72
    support_segment_gap_px: int = 48
    rope_width_px: int = 7
    pulley_radius_px: int = 20
    pulley_hub_radius_px: int = 6
    load_width_px: int = 170
    load_height_px: int = 74
    load_top_gap_px: int = 34
    connector_width_px: int = 5
    effort_arrow_x_gap_px: int = 92
    effort_arrow_length_px: int = 144
    label_font_size_px: int = 28
    small_label_font_size_px: int = 24
    label_stroke_width_px: int = 3
    texture_line_width_px: int = 2
    texture_spacing_px: int = 18
    cut_endpoint_radius_px: int = 6
    connected_support_count_support: tuple[int, ...] = (2, 3, 4, 5, 6)
    disconnected_segment_count_support: tuple[int, ...] = (0, 1, 2, 3, 4)
    effort_force_support: tuple[int, ...] = tuple(range(4, 19))
    load_force_support: tuple[int, ...] = LOAD_FORCE_SUPPORT
    effort_force_min: int = 4
    effort_force_max: int = 18


@dataclass(frozen=True)
class PulleyResolvedAxes:
    """Resolved scene axes and answer support for one instance."""

    scene_variant: str
    solve_for: str
    accent_color_name: str
    target_answer: int
    scene_variant_probabilities: dict[str, float]
    solve_for_probabilities: dict[str, float]
    accent_color_name_probabilities: dict[str, float]
    target_answer_probabilities: dict[str, float]


@dataclass(frozen=True)
class CutSegmentSpec:
    """One visible rope segment that is not connected all the way through."""

    segment_id: str
    attach_side: str
    x_order: int
    cut_fraction: float


@dataclass(frozen=True)
class PulleySceneSpec:
    """Symbolic pulley system that realizes one force-relation answer."""

    scene_variant: str
    solve_for: str
    support_segment_count: int
    disconnected_segment_count: int
    connected_slot_indices: tuple[int, ...]
    cut_segments: tuple[CutSegmentSpec, ...]
    effort_force_value: int
    load_force_value: int
    shown_effort_force_value: int | None
    shown_load_force_value: int | None
    target_answer: int
    annotation_entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class RenderedPulleyScene:
    """Rendered pulley scene plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bboxes: list[list[float]]
    annotation_bbox_map: dict[str, list[float]]
    annotation_entity_ids: list[str]
    annotation_entity_id_map: dict[str, str]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]
    support_segment_bboxes: list[list[float]]
    cut_segment_bboxes: list[list[float]]
    load_label_bbox: list[float]
    effort_label_bbox: list[float]


__all__ = [
    "CutSegmentSpec",
    "LOAD_FORCE_SUPPORT",
    "PULLEY_SEMANTIC_COLORS",
    "PulleyResolvedAxes",
    "PulleySceneSpec",
    "PulleyTaskDefaults",
    "RenderedPulleyScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_SOLVE_FOR_TARGETS",
]
