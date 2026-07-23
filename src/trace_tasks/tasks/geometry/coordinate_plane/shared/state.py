"""Scene-local state records for coordinate-plane task primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults

POST_IMAGE_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id="coordinate_plane")
POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id="coordinate_plane")


GraphPoint = Tuple[int, int]
PixelPoint = Tuple[float, float]
Segment = Tuple[GraphPoint, GraphPoint]
PixelSegment = Tuple[PixelPoint, PixelPoint]


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for geometry coordinate-relation scenes."""

    canvas_size_min: int = 640
    canvas_size_max: int = 720
    graph_cells_min: int = 18
    graph_cells_max: int = 20
    line_width: int = 4
    line_width_min: int = 3
    line_width_max: int = 5
    point_radius_px: int = 4
    point_radius_px_min: int = 3
    point_radius_px_max: int = 5
    label_font_size_min: int = 16
    label_font_size_max: int = 28
    label_stroke_width: int = 1
    label_stroke_width_min: int = 1
    label_stroke_width_max: int = 1
    label_offset_px: int = 14
    segment_candidate_count: int = 6
    segment_endpoint_abs_max: int = 8
    segment_half_vectors: Tuple[Tuple[int, int], ...] = (
        (3, 0),
        (0, 3),
        (3, 1),
        (1, 3),
        (3, -1),
        (1, -3),
        (2, 2),
        (2, -2),
    )
    segment_target_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    collinear_candidate_count: int = 8
    collinear_point_abs_max: int = 8
    collinear_direction_vectors: Tuple[Tuple[int, int], ...] = (
        (1, 0),
        (0, 1),
        (1, 1),
        (1, -1),
        (2, 1),
        (1, 2),
        (2, -1),
        (1, -2),
    )
    collinear_target_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    quadrant_reference_label: str = "A"
    quadrant_candidate_labels: Tuple[str, ...] = ("B", "C", "D", "E", "F", "G", "H", "I")
    same_quadrant_target_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    point_in_shape_target_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved semantic operation and answer support for one coordinate scene."""

    scene_variant: str
    operation_key: str
    scene_variant_probabilities: Dict[str, float]
    operation_key_probabilities: Dict[str, float]
    target_count: int | None = None
    target_count_probabilities: Dict[str, float] = field(default_factory=dict)
    label_pool: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _RenderedCoordinateScene:
    """Rendered coordinate-relation scene plus trace-ready annotation artifacts."""

    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    answer_value: str | int
    annotation_type: str
    annotation_value: Any
    projected_annotation: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    required_annotation_labels: List[str]
    object_count: int
    matching_labels: Tuple[str, ...] = field(default_factory=tuple)


_DEFAULTS = _TaskDefaults()
_SCENE_DEFAULTS = get_scene_defaults("geometry", "coordinate_plane")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)

__all__ = [
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "GraphPoint",
    "PixelPoint",
    "Segment",
    "PixelSegment",
    "_TaskDefaults",
    "_ResolvedQuery",
    "_RenderedCoordinateScene",
    "_DEFAULTS",
    "_SCENE_DEFAULTS",
    "_GEN_DEFAULTS",
    "_RENDER_DEFAULTS",
    "_PROMPT_DEFAULTS",
]
