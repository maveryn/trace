"""State dataclasses and constants for ray-optics scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "ray_optics"
SCENE_NAMESPACE = "physics_ray_optics"

RAY_EVENT_BOUNCE = "bounce_count"
RAY_EVENT_TARGET_HIT = "target_hit_count"

BOUNCE_SCENE_VARIANTS: Tuple[str, ...] = ("quad_mirror", "five_mirror")
TARGET_SCENE_VARIANTS: Tuple[str, ...] = (
    "single_mirror",
    "double_mirror",
    "triple_mirror",
)
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "single_mirror",
    "double_mirror",
    "triple_mirror",
    "quad_mirror",
    "five_mirror",
)

SCENE_MIRROR_COUNT = {
    "single_mirror": 1,
    "double_mirror": 2,
    "triple_mirror": 3,
    "quad_mirror": 4,
    "five_mirror": 5,
}

DIRECTION_STEP = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, -1),
    "S": (0, 1),
}
REFLECT_SLASH = {"E": "N", "N": "E", "W": "S", "S": "W"}
REFLECT_BACKSLASH = {"E": "S", "S": "E", "W": "N", "N": "W"}


@dataclass(frozen=True)
class RayOpticsTaskDefaults:
    """Stable fallback defaults for ray-optics diagrams."""

    canvas_width: int = 920
    canvas_height: int = 650
    board_left_px: int = 168
    board_top_px: int = 69
    board_cols: int = 8
    board_rows: int = 8
    cell_size_px: int = 64
    board_grid_width_px: int = 1
    board_outline_width_px: int = 3
    mirror_width_px: int = 7
    mirror_padding_px: int = 6
    ray_width_px: int = 6
    ray_head_length_px: int = 16
    ray_head_width_px: int = 16
    target_radius_px: int = 18
    source_radius_px: int = 14
    bounce_radius_px: int = 8
    target_font_size_px: int = 18
    source_font_size_px: int = 16
    label_stroke_width_px: int = 3
    bounce_count_support_single_mirror: Tuple[int, ...] = (0, 1)
    bounce_count_support_double_mirror: Tuple[int, ...] = (0, 1, 2)
    bounce_count_support_triple_mirror: Tuple[int, ...] = (0, 1, 2, 3)
    bounce_count_support_quad_mirror: Tuple[int, ...] = (0, 1, 2, 3, 4)
    bounce_count_support_five_mirror: Tuple[int, ...] = (1, 2, 3, 4, 5)
    target_hit_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
    target_count_min: int = 4
    target_count_max: int = 5


@dataclass(frozen=True)
class RayAxes:
    """Resolved sampling axes for one ray-optics instance."""

    scene_variant: str
    ray_event_kind: str
    accent_color_name: str
    target_answer: int
    scene_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class MirrorPlacement:
    """One logical mirror placement on the board."""

    mirror_id: str
    col: int
    row: int
    orientation: str
    hit: bool


@dataclass(frozen=True)
class TargetPlacement:
    """One logical target placement on the board."""

    target_id: str
    col: int
    row: int
    label: int
    hit: bool


@dataclass(frozen=True)
class RaySceneLayout:
    """One fully resolved ray-optics board before rendering."""

    scene_variant: str
    ray_event_kind: str
    target_answer: int
    source_row: int
    mirrors: Tuple[MirrorPlacement, ...]
    targets: Tuple[TargetPlacement, ...]
    path_cells: Tuple[Tuple[int, int], ...]
    bounce_cells: Tuple[Tuple[int, int], ...]
    source_point_px: Tuple[float, float]
    exit_point_px: Tuple[float, float]
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class OpticsMirrorSpec:
    """One rendered diagonal mirror cell."""

    mirror_id: str
    col: int
    row: int
    orientation: str
    hit: bool
    bbox_px: List[float]


@dataclass(frozen=True)
class OpticsTargetSpec:
    """One rendered target point on the graph board."""

    target_id: str
    col: int
    row: int
    hit: bool
    bbox_px: List[float]
    point_px: List[float]


@dataclass(frozen=True)
class OpticsBounceSpec:
    """One logical bounce point on the solved ray path."""

    bounce_id: str
    col: int
    row: int
    bbox_px: List[float]
    point_px: List[float]


@dataclass(frozen=True)
class RenderedOpticsScene:
    """Rendered ray-optics scene plus graph-point projection metadata."""

    image: Image.Image
    mirror_specs: List[OpticsMirrorSpec]
    target_specs: List[OpticsTargetSpec]
    bounce_specs: List[OpticsBounceSpec]
    graph_origin_px: List[float]
    graph_spacing_px: int
    annotation_entity_ids: List[str]
    render_map: Dict[str, Any]
    scene_entities: List[Dict[str, Any]]


__all__ = [
    "BOUNCE_SCENE_VARIANTS",
    "DIRECTION_STEP",
    "MirrorPlacement",
    "OpticsBounceSpec",
    "OpticsMirrorSpec",
    "OpticsTargetSpec",
    "RAY_EVENT_BOUNCE",
    "RAY_EVENT_TARGET_HIT",
    "REFLECT_BACKSLASH",
    "REFLECT_SLASH",
    "RayAxes",
    "RayOpticsTaskDefaults",
    "RaySceneLayout",
    "RenderedOpticsScene",
    "SCENE_ID",
    "SCENE_MIRROR_COUNT",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "TARGET_SCENE_VARIANTS",
    "TargetPlacement",
]
