"""State containers for printed map scene packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]
Cell = Tuple[int, int]


@dataclass(frozen=True)
class MapDefaults:
    """Default generation bounds for printed map route tasks."""

    landmark_count_min: int = 8
    landmark_count_max: int = 12
    direction_step_count_min: int = 2
    direction_step_count_max: int = 4
    highlighted_route_step_min: int = 2
    highlighted_route_step_max: int = 5


@dataclass(frozen=True)
class MapRenderParams:
    """Resolved rendering knobs for one printed campus-map scene."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    title_font_size_px: int
    title_band_height_px: int
    map_corner_radius_px: int
    map_border_width_px: int
    path_width_px: int
    highlighted_path_width_px: int
    landmark_width_px: int
    landmark_height_px: int
    landmark_corner_radius_px: int
    landmark_border_width_px: int
    landmark_label_font_size_px: int
    zone_label_font_size_px: int
    legend_font_size_px: int
    compass_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    title_color_rgb: Tuple[int, int, int]
    map_fill_rgb: Tuple[int, int, int]
    map_border_rgb: Tuple[int, int, int]
    zone_border_rgb: Tuple[int, int, int]
    path_rgb: Tuple[int, int, int]
    highlighted_path_rgb: Tuple[int, int, int]
    landmark_fill_rgb: Tuple[int, int, int]
    landmark_border_rgb: Tuple[int, int, int]
    landmark_label_rgb: Tuple[int, int, int]
    label_stroke_rgb: Tuple[int, int, int]
    zone_label_rgb: Tuple[int, int, int]
    compass_rgb: Tuple[int, int, int]
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class MapSceneCase:
    """Sampled visible map graph before objective-specific route binding."""

    scene_title: str
    scene_variant: str
    scene_variant_probabilities: Mapping[str, float]
    grid_cols: int
    grid_rows: int
    landmark_count: int
    zone_specs: Tuple[Mapping[str, object], ...]
    landmark_specs: Tuple[Mapping[str, object], ...]
    path_specs: Tuple[Mapping[str, object], ...]
    cells: Tuple[Cell, ...]
    adjacency: Mapping[Cell, Tuple[Cell, ...]]
    cell_to_landmark_id: Mapping[Cell, str]


@dataclass(frozen=True)
class RenderedMapScene:
    """Rendered printed-map scene plus traced map geometry."""

    image: Image.Image
    entities: List[Dict[str, object]]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    map_bbox_px: List[float]
    landmark_bbox_map: Dict[str, List[float]]
    landmark_label_bbox_map: Dict[str, List[float]]
    zone_label_bbox_map: Dict[str, List[float]]
    path_bbox_map: Dict[str, List[float]]
    highlighted_route_bbox_map: Dict[str, List[float]]
    layout_jitter_meta: Dict[str, object]


@dataclass(frozen=True)
class RenderedMapBundle:
    """Rendered map image and metadata used by public objective files."""

    image: Image.Image
    render_params: MapRenderParams
    rendered_scene: RenderedMapScene
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]


__all__ = [
    "BBox",
    "Cell",
    "MapDefaults",
    "MapRenderParams",
    "MapSceneCase",
    "Point",
    "RenderedMapBundle",
    "RenderedMapScene",
]
