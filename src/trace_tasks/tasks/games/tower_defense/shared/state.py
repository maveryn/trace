"""Passive state and constants for tower-defense scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "games"
SCENE_ID = "tower_defense"
SCENE_NAMESPACE = "games.tower_defense"
Point = Tuple[float, float]
Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("winding_path", "switchback_path")
SUPPORTED_STYLE_VARIANTS: Tuple[str, ...] = (
    "grass_field",
    "desert_path",
    "blueprint_grid",
    "night_ops",
    "paper_map",
)


@dataclass(frozen=True)
class TowerDefenseDefaults:
    """Stable fallback defaults for tower-defense scenes."""

    covered_path_tower_count_support: Tuple[int, ...] = (3, 4, 5, 6)
    best_position_candidate_count_support: Tuple[int, ...] = (4,)
    nearest_exit_tower_count_support: Tuple[int, ...] = (1, 2)
    nearest_exit_option_count_support: Tuple[int, ...] = (6,)
    path_segment_count_support: Tuple[int, ...] = (10, 11, 12, 13, 14, 15, 16)
    covered_path_target_answer_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    best_position_target_answer_support: Tuple[int, ...] = (2,)
    best_position_answer_option_index_support: Tuple[int, ...] = (0, 1, 2, 3)
    nearest_exit_answer_option_index_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    canvas_width: int = 980
    canvas_height: int = 760
    map_width_px: int = 820
    map_height_px: int = 570
    panel_margin_px: int = 38
    path_width_px: int = 28
    path_node_radius_px: int = 17
    tower_radius_px: int = 19
    enemy_radius_px: int = 13
    range_outline_width_px: int = 3
    label_font_size_px: int = 25
    covered_path_range_radius_min_px: int = 88
    covered_path_range_radius_max_px: int = 168
    best_position_range_radius_min_px: int = 86
    best_position_range_radius_max_px: int = 104
    tower_path_clearance_px: int = 42
    tower_min_gap_px: int = 46
    uncovered_margin_px: int = 28


@dataclass(frozen=True)
class TowerDefenseAxes:
    """Resolved visual and construction axes for one instance."""

    scene_variant: str
    style_variant: str
    tower_count: int
    path_segment_count: int
    target_answer: int
    target_answer_support: Tuple[int, ...]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    tower_count_probabilities: Dict[str, float]
    path_segment_count_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class TowerDefenseTower:
    """One visible tower and circular range."""

    tower_id: str
    center_px: Point
    range_radius_px: float
    covers_target: bool


@dataclass(frozen=True)
class TowerDefenseEnemy:
    """The marked enemy target on the path."""

    enemy_id: str
    center_px: Point
    path_index: int


@dataclass(frozen=True)
class TowerDefenseSample:
    """Generated tower-defense map state and answer."""

    mode: str
    scene_variant: str
    style_variant: str
    map_width_px: int
    map_height_px: int
    path_points_px: Tuple[Point, ...]
    towers: Tuple[TowerDefenseTower, ...]
    enemy: TowerDefenseEnemy | None
    answer: int | str
    target_answer: int
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str
    labeled_path_enemy_options: Tuple[Tuple[int, str], ...] = ()
    show_exit_marker: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TowerDefenseRenderParams:
    """Resolved render controls for one tower-defense scene."""

    canvas_width: int
    canvas_height: int
    map_width_px: int
    map_height_px: int
    panel_margin_px: int
    path_width_px: int
    path_node_radius_px: int
    tower_radius_px: int
    enemy_radius_px: int
    range_outline_width_px: int
    label_font_size_px: int
    layout_jitter_meta: Dict[str, Any] | None = None
    font_family: str = ""


@dataclass(frozen=True)
class TowerDefenseTheme:
    """Resolved tower-defense visual theme."""

    map_fill_rgb: Color
    map_outline_rgb: Color
    terrain_accent_rgb: Color
    path_fill_rgb: Color
    path_outline_rgb: Color
    path_node_rgb: Color
    tower_fill_rgb: Color
    tower_inner_rgb: Color
    tower_outline_rgb: Color
    enemy_fill_rgb: Color
    enemy_outline_rgb: Color
    range_palette_rgb: Tuple[Color, ...]
    terrain_pattern: str


@dataclass(frozen=True)
class RenderedTowerDefenseScene:
    """Rendered tower-defense image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "DOMAIN",
    "Point",
    "RenderedTowerDefenseScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
    "TowerDefenseAxes",
    "TowerDefenseDefaults",
    "TowerDefenseEnemy",
    "TowerDefenseRenderParams",
    "TowerDefenseSample",
    "TowerDefenseTheme",
    "TowerDefenseTower",
]
