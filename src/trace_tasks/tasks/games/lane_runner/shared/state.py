"""Passive state contracts for lane-runner game scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SCENE_ID = "lane_runner"
SCENE_NAMESPACE = "games.lane_runner"
SUPPORTED_LANE_RUNNER_SCENE_VARIANTS: Tuple[str, ...] = ("two_lane_track",)
SUPPORTED_LANE_RUNNER_STYLE_VARIANTS: Tuple[str, ...] = (
    "arcade_lane",
    "city_road",
    "forest_path",
    "neon_track",
    "paper_course",
)
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class LaneRunnerDefaults:
    """Stable scene-level fallback defaults for lane-runner rendering."""

    row_count_support: Tuple[int, ...] = (5, 6, 7, 8)
    start_lane_support: Tuple[int, ...] = (0, 1)
    lane_count: int = 2
    cell_size_px: int = 80
    cell_gap_px: int = 7
    panel_margin_px: int = 28
    start_band_height_px: int = 44
    finish_band_height_px: int = 42
    coin_radius_px: int = 17
    runner_radius_px: int = 20
    hazard_radius_px: int = 18
    path_line_width_px: int = 7
    path_label_font_size_px: int = 18
    option_card_cell_size_px: int = 80
    option_card_gap_px: int = 7
    option_card_margin_px: int = 10
    option_card_area_gap_px: int = 28
    grid_line_width_px: int = 3
    label_font_size_px: int = 20
    canvas_outer_margin_px: int = 72
    canvas_min_width: int = 360
    canvas_min_height: int = 430


@dataclass(frozen=True)
class LaneRunnerSceneAxes:
    """Resolved scene axes shared by lane-runner public tasks."""

    scene_variant: str
    style_variant: str
    row_count: int
    lane_count: int
    start_lane: int
    row_count_support: Tuple[int, ...]
    start_lane_support: Tuple[int, ...]
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    row_count_probabilities: Dict[str, float]
    start_lane_probabilities: Dict[str, float]


@dataclass(frozen=True)
class LaneRunnerIntegerAxis:
    """Resolved integer target axis owned by a public task."""

    value: int
    support: Tuple[int, ...]
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class LaneRunnerCoin:
    """One visible coin in the lane-runner grid."""

    coin_id: str
    row: int
    lane: int


@dataclass(frozen=True)
class LaneRunnerHazard:
    """One visible hazard in the lane-runner grid."""

    hazard_id: str
    row: int
    lane: int


@dataclass(frozen=True)
class LaneRunnerPathOption:
    """One labeled candidate route in the lane-runner grid."""

    label: str
    lanes_by_row: Tuple[int, ...]


@dataclass(frozen=True)
class LaneRunnerPathCoinSample:
    """Symbolic shown-path lane-runner instance before rendering."""

    scene_variant: str
    style_variant: str
    row_count: int
    lane_count: int
    start_lane: int
    coins: Tuple[LaneRunnerCoin, ...]
    shown_path_lanes: Tuple[int, ...]
    answer: int
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str


@dataclass(frozen=True)
class LaneRunnerSafePathSample:
    """Symbolic safe-path option-card instance before rendering."""

    scene_variant: str
    style_variant: str
    row_count: int
    lane_count: int
    start_lane: int
    hazards: Tuple[LaneRunnerHazard, ...]
    path_options: Tuple[LaneRunnerPathOption, ...]
    answer_label: str
    safe_path_cell_ids: Tuple[str, ...]
    construction_mode: str


DEFAULTS = LaneRunnerDefaults()


__all__ = [
    "DEFAULTS",
    "OPTION_LABELS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_LANE_RUNNER_SCENE_VARIANTS",
    "SUPPORTED_LANE_RUNNER_STYLE_VARIANTS",
    "LaneRunnerCoin",
    "LaneRunnerDefaults",
    "LaneRunnerHazard",
    "LaneRunnerIntegerAxis",
    "LaneRunnerPathCoinSample",
    "LaneRunnerPathOption",
    "LaneRunnerSafePathSample",
    "LaneRunnerSceneAxes",
]
