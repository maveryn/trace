"""State objects for the physics motion-graph scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "motion_graph"
SCENE_NAMESPACE = "physics_motion_graph"

SUPPORTED_SCENE_STYLES: Tuple[str, ...] = (
    "clean_grid",
    "paper_grid",
    "bold_grid",
)
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")
SPEED_CHANGE_STATES: Tuple[str, ...] = (
    "speeding_up",
    "slowing_down",
    "constant_speed",
)
STATE_LABELS: Dict[str, str] = {
    "speeding_up": "speeding up",
    "slowing_down": "slowing down",
    "constant_speed": "constant speed",
    "changing_direction": "changing direction",
}


@dataclass(frozen=True)
class MotionGraphRenderDefaults:
    """Stable fallback defaults for motion-graph diagrams."""

    canvas_width: int = 1120
    canvas_height: int = 760
    plot_left_px: int = 132
    plot_top_px: int = 74
    plot_width_px: int = 856
    plot_height_px: int = 500
    option_panel_top_px: int = 610
    option_cell_left_px: int = 70
    option_cell_width_px: int = 468
    option_cell_height_px: int = 66
    option_cell_gap_x_px: int = 24
    option_cell_gap_y_px: int = 14
    axis_width_px: int = 5
    curve_width_px: int = 7
    grid_line_width_px: int = 1
    bold_grid_line_width_px: int = 2
    label_font_size_px: int = 25
    tick_font_size_px: int = 18
    option_font_size_px: int = 22
    title_font_size_px: int = 28
    label_stroke_width_px: int = 2
    point_radius_px: int = 5
    y_min: int = -5
    y_max: int = 5
    t_min: int = 0
    t_max: int = 6


@dataclass(frozen=True)
class StateChoiceAxes:
    """Resolved state-choice sampling axes."""

    scene_style: str
    motion_state: str
    correct_option_letter: str
    scene_style_probabilities: Dict[str, float]
    motion_state_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]


@dataclass(frozen=True)
class StateGraphSpec:
    """Symbolic position-time or velocity-time graph for state-choice tasks."""

    scene_style: str
    operation: str
    graph_kind: str
    motion_state: str
    correct_option_letter: str
    option_map: Dict[str, str]
    t_values: Tuple[int, ...]
    y_values: Tuple[int, ...]
    target_segment_index: int
    y_axis_label: str
    title: str


@dataclass(frozen=True)
class IntervalAxes:
    """Resolved interval-displacement scene axes."""

    scene_style: str
    scene_style_probabilities: Dict[str, float]


@dataclass(frozen=True)
class IntervalGraphSpec:
    """Symbolic velocity-time graph for interval displacement."""

    scene_style: str
    segment_mode: str
    graph_kind: str
    t_values: Tuple[int, ...]
    velocity_values: Tuple[int, ...]
    t_start: int
    t_end: int
    v_start: int
    v_end: int
    displacement_m: int
    y_axis_label: str
    title: str


@dataclass(frozen=True)
class AverageSpeedGraphSpec:
    """Symbolic distance-time graph for average-speed readout."""

    scene_style: str
    graph_kind: str
    t_values: Tuple[int, ...]
    distance_values: Tuple[int, ...]
    t_start: int
    t_end: int
    d_start: int
    d_end: int
    average_speed_m_s: int
    y_axis_label: str
    title: str


@dataclass(frozen=True)
class RenderedMotionGraph:
    """Rendered graph plus projected annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


__all__ = [
    "AverageSpeedGraphSpec",
    "IntervalAxes",
    "IntervalGraphSpec",
    "MotionGraphRenderDefaults",
    "OPTION_LETTERS",
    "RenderedMotionGraph",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SPEED_CHANGE_STATES",
    "STATE_LABELS",
    "SUPPORTED_SCENE_STYLES",
    "StateChoiceAxes",
    "StateGraphSpec",
]
