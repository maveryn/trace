"""Passive state containers for symbolic agent-automaton rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class AgentStepTrace:
    """One update step in the turning-agent simulation."""

    step: int
    row: int
    col: int
    direction: int
    state_before: int
    state_after: int


@dataclass(frozen=True)
class AgentPoseOption:
    """One visual final-pose option."""

    option_id: str
    label: str
    row: int
    col: int
    direction: int
    pose_text: str
    is_correct: bool


@dataclass(frozen=True)
class AgentGridOption:
    """One visual future-grid option."""

    option_id: str
    label: str
    grid: Tuple[Tuple[int, ...], ...]
    is_correct: bool


@dataclass(frozen=True)
class AgentSceneSpec:
    """Rendered scene inputs shared by both agent-automaton tasks."""

    rows: int
    cols: int
    state_count: int
    initial_grid: Tuple[Tuple[int, ...], ...]
    start_row: int
    start_col: int
    start_direction: int
    traces: Tuple[AgentStepTrace, ...]
    option_specs: Tuple[AgentPoseOption, ...] = tuple()
    grid_option_specs: Tuple[AgentGridOption, ...] = tuple()
    source_marker_label: str = ""


@dataclass(frozen=True)
class AgentSimulationSample:
    """Base simulated agent run before objective-specific target binding."""

    rows: int
    cols: int
    steps: int
    initial_grid: Tuple[Tuple[int, ...], ...]
    final_grid: Tuple[Tuple[int, ...], ...]
    start_row: int
    start_col: int
    start_direction: int
    final_row: int
    final_col: int
    final_direction: int
    traces: Tuple[AgentStepTrace, ...]


@dataclass(frozen=True)
class AgentRenderParams:
    """Resolved rendering parameters for one agent-automaton instance."""

    canvas_width: int
    canvas_height: int
    cell_size_px: int
    grid_gap_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    grid_line_width_px: int
    option_card_width_px: int
    option_card_height_px: int
    option_gap_px: int
    option_grid_cell_px: int
    label_font_size_px: int
    small_font_size_px: int
    arrow_width_px: int
    unit_size_jitter: Dict[str, Any]
    layout_seed: int
    font_family: str


@dataclass(frozen=True)
class RenderedAgentScene:
    """Rendered image metadata for the agent-automaton scene."""

    image: Image.Image
    scene_bbox_px: Tuple[int, int, int, int]
    item_bboxes: Dict[str, Tuple[int, int, int, int]]
    entities: Tuple[Dict[str, Any], ...]
    layout_jitter: Dict[str, Any]
    style_metadata: Dict[str, Any]


@dataclass(frozen=True)
class AgentRenderBundle:
    """Rendered image plus reusable trace metadata from scene rendering."""

    image: Image.Image
    rendered: RenderedAgentScene
    render_params: AgentRenderParams
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    board_style: str
    board_style_probabilities: Dict[str, float]
