"""State and dataclasses for waveform-panel diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "waveform_panel"
SCENE_NAMESPACE = "physics_waveform_panel"
SCENE_PROMPT_KEY = "waveform_panel_diagram"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("clean_stack", "grid_stack", "lab_sheet")
PANEL_COUNT_SUPPORT: Tuple[int, ...] = (4, 5, 6)
PANEL_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class WaveformPanelDefaults:
    """Stable fallback defaults for waveform-panel diagrams."""

    canvas_width: int = 1180
    canvas_height: int = 760
    sheet_left_px: int = 58
    sheet_top_px: int = 52
    sheet_right_margin_px: int = 58
    sheet_bottom_margin_px: int = 44
    stack_left_px: int = 96
    stack_top_px: int = 112
    stack_width_px: int = 988
    stack_height_px: int = 570
    panel_gap_px: int = 10
    label_font_size_px: int = 24
    title_font_size_px: int = 27
    wave_line_width_px: int = 4
    grid_line_width_px: int = 1
    midline_width_px: int = 2


@dataclass(frozen=True)
class WaveformAxes:
    """Resolved public-task-independent scene axes."""

    scene_variant: str
    panel_count: int
    correct_option_letter: str
    scene_variant_probabilities: Dict[str, float]
    panel_count_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class WaveformPanelSemanticSpec:
    """One labeled waveform panel before pixel layout is known."""

    label: str
    amplitude_rank: int
    cycle_count: int
    is_correct: bool


@dataclass(frozen=True)
class RenderedWaveformPanel:
    """One rendered waveform panel with final-image geometry."""

    label: str
    amplitude_rank: int
    cycle_count: int
    amplitude_px: float
    bbox_px: List[float]
    wave_bbox_px: List[float]
    label_bbox_px: List[float]
    is_correct: bool


@dataclass(frozen=True)
class RenderedWaveformPanelScene:
    """Rendered waveform-panel scene plus verifier-facing metadata."""

    image: Image.Image
    selected_panel_bbox: List[float]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


__all__ = [
    "PANEL_COUNT_SUPPORT",
    "PANEL_LABELS",
    "RenderedWaveformPanel",
    "RenderedWaveformPanelScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "SUPPORTED_SCENE_VARIANTS",
    "WaveformAxes",
    "WaveformPanelDefaults",
    "WaveformPanelSemanticSpec",
]
