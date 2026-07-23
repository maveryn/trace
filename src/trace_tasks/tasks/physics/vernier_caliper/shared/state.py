"""Scene state models and constants for Vernier-caliper diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "vernier_caliper"
SCENE_NAMESPACE = "physics.vernier_caliper"
VERNIER_DIVISIONS = 10
VERNIER_RESOLUTION_MM = 0.1
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class CaliperScenario:
    """Resolved Vernier-caliper measurement."""

    main_mm: int
    aligned_vernier_tick: int
    answer_mm: float
    option_values_mm: Dict[str, float]
    correct_option_letter: str
    correct_option_letter_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    main_mm_probabilities: Dict[str, float]
    aligned_vernier_tick_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedScene:
    """Rendered caliper and verifier payload fragments."""

    image: Image.Image
    annotation_point_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class TaskDefaults:
    """Stable fallback defaults for Vernier-caliper scenes."""

    canvas_width: int = 1180
    canvas_height: int = 720
    panel_left_px: int = 58
    panel_top_px: int = 54
    panel_right_margin_px: int = 58
    panel_bottom_margin_px: int = 58
    main_scale_left_px: int = 174
    main_scale_y_px: int = 266
    mm_px: int = 13
    main_scale_max_mm: int = 62
    label_font_size_px: int = 22
    small_font_size_px: int = 17
    title_font_size_px: int = 28
    jaw_top_px: int = 170
    jaw_bottom_px: int = 510
    option_panel_top_px: int = 565
    option_cell_height_px: int = 42
    main_mm_support: Tuple[int, ...] = tuple(range(8, 56))
    aligned_vernier_tick_support: Tuple[int, ...] = tuple(range(1, VERNIER_DIVISIONS))


DEFAULTS = TaskDefaults()
