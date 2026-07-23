"""State and dataclasses for thermal-mixing diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "thermal_mixing"
SCENE_NAMESPACE = "physics_thermal_mixing"
SCENE_PROMPT_KEY = "thermal_mixing_setup"
CUP_COUNTS: Tuple[str, ...] = ("2", "3", "4")
FINAL_TEMPERATURE_SUPPORT: Tuple[int, ...] = (20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)
OFFSET_PATTERNS: Dict[int, Tuple[Tuple[int, ...], ...]] = {
    2: ((-30, 30), (-25, 25), (-20, 20), (-15, 15), (-10, 10)),
    3: ((-30, 0, 30), (-25, -5, 30), (-20, 0, 20), (-15, -5, 20), (-10, -10, 20)),
    4: ((-30, -10, 15, 25), (-25, -15, 15, 25), (-20, -10, 10, 20), (-15, -15, 10, 20), (-30, 0, 10, 20)),
}
LIQUID_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (93, 159, 224),
    (86, 180, 170),
    (218, 144, 86),
    (160, 141, 219),
    (97, 169, 113),
)


@dataclass(frozen=True)
class ThermalMixingDefaults:
    canvas_width: int = 1180
    canvas_height: int = 760
    panel_left_px: int = 58
    panel_top_px: int = 52
    panel_right_margin_px: int = 58
    panel_bottom_margin_px: int = 56
    cup_width_px: int = 138
    cup_height_px: int = 170
    cup_top_px: int = 176
    cup_gap_px: int = 42
    mixer_width_px: int = 330
    mixer_height_px: int = 172
    mixer_top_px: int = 486
    title_font_size_px: int = 27
    label_font_size_px: int = 24
    temp_font_size_px: int = 25
    note_font_size_px: int = 20
    label_stroke_width_px: int = 2


@dataclass(frozen=True)
class ThermalMixingScenario:
    cup_count: int
    initial_temperatures_c: Tuple[int, ...]
    final_temperature_c: int
    cup_count_probabilities: Dict[str, float]
    final_temperature_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedThermalMixingScene:
    image: Image.Image
    annotation_bboxes: List[List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


__all__ = [
    "CUP_COUNTS",
    "FINAL_TEMPERATURE_SUPPORT",
    "LIQUID_COLORS",
    "OFFSET_PATTERNS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "RenderedThermalMixingScene",
    "ThermalMixingDefaults",
    "ThermalMixingScenario",
]
