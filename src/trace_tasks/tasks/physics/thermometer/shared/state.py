"""State and dataclasses for thermometer diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "thermometer"
SCENE_NAMESPACE = "physics_thermometer"
SCENE_PROMPT_KEY = "thermometer_scale"
LIQUID_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (218, 64, 80),
    (224, 92, 48),
    (198, 54, 92),
    (230, 116, 58),
)


@dataclass(frozen=True)
class ThermometerProfile:
    profile_id: str
    source_unit: str
    target_unit: str
    scale_min: int
    scale_max: int
    major_step: int
    minor_step: int
    source_support: Tuple[int, ...]


PROFILES: Dict[str, ThermometerProfile] = {
    "celsius_weather": ThermometerProfile(
        profile_id="celsius_weather",
        source_unit="C",
        target_unit="F",
        scale_min=-20,
        scale_max=50,
        major_step=10,
        minor_step=5,
        source_support=tuple(range(-15, 50, 5)),
    ),
    "celsius_lab": ThermometerProfile(
        profile_id="celsius_lab",
        source_unit="C",
        target_unit="F",
        scale_min=0,
        scale_max=100,
        major_step=20,
        minor_step=5,
        source_support=tuple(range(5, 100, 5)),
    ),
    "fahrenheit_weather": ThermometerProfile(
        profile_id="fahrenheit_weather",
        source_unit="F",
        target_unit="C",
        scale_min=20,
        scale_max=120,
        major_step=20,
        minor_step=2,
        source_support=(32, 50, 68, 86, 104),
    ),
    "fahrenheit_compact": ThermometerProfile(
        profile_id="fahrenheit_compact",
        source_unit="F",
        target_unit="C",
        scale_min=30,
        scale_max=110,
        major_step=10,
        minor_step=2,
        source_support=(32, 50, 68, 86, 104),
    ),
}


@dataclass(frozen=True)
class ThermometerScenario:
    profile: ThermometerProfile
    source_temperature: int
    target_temperature: int
    scale_profile_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ThermometerGeometry:
    center_x: float
    scale_top: float
    scale_bottom: float
    tube_width: float
    bulb_radius: float
    scale_left: bool


@dataclass(frozen=True)
class RenderedThermometerScene:
    image: Image.Image
    annotation_segment: List[List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class ThermometerDefaults:
    canvas_width: int = 1100
    canvas_height: int = 720
    panel_left_px: int = 58
    panel_top_px: int = 52
    panel_right_margin_px: int = 58
    panel_bottom_margin_px: int = 56
    thermometer_center_x_px: int = 550
    thermometer_scale_top_px: int = 120
    thermometer_scale_bottom_px: int = 562
    tube_width_px: int = 46
    bulb_radius_px: int = 58
    title_font_size_px: int = 28
    tick_font_size_px: int = 22
    unit_font_size_px: int = 34


__all__ = [
    "LIQUID_COLORS",
    "PROFILES",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "RenderedThermometerScene",
    "ThermometerDefaults",
    "ThermometerGeometry",
    "ThermometerProfile",
    "ThermometerScenario",
]
