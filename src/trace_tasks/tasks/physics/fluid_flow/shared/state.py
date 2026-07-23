"""Passive state records for fluid-flow continuity diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "fluid_flow"
SCENE_NAMESPACE = "physics_fluid_flow"

SUPPORTED_ORIENTATIONS: tuple[str, ...] = ("horizontal_pipe", "vertical_pipe")
SUPPORTED_MISSING_STATIONS: tuple[str, ...] = ("v1", "v2")
ANNOTATION_KEYS: tuple[str, ...] = ("missing_speed_label",)

DEFAULT_AREA_CM2_SUPPORT: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 9, 10, 12)
DEFAULT_SPEED_M_S_SUPPORT: tuple[int, ...] = (
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    15,
    16,
    18,
    20,
    24,
)


@dataclass(frozen=True)
class FluidFlowRenderDefaults:
    """Stable code fallbacks for fluid-flow rendering."""

    canvas_width: int = 1120
    canvas_height: int = 720
    panel_margin_x_px: int = 58
    panel_margin_top_px: int = 52
    panel_margin_bottom_px: int = 58
    label_font_size_px: int = 24
    station_font_size_px: int = 28
    title_font_size_px: int = 28


@dataclass(frozen=True)
class FlowScenario:
    """Resolved continuity scenario for a two-station flow diagram."""

    orientation: str
    missing_station: str
    area_1_cm2: int
    area_2_cm2: int
    speed_1_m_s: int
    speed_2_m_s: int
    target_answer: int
    orientation_probabilities: dict[str, float]
    missing_station_probabilities: dict[str, float]
    area_1_cm2_probabilities: dict[str, float]
    area_2_cm2_probabilities: dict[str, float]
    speed_1_m_s_probabilities: dict[str, float]
    speed_2_m_s_probabilities: dict[str, float]
    target_answer_probabilities: dict[str, float]


@dataclass(frozen=True)
class RenderedFlowScene:
    """Rendered fluid-flow scene and projected role annotations."""

    image: Image.Image
    annotation_bbox_map: dict[str, list[float]]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str
