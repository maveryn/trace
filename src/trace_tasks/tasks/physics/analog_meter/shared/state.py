"""Scene state models and static meter profiles for analog-meter diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "analog_meter"
SCENE_NAMESPACE = "physics.analog_meter"


@dataclass(frozen=True)
class MeterProfile:
    """One analog meter scale profile."""

    profile_id: str
    meter_kind: str
    meter_name: str
    unit: str
    scale_max: int
    major_step: int
    minor_step: int
    answer_support: Tuple[int, ...]


@dataclass(frozen=True)
class MeterScenario:
    """Resolved apparatus state for one generated analog meter."""

    profile: MeterProfile
    readout_value: int
    meter_profile_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedScene:
    """Rendered analog meter plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class TaskDefaults:
    """Stable fallback defaults for analog meter scenes."""

    canvas_width: int = 1120
    canvas_height: int = 720
    panel_left_px: int = 64
    panel_top_px: int = 54
    panel_right_margin_px: int = 64
    panel_bottom_margin_px: int = 56
    meter_center_x_px: int = 560
    meter_center_y_px: int = 472
    scale_radius_px: int = 270
    needle_radius_px: int = 222
    face_radius_px: int = 318
    label_font_size_px: int = 28
    tick_font_size_px: int = 20
    unit_font_size_px: int = 34
    title_font_size_px: int = 24


DEFAULTS = TaskDefaults()
METER_PROFILES: Dict[str, MeterProfile] = {
    "ammeter_a": MeterProfile(
        profile_id="ammeter_a",
        meter_kind="ammeter",
        meter_name="ammeter",
        unit="A",
        scale_max=10,
        major_step=2,
        minor_step=1,
        answer_support=tuple(range(1, 10)),
    ),
    "ammeter_ma": MeterProfile(
        profile_id="ammeter_ma",
        meter_kind="ammeter",
        meter_name="ammeter",
        unit="mA",
        scale_max=100,
        major_step=20,
        minor_step=10,
        answer_support=tuple(range(10, 100, 10)),
    ),
    "voltmeter_v": MeterProfile(
        profile_id="voltmeter_v",
        meter_kind="voltmeter",
        meter_name="voltmeter",
        unit="V",
        scale_max=12,
        major_step=2,
        minor_step=1,
        answer_support=tuple(range(1, 12)),
    ),
}
PROFILE_IDS_BY_METER_KIND: Dict[str, Tuple[str, ...]] = {
    "ammeter": ("ammeter_a", "ammeter_ma"),
    "voltmeter": ("voltmeter_v",),
}


__all__ = [
    "DEFAULTS",
    "METER_PROFILES",
    "PROFILE_IDS_BY_METER_KIND",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "MeterProfile",
    "MeterScenario",
    "RenderedScene",
    "TaskDefaults",
]
