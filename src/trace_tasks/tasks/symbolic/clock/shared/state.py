"""Passive state objects for symbolic clock-display scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image


SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "minimal",
    "outline",
)


@dataclass(frozen=True)
class ClockRenderParams:
    """Concrete analog-clock render parameters resolved from config."""

    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    face_radius_px: int
    bezel_width_px: int
    numeral_font_size_px: int
    major_tick_length_px: int
    minor_tick_length_px: int
    major_tick_width_px: int
    minor_tick_width_px: int
    minor_tick_dot_radius_px: int
    hour_hand_width_px: int
    minute_hand_width_px: int
    second_hand_width_px: int
    hand_bbox_padding_px: int
    center_dot_radius_px: int
    inner_ring_inset_px: int
    inner_ring_width_px: int


@dataclass(frozen=True)
class RenderedClockGeometry:
    """Rendered analog-clock geometry plus scene entities."""

    face_bbox_px: Tuple[float, float, float, float]
    center_px: Tuple[float, float]
    hour_hand_bbox_px: Tuple[float, float, float, float]
    minute_hand_bbox_px: Tuple[float, float, float, float]
    second_hand_bbox_px: Tuple[float, float, float, float] | None
    alarm_hand_bbox_px: Tuple[float, float, float, float] | None
    hour_hand_tip_px: Tuple[float, float]
    minute_hand_tip_px: Tuple[float, float]
    second_hand_tip_px: Tuple[float, float] | None
    alarm_hand_tip_px: Tuple[float, float] | None
    entities: List[Dict[str, Any]]


@dataclass(frozen=True)
class RenderedClockScene:
    """Rendered analog-clock image plus witness geometry."""

    image: Image.Image
    scene_bbox_px: Tuple[float, float, float, float]
    face_bbox_px: Tuple[float, float, float, float]
    center_px: Tuple[float, float]
    hour_hand_bbox_px: Tuple[float, float, float, float]
    minute_hand_bbox_px: Tuple[float, float, float, float]
    second_hand_bbox_px: Tuple[float, float, float, float] | None
    alarm_hand_bbox_px: Tuple[float, float, float, float] | None
    hour_hand_tip_px: Tuple[float, float]
    minute_hand_tip_px: Tuple[float, float]
    second_hand_tip_px: Tuple[float, float] | None
    alarm_hand_tip_px: Tuple[float, float] | None
    entities: List[Dict[str, Any]]


@dataclass(frozen=True)
class ClockTextOptionSpec:
    """Visible text answer cards for clock MCQ readout/value tasks."""

    labels: Tuple[str, ...]
    correct_label: str
    text_by_label: Mapping[str, str]
    value_by_label: Mapping[str, Any]


@dataclass(frozen=True)
class ClockStyleResolution:
    """Resolved visual axes for one clock rendering."""

    scene_variant: str
    style_variant: str
    accent_color_name: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


__all__ = [
    "ClockRenderParams",
    "ClockStyleResolution",
    "ClockTextOptionSpec",
    "RenderedClockGeometry",
    "RenderedClockScene",
    "SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS",
]
