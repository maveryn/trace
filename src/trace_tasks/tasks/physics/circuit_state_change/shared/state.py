"""State models and constants for circuit state-change diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "circuit_state_change"
SCENE_NAMESPACE = "physics.circuit_state_change"
BULB_LABELS: Tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")
BULB_ROLES: Tuple[str, ...] = (
    "series_bulb",
    "main_branch_bulb",
    "switched_branch_bulb",
    "reference_branch_bulb_1",
    "reference_branch_bulb_2",
)
CHANGE_CLASSES: Tuple[str, ...] = ("brightens", "dims", "turns_on", "turns_off")
DEFAULT_RESISTANCE_OPTIONS: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)


@dataclass(frozen=True)
class BulbStateChangeSpec:
    """One visible bulb with before/after power metadata."""

    role: str
    label: str
    resistance_ohm: int
    power_before: float
    power_after: float
    change_class: str


@dataclass(frozen=True)
class CircuitStateChangeScenario:
    """Resolved symbolic state-change circuit scenario."""

    switch_action: str
    target_change_class: str
    accent_color_name: str
    target_label: str
    bulbs: Tuple[BulbStateChangeSpec, ...]
    correct_label: str
    switch_action_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_label_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedCircuitStateChangeScene:
    """Rendered circuit state-change scene plus projected annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class CircuitStateChangeRenderDefaults:
    """Stable fallback rendering defaults for circuit state-change diagrams."""

    canvas_width: int = 1280
    canvas_height: int = 720
    panel_left_px: int = 56
    panel_top_px: int = 58
    panel_right_px: int = 1224
    panel_bottom_px: int = 662
    wire_width_px: int = 5
    component_label_font_size_px: int = 20
    title_font_size_px: int = 28


DEFAULT_RENDERING = CircuitStateChangeRenderDefaults()


__all__ = [
    "BULB_LABELS",
    "BULB_ROLES",
    "CHANGE_CLASSES",
    "DEFAULT_RENDERING",
    "DEFAULT_RESISTANCE_OPTIONS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "BulbStateChangeSpec",
    "CircuitStateChangeRenderDefaults",
    "CircuitStateChangeScenario",
    "RenderedCircuitStateChangeScene",
]
