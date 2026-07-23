"""State models and constants for bridge-circuit diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "bridge_circuit"
SCENE_NAMESPACE = "physics.bridge_circuit"
SCENE_VARIANT = "rectangular_bridge"
BRIDGE_EQUATION = "R1*R4 = R2*R3"
RESISTOR_LABELS: Tuple[str, ...] = ("R1", "R2", "R3", "R4")
DEFAULT_TARGET_SUPPORT: Tuple[int, ...] = tuple(range(1, 21))
DEFAULT_MAX_RESISTANCE = 60


@dataclass(frozen=True)
class BridgeResistor:
    """One semantic bridge resistor."""

    label: str
    value_ohm: int
    is_missing: bool


@dataclass(frozen=True)
class BridgeScenario:
    """Resolved bridge-circuit state."""

    scene_variant: str
    accent_color_name: str
    missing_resistor: str
    target_answer: int
    resistors: Tuple[BridgeResistor, ...]
    scene_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    missing_resistor_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedBridgeScene:
    """Rendered bridge circuit plus projected witness metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class BridgeRenderDefaults:
    """Stable fallback rendering defaults for the bridge-circuit scene."""

    canvas_width: int = 1280
    canvas_height: int = 720
    panel_left_px: int = 56
    panel_top_px: int = 58
    panel_right_px: int = 1224
    panel_bottom_px: int = 662
    wire_width_px: int = 5
    component_label_font_size_px: int = 20
    title_font_size_px: int = 28


DEFAULT_RENDERING = BridgeRenderDefaults()


__all__ = [
    "BRIDGE_EQUATION",
    "DEFAULT_MAX_RESISTANCE",
    "DEFAULT_RENDERING",
    "DEFAULT_TARGET_SUPPORT",
    "RESISTOR_LABELS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "BridgeRenderDefaults",
    "BridgeResistor",
    "BridgeScenario",
    "RenderedBridgeScene",
]
