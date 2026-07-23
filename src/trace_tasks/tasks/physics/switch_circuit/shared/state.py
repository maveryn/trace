"""State and dataclasses for switch-circuit diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "switch_circuit"
SCENE_NAMESPACE = "physics_switch_circuit"
SCENE_PROMPT_KEY = "switch_circuit_diagram"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("mixed_branch",)
SWITCH_LABELS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")
BULB_LABELS: Tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")
TARGET_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
POS_NODE = "P"
NEG_NODE = "N"


@dataclass(frozen=True)
class SwitchCircuitDefaults:
    canvas_width: int = 1280
    canvas_height: int = 720
    panel_left_px: int = 56
    panel_top_px: int = 58
    panel_right_px: int = 1224
    panel_bottom_px: int = 662
    wire_width_px: int = 5
    bulb_radius_px: int = 27
    switch_width_px: int = 74
    switch_height_px: int = 46
    component_label_font_size_px: int = 20
    title_font_size_px: int = 28
    label_stroke_width_px: int = 2


@dataclass(frozen=True)
class CircuitEdge:
    edge_id: str
    kind: str
    node_a: str
    node_b: str
    label: str
    conductive: bool


@dataclass(frozen=True)
class SwitchCircuitScenario:
    public_query_id: str
    scene_variant: str
    target_answer: int
    accent_color_name: str
    switch_states: Dict[str, bool]
    edges: Tuple[CircuitEdge, ...]
    lit_bulbs: Tuple[str, ...]
    query_id_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedSwitchCircuitScene:
    image: Image.Image
    annotation_bboxes: List[List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


__all__ = [
    "BULB_LABELS",
    "NEG_NODE",
    "POS_NODE",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "SUPPORTED_SCENE_VARIANTS",
    "SWITCH_LABELS",
    "TARGET_SUPPORT",
    "CircuitEdge",
    "RenderedSwitchCircuitScene",
    "SwitchCircuitDefaults",
    "SwitchCircuitScenario",
]
