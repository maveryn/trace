"""State models and constants for equivalent circuit diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "circuit_equivalent"
SCENE_NAMESPACE = "physics.circuit_equivalent"
SCENE_VARIANT = "series_parallel"
SCENE_VARIANTS: Tuple[str, ...] = (SCENE_VARIANT,)
COMPONENT_KINDS: Tuple[str, str] = ("resistor", "capacitor")
RESISTANCE_SUPPORT_KEY = "total_resistance_target_answer_support"
CAPACITANCE_SUPPORT_KEY = "total_capacitance_target_answer_support"


@dataclass(frozen=True)
class EquivalentCircuitDefaults:
    """Stable fallback defaults for equivalent circuit scenes."""

    canvas_width: int = 1280
    canvas_height: int = 720
    terminal_left_x_px: int = 96
    terminal_radius_px: int = 12
    terminal_font_size_px: int = 24
    wire_width_px: int = 5
    component_symbol_width_px: int = 118
    component_symbol_height_px: int = 56
    component_label_font_size_px: int = 20
    label_stroke_width_px: int = 0
    parallel_rail_left_x_px: int = 268
    parallel_branch_top_y_px: int = 180
    parallel_branch_bottom_y_px: int = 470
    component_value_min: int = 1
    component_value_max: int = 60
    total_resistance_target_answer_support: Tuple[int, ...] = tuple(range(1, 21))
    total_capacitance_target_answer_support: Tuple[int, ...] = tuple(range(1, 21))
    parallel_component_count_options: Tuple[int, ...] = (2, 3, 4)
    parallel_block_count_options: Tuple[int, ...] = (1, 2)
    series_parallel_branch_count_options: Tuple[int, ...] = (2, 3)


@dataclass(frozen=True)
class EquivalentCircuitScenario:
    """Resolved symbolic equivalent-circuit scenario."""

    scene_variant: str
    component_kind: str
    accent_color_name: str
    target_answer: int
    target_answer_support: Tuple[int, ...]
    scene_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class EquivalentCircuitLayout:
    """One sampled equivalent-circuit topology with exact equivalent value."""

    scene_variant: str
    component_kind: str
    series_values: Tuple[int, ...]
    parallel_values: Tuple[int, ...]
    target_answer: int
    equivalent_value: Fraction
    parallel_blocks: Tuple[Tuple[int, ...], ...] = tuple()
    inter_block_series_values: Tuple[int, ...] = tuple()
    outer_series_values: Tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class RenderedEquivalentCircuitScene:
    """Rendered equivalent circuit plus projected witness metadata."""

    image: Image.Image
    component_specs: List[Any]
    annotation_bbox: List[float]
    annotation_bbox_map: Dict[str, List[float]]
    annotation_entity_id_map: Dict[str, str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    render_spec: Dict[str, Any]
    font_family: str


DEFAULT_RENDERING = EquivalentCircuitDefaults()


__all__ = [
    "CAPACITANCE_SUPPORT_KEY",
    "COMPONENT_KINDS",
    "DEFAULT_RENDERING",
    "EquivalentCircuitDefaults",
    "EquivalentCircuitLayout",
    "EquivalentCircuitScenario",
    "RenderedEquivalentCircuitScene",
    "RESISTANCE_SUPPORT_KEY",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANT",
    "SCENE_VARIANTS",
]
