"""Passive state containers for symbolic logic-gate circuit scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image


SCENE_ID = "logic_gate_circuit"
SUPPORTED_GATE_TYPES: tuple[str, ...] = ("AND", "OR", "NOT", "XOR", "NAND", "NOR")
SCENE_VARIANTS: tuple[str, ...] = ("clean_worksheet", "notebook_problem", "exam_scan")
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
INPUT_LABELS: tuple[str, ...] = ("x", "y", "z")


@dataclass(frozen=True)
class LogicInputSpec:
    item_id: str
    label: str
    value: int | None = None


@dataclass(frozen=True)
class LogicGateSpec:
    item_id: str
    gate_type: str
    input_signal_ids: tuple[str, ...]
    output_signal_id: str


@dataclass(frozen=True)
class LogicCircuitSpec:
    item_id: str
    label: str
    inputs: tuple[LogicInputSpec, ...]
    gates: tuple[LogicGateSpec, ...]
    output_signal_id: str
    output_value: int | None = None
    role: str = "circuit"


@dataclass(frozen=True)
class CandidateAssignmentSpec:
    item_id: str
    label: str
    values: Mapping[str, int]
    output_value: int
    is_correct: bool = False


@dataclass(frozen=True)
class LogicGateRenderParams:
    canvas_width: int = 1180
    canvas_height: int = 820
    card_corner_radius_px: int = 18
    card_border_width_px: int = 2
    gate_width_px: int = 72
    gate_height_px: int = 46
    wire_width_px: int = 3
    node_radius_px: int = 5
    label_font_size_px: int = 22
    small_font_size_px: int = 16
    gate_font_size_px: int = 15
    table_font_size_px: int = 20


@dataclass(frozen=True)
class RenderedLogicGateScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    output_points: dict[str, list[float]]
    signal_points: dict[str, list[float]]
    wire_segments: tuple[dict[str, Any], ...]
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


@dataclass(frozen=True)
class LogicGateRenderBundle:
    image: Image.Image
    rendered: RenderedLogicGateScene
    render_params: LogicGateRenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    scene_style_meta: dict[str, Any]
