"""State containers for circular-sector formula diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class SectorValues:
    """Numeric values for one circular-sector formula case."""

    radius_units: int
    theta_degrees: int
    arc_length: float
    sector_area: float
    angle_from_arc_length: float
    angle_from_sector_area: float
    adjacent_angle_degrees: int | None = None
    target_angle_total: int | None = None


@dataclass(frozen=True)
class SectorProblem:
    """One solved sector objective with semantic render instructions."""

    answer: float
    values: SectorValues
    formula_family: str
    visual_case: str
    target_kind: str
    visible_measure_kind: str
    reasoning_steps: int
    case_index: int
    layout_seed: int


@dataclass
class RenderContext:
    """Styled canvas resources for one sector rendering."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    fill_color: Color
    secondary_fill_color: Color
    accent_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    fill_style_meta: dict[str, Any]
    scene_transform: LazySceneTransform


@dataclass(frozen=True)
class RenderedSectorScene:
    """Rendered image plus projected sector annotation geometry."""

    image: Image.Image
    annotation_bboxes: dict[str, BBox]
    scene_entities: tuple[dict[str, Any], ...]
    render_map: dict[str, Any]
    witness: dict[str, Any]
    reasoning_steps: int


@dataclass(frozen=True)
class SectorObjectivePlan:
    """Task-owned objective binding prepared after query selection."""

    prompt_task_key: str
    prompt_branch_key: str
    problem: SectorProblem
    answer_type: str
    annotation_roles: tuple[str, ...]
    replay_params: Mapping[str, Any]
    trace_values: Mapping[str, Any]


__all__ = [
    "BBox",
    "Color",
    "Point",
    "RenderContext",
    "RenderedSectorScene",
    "SectorObjectivePlan",
    "SectorProblem",
    "SectorValues",
]
