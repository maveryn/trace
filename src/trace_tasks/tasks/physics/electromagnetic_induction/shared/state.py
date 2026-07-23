"""Passive state records for electromagnetic induction panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "electromagnetic_induction"
SCENE_NAMESPACE = "physics_electromagnetic_induction"

CURRENT_CLASSES: tuple[str, ...] = ("clockwise", "counterclockwise", "no_current")
FIELD_ORIENTATIONS: tuple[str, ...] = ("into_page", "out_of_page")
ANSWER_SUPPORT: tuple[int, ...] = tuple(range(7))
PANEL_COUNT = 6
PANEL_MECHANISMS_BY_FLUX_CHANGE: dict[str, tuple[str, ...]] = {
    "increasing": ("loop_enters_field", "field_strength_increases", "loop_area_expands"),
    "decreasing": ("loop_leaves_field", "field_strength_decreases", "loop_area_contracts"),
    "none": ("loop_slides_inside_uniform_field", "stationary_constant_field"),
}


@dataclass(frozen=True)
class PanelSpec:
    """Symbolic state for one visible induction mini-panel."""

    panel_id: str
    current_class: str
    field_orientation: str
    flux_change: str
    mechanism: str
    region_side: str
    bbox_px: list[float]


@dataclass(frozen=True)
class InductionScenario:
    """Resolved six-panel induction scene for one target current class."""

    target_current_class: str
    target_answer: int
    panels: tuple[PanelSpec, ...]
    target_answer_probabilities: dict[str, float]


@dataclass(frozen=True)
class RenderedInductionScene:
    """Rendered induction diagram and projected panel witnesses."""

    image: Image.Image
    annotation_bboxes: list[list[float]]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str
