"""State objects for trapezoid-extension diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

DOMAIN = "geometry"
SCENE_ID = "trapezoid_extension"
SCENE_KIND = "geometry_trapezoid_extension"
PROMPT_BUNDLE_ID = "geometry_trapezoid_extension_v1"
SCENE_PROMPT_KEY = "trapezoid_extension_scene"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class TrapezoidExtensionCase:
    """Numeric construction values for one trapezoid-completion diagram."""

    top_base: int
    extension: int
    height: int
    side: int

    @property
    def bottom_base(self) -> int:
        return int(self.top_base) + int(self.extension)

    @property
    def parallelogram_area(self) -> int:
        return int(self.bottom_base) * int(self.height)

    @property
    def parallelogram_perimeter(self) -> int:
        return 2 * (int(self.bottom_base) + int(self.side))

    @property
    def trapezoid_area(self) -> float:
        return float((int(self.top_base) + int(self.bottom_base)) * int(self.height) / 2.0)


@dataclass(frozen=True)
class LabelSpec:
    """One task-bound measurement label to draw at a named scene position."""

    role: str
    text: str
    position_key: str


@dataclass(frozen=True)
class TrapezoidExtensionProblem:
    """Resolved objective data after a public task selects a formula case."""

    target_text: str
    target_position_key: str
    answer: float
    case: TrapezoidExtensionCase
    formula_family: str
    formula_text: str
    reasoning_steps: int
    support_labels: tuple[LabelSpec, ...]
    target_support_probabilities: Mapping[str, float]
    include_height_in_support: bool = True
    annotation_mode: str = "original_trapezoid_bbox"


@dataclass
class RenderContext:
    """Mutable PIL context plus sampled visual style."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    fill_color: Color
    extension_fill_color: Color
    accent_color: Color
    muted_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    scene_transform: Any


@dataclass(frozen=True)
class RenderedTrapezoidExtensionScene:
    """Rendered diagram and projected role-bound witnesses."""

    image: Image.Image
    answer: float
    annotation_bboxes: Mapping[str, BBox]
    annotation_roles: tuple[str, ...]
    annotation_mode: str
    annotation_segment: tuple[Point, Point] | None
    annotation_bbox: BBox | None
    label_bboxes: Mapping[str, BBox]
    scene_entities: tuple[dict[str, Any], ...]
    render_map: Mapping[str, Any]
    witness: Mapping[str, Any]


__all__ = [
    "BBox",
    "Color",
    "DOMAIN",
    "LabelSpec",
    "PROMPT_BUNDLE_ID",
    "Point",
    "RenderContext",
    "RenderedTrapezoidExtensionScene",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_PROMPT_KEY",
    "TrapezoidExtensionCase",
    "TrapezoidExtensionProblem",
]
