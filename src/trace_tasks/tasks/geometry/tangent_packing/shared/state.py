"""State objects for tangent-packing geometry diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

DOMAIN = "geometry"
SCENE_ID = "tangent_packing"
SCENE_KIND = "geometry_tangent_packing"
PROMPT_BUNDLE_ID = "geometry_tangent_packing_v1"
SCENE_PROMPT_KEY = "tangent_packing_scene"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class TangentPackingCase:
    """Numeric construction case shared by the six tangent-packing objectives."""

    radius: int

    @property
    def square_side(self) -> int:
        return 2 * int(self.radius)

    @property
    def packed_rectangle_width(self) -> int:
        return 4 * int(self.radius)

    @property
    def packed_rectangle_height(self) -> int:
        return 2 * int(self.radius)


@dataclass(frozen=True)
class TangentPackingProblem:
    """Task-bound case after one public objective has selected its formula."""

    construction_kind: str
    target_kind: str
    support_kind: str
    target_text: str
    support_text: str
    answer: float | int
    case: TangentPackingCase
    formula_family: str
    formula_text: str
    reasoning_steps: int
    answer_type: str = "number"
    answer_rounding: str = "one_decimal"
    radius_probabilities: Mapping[str, float] = field(default_factory=dict)
    answer_support_probabilities: Mapping[str, float] = field(default_factory=dict)


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
    shaded_color: Color
    accent_color: Color
    muted_color: Color
    line_width: int
    font: Any
    small_font: Any
    scene_transform: Any


@dataclass(frozen=True)
class RenderedTangentPackingScene:
    """Rendered diagram and projected witnesses before final output wrapping."""

    image: Image.Image
    answer: float | int
    annotation_bboxes: Mapping[str, BBox]
    annotation_roles: tuple[str, ...]
    label_bboxes: Mapping[str, BBox]
    scene_entities: tuple[dict[str, Any], ...]
    render_map: Mapping[str, Any]
    witness: Mapping[str, Any]


__all__ = [
    "BBox",
    "Color",
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "Point",
    "RenderContext",
    "RenderedTangentPackingScene",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_PROMPT_KEY",
    "TangentPackingCase",
    "TangentPackingProblem",
]
