"""State models and constants for buoyancy-density diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "buoyancy_density"
SCENE_NAMESPACE = "physics.buoyancy_density"
SCENE_VARIANTS: Tuple[str, ...] = (
    "rectangular_tank",
    "beaker_tank",
    "wide_tank",
)
OBJECT_SHAPES: Tuple[str, ...] = (
    "block",
    "rounded_block",
)
DEFAULT_FRACTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (9, 10),
)
ANNOTATION_KEYS: Tuple[str, ...] = (
    "floating_object",
    "waterline",
    "fluid_density_label",
    "submerged_fraction_marker",
)


@dataclass(frozen=True)
class BuoyancyRenderDefaults:
    """Stable fallback defaults for buoyancy-density scenes."""

    canvas_width: int = 1120
    canvas_height: int = 720
    panel_left_px: int = 54
    panel_top_px: int = 52
    panel_right_margin_px: int = 54
    panel_bottom_margin_px: int = 58
    tank_left_px: int = 172
    tank_top_px: int = 160
    tank_width_px: int = 700
    tank_height_px: int = 410
    waterline_y_px: int = 346
    object_center_x_px: int = 520
    object_width_px: int = 118
    object_part_height_px: int = 44
    label_font_size_px: int = 25
    small_font_size_px: int = 20
    title_font_size_px: int = 28
    label_stroke_width_px: int = 2
    marker_width_px: int = 4


@dataclass(frozen=True)
class BuoyancyScenario:
    """Symbolic floating-object scenario."""

    scene_variant: str
    object_shape: str
    fraction_num: int
    fraction_den: int
    liquid_density_tenths: int
    object_density_tenths: int
    target_answer: float
    answer_support: Tuple[float, ...]
    target_answer_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    object_shape_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedBuoyancyScene:
    """Rendered buoyancy diagram plus annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    render_spec: Dict[str, Any]
    font_family: str


DEFAULT_RENDERING = BuoyancyRenderDefaults()


__all__ = [
    "ANNOTATION_KEYS",
    "DEFAULT_FRACTIONS",
    "DEFAULT_RENDERING",
    "OBJECT_SHAPES",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANTS",
    "BuoyancyRenderDefaults",
    "BuoyancyScenario",
    "RenderedBuoyancyScene",
]
