"""Passive state for the lens-optics scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "lens_optics"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("clean_axis", "paper_grid", "lab_card")
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")
OBJECT_POSITION_CASES: Tuple[str, ...] = (
    "beyond_2f",
    "at_2f",
    "between_f_2f",
    "inside_f",
)
PROPERTY_TEXT: Dict[str, str] = {
    "real_inverted_smaller": "real, inverted,\nsmaller",
    "real_inverted_same_size": "real, inverted,\nsame size",
    "real_inverted_larger": "real, inverted,\nlarger",
    "virtual_upright_larger": "virtual, upright,\nlarger",
}


@dataclass(frozen=True)
class LensOpticsVisualScenario:
    """Resolved visual inputs needed to render one lens diagram."""

    scene_variant: str
    object_position_case: str
    image_property: str
    correct_option_letter: str
    accent_color_name: str
    option_text_map: Dict[str, str]
    focal_length_px: float
    object_x_factor: float


@dataclass(frozen=True)
class RenderedLensOpticsScene:
    """Rendered lens image plus projected verifier geometry."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
