"""State models and constants for bulb-circuit diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "bulb_circuit"
SCENE_NAMESPACE = "physics.bulb_circuit"
SCENE_VARIANTS: Tuple[str, ...] = (
    "series_unequal",
    "parallel_unequal",
    "mixed_branch",
)
BULB_LABELS: Tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")
TARGET_DIRECTIONS: Tuple[str, ...] = ("brightest", "dimmest")
DEFAULT_RESISTANCE_OPTIONS: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)


@dataclass(frozen=True)
class BulbSpec:
    """One visible bulb with a computed relative power."""

    slot_id: str
    label: str
    resistance_ohm: int
    relative_power: float


@dataclass(frozen=True)
class BulbScenario:
    """Resolved semantic bulb-circuit scenario."""

    target_direction: str
    scene_variant: str
    accent_color_name: str
    branch_single_position: str
    bulbs: Tuple[BulbSpec, ...]
    correct_label: str
    brightest_label: str
    dimmest_label: str
    scene_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_label_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedBulbScene:
    """Rendered bulb circuit plus projected witness metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class BulbRenderDefaults:
    """Stable fallback rendering defaults for bulb-circuit diagrams."""

    canvas_width: int = 1280
    canvas_height: int = 720
    panel_left_px: int = 56
    panel_top_px: int = 58
    panel_right_px: int = 1224
    panel_bottom_px: int = 662
    wire_width_px: int = 5
    bulb_label_font_size_px: int = 20
    title_font_size_px: int = 28


DEFAULT_RENDERING = BulbRenderDefaults()


__all__ = [
    "BULB_LABELS",
    "DEFAULT_RENDERING",
    "DEFAULT_RESISTANCE_OPTIONS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANTS",
    "TARGET_DIRECTIONS",
    "BulbRenderDefaults",
    "BulbScenario",
    "BulbSpec",
    "RenderedBulbScene",
]
