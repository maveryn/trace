"""Passive state records for graduated-cylinder diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


SCENE_ID = "graduated_cylinder"
SCENE_NAMESPACE = "physics_graduated_cylinder"


@dataclass(frozen=True)
class CylinderScale:
    """Readable graduated-cylinder scale definition."""

    capacity_ml: int
    major_tick_ml: int
    minor_tick_ml: int


@dataclass(frozen=True)
class CylinderGeometry:
    """Pixel geometry for one drawn graduated cylinder."""

    left: float
    top: float
    width: float
    height: float
    bottom: float
    scale_left: bool


@dataclass(frozen=True)
class GraduatedCylinderRenderDefaults:
    """Stable code fallbacks for graduated-cylinder rendering."""

    canvas_width: int = 1040
    canvas_height: int = 720


@dataclass(frozen=True)
class RenderedCylinderScene:
    """Rendered graduated-cylinder scene and projected annotations."""

    image: Image.Image
    annotation_bbox_map: dict[str, list[float]]
    scene_entities: list[dict[str, Any]]
    render_map: dict[str, Any]
    background_meta: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    font_family: str


SCALE_OPTIONS: tuple[CylinderScale, ...] = (
    CylinderScale(50, 10, 5),
    CylinderScale(80, 20, 5),
    CylinderScale(100, 20, 5),
    CylinderScale(120, 20, 10),
)
LIQUID_PALETTE: tuple[tuple[int, int, int], ...] = (
    (93, 168, 218),
    (72, 181, 154),
    (146, 183, 77),
    (120, 148, 226),
)
