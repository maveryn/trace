"""Passive state objects for orbital-motion diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "orbital_motion"
SCENE_NAMESPACE = "physics_orbital_motion"
FOCUS_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SPEED_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
SPEED_DIRECTIONS: Tuple[str, str] = ("greatest", "least")


@dataclass(frozen=True)
class OrbitRenderDefaults:
    """Stable fallback defaults for orbital-motion diagrams."""

    canvas_width: int = 1040
    canvas_height: int = 720


@dataclass(frozen=True)
class OrbitSpec:
    """Symbolic ellipse and candidate-point layout for one orbit diagram."""

    center: Tuple[float, float]
    semi_major: float
    semi_minor: float
    rotation_rad: float
    focus_side: int
    candidate_points: Dict[str, Tuple[float, float]]
    selected_label: str
    selected_point: Tuple[float, float]
    sun_point: Tuple[float, float] | None
    major_axis_endpoints: Tuple[Tuple[float, float], Tuple[float, float]]
    eccentricity: float


@dataclass(frozen=True)
class RenderedOrbitScene:
    """Rendered orbit diagram plus projected render metadata."""

    image: Image.Image
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str


__all__ = [
    "FOCUS_OPTION_LABELS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SPEED_DIRECTIONS",
    "SPEED_OPTION_LABELS",
    "OrbitRenderDefaults",
    "OrbitSpec",
    "RenderedOrbitScene",
]
