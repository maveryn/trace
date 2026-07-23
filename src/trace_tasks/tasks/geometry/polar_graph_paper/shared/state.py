"""State objects for polar graph paper rendering and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ReadoutComponent = Literal["radius", "angle_degrees"]


@dataclass(frozen=True)
class PolarPointSpec:
    label: str
    radius: int
    theta_degrees: int


@dataclass(frozen=True)
class PolarReadoutCase:
    component: ReadoutComponent
    radius: int
    theta_degrees: int
    correct_value: int


@dataclass(frozen=True)
class PolarDifferenceCase:
    component: ReadoutComponent
    point_p: PolarPointSpec
    point_q: PolarPointSpec
    correct_value: int


@dataclass(frozen=True)
class PolarCoordinateCountCase:
    component: ReadoutComponent
    target_value: int
    points: tuple[PolarPointSpec, ...]
    matching_labels: tuple[str, ...]
    correct_value: int


@dataclass(frozen=True)
class RenderedPolarGraphPaperScene:
    image: Any
    render_map: dict[str, Any]
    render_spec: dict[str, Any]
    style_metadata: dict[str, Any]
