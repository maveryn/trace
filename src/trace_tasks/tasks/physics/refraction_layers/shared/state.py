"""Passive state for the refraction-layers scene."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "refraction_layers"
SCENE_NAMESPACE = "physics_refraction_layers"
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
MEDIUM_LABELS: Tuple[str, ...] = ("M1", "M2", "M3")
ALL_SPEED_ORDERS: Tuple[Tuple[str, str, str], ...] = tuple(itertools.permutations(MEDIUM_LABELS))
SPEED_VALUES_BY_RANK: Tuple[float, float, float] = (1.0, 0.76, 0.52)


@dataclass(frozen=True)
class RefractionScenario:
    """Resolved physical and answer state for one refraction diagram."""

    orientation: str
    entry_side: str
    transverse_sign: int
    speed_order: Tuple[str, str, str]
    medium_speeds: Dict[str, float]
    angle_by_medium_deg: Dict[str, float]
    option_map: Dict[str, str]
    correct_label: str


@dataclass(frozen=True)
class RayGeometry:
    """Projected ray geometry through the three media."""

    points: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    bend_points: Tuple[Tuple[float, float], Tuple[float, float]]
    segment_mediums: Tuple[str, str, str]
    segment_angles_deg: Tuple[float, float, float]


@dataclass(frozen=True)
class RenderedRefractionScene:
    """Rendered refraction image plus projected verifier geometry."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    render_map: Dict[str, Any]
