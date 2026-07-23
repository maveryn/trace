"""State contracts for the area-partition geometry scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SCENE_ID = "area_partition"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]
PartitionCase = Tuple[str, int, int]

OUTER_SHAPE_ID = "outer_shape"
SHADED_REGION_ID = "shaded_region"
ANNOTATION_ROLES: tuple[str, str] = (OUTER_SHAPE_ID, SHADED_REGION_ID)


@dataclass(frozen=True)
class AreaPartitionProblem:
    """Resolved area-partition construction before rendering."""

    scene_variant: str
    answer: float
    shaded_area: int
    denominator: int
    formula: str
    support_probabilities: Dict[str, float]


@dataclass(frozen=True)
class AreaPartitionWitness:
    """Task-neutral symbolic facts exposed in trace metadata."""

    scene_variant: str
    shape_type: str
    shaded_area: int
    shaded_fraction_numerator: int
    shaded_fraction_denominator: int
    formula: str
    answer_value: float

    def to_trace(self) -> dict[str, object]:
        return {
            "scene_variant": str(self.scene_variant),
            "shape_type": str(self.shape_type),
            "shaded_area": int(self.shaded_area),
            "shaded_fraction_numerator": int(self.shaded_fraction_numerator),
            "shaded_fraction_denominator": int(self.shaded_fraction_denominator),
            "formula": str(self.formula),
            "answer_value": float(self.answer_value),
        }


__all__ = [
    "ANNOTATION_ROLES",
    "AreaPartitionProblem",
    "AreaPartitionWitness",
    "BBox",
    "Color",
    "OUTER_SHAPE_ID",
    "PartitionCase",
    "Point",
    "SCENE_ID",
    "SHADED_REGION_ID",
]
