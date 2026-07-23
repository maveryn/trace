"""Shared object taxonomy records for three_d scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple


BBox = Tuple[float, float, float, float]
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe copy of a metadata value."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 3)
    return value


def rounded_bbox(bbox: Sequence[float] | None) -> list[float] | None:
    """Return a rounded bbox payload, or ``None`` when no bbox is available."""

    if bbox is None:
        return None
    values = [float(value) for value in bbox]
    if len(values) != 4:
        raise ValueError("bbox must contain exactly four values")
    return [round(float(value), 3) for value in values]


def rounded_point2(point: Sequence[float] | None) -> list[float] | None:
    """Return a rounded 2D point payload, or ``None`` when unavailable."""

    if point is None:
        return None
    values = [float(value) for value in point]
    if len(values) != 2:
        raise ValueError("2D point must contain exactly two values")
    return [round(float(value), 3) for value in values]


def rounded_point3(point: Sequence[float] | None) -> list[float] | None:
    """Return a rounded 3D point payload, or ``None`` when unavailable."""

    if point is None:
        return None
    values = [float(value) for value in point]
    if len(values) != 3:
        raise ValueError("3D point must contain exactly three values")
    return [round(float(value), 3) for value in values]


@dataclass(frozen=True)
class ThreeDObjectRecord:
    """Normalized object payload embedded in three_d scene entities."""

    object_id: str
    object_type: str
    public_name: str
    family: str
    canonical_id: str = ""
    bbox_xyxy: BBox | None = None
    center_xy: Point2D | None = None
    world_xyz: Point3D | None = None
    base_xyz: Point3D | None = None
    dimensions_xyz: Point3D | None = None
    semantic_attributes: Mapping[str, Any] = field(default_factory=dict)
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)
    role: str = "distractor"
    source_entity_type: str = ""
    parts: Tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a trace-ready object record."""

        return {
            "object_id": str(self.object_id),
            "object_type": str(self.object_type),
            "canonical_id": str(self.canonical_id or self.object_type),
            "public_name": str(self.public_name),
            "family": str(self.family),
            "bbox": rounded_bbox(self.bbox_xyxy),
            "center_xy": rounded_point2(self.center_xy),
            "world_xyz": rounded_point3(self.world_xyz),
            "base_xyz": rounded_point3(self.base_xyz),
            "dimensions_xyz": rounded_point3(self.dimensions_xyz),
            "semantic_attributes": json_safe(self.semantic_attributes),
            "visual_attributes": json_safe(self.visual_attributes),
            "role": str(self.role),
            "source_entity_type": str(self.source_entity_type),
            "parts": [json_safe(part) for part in self.parts],
        }


__all__ = [
    "BBox",
    "Point2D",
    "Point3D",
    "ThreeDObjectRecord",
    "json_safe",
    "rounded_bbox",
    "rounded_point2",
    "rounded_point3",
]
