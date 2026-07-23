"""Shared object taxonomy records for illustration scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple


BBox = Tuple[float, float, float, float]


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


@dataclass(frozen=True)
class ObjectAttributeDef:
    """One registered object attribute."""

    name: str
    public: bool = True
    values: Tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ObjectTypeDef:
    """Prompt-facing object type definition shared across scenes."""

    object_type: str
    public_name: str
    family: str
    render_layer: str = "foreground"
    size_class: str = "medium"
    placement_tags: Tuple[str, ...] = ()
    scene_tags: Tuple[str, ...] = ()
    semantic_attributes: Tuple[ObjectAttributeDef, ...] = ()
    visual_attributes: Tuple[ObjectAttributeDef, ...] = ()
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectRecord:
    """Normalized object payload embedded in scene entities."""

    object_id: str
    object_type: str
    public_name: str
    family: str
    bbox_xyxy: BBox | None = None
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
            "public_name": str(self.public_name),
            "family": str(self.family),
            "bbox": rounded_bbox(self.bbox_xyxy),
            "semantic_attributes": json_safe(self.semantic_attributes),
            "visual_attributes": json_safe(self.visual_attributes),
            "role": str(self.role),
            "source_entity_type": str(self.source_entity_type),
            "parts": [json_safe(part) for part in self.parts],
        }


__all__ = [
    "BBox",
    "ObjectAttributeDef",
    "ObjectRecord",
    "ObjectTypeDef",
    "json_safe",
    "rounded_bbox",
]
