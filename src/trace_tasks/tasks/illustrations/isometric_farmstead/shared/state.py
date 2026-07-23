"""State containers for the isometric farmstead scene package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BBox = tuple[float, float, float, float]
IsoPoint = tuple[float, float]
IsoPolygon = tuple[IsoPoint, ...]
Tile = tuple[int, int]


@dataclass(frozen=True)
class IsoFarmsteadTile:
    """One visible isometric terrain tile."""

    tile_id: str
    col: int
    row: int
    level: int
    terrain: str
    polygon_xy: IsoPolygon
    bbox_xyxy: BBox
    center_xy: IsoPoint
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_id": str(self.tile_id),
            "col": int(self.col),
            "row": int(self.row),
            "level": int(self.level),
            "terrain": str(self.terrain),
            "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in self.polygon_xy],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "center": [round(float(self.center_xy[0]), 3), round(float(self.center_xy[1]), 3)],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoFarmsteadEntity:
    """One contextual farmstead object."""

    entity_id: str
    public_name: str
    object_type: str
    tile_ids: tuple[str, ...]
    level: int
    bbox_xyxy: BBox
    point_xy: IsoPoint
    role: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_id": str(self.entity_id),
            "public_name": str(self.public_name),
            "object_type": str(self.object_type),
            "tile_ids": [str(value) for value in self.tile_ids],
            "level": int(self.level),
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(self.point_xy[0]), 3), round(float(self.point_xy[1]), 3)],
            "role": str(self.role),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoFarmsteadTransition:
    """Optional visible transition record between adjacent terrain levels."""

    transition_id: str
    transition_type: str
    lower_tile_id: str
    upper_tile_id: str
    lower_level: int
    upper_level: int
    polygon_xy: IsoPolygon
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "transition_id": str(self.transition_id),
            "transition_type": str(self.transition_type),
            "lower_tile_id": str(self.lower_tile_id),
            "upper_tile_id": str(self.upper_tile_id),
            "lower_level": int(self.lower_level),
            "upper_level": int(self.upper_level),
            "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in self.polygon_xy],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoFarmsteadScene:
    """Rendered isometric farmstead plus verifier metadata."""

    image: Any
    tiles: tuple[IsoFarmsteadTile, ...]
    entities: tuple[IsoFarmsteadEntity, ...]
    transitions: tuple[IsoFarmsteadTransition, ...]
    label_bboxes_by_tile_id: Mapping[str, BBox]
    trace: Mapping[str, Any]


__all__ = [
    "BBox",
    "IsoFarmsteadEntity",
    "IsoFarmsteadScene",
    "IsoFarmsteadTile",
    "IsoFarmsteadTransition",
    "IsoPoint",
    "IsoPolygon",
    "Tile",
]
