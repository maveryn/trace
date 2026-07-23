"""State containers for the isometric harbor scene package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BBox = tuple[float, float, float, float]
IsoPoint = tuple[float, float]
IsoPolygon = tuple[IsoPoint, ...]


@dataclass(frozen=True)
class IsoHarborTile:
    """One visible isometric harbor tile."""

    tile_id: str
    col: int
    row: int
    terrain: str
    walkable: bool
    polygon_xy: IsoPolygon
    bbox_xyxy: BBox
    center_xy: IsoPoint
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_id": str(self.tile_id),
            "col": int(self.col),
            "row": int(self.row),
            "terrain": str(self.terrain),
            "walkable": bool(self.walkable),
            "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in self.polygon_xy],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "center": [round(float(self.center_xy[0]), 3), round(float(self.center_xy[1]), 3)],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoHarborEntity:
    """One boat or contextual dock object in the harbor scene."""

    entity_id: str
    public_name: str
    object_type: str
    tile_ids: tuple[str, ...]
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
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(self.point_xy[0]), 3), round(float(self.point_xy[1]), 3)],
            "role": str(self.role),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoHarborScene:
    """Rendered isometric harbor plus verifier metadata."""

    image: Any
    tiles: tuple[IsoHarborTile, ...]
    entities: tuple[IsoHarborEntity, ...]
    trace: Mapping[str, Any]


__all__ = [
    "BBox",
    "IsoHarborEntity",
    "IsoHarborScene",
    "IsoHarborTile",
    "IsoPoint",
    "IsoPolygon",
]
