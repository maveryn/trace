"""State containers for the RPG tactical map scene package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
TileCoord = tuple[int, int]


@dataclass(frozen=True)
class RpgTacticalTile:
    """One terrain tile in the tactical map."""

    tile_id: str
    row: int
    col: int
    terrain: str
    movement_cost: int | None
    passable: bool
    bbox_xyxy: BBox
    point_xy: Point
    metadata: Mapping[str, Any]

    @property
    def coord(self) -> TileCoord:
        return (int(self.row), int(self.col))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_id": str(self.tile_id),
            "row": int(self.row),
            "col": int(self.col),
            "terrain": str(self.terrain),
            "movement_cost": None if self.movement_cost is None else int(self.movement_cost),
            "passable": bool(self.passable),
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(value), 3) for value in self.point_xy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RpgTacticalUnit:
    """One reference unit on the tactical map."""

    unit_id: str
    public_name: str
    team: str
    tile_id: str
    bbox_xyxy: BBox
    point_xy: Point
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": str(self.unit_id),
            "public_name": str(self.public_name),
            "team": str(self.team),
            "tile_id": str(self.tile_id),
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(value), 3) for value in self.point_xy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RpgTacticalMapScene:
    """Rendered tactical map plus verifier metadata."""

    image: Any
    tiles: tuple[RpgTacticalTile, ...]
    units: tuple[RpgTacticalUnit, ...]
    label_bboxes_by_tile_id: Mapping[str, BBox]
    trace: Mapping[str, Any]


__all__ = [
    "BBox",
    "Point",
    "RpgTacticalMapScene",
    "RpgTacticalTile",
    "RpgTacticalUnit",
    "TileCoord",
]
