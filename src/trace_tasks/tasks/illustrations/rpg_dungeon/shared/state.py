"""State containers for the RPG dungeon scene package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Tile = tuple[int, int]
TileBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class RpgDungeonChamber:
    """One carved dungeon chamber in tile coordinates."""

    chamber_id: str
    public_name: str
    tile_xywh: TileBox
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chamber_id": str(self.chamber_id),
            "public_name": str(self.public_name),
            "tile_xywh": [int(value) for value in self.tile_xywh],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RpgDungeonEntity:
    """One rendered dungeon entity such as the player, chest, or fixture."""

    entity_id: str
    public_name: str
    object_type: str
    chamber_id: str | None
    tile_xywh: TileBox
    bbox_xyxy: BBox
    point_xy: Point
    role: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        metadata = {
            str(key): value
            for key, value in self.metadata.items()
            if str(key) != "object_record"
        }
        payload = {
            "entity_id": str(self.entity_id),
            "public_name": str(self.public_name),
            "object_type": str(self.object_type),
            "chamber_id": None if self.chamber_id is None else str(self.chamber_id),
            "tile_xywh": [int(value) for value in self.tile_xywh],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(value), 3) for value in self.point_xy],
            "role": str(self.role),
            "metadata": metadata,
        }
        if "object_record" in self.metadata:
            payload["object_record"] = self.metadata["object_record"]
        return payload


@dataclass(frozen=True)
class RpgDungeonBlocker:
    """One impassable corridor blocker."""

    blocker_id: str
    blocker_type: str
    tile_xy: Tile
    tile_xywh: TileBox
    bbox_xyxy: BBox
    point_xy: Point
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocker_id": str(self.blocker_id),
            "blocker_type": str(self.blocker_type),
            "tile_xy": [int(value) for value in self.tile_xy],
            "tile_xywh": [int(value) for value in self.tile_xywh],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(value), 3) for value in self.point_xy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RpgDungeonScene:
    """Rendered RPG dungeon plus verifier metadata."""

    image: Any
    chambers: tuple[RpgDungeonChamber, ...]
    floor_tiles: tuple[Tile, ...]
    blocked_tiles: tuple[Tile, ...]
    corridor_tiles: tuple[Tile, ...]
    blockers: tuple[RpgDungeonBlocker, ...]
    entities: tuple[RpgDungeonEntity, ...]
    player_entity_id: str
    chest_entity_ids: tuple[str, ...]
    reachable_chest_ids: tuple[str, ...]
    trace: Mapping[str, Any]


__all__ = [
    "BBox",
    "Point",
    "RpgDungeonBlocker",
    "RpgDungeonChamber",
    "RpgDungeonEntity",
    "RpgDungeonScene",
    "Tile",
    "TileBox",
]

