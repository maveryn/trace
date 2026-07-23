"""State containers for the RPG house scene package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
TileBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class RpgHouseRoom:
    """One room in the top-down house layout."""

    room_id: str
    public_name: str
    label: str | None
    tile_xywh: TileBox
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "room_id": str(self.room_id),
            "public_name": str(self.public_name),
            "label": None if self.label is None else str(self.label),
            "tile_xywh": [int(value) for value in self.tile_xywh],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RpgHouseDoor:
    """One doorway connecting two rooms."""

    door_id: str
    room_a_id: str
    room_b_id: str
    state: str
    orientation: str
    tile_xy: tuple[int, int]
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "door_id": str(self.door_id),
            "room_a_id": str(self.room_a_id),
            "room_b_id": str(self.room_b_id),
            "state": str(self.state),
            "orientation": str(self.orientation),
            "tile_xy": [int(value) for value in self.tile_xy],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RpgHouseEntity:
    """One large fixture or context object in a room."""

    entity_id: str
    public_name: str
    object_type: str
    room_id: str
    tile_xywh: TileBox
    bbox_xyxy: BBox
    point_xy: Point
    layer: str
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
            "room_id": str(self.room_id),
            "tile_xywh": [int(value) for value in self.tile_xywh],
            "bbox": [round(float(value), 3) for value in self.bbox_xyxy],
            "point": [round(float(value), 3) for value in self.point_xy],
            "layer": str(self.layer),
            "metadata": metadata,
        }
        if "object_record" in self.metadata:
            payload["object_record"] = self.metadata["object_record"]
        return payload


@dataclass(frozen=True)
class RpgHouseScene:
    """Rendered RPG house plus verifier metadata."""

    image: Any
    rooms: tuple[RpgHouseRoom, ...]
    doors: tuple[RpgHouseDoor, ...]
    entities: tuple[RpgHouseEntity, ...]
    trace: Mapping[str, Any]


__all__ = [
    "BBox",
    "Point",
    "RpgHouseDoor",
    "RpgHouseEntity",
    "RpgHouseRoom",
    "RpgHouseScene",
    "TileBox",
]
