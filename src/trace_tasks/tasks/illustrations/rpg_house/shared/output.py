"""Trace fragment helpers for RPG house public tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .rendering import SCENE_ID
from .state import RpgHouseScene


def rpg_house_scene_ir(
    *,
    domain: str,
    scene_id: str,
    scene: RpgHouseScene,
    relations: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the common scene-IR fragment for an RPG house scene."""

    return {
        "domain": str(domain),
        "scene_id": str(scene_id),
        "rooms": [room.as_dict() for room in scene.rooms],
        "doors": [door.as_dict() for door in scene.doors],
        "entities": [entity.as_dict() for entity in scene.entities],
        "relations": dict(relations),
    }


def rpg_house_render_spec(scene: RpgHouseScene, *, scene_id: str = SCENE_ID) -> dict[str, Any]:
    """Return the render-spec fragment for one scene."""

    return {
        "canvas_size": [int(scene.image.size[0]), int(scene.image.size[1])],
        "coord_space": "pixel",
        "scene_id": str(scene_id),
        "style": {
            "renderer_id": str(scene.trace.get("renderer_id", "")),
            "style_id": str(scene.trace.get("renderer_style", "")),
            "theme_id": str(scene.trace.get("theme_id", "")),
            "tile_px": int(scene.trace.get("tile_px", 0)),
            "grid_cols": int(scene.trace.get("grid_cols", 0)),
            "grid_rows": int(scene.trace.get("grid_rows", 0)),
            "canvas_profile": str(scene.trace.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(scene.trace.get("canvas_profile_probabilities", {})),
            "label_font": dict(scene.trace.get("label_font", {})),
        },
    }


def rpg_house_reachability_render_map(
    *,
    scene: RpgHouseScene,
    start_room_id: str,
    candidate_room_ids: Sequence[str],
    answer_room_id: str,
) -> dict[str, Any]:
    """Return task render-map fields for the room-reachability task."""

    room_boxes = room_bbox_map(scene)
    door_boxes = door_bbox_map(scene)
    candidate_ids = [str(room_id) for room_id in candidate_room_ids]
    return {
        "image_id": "img0",
        "room_bboxes_px": room_boxes,
        "door_bboxes_px": door_boxes,
        "start_room_id": str(start_room_id),
        "start_room_bbox_px": room_boxes[str(start_room_id)],
        "candidate_room_ids": candidate_ids,
        "candidate_room_bboxes_px": {room_id: room_boxes[room_id] for room_id in candidate_ids},
        "answer_room_id": str(answer_room_id),
        "answer_room_bbox_px": room_boxes[str(answer_room_id)],
    }


def rpg_house_room_count_render_map(*, scene: RpgHouseScene) -> dict[str, Any]:
    """Return task render-map fields for total-room counting."""

    return {
        "image_id": "img0",
        "room_bboxes_px": room_bbox_map(scene),
        "room_points_px": room_point_map(scene),
        "counted_room_ids": [str(room.room_id) for room in scene.rooms],
        "counted_room_count": len(scene.rooms),
    }


def rpg_house_reachable_room_count_render_map(
    *,
    scene: RpgHouseScene,
    player_room_id: str,
    reachable_room_ids: Sequence[str],
) -> dict[str, Any]:
    """Return task render-map fields for reachable-room counting."""

    room_points = room_point_map(scene)
    player = player_entity(scene)
    if player is None:
        raise ValueError("reachable-room count render map requires a player entity")
    reachable_ids = [str(room_id) for room_id in reachable_room_ids]
    return {
        "image_id": "img0",
        "room_bboxes_px": room_bbox_map(scene),
        "room_points_px": room_points,
        "door_bboxes_px": door_bbox_map(scene),
        "door_points_px": door_point_map(scene),
        "player_room_id": str(player_room_id),
        "player_point_px": [round(float(value), 3) for value in player.point_xy],
        "reachable_room_ids": reachable_ids,
        "reachable_room_points_px": {room_id: room_points[room_id] for room_id in reachable_ids},
        "reachable_count": len(reachable_ids),
    }


def rpg_house_adjacent_room_count_render_map(
    *,
    scene: RpgHouseScene,
    player_room_id: str,
    adjacent_room_ids: Sequence[str],
) -> dict[str, Any]:
    """Return task render-map fields for adjacent-room counting."""

    room_points = room_point_map(scene)
    player = player_entity(scene)
    if player is None:
        raise ValueError("adjacent-room count render map requires a player entity")
    adjacent_ids = [str(room_id) for room_id in adjacent_room_ids]
    return {
        "image_id": "img0",
        "room_bboxes_px": room_bbox_map(scene),
        "room_points_px": room_points,
        "door_bboxes_px": door_bbox_map(scene),
        "door_points_px": door_point_map(scene),
        "player_room_id": str(player_room_id),
        "player_point_px": [round(float(value), 3) for value in player.point_xy],
        "adjacent_room_ids": adjacent_ids,
        "adjacent_room_points_px": {room_id: room_points[room_id] for room_id in adjacent_ids},
        "adjacent_count": len(adjacent_ids),
    }


def room_bbox_map(scene: RpgHouseScene) -> dict[str, list[float]]:
    return {
        str(room.room_id): [round(float(value), 3) for value in room.bbox_xyxy]
        for room in scene.rooms
    }


def room_point_map(scene: RpgHouseScene) -> dict[str, list[float]]:
    return {
        str(room.room_id): [
            round((float(room.bbox_xyxy[0]) + float(room.bbox_xyxy[2])) * 0.5, 3),
            round((float(room.bbox_xyxy[1]) + float(room.bbox_xyxy[3])) * 0.5, 3),
        ]
        for room in scene.rooms
    }


def door_bbox_map(scene: RpgHouseScene) -> dict[str, list[float]]:
    return {
        str(door.door_id): [round(float(value), 3) for value in door.bbox_xyxy]
        for door in scene.doors
    }


def door_point_map(scene: RpgHouseScene) -> dict[str, list[float]]:
    return {
        str(door.door_id): [
            round((float(door.bbox_xyxy[0]) + float(door.bbox_xyxy[2])) * 0.5, 3),
            round((float(door.bbox_xyxy[1]) + float(door.bbox_xyxy[3])) * 0.5, 3),
        ]
        for door in scene.doors
    }


def player_entity(scene: RpgHouseScene) -> Any | None:
    for entity in scene.entities:
        if str(entity.public_name) == "player" and str(entity.metadata.get("role", "")) == "reference":
            return entity
    return None


def bbox_projection(bbox: Sequence[float]) -> dict[str, Any]:
    values = [round(float(value), 3) for value in bbox[:4]]
    return {"type": "bbox", "bbox": values, "pixel_bbox": values}


def point_set_projection(points: Sequence[Sequence[float]]) -> dict[str, Any]:
    values = [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in points]
    return {"type": "point_set", "point_set": values, "pixel_point_set": values}


def point_set_map_projection(keyed_points: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
    values = {
        str(key): [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in points]
        for key, points in keyed_points.items()
    }
    return {"type": "point_set_map", "point_set_map": values, "pixel_point_set_map": values}


__all__ = [
    "bbox_projection",
    "rpg_house_adjacent_room_count_render_map",
    "door_point_map",
    "door_bbox_map",
    "point_set_map_projection",
    "player_entity",
    "point_set_projection",
    "room_bbox_map",
    "room_point_map",
    "rpg_house_reachable_room_count_render_map",
    "rpg_house_room_count_render_map",
    "rpg_house_reachability_render_map",
    "rpg_house_render_spec",
    "rpg_house_scene_ir",
]
