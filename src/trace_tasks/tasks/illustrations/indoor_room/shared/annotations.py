"""Annotation and render-map helpers for indoor-room scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ...shared.object_rendering import serialize_rendered_illustration_object
from .state import RenderedIndoorRoomScene


def serialize_indoor_scene(scene: RenderedIndoorRoomScene) -> tuple[list[dict[str, Any]], Dict[str, list[float]], Dict[str, list[float]]]:
    serialized_objects = [serialize_rendered_illustration_object(obj) for obj in scene.objects]
    object_bboxes = {str(obj["object_id"]): list(obj["bbox"]) for obj in serialized_objects}
    part_bboxes = {
        str(part["part_id"]): list(part["bbox"])
        for obj in serialized_objects
        for part in obj["parts"]
    }
    return serialized_objects, object_bboxes, part_bboxes


def surface_bbox_map(scene: RenderedIndoorRoomScene) -> Dict[str, list[float]]:
    return {str(surface.surface_id): [round(float(v), 3) for v in surface.bbox_xyxy] for surface in scene.surfaces}


def surface_support_bbox_map(scene: RenderedIndoorRoomScene) -> Dict[str, list[float]]:
    return {str(surface.surface_id): [round(float(v), 3) for v in surface.support_bbox_xyxy] for surface in scene.surfaces}


def container_bbox_map(scene: RenderedIndoorRoomScene) -> Dict[str, list[float]]:
    return {str(container.container_id): [round(float(v), 3) for v in container.bbox_xyxy] for container in scene.containers}


def container_interior_bbox_map(scene: RenderedIndoorRoomScene) -> Dict[str, list[float]]:
    return {str(container.container_id): [round(float(v), 3) for v in container.interior_bbox_xyxy] for container in scene.containers}


def furniture_bbox_map(scene: RenderedIndoorRoomScene) -> Dict[str, list[float]]:
    return {str(furniture.furniture_id): [round(float(v), 3) for v in furniture.bbox_xyxy] for furniture in scene.furniture}


def placement_map(scene: RenderedIndoorRoomScene) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for placement in scene.placements:
        result[str(placement.object_id)] = {
            "object_type": str(placement.object_type),
            "placement_kind": str(placement.placement_kind),
            "surface_id": placement.surface_id,
            "surface_type": placement.surface_type,
            "surface_contact_px": [round(float(v), 3) for v in placement.surface_contact_px] if placement.surface_contact_px else None,
            "surface_depth": round(float(placement.surface_depth), 4) if placement.surface_depth is not None else None,
            "container_id": placement.container_id,
            "container_type": placement.container_type,
            "region_relation": placement.region_relation,
            "region_furniture_id": placement.region_furniture_id,
            "relations": dict(placement.relations),
            "role": str(placement.role),
        }
    return result


def sort_bboxes_by_ids(bbox_map: Mapping[str, Sequence[float]], ids: Sequence[str]) -> list[list[float]]:
    boxes = [(str(item_id), [round(float(v), 3) for v in bbox_map[str(item_id)]]) for item_id in ids]
    ordered = sorted(boxes, key=lambda item: (float(item[1][1]), float(item[1][0]), str(item[0])))
    return [box for _item_id, box in ordered]


def sort_bbox_centers_by_ids(bbox_map: Mapping[str, Sequence[float]], ids: Sequence[str]) -> list[list[float]]:
    boxes = [(str(item_id), [float(v) for v in bbox_map[str(item_id)]]) for item_id in ids]
    ordered = sorted(boxes, key=lambda item: (float(item[1][1]), float(item[1][0]), str(item[0])))
    return [
        [
            round((float(box[0]) + float(box[2])) / 2.0, 3),
            round((float(box[1]) + float(box[3])) / 2.0, 3),
        ]
        for _item_id, box in ordered
    ]


__all__ = [
    "container_bbox_map",
    "container_interior_bbox_map",
    "furniture_bbox_map",
    "placement_map",
    "serialize_indoor_scene",
    "sort_bbox_centers_by_ids",
    "sort_bboxes_by_ids",
    "surface_bbox_map",
    "surface_support_bbox_map",
]
