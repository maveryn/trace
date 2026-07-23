"""Annotation projection helpers for construction-site illustrations."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

from .state import RenderedConstructionSiteScene


def construction_worker_bbox_map(scene: RenderedConstructionSiteScene) -> Dict[str, List[float]]:
    """Return worker bbox map keyed by worker id."""

    return {str(worker.worker_id): [round(float(v), 3) for v in worker.bbox_xyxy] for worker in scene.workers}


def construction_material_bbox_map(scene: RenderedConstructionSiteScene) -> Dict[str, List[float]]:
    """Return material bbox map keyed by material id."""

    return {str(material.material_id): [round(float(v), 3) for v in material.bbox_xyxy] for material in scene.materials}


def construction_equipment_bbox_map(scene: RenderedConstructionSiteScene) -> Dict[str, List[float]]:
    """Return equipment bbox map keyed by equipment id."""

    return {str(equipment.equipment_id): [round(float(v), 3) for v in equipment.bbox_xyxy] for equipment in scene.equipment}


def construction_zone_bbox_map(scene: RenderedConstructionSiteScene) -> Dict[str, List[float]]:
    """Return construction-zone bbox map keyed by zone id."""

    return {str(zone.zone_id): [round(float(v), 3) for v in zone.bbox_xyxy] for zone in scene.zones}


def sort_construction_bboxes(bbox_map: Mapping[str, Sequence[float]], ids: Iterable[str]) -> List[List[float]]:
    """Return bbox values sorted by top-left position."""

    boxes = [list(float(v) for v in bbox_map[str(item_id)]) for item_id in ids]
    boxes.sort(key=lambda box: (float(box[1]), float(box[0]), float(box[3]), float(box[2])))
    return [[round(float(v), 3) for v in box] for box in boxes]


def sort_construction_bbox_centers(bbox_map: Mapping[str, Sequence[float]], ids: Iterable[str]) -> List[List[float]]:
    """Return bbox center points sorted by the source bbox top-left position."""

    points: List[tuple[List[float], List[float]]] = []
    for item_id in ids:
        box = [float(v) for v in bbox_map[str(item_id)]]
        point = [
            round((float(box[0]) + float(box[2])) / 2.0, 3),
            round((float(box[1]) + float(box[3])) / 2.0, 3),
        ]
        points.append((box, point))
    points.sort(key=lambda item: (float(item[0][1]), float(item[0][0]), float(item[0][3]), float(item[0][2])))
    return [list(point) for _, point in points]


__all__ = [
    "construction_equipment_bbox_map",
    "construction_material_bbox_map",
    "construction_worker_bbox_map",
    "construction_zone_bbox_map",
    "sort_construction_bbox_centers",
    "sort_construction_bboxes",
]
