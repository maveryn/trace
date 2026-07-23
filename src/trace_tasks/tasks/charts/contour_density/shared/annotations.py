"""Annotation projection helpers for contour-density chart scenes."""

from __future__ import annotations

from typing import Any, Dict, List

from trace_tasks.tasks.charts.contour_density.shared.state import ContourDataset, RenderedContourScene


def bbox_map(dataset: ContourDataset, rendered: RenderedContourScene) -> Dict[str, List[float]]:
    annotation: Dict[str, List[float]] = {}
    for role, entity_id in dataset.query.annotation_roles.items():
        if str(entity_id) == "reference":
            box = rendered.reference_bboxes.get("reference")
        else:
            box = rendered.region_bboxes.get(str(entity_id))
        if box is None:
            raise RuntimeError(f"missing annotation bbox for role {role}: {entity_id}")
        annotation[str(role)] = list(box)
    return annotation


def _bbox_center(bbox: List[float]) -> List[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def point_map(dataset: ContourDataset, rendered: RenderedContourScene) -> Dict[str, List[float]]:
    annotation: Dict[str, List[float]] = {}
    for role, entity_id in dataset.query.annotation_roles.items():
        if str(entity_id) == "reference":
            box = rendered.reference_bboxes.get("reference")
        else:
            box = rendered.region_bboxes.get(str(entity_id))
        if box is None:
            raise RuntimeError(f"missing annotation point for role {role}: {entity_id}")
        annotation[str(role)] = _bbox_center(list(box))
    return annotation


def bbox_set(dataset: ContourDataset, rendered: RenderedContourScene) -> List[List[float]]:
    boxes: List[List[float]] = []
    for region_id in dataset.query.annotation_region_ids:
        box = rendered.region_bboxes.get(str(region_id))
        if box is None:
            raise RuntimeError(f"missing bbox-set annotation for region: {region_id}")
        boxes.append(list(box))
    return boxes


def projected_annotation_payload(dataset: ContourDataset, annotation: Any) -> Dict[str, Any]:
    if str(dataset.query.annotation_type) == "bbox":
        bbox = list(annotation)
        return {
            "type": "bbox",
            "bbox": list(bbox),
            "pixel_bbox": list(bbox),
            "annotation_region_ids": list(dataset.query.annotation_region_ids),
        }
    if str(dataset.query.annotation_type) == "bbox_set":
        return {
            "type": "bbox_set",
            "bbox_set": [list(value) for value in annotation],
            "pixel_bbox_set": [list(value) for value in annotation],
            "annotation_region_ids": list(dataset.query.annotation_region_ids),
        }
    if str(dataset.query.annotation_type) == "bbox_map":
        return {
            "type": "bbox_map",
            "bbox_map": {key: list(value) for key, value in annotation.items()},
            "pixel_bbox_map": {key: list(value) for key, value in annotation.items()},
            "annotation_roles": dict(dataset.query.annotation_roles),
        }
    if str(dataset.query.annotation_type) == "point_map":
        return {
            "type": "point_map",
            "point_map": {key: list(value) for key, value in annotation.items()},
            "pixel_point_map": {key: list(value) for key, value in annotation.items()},
            "annotation_roles": dict(dataset.query.annotation_roles),
        }
    raise ValueError(f"unsupported annotation type: {dataset.query.annotation_type}")


def annotation_value(dataset: ContourDataset, rendered: RenderedContourScene) -> Any:
    if str(dataset.query.annotation_type) == "bbox":
        boxes = bbox_set(dataset, rendered)
        if len(boxes) != 1:
            raise RuntimeError("contour-density scalar bbox annotation must contain exactly one box")
        return list(boxes[0])
    if str(dataset.query.annotation_type) == "bbox_set":
        return bbox_set(dataset, rendered)
    if str(dataset.query.annotation_type) == "point_map":
        return point_map(dataset, rendered)
    return bbox_map(dataset, rendered)


__all__ = ["annotation_value", "bbox_set", "bbox_map", "point_map", "projected_annotation_payload"]
