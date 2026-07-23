"""Annotation projection helpers for curve-panel charts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .state import CurvePanelDataset, RenderedCurvePanels


def bbox_center(bbox: Sequence[float]) -> List[float]:
    """Return the center point for one rendered bbox."""

    if len(bbox) < 4:
        raise ValueError("bbox must have at least four coordinates")
    x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    return [round((x0 + x1) * 0.5, 3), round((y0 + y1) * 0.5, 3)]


def point_set_from_ids(
    *,
    rendered: RenderedCurvePanels,
    point_ids: Sequence[str] = (),
    intersection_ids: Sequence[str] = (),
    crossing_ids: Sequence[str] = (),
) -> List[List[float]]:
    """Project rendered marker/intersection/crossing ids to a point set."""

    points: List[List[float]] = []
    for point_id in point_ids:
        point_box = rendered.point_bboxes.get(str(point_id))
        if point_box is not None:
            points.append(bbox_center(point_box))
    for intersection_id in intersection_ids:
        intersection_box = rendered.intersection_bboxes.get(str(intersection_id))
        if intersection_box is not None:
            points.append(bbox_center(intersection_box))
    for crossing_id in crossing_ids:
        crossing_box = rendered.threshold_crossing_bboxes.get(str(crossing_id))
        if crossing_box is not None:
            points.append(bbox_center(crossing_box))
    return points


def point_map_from_ids(
    *,
    rendered: RenderedCurvePanels,
    keyed_point_ids: Mapping[str, str],
) -> Dict[str, List[float]]:
    """Project rendered marker ids to a role-keyed point map."""

    points: Dict[str, List[float]] = {}
    for role, point_id in keyed_point_ids.items():
        point_box = rendered.point_bboxes.get(str(point_id))
        if point_box is not None:
            points[str(role)] = bbox_center(point_box)
    return points


def bbox_set_from_panel_labels(
    *,
    rendered: RenderedCurvePanels,
    panel_labels: Sequence[str],
) -> List[List[float]]:
    """Project panel labels to whole-panel bbox annotation."""

    boxes: List[List[float]] = []
    for panel_label in panel_labels:
        panel_box = rendered.panel_bboxes.get(str(panel_label))
        if panel_box is not None:
            boxes.append([round(float(value), 3) for value in panel_box])
    return boxes


def projected_annotation_payload(
    *,
    dataset: CurvePanelDataset,
    annotation_type: str,
    annotation: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
) -> Dict[str, Any]:
    """Build the projected annotation trace payload for one public task."""

    if str(annotation_type) == "bbox_set":
        bbox_set = [list(box) for box in list(annotation)]
        return {
            "type": "bbox_set",
            "bbox_set": list(bbox_set),
            "pixel_bbox_set": list(bbox_set),
            "panel_labels": list(dataset.query.annotation_panel_labels),
        }
    if str(annotation_type) == "point_map":
        point_map = {
            str(key): list(point) for key, point in dict(annotation).items()
        }
        return {
            "type": "point_map",
            "point_map": dict(point_map),
            "pixel_point_map": dict(point_map),
            "panel_labels": list(dataset.query.annotation_panel_labels),
            "keyed_point_ids": dict(dataset.query.annotation_keyed_point_ids),
            "point_ids": list(dataset.query.annotation_keyed_point_ids.values()),
            "intersection_ids": [],
            "threshold_crossing_ids": [],
        }
    if str(annotation_type) == "point":
        point = list(annotation)
        return {
            "type": "point",
            "point": list(point),
            "pixel_point": list(point),
            "panel_labels": list(dataset.query.annotation_panel_labels),
            "point_ids": list(dataset.query.annotation_point_ids),
            "intersection_ids": list(dataset.query.annotation_intersection_ids),
            "threshold_crossing_ids": list(dataset.query.annotation_threshold_crossing_ids),
        }
    point_set = [list(point) for point in list(annotation)]
    return {
        "type": "point_set",
        "point_set": list(point_set),
        "pixel_point_set": list(point_set),
        "panel_labels": list(dataset.query.annotation_panel_labels),
        "point_ids": list(dataset.query.annotation_point_ids),
        "intersection_ids": list(dataset.query.annotation_intersection_ids),
        "threshold_crossing_ids": list(dataset.query.annotation_threshold_crossing_ids),
    }


__all__ = [
    "bbox_center",
    "bbox_set_from_panel_labels",
    "point_map_from_ids",
    "point_set_from_ids",
    "projected_annotation_payload",
]
