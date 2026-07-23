"""Annotation projection helpers for multiseries chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple


def projected_category_annotation(
    rendered_scene,
    category_labels: Sequence[str],
) -> Dict[str, Any]:
    """Project one ordered category-label annotation list into pixel-space chart annotation."""

    requested = [str(label) for label in category_labels]
    bbox_by_category: Dict[str, List[float]] = {}
    point_by_category: Dict[str, List[float]] = {}
    for mark_trace in rendered_scene.mark_traces:
        category_label = str(mark_trace.get("category_label", ""))
        if category_label in bbox_by_category:
            continue
        group_bbox = mark_trace.get("category_group_bbox_px")
        label_center = mark_trace.get("category_label_center_px")
        if isinstance(group_bbox, list) and len(group_bbox) == 4:
            bbox_by_category[category_label] = [float(value) for value in group_bbox]
        if isinstance(label_center, list) and len(label_center) == 2:
            point_by_category[category_label] = [float(value) for value in label_center]
    return {
        "pixel_point_map": {
            str(label): list(point_by_category[str(label)])
            for label in requested
            if str(label) in point_by_category
        },
        "pixel_point_set": [
            list(point_by_category[str(label)])
            for label in requested
            if str(label) in point_by_category
        ],
        "bbox_set": [
            list(bbox_by_category[str(label)])
            for label in requested
            if str(label) in bbox_by_category
        ],
    }


def projected_multiseries_mark_annotation(
    rendered_scene,
    category_labels: Sequence[str],
    series_labels: Sequence[str] | Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Project category/series mark witnesses into pixel-space annotation."""

    requested_categories = [str(label) for label in category_labels]
    if isinstance(series_labels, Mapping):
        requested_series_by_category = {
            str(category): [str(series_label) for series_label in labels]
            for category, labels in series_labels.items()
        }
    else:
        shared_series = [str(series_label) for series_label in series_labels]
        requested_series_by_category = {
            str(category): list(shared_series)
            for category in requested_categories
        }

    mark_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for mark_trace in rendered_scene.mark_traces:
        category_label = str(mark_trace.get("category_label", ""))
        series_label = str(mark_trace.get("series_label", ""))
        mark_by_key[(category_label, series_label)] = mark_trace

    pixel_point_map: Dict[str, List[float]] = {}
    pixel_point_set: List[List[float]] = []
    bbox_set: List[List[float]] = []
    for category_label in requested_categories:
        for series_label in requested_series_by_category.get(str(category_label), []):
            mark_trace = mark_by_key.get((str(category_label), str(series_label)))
            if mark_trace is None:
                continue
            center = mark_trace.get("mark_center_px")
            bbox = mark_trace.get("mark_bbox_px")
            key = f"{str(category_label)}:{str(series_label)}"
            if isinstance(center, list) and len(center) == 2:
                point = [float(center[0]), float(center[1])]
                pixel_point_map[str(key)] = list(point)
                pixel_point_set.append(list(point))
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox_set.append([float(value) for value in bbox])

    return {
        "pixel_point_map": pixel_point_map,
        "pixel_point_set": pixel_point_set,
        "bbox_set": bbox_set,
    }

def keyed_points_from_projection(annotation_projection: Mapping[str, Any]) -> Dict[str, list[float]]:
    """Normalize projected category/series mark centers into a keyed point map."""

    return {
        str(key): [float(point[0]), float(point[1])]
        for key, point in dict(annotation_projection.get("pixel_point_map", {})).items()
    }


def projected_keyed_point_annotation(
    annotation_projection: Mapping[str, Any],
    keyed_points: Mapping[str, list[float]],
) -> Dict[str, Any]:
    """Build the public projected-annotation payload for keyed mark witnesses."""

    point_set = [[float(point[0]), float(point[1])] for point in annotation_projection.get("pixel_point_set", [])]
    return {
        "type": "point_map",
        "point_map": {str(key): list(point) for key, point in keyed_points.items()},
        "point_set": list(point_set),
        "pixel_point_set": list(point_set),
        "bbox_set": [list(bbox) for bbox in annotation_projection.get("bbox_set", [])],
        "pixel_point_map": {str(key): list(point) for key, point in keyed_points.items()},
    }

def mark_annotation_payload(
    *,
    rendered_scene: Any,
    category_labels: Sequence[str],
    series_labels: Sequence[str] | Mapping[str, Sequence[str]],
) -> tuple[Dict[str, list[float]], Dict[str, Any]]:
    """Return task annotation points and projected point-map payload."""

    projection = projected_multiseries_mark_annotation(
        rendered_scene,
        [str(label) for label in category_labels],
        series_labels,
    )
    points = keyed_points_from_projection(projection)
    return points, projected_keyed_point_annotation(projection, points)


__all__ = [
    "keyed_points_from_projection",
    "mark_annotation_payload",
    "projected_category_annotation",
    "projected_keyed_point_annotation",
    "projected_multiseries_mark_annotation",
]
