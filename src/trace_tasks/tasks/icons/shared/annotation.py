"""Shared public annotation helpers for icon tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


ICON_OBJECT_ANNOTATION_MIN_SIDE_PX = 24


def _bbox_center(bbox: Sequence[int | float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _normalize_point(point: Sequence[int | float], *, context: str) -> list[float]:
    if not isinstance(point, Sequence) or len(point) != 2:
        raise RuntimeError(f"invalid point annotation for {context}: {point}")
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _normalize_bbox(bbox: Sequence[int | float], *, context: str) -> list[int]:
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise RuntimeError(f"invalid bbox annotation for {context}: {bbox}")
    normalized = [int(round(float(value))) for value in bbox]
    if normalized[2] < normalized[0] or normalized[3] < normalized[1]:
        raise RuntimeError(f"invalid bbox bounds for {context}: {bbox}")
    return normalized


def _expand_bbox_to_min_side(
    bbox: Sequence[int | float],
    *,
    min_side_px: int,
    clip_bbox: Sequence[int | float] | None = None,
    context: str,
) -> list[int]:
    """Return `bbox` expanded around center until both sides meet `min_side_px`."""

    x0, y0, x1, y1 = _normalize_bbox(bbox, context=context)
    min_side = max(1, int(min_side_px))
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    target_w = max(width, min_side)
    target_h = max(height, min_side)
    cx = 0.5 * float(x0 + x1)
    cy = 0.5 * float(y0 + y1)
    expanded = [
        int(round(cx - 0.5 * float(target_w))),
        int(round(cy - 0.5 * float(target_h))),
        int(round(cx - 0.5 * float(target_w))) + int(target_w),
        int(round(cy - 0.5 * float(target_h))) + int(target_h),
    ]
    if clip_bbox is not None:
        cx0, cy0, cx1, cy1 = _normalize_bbox(clip_bbox, context=f"clip bbox for {context}")
        clip_w = int(cx1 - cx0)
        clip_h = int(cy1 - cy0)
        if clip_w < target_w or clip_h < target_h:
            raise RuntimeError(f"clip bbox too small for {min_side}px icon annotation at {context}")
        if expanded[0] < cx0:
            delta = int(cx0 - expanded[0])
            expanded[0] += delta
            expanded[2] += delta
        if expanded[2] > cx1:
            delta = int(expanded[2] - cx1)
            expanded[0] -= delta
            expanded[2] -= delta
        if expanded[1] < cy0:
            delta = int(cy0 - expanded[1])
            expanded[1] += delta
            expanded[3] += delta
        if expanded[3] > cy1:
            delta = int(expanded[3] - cy1)
            expanded[1] -= delta
            expanded[3] -= delta
    else:
        if expanded[0] < 0:
            expanded[2] -= expanded[0]
            expanded[0] = 0
        if expanded[1] < 0:
            expanded[3] -= expanded[1]
            expanded[1] = 0
    if min(int(expanded[2] - expanded[0]), int(expanded[3] - expanded[1])) < min_side:
        raise RuntimeError(f"expanded icon annotation is smaller than {min_side}px at {context}")
    return [int(value) for value in expanded]


def icon_object_bbox(bbox: Sequence[int | float], *, clip_bbox: Sequence[int | float] | None = None) -> list[int]:
    """Return an icon-object bbox expanded to the domain minimum annotation side."""

    return _expand_bbox_to_min_side(
        bbox,
        min_side_px=ICON_OBJECT_ANNOTATION_MIN_SIDE_PX,
        clip_bbox=clip_bbox,
        context="icon object",
    )


def matching_scene_cell_bbox_annotation(
    *,
    scene_cells: Sequence[Mapping[str, Any]],
    matching_labels: Sequence[str],
) -> Dict[str, Any]:
    """Return bbox-set annotation for matching labeled Scene cells."""

    requested_labels = [str(label) for label in matching_labels]
    requested_set = set(requested_labels)
    cells_by_label: Dict[str, Mapping[str, Any]] = {
        str(cell.get("label")): cell
        for cell in scene_cells
        if isinstance(cell, Mapping) and str(cell.get("label")) in requested_set
    }
    missing = [label for label in requested_labels if label not in cells_by_label]
    if missing:
        raise RuntimeError(f"missing Scene cell bbox annotation for labels: {missing}")

    matched_cells = list(cells_by_label.values())
    matched_cells.sort(
        key=lambda cell: (
            float((cell.get("cell_bbox_xyxy") or [0, 0, 0, 0])[1]),
            float((cell.get("cell_bbox_xyxy") or [0, 0, 0, 0])[0]),
            str(cell.get("label")),
        )
    )
    bboxes: list[list[int]] = []
    labels_top_left: list[str] = []
    for cell in matched_cells:
        bbox = cell.get("cell_bbox_xyxy")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise RuntimeError(f"invalid Scene cell bbox for label {cell.get('label')}: {bbox}")
        bboxes.append([int(round(float(value))) for value in bbox])
        labels_top_left.append(str(cell.get("label")))

    return {
        "annotation_type": "bbox_set",
        "annotation_value": [list(bbox) for bbox in bboxes],
        "labels_top_left": list(labels_top_left),
        "witness_symbolic": {
            "matching_cell_labels": list(requested_labels),
            "matching_cell_labels_top_left": list(labels_top_left),
        },
        "projected_annotation": {
            "type": "bbox_set",
            "bbox_set": [list(bbox) for bbox in bboxes],
            "pixel_bbox_set": [list(bbox) for bbox in bboxes],
            "pixel_point_set": [_bbox_center(bbox) for bbox in bboxes],
        },
    }


def point_set_annotation(
    points: Sequence[Sequence[int | float]],
) -> Dict[str, Any]:
    """Return typed point-set annotation for homogeneous icon witnesses."""

    normalized_points = [
        _normalize_point(point, context=f"point_set index {index}")
        for index, point in enumerate(points)
    ]
    return {
        "annotation_type": "point_set",
        "annotation_value": [list(point) for point in normalized_points],
        "projected_annotation": {
            "type": "point_set",
            "point_set": [list(point) for point in normalized_points],
            "pixel_point_set": [list(point) for point in normalized_points],
        },
    }


def point_annotation(point: Sequence[int | float]) -> Dict[str, Any]:
    """Return typed scalar point annotation for one icon witness."""

    normalized_point = _normalize_point(point, context="point")
    return {
        "annotation_type": "point",
        "annotation_value": list(normalized_point),
        "projected_annotation": {
            "type": "point",
            "point": list(normalized_point),
            "pixel_point": list(normalized_point),
        },
    }


def point_from_bbox(bbox: Sequence[int | float]) -> Dict[str, Any]:
    """Return scalar point annotation using one bbox center as witness."""

    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise RuntimeError(f"invalid source bbox for point annotation: {bbox}")
    return point_annotation(_bbox_center(bbox))


def point_set_from_bboxes(
    bboxes: Sequence[Sequence[int | float]],
) -> Dict[str, Any]:
    """Return point-set annotation using bbox centers as witnesses."""

    normalized_bboxes: list[list[int]] = []
    for index, bbox in enumerate(bboxes):
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise RuntimeError(f"invalid source bbox for point_set annotation at index {index}: {bbox}")
        normalized_bboxes.append([int(round(float(value))) for value in bbox])
    return point_set_annotation(_bbox_center(bbox) for bbox in normalized_bboxes)


def bbox_set_annotation(
    bboxes: Sequence[Sequence[int | float]],
) -> Dict[str, Any]:
    """Return typed bbox-set annotation for homogeneous icon witnesses."""

    normalized_bboxes: list[list[int]] = []
    for index, bbox in enumerate(bboxes):
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise RuntimeError(f"invalid bbox_set annotation at index {index}: {bbox}")
        normalized_bboxes.append([int(round(float(value))) for value in bbox])
    return {
        "annotation_type": "bbox_set",
        "annotation_value": [list(bbox) for bbox in normalized_bboxes],
        "projected_annotation": {
            "type": "bbox_set",
            "bbox_set": [list(bbox) for bbox in normalized_bboxes],
            "pixel_bbox_set": [list(bbox) for bbox in normalized_bboxes],
            "pixel_point_set": [_bbox_center(bbox) for bbox in normalized_bboxes],
        },
    }


def icon_bbox_set_annotation(
    bboxes: Sequence[Sequence[int | float]],
    *,
    clip_bbox: Sequence[int | float] | None = None,
) -> Dict[str, Any]:
    """Return bbox-set annotation for icon objects with a 24px minimum side."""

    return bbox_set_annotation(
        [
            icon_object_bbox(bbox, clip_bbox=clip_bbox)
            for bbox in bboxes
        ]
    )


def bbox_annotation(bbox: Sequence[int | float]) -> Dict[str, Any]:
    """Return typed scalar bbox annotation for one icon witness."""

    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise RuntimeError(f"invalid bbox annotation: {bbox}")
    normalized_bbox = [int(round(float(value))) for value in bbox]
    return {
        "annotation_type": "bbox",
        "annotation_value": list(normalized_bbox),
        "projected_annotation": {
            "type": "bbox",
            "bbox": list(normalized_bbox),
            "pixel_bbox": list(normalized_bbox),
            "pixel_point": _bbox_center(normalized_bbox),
        },
    }


def icon_bbox_annotation(
    bbox: Sequence[int | float],
    *,
    clip_bbox: Sequence[int | float] | None = None,
) -> Dict[str, Any]:
    """Return scalar bbox annotation for one icon object with a 24px minimum side."""

    return bbox_annotation(icon_object_bbox(bbox, clip_bbox=clip_bbox))


def point_map_annotation(
    role_points: Mapping[str, Sequence[int | float]],
) -> Dict[str, Any]:
    """Return keyed-point annotation for role-bound icon witnesses."""

    keyed_points = {
        str(role): _normalize_point(point, context=f"role {role!r}")
        for role, point in role_points.items()
    }
    return {
        "annotation_type": "point_map",
        "annotation_value": {str(key): list(value) for key, value in keyed_points.items()},
        "projected_annotation": {
            "type": "point_map",
            "point_map": {str(key): list(value) for key, value in keyed_points.items()},
            "pixel_point_map": {str(key): list(value) for key, value in keyed_points.items()},
        },
    }


def point_map_from_bboxes(
    role_bboxes: Mapping[str, Sequence[int | float]],
) -> Dict[str, Any]:
    """Return keyed-point annotation using role bbox centers as witnesses."""

    role_points: dict[str, list[float]] = {}
    for role, bbox in role_bboxes.items():
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise RuntimeError(f"invalid keyed source bbox for role {role!r}: {bbox}")
        role_points[str(role)] = _bbox_center(bbox)
    return point_map_annotation(role_points)


def bbox_map_annotation(
    role_bboxes: Mapping[str, Sequence[int | float]],
) -> Dict[str, Any]:
    """Return keyed-bbox annotation for role-bound icon witnesses."""

    keyed_bboxes: dict[str, list[int]] = {}
    for role, bbox in role_bboxes.items():
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise RuntimeError(f"invalid keyed bbox for role {role!r}: {bbox}")
        keyed_bboxes[str(role)] = [int(round(float(value))) for value in bbox]
    return {
        "annotation_type": "bbox_map",
        "annotation_value": {str(key): list(value) for key, value in keyed_bboxes.items()},
        "projected_annotation": {
            "type": "bbox_map",
            "bbox_map": {str(key): list(value) for key, value in keyed_bboxes.items()},
            "pixel_bbox_map": {str(key): list(value) for key, value in keyed_bboxes.items()},
        },
    }


def icon_bbox_map_annotation(
    role_bboxes: Mapping[str, Sequence[int | float]],
    *,
    clip_bbox: Sequence[int | float] | None = None,
) -> Dict[str, Any]:
    """Return keyed icon-object bboxes with a 24px minimum side."""

    return bbox_map_annotation(
        {
            str(role): icon_object_bbox(bbox, clip_bbox=clip_bbox)
            for role, bbox in role_bboxes.items()
        }
    )


def bbox_set_map_annotation(
    role_bbox_sets: Mapping[str, Sequence[Sequence[int | float]]],
) -> Dict[str, Any]:
    """Return keyed bbox-set annotation for role-bound witness groups."""

    keyed_bbox_sets: dict[str, list[list[int]]] = {}
    for role, bboxes in role_bbox_sets.items():
        if not isinstance(bboxes, Sequence):
            raise RuntimeError(f"invalid keyed bbox set for role {role!r}: {bboxes}")
        normalized_bboxes: list[list[int]] = []
        for index, bbox in enumerate(bboxes):
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                raise RuntimeError(f"invalid keyed bbox for role {role!r} at index {index}: {bbox}")
            normalized_bboxes.append([int(round(float(value))) for value in bbox])
        keyed_bbox_sets[str(role)] = list(normalized_bboxes)
    return {
        "annotation_type": "bbox_set_map",
        "annotation_value": {str(key): [list(bbox) for bbox in value] for key, value in keyed_bbox_sets.items()},
        "projected_annotation": {
            "type": "bbox_set_map",
            "bbox_set_map": {str(key): [list(bbox) for bbox in value] for key, value in keyed_bbox_sets.items()},
            "pixel_bbox_set_map": {str(key): [list(bbox) for bbox in value] for key, value in keyed_bbox_sets.items()},
        },
    }


__all__ = [
    "bbox_annotation",
    "icon_bbox_annotation",
    "icon_bbox_map_annotation",
    "icon_bbox_set_annotation",
    "icon_object_bbox",
    "ICON_OBJECT_ANNOTATION_MIN_SIDE_PX",
    "bbox_set_annotation",
    "bbox_map_annotation",
    "bbox_set_map_annotation",
    "point_map_annotation",
    "point_map_from_bboxes",
    "matching_scene_cell_bbox_annotation",
    "point_annotation",
    "point_from_bbox",
    "point_set_annotation",
    "point_set_from_bboxes",
]
