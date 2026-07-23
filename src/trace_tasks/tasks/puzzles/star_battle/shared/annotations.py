"""Annotation projection helpers for Star Battle puzzle tasks."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts


def round_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one pixel bbox into public annotation form."""

    return [round(float(value), 3) for value in bbox[:4]]


def bbox_artifacts(
    item_bbox_map: Mapping[str, Sequence[float]],
    item_id: str,
) -> AnnotationArtifacts:
    """Project one item bbox as a scalar bbox annotation."""

    key = str(item_id)
    if key not in item_bbox_map:
        raise RuntimeError(f"missing bbox annotation for item {key!r}")
    value = round_bbox(item_bbox_map[key])
    projected = {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
        "value": list(value),
    }
    return AnnotationArtifacts(
        annotation_type="bbox",
        value=list(value),
        annotation_gt=TypedValue(type="bbox", value=list(value)),
        projected_annotation=projected,
    )


def bbox_set_artifacts(
    item_bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project a homogeneous set of item bboxes as bbox_set annotation."""

    values: list[list[float]] = []
    for item_id in item_ids:
        key = str(item_id)
        if key not in item_bbox_map:
            raise RuntimeError(f"missing bbox annotation for item {key!r}")
        values.append(round_bbox(item_bbox_map[key]))
    projected = {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in values],
        "pixel_bbox_set": [list(bbox) for bbox in values],
        "value": [list(bbox) for bbox in values],
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set",
        value=[list(bbox) for bbox in values],
        annotation_gt=TypedValue(type="bbox_set", value=[list(bbox) for bbox in values]),
        projected_annotation=projected,
    )


def bbox_map_artifacts(
    item_bbox_map: Mapping[str, Sequence[float]],
    role_item_ids: Mapping[str, str],
) -> AnnotationArtifacts:
    """Project role-bound item bboxes as bbox_map annotation."""

    value: Dict[str, list[float]] = {}
    for role, item_id in role_item_ids.items():
        key = str(item_id)
        if key not in item_bbox_map:
            raise RuntimeError(f"missing bbox annotation for role {role!r}: item {key!r}")
        value[str(role)] = round_bbox(item_bbox_map[key])
    projected = {
        "type": "bbox_map",
        "bbox_map": {str(role): list(bbox) for role, bbox in value.items()},
        "pixel_bbox_map": {str(role): list(bbox) for role, bbox in value.items()},
        "value": {str(role): list(bbox) for role, bbox in value.items()},
    }
    return AnnotationArtifacts(
        annotation_type="bbox_map",
        value={str(role): list(bbox) for role, bbox in value.items()},
        annotation_gt=TypedValue(
            type="bbox_map",
            value={str(role): list(bbox) for role, bbox in value.items()},
        ),
        projected_annotation=projected,
    )


__all__ = ["bbox_artifacts", "bbox_map_artifacts", "bbox_set_artifacts", "round_bbox"]
