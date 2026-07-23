"""Annotation projection helpers for balance-scale query witnesses."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts


def bbox_annotation_artifacts(
    item_bbox_map: Mapping[str, Sequence[float]],
    item_id: str,
) -> AnnotationArtifacts:
    """Project one target item box as a scalar bbox annotation."""

    key = str(item_id)
    if key not in item_bbox_map:
        raise RuntimeError(f"missing bbox annotation for item {key!r}")
    value = [round(float(coord), 3) for coord in item_bbox_map[key][:4]]
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


def bbox_set_annotation_artifacts(
    item_bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project multiple target item boxes as an unordered bbox set."""

    value = []
    for item_id in item_ids:
        key = str(item_id)
        if key not in item_bbox_map:
            raise RuntimeError(f"missing bbox annotation for item {key!r}")
        value.append([round(float(coord), 3) for coord in item_bbox_map[key][:4]])
    projected = {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in value],
        "pixel_bbox_set": [list(bbox) for bbox in value],
        "value": [list(bbox) for bbox in value],
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set",
        value=[list(bbox) for bbox in value],
        annotation_gt=TypedValue(
            type="bbox_set",
            value=[list(bbox) for bbox in value],
        ),
        projected_annotation=projected,
    )


def bbox_map_annotation_artifacts(
    item_bbox_map: Mapping[str, Sequence[float]],
    role_item_ids: Mapping[str, str],
) -> AnnotationArtifacts:
    """Project role-bound item boxes as a public scalar bbox map."""

    value: Dict[str, list[float]] = {}
    for role, item_name in role_item_ids.items():
        key = str(item_name)
        if key not in item_bbox_map:
            raise RuntimeError(
                f"missing bbox annotation for role {role!r}: item {key!r}"
            )
        value[str(role)] = [round(float(coord), 3) for coord in item_bbox_map[key][:4]]
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


__all__ = [
    "bbox_annotation_artifacts",
    "bbox_map_annotation_artifacts",
    "bbox_set_annotation_artifacts",
]
