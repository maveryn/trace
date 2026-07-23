"""Annotation projection helpers for Tents candidate cells."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue


def candidate_bbox_annotation(
    item_bbox_map: Mapping[str, Sequence[float]],
    label: str,
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Project one labeled candidate cell to scalar bbox annotation."""

    item_id = f"candidate_{str(label)}"
    if item_id not in item_bbox_map:
        raise RuntimeError(f"missing candidate bbox for label {label!r}")
    bbox = [round(float(value), 3) for value in item_bbox_map[item_id]]
    annotation_gt = TypedValue(type="bbox", value=list(bbox))
    projected = {
        "type": "bbox",
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
        "value": list(bbox),
    }
    witness = {
        "type": "bbox",
        "item_id": item_id,
        "label": str(label),
        "value": list(bbox),
    }
    return annotation_gt, projected, witness


def candidate_bbox_set_annotation(
    item_bbox_map: Mapping[str, Sequence[float]],
    labels: Sequence[str],
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Project zero or more labeled candidate cells to bbox_set annotation."""

    bboxes: list[list[float]] = []
    item_ids: list[str] = []
    for label in labels:
        item_id = f"candidate_{str(label)}"
        if item_id not in item_bbox_map:
            raise RuntimeError(f"missing candidate bbox for label {label!r}")
        item_ids.append(item_id)
        bboxes.append([round(float(value), 3) for value in item_bbox_map[item_id]])
    annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in bboxes])
    projected = {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in bboxes],
        "pixel_bbox_set": [list(bbox) for bbox in bboxes],
        "value": [list(bbox) for bbox in bboxes],
    }
    witness = {
        "type": "bbox_set",
        "item_ids": list(item_ids),
        "labels": [str(label) for label in labels],
        "value": [list(bbox) for bbox in bboxes],
    }
    return annotation_gt, projected, witness


def labeled_tent_bbox_annotation(
    item_bbox_map: Mapping[str, Sequence[float]],
    label: str,
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    """Project one labeled tent cell to scalar bbox annotation."""

    item_id = f"labeled_tent_{str(label)}"
    if item_id not in item_bbox_map:
        raise RuntimeError(f"missing labeled tent bbox for label {label!r}")
    bbox = [round(float(value), 3) for value in item_bbox_map[item_id]]
    annotation_gt = TypedValue(type="bbox", value=list(bbox))
    projected = {
        "type": "bbox",
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
        "value": list(bbox),
    }
    witness = {
        "type": "bbox",
        "item_id": item_id,
        "label": str(label),
        "value": list(bbox),
    }
    return annotation_gt, projected, witness


__all__ = [
    "candidate_bbox_annotation",
    "candidate_bbox_set_annotation",
    "labeled_tent_bbox_annotation",
]
