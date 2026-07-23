"""Annotation projection helpers for voxel-cube puzzle tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
)

from .state import BBox


def scalar_bbox_annotation(
    bbox: BBox | None,
    *,
    role: str,
) -> tuple[object, dict[str, object], dict[str, object]]:
    """Build scalar bbox artifacts for one required visual witness."""

    if bbox is None:
        raise ValueError(f"missing bbox for annotation role: {role}")
    artifacts = bbox_annotation_artifacts(bbox)
    return (
        artifacts.annotation_gt,
        artifacts.projected_annotation,
        {"type": "bbox", "role": str(role), "bbox": list(artifacts.value)},
    )


def bbox_set_annotation(
    bboxes: Sequence[BBox],
    *,
    role: str,
) -> tuple[object, dict[str, object], dict[str, object]]:
    """Build bbox-set artifacts for unordered visual witnesses."""

    artifacts = bbox_set_annotation_artifacts(tuple(bboxes))
    return (
        artifacts.annotation_gt,
        artifacts.projected_annotation,
        {
            "type": "bbox_set",
            "role": str(role),
            "bboxes": [list(bbox) for bbox in artifacts.value],
        },
    )


def option_bbox_annotation(
    option_bboxes: Mapping[str, BBox],
    answer_label: str,
) -> tuple[object, dict[str, object], dict[str, object]]:
    """Build scalar bbox artifacts for the selected option panel."""

    label = str(answer_label)
    if label not in option_bboxes:
        raise ValueError(f"missing option bbox for label {label}")
    artifacts = bbox_annotation_artifacts(option_bboxes[label])
    return (
        artifacts.annotation_gt,
        artifacts.projected_annotation,
        {
            "type": "bbox",
            "role": "selected_option_panel",
            "option_label": label,
            "bbox": list(artifacts.value),
        },
    )
