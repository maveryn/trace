"""Annotation projection helpers for the area-partition scene."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .rendering import RenderedAreaPartitionScene


def keyed_region_bboxes(rendered: RenderedAreaPartitionScene) -> dict[str, list[float]]:
    """Return keyed region bboxes from the final rendered scene."""

    return {
        str(key): bbox_to_list(bbox)
        for key, bbox in rendered.annotation_keyed_bboxes.items()
    }


def keyed_region_points(keyed_bboxes: dict[str, list[float]]) -> dict[str, list[float]]:
    """Return bbox centers for trace-side projected point metadata."""

    return {
        str(key): [
            round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
            round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
        ]
        for key, bbox in keyed_bboxes.items()
    }


def keyed_bbox_annotation(rendered: RenderedAreaPartitionScene) -> TypedValue:
    """Build the public keyed-bbox annotation value."""

    return TypedValue(type="bbox_map", value=keyed_region_bboxes(rendered))


__all__ = ["keyed_bbox_annotation", "keyed_region_bboxes", "keyed_region_points"]
