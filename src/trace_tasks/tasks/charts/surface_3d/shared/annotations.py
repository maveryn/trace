"""Annotation artifact helpers for 3D chart scene witnesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    point_annotation_artifacts,
    segment_annotation_artifacts,
)


def bbox_for_single_witness(bbox: Sequence[float]) -> AnnotationArtifacts:
    """Build scalar bbox artifacts for one selected 3D chart witness."""

    return bbox_annotation_artifacts(bbox)


def point_for_single_witness(bbox: Sequence[float]) -> AnnotationArtifacts:
    """Build scalar point artifacts at the center of one selected marker box."""

    values = [float(value) for value in list(bbox)[:4]]
    return point_annotation_artifacts([(values[0] + values[2]) / 2.0, (values[1] + values[3]) / 2.0])


def bbox_map_for_roles(values: Mapping[str, Sequence[float]], *, ndigits: int = 3) -> AnnotationArtifacts:
    """Build keyed bbox artifacts when start/end roles must stay bound."""

    rounded = {
        str(role): [round(float(value), int(ndigits)) for value in list(bbox)[:4]]
        for role, bbox in values.items()
    }
    projected = {
        "type": "bbox_map",
        "bbox_map": {role: list(bbox) for role, bbox in rounded.items()},
        "pixel_bbox_map": {role: list(bbox) for role, bbox in rounded.items()},
    }
    return AnnotationArtifacts(
        annotation_type="bbox_map",
        value={role: list(bbox) for role, bbox in rounded.items()},
        annotation_gt=TypedValue(type="bbox_map", value={role: list(bbox) for role, bbox in rounded.items()}),
        projected_annotation=dict(projected),
    )


def segment_between_bboxes(
    start_bbox: Sequence[float],
    end_bbox: Sequence[float],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build scalar segment artifacts between the centers of two marker boxes."""

    start = [float(value) for value in list(start_bbox)[:4]]
    end = [float(value) for value in list(end_bbox)[:4]]
    return segment_annotation_artifacts(
        [
            [(start[0] + start[2]) / 2.0, (start[1] + start[3]) / 2.0],
            [(end[0] + end[2]) / 2.0, (end[1] + end[3]) / 2.0],
        ],
        ndigits=int(ndigits),
    )


__all__ = [
    "bbox_for_single_witness",
    "bbox_map_for_roles",
    "point_for_single_witness",
    "segment_between_bboxes",
]
