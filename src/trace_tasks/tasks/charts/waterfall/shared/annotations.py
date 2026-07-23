"""Annotation artifact builders for waterfall chart tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts


def _round_bbox(bbox: Sequence[float], *, ndigits: int = 3) -> list[float]:
    return [round(float(value), int(ndigits)) for value in list(bbox)[:4]]


def bbox_map_artifacts(
    values: Mapping[str, Sequence[float]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public bbox-map annotation artifacts."""

    rounded = {str(key): _round_bbox(value, ndigits=int(ndigits)) for key, value in values.items()}
    projected = {
        "type": "bbox_map",
        "bbox_map": {key: list(value) for key, value in rounded.items()},
        "pixel_bbox_map": {key: list(value) for key, value in rounded.items()},
        "bbox_set": [list(value) for value in rounded.values()],
    }
    return AnnotationArtifacts(
        annotation_type="bbox_map",
        value={key: list(value) for key, value in rounded.items()},
        annotation_gt=TypedValue(
            type="bbox_map",
            value={key: list(value) for key, value in rounded.items()},
        ),
        projected_annotation=dict(projected),
    )


def bbox_artifacts(
    value: Sequence[float],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build a scalar bbox annotation artifact for one visual witness."""

    rounded = _round_bbox(value, ndigits=int(ndigits))
    projected = {
        "type": "bbox",
        "bbox": list(rounded),
        "pixel_bbox": list(rounded),
    }
    return AnnotationArtifacts(
        annotation_type="bbox",
        value=list(rounded),
        annotation_gt=TypedValue(type="bbox", value=list(rounded)),
        projected_annotation=dict(projected),
    )


def bbox_set_artifacts(
    values: Sequence[Sequence[float]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build unordered bbox-set annotation artifacts."""

    rounded = [_round_bbox(bbox, ndigits=int(ndigits)) for bbox in values]
    projected = {
        "type": "bbox_set",
        "bbox_set": [list(value) for value in rounded],
        "pixel_bbox_set": [list(value) for value in rounded],
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set",
        value=[list(value) for value in rounded],
        annotation_gt=TypedValue(
            type="bbox_set",
            value=[list(value) for value in rounded],
        ),
        projected_annotation=dict(projected),
    )


def bbox_set_map_artifacts(
    values: Mapping[str, Sequence[Sequence[float]]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build keyed bbox-set artifacts while preserving each semantic role."""

    rounded = {
        str(key): [_round_bbox(bbox, ndigits=int(ndigits)) for bbox in boxes]
        for key, boxes in values.items()
    }
    projected = {
        "type": "bbox_set_map",
        "bbox_set_map": {
            key: [list(bbox) for bbox in boxes]
            for key, boxes in rounded.items()
        },
        "pixel_bbox_set_map": {
            key: [list(bbox) for bbox in boxes]
            for key, boxes in rounded.items()
        },
        "bbox_set": [
            list(bbox)
            for boxes in rounded.values()
            for bbox in boxes
        ],
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set_map",
        value={
            key: [list(bbox) for bbox in boxes]
            for key, boxes in rounded.items()
        },
        annotation_gt=TypedValue(
            type="bbox_set_map",
            value={
                key: [list(bbox) for bbox in boxes]
                for key, boxes in rounded.items()
            },
        ),
        projected_annotation=dict(projected),
    )


__all__ = [
    "bbox_artifacts",
    "bbox_map_artifacts",
    "bbox_set_artifacts",
    "bbox_set_map_artifacts",
]
