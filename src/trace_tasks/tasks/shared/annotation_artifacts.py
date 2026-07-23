"""Shared public annotation artifact builders for task generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...core.types import TypedValue


@dataclass(frozen=True)
class AnnotationArtifacts:
    """Normalized annotation value plus matching public projection payload."""

    annotation_type: str
    value: Any
    annotation_gt: TypedValue
    projected_annotation: dict[str, Any]


def annotation_cardinality(value: Any, *, annotation_type: str = "") -> int:
    """Return the number of visual witnesses represented by an annotation."""

    if str(annotation_type) in {"bbox", "point", "segment"}:
        return 1 if value is not None else 0
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1 if value is not None else 0


def _round_point(point: Sequence[float], *, ndigits: int = 3) -> list[float]:
    return [
        round(float(point[0]), int(ndigits)),
        round(float(point[1]), int(ndigits)),
    ]


def _round_bbox(bbox: Sequence[float], *, ndigits: int = 3) -> list[float]:
    return [round(float(value), int(ndigits)) for value in bbox[:4]]


def bbox_annotation_artifacts(
    bbox: Sequence[float],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for one bbox witness."""

    value = _round_bbox(bbox, ndigits=int(ndigits))
    projected_annotation = {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
    }
    return AnnotationArtifacts(
        annotation_type="bbox",
        value=list(value),
        annotation_gt=TypedValue(type="bbox", value=list(value)),
        projected_annotation=projected_annotation,
    )


def bbox_set_annotation_artifacts(
    bboxes: Sequence[Sequence[float]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for an unordered bbox set."""

    value = [_round_bbox(bbox, ndigits=int(ndigits)) for bbox in bboxes]
    projected_annotation = {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in value],
        "pixel_bbox_set": [list(bbox) for bbox in value],
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set",
        value=[list(bbox) for bbox in value],
        annotation_gt=TypedValue(type="bbox_set", value=[list(bbox) for bbox in value]),
        projected_annotation=projected_annotation,
    )


def bbox_map_annotation_artifacts(
    keyed_bboxes: Mapping[str, Sequence[float]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for role-keyed bbox witnesses."""

    value = {
        str(key): _round_bbox(bbox, ndigits=int(ndigits))
        for key, bbox in dict(keyed_bboxes).items()
    }
    projected_annotation = {
        "type": "bbox_map",
        "bbox_map": {str(key): list(bbox) for key, bbox in value.items()},
        "pixel_bbox_map": {str(key): list(bbox) for key, bbox in value.items()},
    }
    return AnnotationArtifacts(
        annotation_type="bbox_map",
        value={str(key): list(bbox) for key, bbox in value.items()},
        annotation_gt=TypedValue(
            type="bbox_map",
            value={str(key): list(bbox) for key, bbox in value.items()},
        ),
        projected_annotation=projected_annotation,
    )


def bbox_set_map_annotation_artifacts(
    keyed_bboxes: Mapping[str, Sequence[Sequence[float]]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for role-keyed bbox sets."""

    value = {
        str(key): [_round_bbox(bbox, ndigits=int(ndigits)) for bbox in bboxes]
        for key, bboxes in dict(keyed_bboxes).items()
    }
    projected_annotation = {
        "type": "bbox_set_map",
        "bbox_set_map": {str(key): [list(bbox) for bbox in bboxes] for key, bboxes in value.items()},
        "pixel_bbox_set_map": {str(key): [list(bbox) for bbox in bboxes] for key, bboxes in value.items()},
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set_map",
        value={str(key): [list(bbox) for bbox in bboxes] for key, bboxes in value.items()},
        annotation_gt=TypedValue(
            type="bbox_set_map",
            value={str(key): [list(bbox) for bbox in bboxes] for key, bboxes in value.items()},
        ),
        projected_annotation=projected_annotation,
    )


def point_annotation_artifacts(
    point: Sequence[float],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for one point witness."""

    value = _round_point(point, ndigits=int(ndigits))
    projected_annotation = {
        "type": "point",
        "point": list(value),
        "pixel_point": list(value),
    }
    return AnnotationArtifacts(
        annotation_type="point",
        value=list(value),
        annotation_gt=TypedValue(type="point", value=list(value)),
        projected_annotation=projected_annotation,
    )


def point_set_annotation_artifacts(
    points: Sequence[Sequence[float]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for an unordered point set."""

    value = [_round_point(point, ndigits=int(ndigits)) for point in points]
    projected_annotation = {
        "type": "point_set",
        "point_set": [list(point) for point in value],
        "pixel_point_set": [list(point) for point in value],
    }
    return AnnotationArtifacts(
        annotation_type="point_set",
        value=[list(point) for point in value],
        annotation_gt=TypedValue(type="point_set", value=[list(point) for point in value]),
        projected_annotation=projected_annotation,
    )


def segment_set_annotation_artifacts(
    segments: Sequence[Sequence[Sequence[float]]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for unordered segment witnesses."""

    value = [
        [_round_point(point, ndigits=int(ndigits)) for point in segment[:2]]
        for segment in segments
    ]
    projected_annotation = {
        "type": "segment_set",
        "segment_set": [[list(point) for point in pair] for pair in value],
        "pixel_segment_set": [[list(point) for point in pair] for pair in value],
    }
    return AnnotationArtifacts(
        annotation_type="segment_set",
        value=[[list(point) for point in pair] for pair in value],
        annotation_gt=TypedValue(
            type="segment_set",
            value=[[list(point) for point in pair] for pair in value],
        ),
        projected_annotation=projected_annotation,
    )


def segment_annotation_artifacts(
    segment: Sequence[Sequence[float]],
    *,
    ndigits: int = 3,
) -> AnnotationArtifacts:
    """Build public annotation artifacts for one finite segment witness."""

    value = [_round_point(point, ndigits=int(ndigits)) for point in segment[:2]]
    projected_annotation = {
        "type": "segment",
        "segment": [list(point) for point in value],
        "pixel_segment": [list(point) for point in value],
    }
    return AnnotationArtifacts(
        annotation_type="segment",
        value=[list(point) for point in value],
        annotation_gt=TypedValue(
            type="segment",
            value=[list(point) for point in value],
        ),
        projected_annotation=projected_annotation,
    )


__all__ = [
    "AnnotationArtifacts",
    "annotation_cardinality",
    "bbox_annotation_artifacts",
    "bbox_map_annotation_artifacts",
    "bbox_set_annotation_artifacts",
    "bbox_set_map_annotation_artifacts",
    "point_annotation_artifacts",
    "segment_annotation_artifacts",
    "segment_set_annotation_artifacts",
    "point_set_annotation_artifacts",
]
