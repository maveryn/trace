"""Annotation helpers for rendered surface-fixture elements."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts
from trace_tasks.tasks.three_d.shared.annotation_geometry import normalize_annotation_bboxes

from .rendering import RenderedSurfaceFixture


def element_centers_for_ids(rendered: RenderedSurfaceFixture, element_ids: Sequence[str]) -> list[list[float]]:
    """Return pixel center points for selected rendered fixture elements."""

    return [
        list(rendered.element_centers_px[str(element_id)])
        for element_id in element_ids
    ]


def element_bboxes_for_ids(rendered: RenderedSurfaceFixture, element_ids: Sequence[str]) -> list[list[float]]:
    """Return pixel boxes for selected rendered fixture elements."""

    return [
        list(rendered.element_bboxes_px[str(element_id)])
        for element_id in element_ids
    ]


def bbox_set_annotation_for_elements(
    rendered: RenderedSurfaceFixture,
    element_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered bbox-set annotation artifacts for selected fixture elements."""

    artifacts, _bboxes_by_id, _normalization = bbox_set_annotation_for_elements_with_metadata(rendered, element_ids)
    return artifacts


def bbox_set_annotation_for_elements_with_metadata(
    rendered: RenderedSurfaceFixture,
    element_ids: Sequence[str],
) -> tuple[AnnotationArtifacts, dict[str, list[float]], dict[str, object]]:
    """Build min-side-normalized bbox-set artifacts plus per-element trace metadata."""

    raw_bboxes = element_bboxes_for_ids(rendered, element_ids)
    normalized_bboxes, normalization = normalize_annotation_bboxes(
        raw_bboxes,
        bounds_px=rendered.fixture_bbox_px,
    )
    bboxes_by_id = {
        str(element_id): list(bbox)
        for element_id, bbox in zip(element_ids, normalized_bboxes)
    }
    return bbox_set_annotation_artifacts(normalized_bboxes), bboxes_by_id, dict(normalization)


__all__ = [
    "bbox_set_annotation_for_elements",
    "bbox_set_annotation_for_elements_with_metadata",
    "element_bboxes_for_ids",
    "element_centers_for_ids",
]
