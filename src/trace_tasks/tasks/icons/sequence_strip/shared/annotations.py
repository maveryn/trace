"""Annotation helpers for sequence-strip icon scenes."""

from __future__ import annotations

from typing import Sequence

from ....shared.annotation_artifacts import AnnotationArtifacts, bbox_annotation_artifacts
from ...shared.icon_scene import sort_bboxes_reading_order


def scalar_bbox_artifacts(
    bboxes_xyxy: Sequence[Sequence[int]],
    *,
    error_message: str,
) -> AnnotationArtifacts:
    """Project exactly one scalar bbox after stable reading-order normalization."""

    annotation_bboxes = sort_bboxes_reading_order(tuple(tuple(int(value) for value in bbox) for bbox in bboxes_xyxy))
    if len(annotation_bboxes) != 1:
        raise RuntimeError(str(error_message))
    return bbox_annotation_artifacts(annotation_bboxes[0])


__all__ = ["scalar_bbox_artifacts"]
