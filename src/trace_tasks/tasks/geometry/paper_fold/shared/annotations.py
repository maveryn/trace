"""Annotation helpers for paper-fold diagrams."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts, keyed_bbox_annotation_artifacts
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts

from .state import BBox, Segment


def paper_fold_bbox_annotation(
    bboxes: Mapping[str, BBox],
    *,
    roles: Sequence[str],
) -> PixelAnnotationArtifacts:
    """Build role-bound bbox annotation for the angle cue and given label."""

    return keyed_bbox_annotation_artifacts(bboxes, roles=tuple(str(role) for role in roles))


def paper_fold_segment_annotation(segment: Segment) -> PixelAnnotationArtifacts:
    """Build scalar segment annotation for a requested folded side."""

    artifacts = segment_annotation_artifacts(segment)
    return PixelAnnotationArtifacts(
        annotation_type=artifacts.annotation_type,
        value=artifacts.value,
        projected_annotation=artifacts.projected_annotation,
    )


__all__ = ["paper_fold_bbox_annotation", "paper_fold_segment_annotation"]
