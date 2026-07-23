"""Annotation helpers for tangent-packing scenes."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts, bbox_annotation_artifacts

from .state import RenderedTangentPackingScene

ANNOTATION_ROLE = "diagram"


def tangent_packing_annotation(rendered: RenderedTangentPackingScene) -> PixelAnnotationArtifacts:
    """Build one scalar bbox around the target geometric witness."""

    return bbox_annotation_artifacts(rendered.annotation_bboxes[ANNOTATION_ROLE])


__all__ = ["ANNOTATION_ROLE", "tangent_packing_annotation"]
