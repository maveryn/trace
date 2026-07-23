"""Annotation helpers for trapezoid-extension scenes."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts, segment_annotation_artifacts

from .state import RenderedTrapezoidExtensionScene


def trapezoid_extension_annotation(rendered: RenderedTrapezoidExtensionScene) -> PixelAnnotationArtifacts:
    """Build the scalar public annotation for the rendered geometric witness."""

    if str(rendered.annotation_mode) == "extension_segment":
        if rendered.annotation_segment is None:
            raise ValueError("extension_segment annotation requested without a segment")
        artifacts = segment_annotation_artifacts(rendered.annotation_segment)
        return PixelAnnotationArtifacts(
            annotation_type=artifacts.annotation_type,
            value=artifacts.value,
            projected_annotation=artifacts.projected_annotation,
        )
    if rendered.annotation_bbox is None:
        raise ValueError("bbox annotation requested without a bbox")
    artifacts = bbox_annotation_artifacts(rendered.annotation_bbox)
    return PixelAnnotationArtifacts(
        annotation_type=artifacts.annotation_type,
        value=artifacts.value,
        projected_annotation=artifacts.projected_annotation,
    )


__all__ = ["trapezoid_extension_annotation"]
