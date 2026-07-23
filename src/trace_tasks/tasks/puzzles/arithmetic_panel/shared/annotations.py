"""Annotation projection helpers for arithmetic-constraint targets."""

from __future__ import annotations

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
)

from .state import RenderedArithmeticScene, target_bbox


def target_bbox_annotation(
    *,
    rendered_scene: RenderedArithmeticScene,
    target_item_id: str,
) -> AnnotationArtifacts:
    """Project the single marked arithmetic target slot as a scalar bbox."""

    return bbox_annotation_artifacts(target_bbox(rendered_scene, str(target_item_id)))


__all__ = ["target_bbox_annotation"]
