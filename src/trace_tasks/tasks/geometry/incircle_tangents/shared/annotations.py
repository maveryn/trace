"""Annotation helpers for incircle-tangent diagrams."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts, keyed_point_annotation_artifacts

from .state import RenderedIncircleScene


def incircle_point_annotation(rendered: RenderedIncircleScene) -> PixelAnnotationArtifacts:
    """Build keyed point annotation for task-selected construction points."""

    return keyed_point_annotation_artifacts(
        rendered.annotation_points,
        roles=rendered.annotation_roles,
    )


__all__ = ["incircle_point_annotation"]
