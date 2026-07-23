"""Annotation projection helpers for angle-relations tasks."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts, keyed_point_annotation_artifacts

from .state import RenderedAngleRelationScene


def angle_vertex_annotation_artifacts(rendered_scene: RenderedAngleRelationScene) -> PixelAnnotationArtifacts:
    """Build keyed angle-vertex annotation from a rendered angle-relations scene."""

    keyed_points = rendered_scene.annotation_keyed_points
    if not keyed_points:
        raise ValueError("angle-relations scene did not render keyed angle points")
    return keyed_point_annotation_artifacts(keyed_points, roles=rendered_scene.annotation_roles)


__all__ = ["angle_vertex_annotation_artifacts"]
