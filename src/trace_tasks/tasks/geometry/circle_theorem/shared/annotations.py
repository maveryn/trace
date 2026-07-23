"""Annotation helpers for circle-theorem rendered scenes."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import (
    PixelAnnotationArtifacts,
    keyed_point_annotation_artifacts,
)

from .state import RenderedCircleTheoremScene


def circle_theorem_keyed_point_annotation(
    rendered_scene: RenderedCircleTheoremScene,
) -> PixelAnnotationArtifacts:
    """Build keyed point annotation for one rendered circle theorem."""

    keyed_points = {
        str(label): rendered_scene.point_pixels[str(label)]
        for label in rendered_scene.annotation_point_labels
    }
    if not keyed_points:
        raise ValueError("circle theorem scene did not render annotation points")
    return keyed_point_annotation_artifacts(
        keyed_points,
        roles=rendered_scene.annotation_point_labels,
    )


__all__ = ["circle_theorem_keyed_point_annotation"]
