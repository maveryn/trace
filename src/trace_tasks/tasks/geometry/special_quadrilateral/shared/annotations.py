"""Annotation helpers for special-quadrilateral diagrams."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts, keyed_point_annotation_artifacts

from .state import RenderedSpecialQuadrilateralScene


def special_quadrilateral_point_annotation(
    rendered: RenderedSpecialQuadrilateralScene,
) -> PixelAnnotationArtifacts:
    """Use the visible point labels as the public point-map annotation keys."""

    return keyed_point_annotation_artifacts(
        rendered.annotation_points,
        roles=tuple(rendered.annotation_points.keys()),
    )


__all__ = ["special_quadrilateral_point_annotation"]
