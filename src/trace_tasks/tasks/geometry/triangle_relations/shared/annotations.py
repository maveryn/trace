"""Annotation helpers for triangle-relations rendered scenes."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts, keyed_point_annotation_artifacts
from trace_tasks.tasks.shared.annotation_artifacts import point_annotation_artifacts, segment_annotation_artifacts

from .state import RenderedTriangleRelationsScene, TriangleRelationsCase


def triangle_relations_annotation_mode(case: TriangleRelationsCase) -> str:
    """Return the public annotation mode implied by one resolved case."""

    if case.point_annotation_labels:
        return "point_map"
    if case.target_point is not None:
        return "point"
    if case.target_segment is not None:
        return "segment"
    raise ValueError("triangle-relations case must define an annotation witness")


def triangle_relations_annotation(rendered: RenderedTriangleRelationsScene) -> PixelAnnotationArtifacts:
    """Build the public annotation artifact matching the rendered witness type."""

    if rendered.annotation_mode == "segment":
        if rendered.annotation_segment is None:
            raise ValueError("segment annotation requested without a segment")
        artifacts = segment_annotation_artifacts(rendered.annotation_segment)
        return PixelAnnotationArtifacts(
            annotation_type=artifacts.annotation_type,
            value=artifacts.value,
            projected_annotation=artifacts.projected_annotation,
        )
    if rendered.annotation_mode == "point":
        if rendered.annotation_point is None:
            raise ValueError("point annotation requested without a point")
        artifacts = point_annotation_artifacts(rendered.annotation_point)
        return PixelAnnotationArtifacts(
            annotation_type=artifacts.annotation_type,
            value=artifacts.value,
            projected_annotation=artifacts.projected_annotation,
        )
    if rendered.annotation_mode == "point_map":
        return keyed_point_annotation_artifacts(rendered.annotation_points, roles=rendered.annotation_roles)
    raise ValueError(f"unsupported triangle-relations annotation mode: {rendered.annotation_mode}")


__all__ = ["triangle_relations_annotation", "triangle_relations_annotation_mode"]
