"""Annotation projection helpers for composite-shape render outputs."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import (
    PixelAnnotationArtifacts,
    keyed_bbox_annotation_artifacts,
    keyed_point_annotation_artifacts,
)

from .state import RenderedCompositeShape


def composite_shape_annotation(rendered: RenderedCompositeShape) -> PixelAnnotationArtifacts:
    """Build the public annotation payload from rendered witness primitives."""

    if rendered.annotation_keyed_points:
        return keyed_point_annotation_artifacts(
            rendered.annotation_keyed_points,
            roles=rendered.annotation_roles,
        )
    if rendered.annotation_keyed_bboxes:
        return keyed_bbox_annotation_artifacts(
            rendered.annotation_keyed_bboxes,
            roles=rendered.annotation_roles,
        )
    raise ValueError("composite-shape render produced no keyed annotation")
