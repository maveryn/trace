"""Annotation projection helpers for darts scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_annotation_artifacts, point_set_annotation_artifacts

from .rendering import RenderedDartsTaskContext


def dart_center_point_set_annotation(
    rendered_context: RenderedDartsTaskContext,
    dart_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected dart ids to center-point annotation artifacts."""

    centers = rendered_context.rendered_scene.render_map["dart_centers_px"]
    return point_set_annotation_artifacts(
        [
            [
                round(float(centers[str(dart_id)][0]), 3),
                round(float(centers[str(dart_id)][1]), 3),
            ]
            for dart_id in dart_ids
        ]
    )


def dart_bbox_set_annotation(
    rendered_context: RenderedDartsTaskContext,
    dart_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected dart ids to bbox-set annotation artifacts."""

    bboxes = rendered_context.rendered_scene.render_map["dart_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[str(dart_id)] for dart_id in dart_ids])


def dart_center_point_annotation(
    rendered_context: RenderedDartsTaskContext,
    dart_id: str,
) -> AnnotationArtifacts:
    """Project one selected dart id to a center-point annotation artifact."""

    centers = rendered_context.rendered_scene.render_map["dart_centers_px"]
    center = centers[str(dart_id)]
    return point_annotation_artifacts(
        [
            round(float(center[0]), 3),
            round(float(center[1]), 3),
        ]
    )


__all__ = ["dart_bbox_set_annotation", "dart_center_point_annotation", "dart_center_point_set_annotation"]
