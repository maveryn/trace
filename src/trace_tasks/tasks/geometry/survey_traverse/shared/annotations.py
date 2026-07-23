"""Annotation helpers for survey-traverse scenes."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import (
    PixelAnnotationArtifacts,
    keyed_bbox_annotation_artifacts,
)

from .state import RenderedAreaScene


def area_scene_annotation(rendered: RenderedAreaScene) -> PixelAnnotationArtifacts:
    """Build bbox-map annotation artifacts from the rendered area witnesses."""

    return keyed_bbox_annotation_artifacts(
        rendered.annotation_bboxes,
        roles=rendered.annotation_roles,
        include_point_centers=False,
    )


__all__ = ["area_scene_annotation"]
