"""Annotation projection helpers for Mini-golf rendered scenes."""

from __future__ import annotations

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, point_annotation_artifacts, segment_annotation_artifacts

from .rendering import RenderedMinigolfScene


def minigolf_obstacle_point_annotation(
    *,
    rendered: RenderedMinigolfScene,
    obstacle_id: str,
) -> AnnotationArtifacts:
    """Project one obstacle center as a scalar point annotation."""

    point = rendered.render_map["entity_points_px"][str(obstacle_id)]
    return point_annotation_artifacts(point)


def minigolf_path_segment_annotation(
    *,
    rendered: RenderedMinigolfScene,
    path_id: str,
) -> AnnotationArtifacts:
    """Project one visible cue segment as a scalar segment annotation."""

    pair = rendered.render_map["path_point_pairs_px"][str(path_id)]
    return segment_annotation_artifacts(pair)


__all__ = [
    "minigolf_obstacle_point_annotation",
    "minigolf_path_segment_annotation",
]
