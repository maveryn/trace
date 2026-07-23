"""Annotation projection helpers for radial hunt board tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, point_set_annotation_artifacts

from .state import RenderedRadialHuntBoardScene


def radial_hunt_point_set_annotation(
    rendered_scene: RenderedRadialHuntBoardScene,
    point_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected board point ids to center-point annotations."""

    centers = rendered_scene.render_map["point_centers_px"]
    return point_set_annotation_artifacts(
        [
            [
                round(float(centers[str(point_id)][0]), 3),
                round(float(centers[str(point_id)][1]), 3),
            ]
            for point_id in point_ids
        ]
    )


__all__ = ["radial_hunt_point_set_annotation"]
