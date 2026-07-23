"""Annotation projection helpers for Go scene tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_set_annotation_artifacts

from .rendering import RenderedGoScene


def go_point_set_annotation(
    rendered_scene: RenderedGoScene,
    point_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected Go intersection ids to center-point annotations."""

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


def go_stone_bbox_set_annotation(
    rendered_scene: RenderedGoScene,
    stone_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected Go stone ids to stone-bbox annotations."""

    bboxes = rendered_scene.render_map["stone_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[str(stone_id)] for stone_id in stone_ids])


__all__ = ["go_point_set_annotation", "go_stone_bbox_set_annotation"]
