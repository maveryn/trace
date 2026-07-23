"""Annotation projection helpers for Reversi scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_set_annotation_artifacts

from .rendering import RenderedReversiScene


def reversi_cell_bbox_set_annotation(
    rendered_scene: RenderedReversiScene,
    cell_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected board cell ids to bbox-set annotations."""

    cell_bboxes = rendered_scene.render_map["cell_bboxes_px"]
    return bbox_set_annotation_artifacts([cell_bboxes[str(cell_id)] for cell_id in cell_ids])


def reversi_disc_point_set_annotation(
    rendered_scene: RenderedReversiScene,
    cell_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected disc-bearing cell ids to disc-center point annotations."""

    disc_points = rendered_scene.render_map["disc_points_px"]
    return point_set_annotation_artifacts([disc_points[str(cell_id)] for cell_id in cell_ids])


__all__ = ["reversi_cell_bbox_set_annotation", "reversi_disc_point_set_annotation"]
