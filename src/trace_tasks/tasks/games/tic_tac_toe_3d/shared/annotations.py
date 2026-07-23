"""Annotation projection helpers for 3D Tic-Tac-Toe cells."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_set_annotation_artifacts

from .rules import coord_id
from .state import Coord, RenderedTicTacToe3DScene


def cell_bbox_set_annotation(
    rendered_scene: RenderedTicTacToe3DScene,
    coords: Sequence[Coord],
) -> AnnotationArtifacts:
    """Return bbox-set annotation artifacts for selected board cells."""

    bboxes = rendered_scene.render_map["cell_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[coord_id(coord)] for coord in coords])


def cell_point_set_annotation(
    rendered_scene: RenderedTicTacToe3DScene,
    coords: Sequence[Coord],
) -> AnnotationArtifacts:
    """Return point-set annotation artifacts for selected board cells."""

    points = rendered_scene.render_map["cell_centers_px"]
    return point_set_annotation_artifacts([points[coord_id(coord)] for coord in coords])


__all__ = ["cell_bbox_set_annotation", "cell_point_set_annotation"]
