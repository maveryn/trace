"""Annotation projection helpers for Nine Men's Morris render maps."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_set_annotation_artifacts

from .rendering import RenderedNineMensMorrisScene


def morris_piece_point_set_annotation(
    *,
    rendered: RenderedNineMensMorrisScene,
    piece_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project piece ids to center-point annotation artifacts."""

    points = [
        list(rendered.render_map["piece_centers_px"][str(piece_id)])
        for piece_id in piece_ids
    ]
    return point_set_annotation_artifacts(points)


def morris_piece_bbox_set_annotation(
    *,
    rendered: RenderedNineMensMorrisScene,
    piece_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project piece ids to bbox-set annotation artifacts."""

    bboxes = [
        list(rendered.render_map["piece_bboxes_px"][str(piece_id)])
        for piece_id in piece_ids
    ]
    return bbox_set_annotation_artifacts(bboxes)


def morris_node_point_set_annotation(
    *,
    rendered: RenderedNineMensMorrisScene,
    node_labels: Sequence[str],
) -> AnnotationArtifacts:
    """Project board-node labels to center-point annotation artifacts."""

    points = [
        list(rendered.render_map["node_centers_px"][str(node_label)])
        for node_label in node_labels
    ]
    return point_set_annotation_artifacts(points)


__all__ = [
    "morris_node_point_set_annotation",
    "morris_piece_bbox_set_annotation",
    "morris_piece_point_set_annotation",
]
