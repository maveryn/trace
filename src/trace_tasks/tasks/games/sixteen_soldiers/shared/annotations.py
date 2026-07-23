"""Annotation projection helpers for Sixteen Soldiers tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, point_set_annotation_artifacts

from .rules import piece_to_entity_id
from .state import RenderedSixteenSoldiersScene


def sixteen_soldiers_destination_point_annotation(
    rendered_scene: RenderedSixteenSoldiersScene,
    point_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project empty destination point ids to point-set annotations."""

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


def sixteen_soldiers_capture_piece_annotation(
    rendered_scene: RenderedSixteenSoldiersScene,
    point_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project capturable opponent piece point ids to piece-center annotations."""

    centers = rendered_scene.render_map["piece_centers_px"]
    return point_set_annotation_artifacts(
        [
            [
                round(float(centers[piece_to_entity_id(str(point_id))][0]), 3),
                round(float(centers[piece_to_entity_id(str(point_id))][1]), 3),
            ]
            for point_id in point_ids
        ]
    )


__all__ = [
    "sixteen_soldiers_capture_piece_annotation",
    "sixteen_soldiers_destination_point_annotation",
]
