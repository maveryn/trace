"""Annotation projection helpers for the Checkers games scene."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
)

from .rendering import RenderedCheckersScene


def _bbox_center(bbox: Sequence[float]) -> list[float]:
    """Return the center point of one rendered bbox."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def cell_bboxes_for_ids(rendered_scene: RenderedCheckersScene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Return rendered cell bboxes for stable cell ids."""

    cell_map = rendered_scene.render_map["cell_bboxes_px"]
    return [list(cell_map[str(entity_id)]) for entity_id in entity_ids]


def piece_bboxes_for_ids(rendered_scene: RenderedCheckersScene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Return rendered piece bboxes for stable piece ids."""

    piece_map = rendered_scene.render_map["piece_bboxes_px"]
    return [list(piece_map[str(entity_id)]) for entity_id in entity_ids]


def piece_points_for_ids(rendered_scene: RenderedCheckersScene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Return rendered piece-center points for stable piece ids."""

    return [_bbox_center(bbox) for bbox in piece_bboxes_for_ids(rendered_scene, entity_ids)]


def cell_points_for_ids(rendered_scene: RenderedCheckersScene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Return rendered cell-center points for stable cell ids."""

    return [_bbox_center(bbox) for bbox in cell_bboxes_for_ids(rendered_scene, entity_ids)]


def checkers_annotation_artifacts(
    *,
    rendered_scene: RenderedCheckersScene,
    entity_ids: Sequence[str],
    annotation_kind: str,
) -> AnnotationArtifacts:
    """Build public annotation artifacts from rendered Checkers witnesses."""

    if str(annotation_kind) == "piece_point":
        return bbox_set_annotation_artifacts(piece_bboxes_for_ids(rendered_scene, entity_ids))
    if str(annotation_kind) == "piece":
        return bbox_set_annotation_artifacts(piece_bboxes_for_ids(rendered_scene, entity_ids))
    return bbox_set_annotation_artifacts(cell_bboxes_for_ids(rendered_scene, entity_ids))


__all__ = [
    "cell_bboxes_for_ids",
    "cell_points_for_ids",
    "checkers_annotation_artifacts",
    "piece_bboxes_for_ids",
    "piece_points_for_ids",
]
