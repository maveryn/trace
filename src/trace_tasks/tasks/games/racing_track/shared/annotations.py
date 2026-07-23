"""Annotation projection helpers for racing-track tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .rendering import RenderedRacingTrackScene


def point_for_entity_id(rendered_scene: RenderedRacingTrackScene, entity_id: str) -> list[float]:
    """Return one rendered racing-track entity center point by id."""

    point_map = rendered_scene.render_map.get("entity_points_px", {})
    if str(entity_id) not in point_map:
        raise ValueError(f"missing racing-track rendered entity point for {entity_id!r}")
    point = point_map[str(entity_id)]
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def point_annotation_for_entity_id(
    rendered_scene: RenderedRacingTrackScene,
    entity_id: str,
) -> AnnotationArtifacts:
    """Build scalar point annotation for one selected racing-track entity."""

    return point_annotation_artifacts(point_for_entity_id(rendered_scene, str(entity_id)))


def point_set_for_entity_ids(
    rendered_scene: RenderedRacingTrackScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered point-set annotation for selected racing-track entities."""

    points = [point_for_entity_id(rendered_scene, str(entity_id)) for entity_id in entity_ids]
    return point_set_annotation_artifacts(points)


def bbox_set_for_entity_ids(
    rendered_scene: RenderedRacingTrackScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered bbox-set annotation for selected racing-track entities."""

    bbox_map = rendered_scene.render_map.get("entity_bboxes_px", {})
    bboxes = []
    for entity_id in entity_ids:
        if str(entity_id) not in bbox_map:
            raise ValueError(f"missing racing-track rendered entity bbox for {entity_id!r}")
        bboxes.append([round(float(value), 3) for value in bbox_map[str(entity_id)][:4]])
    return bbox_set_annotation_artifacts(bboxes)


__all__ = [
    "bbox_set_for_entity_ids",
    "point_annotation_for_entity_id",
    "point_for_entity_id",
    "point_set_for_entity_ids",
]
