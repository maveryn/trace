"""Annotation projection helpers for platformer tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .rendering import RenderedPlatformerScene


def _bbox_for_entity_id(rendered_scene: RenderedPlatformerScene, entity_id: str) -> list[float]:
    """Return a rendered entity bbox by stable entity id."""

    try:
        bbox = rendered_scene.render_map["entity_bboxes_px"][str(entity_id)]
    except KeyError as exc:
        raise ValueError(f"missing platformer rendered entity bbox for {entity_id!r}") from exc
    return [round(float(value), 3) for value in bbox[:4]]


def _point_for_entity_id(rendered_scene: RenderedPlatformerScene, entity_id: str) -> list[float]:
    """Return a rendered entity center point by stable entity id."""

    try:
        point = rendered_scene.render_map["entity_points_px"][str(entity_id)]
    except KeyError as exc:
        raise ValueError(f"missing platformer rendered entity point for {entity_id!r}") from exc
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def bbox_for_entity_id(rendered_scene: RenderedPlatformerScene, entity_id: str) -> AnnotationArtifacts:
    """Build scalar bbox annotation for one selected platformer entity."""

    return bbox_annotation_artifacts(_bbox_for_entity_id(rendered_scene, str(entity_id)))


def point_set_for_entity_ids(
    rendered_scene: RenderedPlatformerScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered point-set annotation for selected platformer entities."""

    points = [_point_for_entity_id(rendered_scene, str(entity_id)) for entity_id in entity_ids]
    return point_set_annotation_artifacts(points)


def bbox_set_for_entity_ids(
    rendered_scene: RenderedPlatformerScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered bbox-set annotation for selected platformer entities."""

    bboxes = [_bbox_for_entity_id(rendered_scene, str(entity_id)) for entity_id in entity_ids]
    return bbox_set_annotation_artifacts(bboxes)


__all__ = [
    "bbox_set_for_entity_ids",
    "bbox_for_entity_id",
    "point_set_for_entity_ids",
]
