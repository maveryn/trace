"""Annotation projection helpers for Pac-Man scenes."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)


def entity_points_for_ids(rendered_scene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Return rendered entity center points for the requested ids."""

    entity_points = rendered_scene.render_map["entity_points_px"]
    return [list(entity_points[str(entity_id)]) for entity_id in entity_ids]


def entity_bboxes_for_ids(rendered_scene, entity_ids: Sequence[str]) -> list[list[float]]:
    """Return rendered entity bboxes for the requested ids."""

    entity_bboxes = rendered_scene.render_map["entity_bboxes_px"]
    return [list(entity_bboxes[str(entity_id)]) for entity_id in entity_ids]


def point_set_for_entity_ids(rendered_scene, entity_ids: Sequence[str]) -> AnnotationArtifacts:
    """Project entity ids to an unordered point-set annotation."""

    return point_set_annotation_artifacts(entity_points_for_ids(rendered_scene, entity_ids))


def bbox_set_for_entity_ids(rendered_scene, entity_ids: Sequence[str]) -> AnnotationArtifacts:
    """Project entity ids to an unordered bbox-set annotation."""

    return bbox_set_annotation_artifacts(entity_bboxes_for_ids(rendered_scene, entity_ids))


def point_for_entity_id(rendered_scene, entity_id: str) -> AnnotationArtifacts:
    """Project one entity id to a scalar point annotation."""

    point = rendered_scene.render_map["entity_points_px"][str(entity_id)]
    return point_annotation_artifacts(point)


def point_set_map_for_entity_ids(
    rendered_scene,
    entity_ids_by_key: Mapping[str, Sequence[str]],
) -> AnnotationArtifacts:
    """Project map annotation roles to point-set maps."""

    point_sets_by_key = {
        str(key): entity_points_for_ids(rendered_scene, tuple(str(entity_id) for entity_id in entity_ids))
        for key, entity_ids in entity_ids_by_key.items()
    }
    projected = {
        "type": "point_set_map",
        "point_set_map": {key: [list(point) for point in points] for key, points in point_sets_by_key.items()},
        "pixel_point_set_map": {key: [list(point) for point in points] for key, points in point_sets_by_key.items()},
    }
    value = {key: [list(point) for point in points] for key, points in point_sets_by_key.items()}
    return AnnotationArtifacts(
        annotation_type="point_set_map",
        value=value,
        annotation_gt=TypedValue(type="point_set_map", value=value),
        projected_annotation=projected,
    )


def bbox_set_map_for_entity_ids(
    rendered_scene,
    entity_ids_by_key: Mapping[str, Sequence[str]],
) -> AnnotationArtifacts:
    """Project map annotation roles to bbox-set maps."""

    bbox_sets_by_key = {
        str(key): entity_bboxes_for_ids(rendered_scene, tuple(str(entity_id) for entity_id in entity_ids))
        for key, entity_ids in entity_ids_by_key.items()
    }
    value = {key: [list(bbox) for bbox in bboxes] for key, bboxes in bbox_sets_by_key.items()}
    projected = {
        "type": "bbox_set_map",
        "bbox_set_map": {key: [list(bbox) for bbox in bboxes] for key, bboxes in value.items()},
        "pixel_bbox_set_map": {key: [list(bbox) for bbox in bboxes] for key, bboxes in value.items()},
    }
    return AnnotationArtifacts(
        annotation_type="bbox_set_map",
        value=value,
        annotation_gt=TypedValue(type="bbox_set_map", value=value),
        projected_annotation=projected,
    )


__all__ = [
    "bbox_set_for_entity_ids",
    "entity_bboxes_for_ids",
    "entity_points_for_ids",
    "bbox_set_map_for_entity_ids",
    "point_set_map_for_entity_ids",
    "point_for_entity_id",
    "point_set_for_entity_ids",
]
