"""Annotation projection helpers for pinball-table tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .rendering import RenderedPinballScene


def _point_for_entity_id(rendered_scene: RenderedPinballScene, entity_id: str) -> list[float]:
    """Return a rendered entity center point by stable entity id."""

    try:
        point = rendered_scene.render_map["entity_points_px"][str(entity_id)]
    except KeyError as exc:
        raise ValueError(f"missing pinball rendered entity point for {entity_id!r}") from exc
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def point_for_entity_id(rendered_scene: RenderedPinballScene, entity_id: str) -> AnnotationArtifacts:
    """Build scalar point annotation for one selected pinball object."""

    return point_annotation_artifacts(_point_for_entity_id(rendered_scene, str(entity_id)))


def point_sequence_for_entity_ids(
    rendered_scene: RenderedPinballScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build ordered point-sequence annotation for path hit objects."""

    value = [_point_for_entity_id(rendered_scene, str(entity_id)) for entity_id in entity_ids]
    projected_annotation = {
        "type": "point_sequence",
        "point_sequence": [list(point) for point in value],
        "pixel_point_sequence": [list(point) for point in value],
    }
    return AnnotationArtifacts(
        annotation_type="point_sequence",
        value=[list(point) for point in value],
        annotation_gt=TypedValue(type="point_sequence", value=[list(point) for point in value]),
        projected_annotation=projected_annotation,
    )


def point_set_for_entity_ids(
    rendered_scene: RenderedPinballScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered point-set annotation for selected pinball objects."""

    return point_set_annotation_artifacts(
        [_point_for_entity_id(rendered_scene, str(entity_id)) for entity_id in entity_ids]
    )


def bbox_set_for_entity_ids(
    rendered_scene: RenderedPinballScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered bbox-set annotation for selected pinball objects."""

    entity_bboxes = rendered_scene.render_map["entity_bboxes_px"]
    bboxes = [list(entity_bboxes[str(entity_id)]) for entity_id in entity_ids]
    return bbox_set_annotation_artifacts(bboxes)


__all__ = [
    "bbox_set_for_entity_ids",
    "point_for_entity_id",
    "point_set_for_entity_ids",
    "point_sequence_for_entity_ids",
]
