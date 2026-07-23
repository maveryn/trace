"""Annotation projection helpers for solitaire tableau scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .state import RenderedSolitaireScene, SolitaireSample


def move_legality_bbox_map(sample: SolitaireSample, rendered: RenderedSolitaireScene) -> AnnotationArtifacts:
    """Bind source-card and target roles to their rendered bounding boxes."""

    entity_bboxes = rendered.render_map["entity_bboxes_px"]
    value = {
        "source_card": list(entity_bboxes[str(sample.metadata["legal_source_id"])]),
        "target": list(entity_bboxes[str(sample.metadata["legal_target_id"])]),
    }
    projected = {
        "type": "bbox_map",
        "bbox_map": dict(value),
        "pixel_bbox_map": dict(value),
    }
    return AnnotationArtifacts(
        annotation_type="bbox_map",
        value=dict(value),
        annotation_gt=TypedValue(type="bbox_map", value=dict(value)),
        projected_annotation=projected,
    )


def entity_bbox_set(sample: SolitaireSample, rendered: RenderedSolitaireScene) -> AnnotationArtifacts:
    """Bind all sampled annotation entity ids to an unordered bbox set."""

    entity_bboxes: Mapping[str, Any] = rendered.render_map["entity_bboxes_px"]
    bboxes = [
        list(entity_bboxes[str(entity_id)])
        for entity_id in sample.annotation_entity_ids
        if str(entity_id) in entity_bboxes
    ]
    return bbox_set_annotation_artifacts(bboxes)


def entity_bbox(sample: SolitaireSample, rendered: RenderedSolitaireScene) -> AnnotationArtifacts:
    """Bind exactly one sampled annotation entity id to a scalar bbox."""

    entity_bboxes: Mapping[str, Any] = rendered.render_map["entity_bboxes_px"]
    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    if len(entity_ids) != 1:
        raise ValueError("scalar solitaire bbox annotation requires exactly one entity id")
    return bbox_annotation_artifacts(list(entity_bboxes[str(entity_ids[0])]))


def entity_point(sample: SolitaireSample, rendered: RenderedSolitaireScene) -> AnnotationArtifacts:
    """Bind exactly one sampled card id to a point on its visible card strip."""

    entity_points: Mapping[str, Any] = rendered.render_map["entity_points_px"]
    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    if len(entity_ids) != 1:
        raise ValueError("scalar solitaire point annotation requires exactly one entity id")
    return point_annotation_artifacts(list(entity_points[str(entity_ids[0])]))


def entity_point_set(sample: SolitaireSample, rendered: RenderedSolitaireScene) -> AnnotationArtifacts:
    """Bind sampled card ids to visible-card points in an unordered point set."""

    entity_points: Mapping[str, Any] = rendered.render_map["entity_points_px"]
    points = [
        list(entity_points[str(entity_id)])
        for entity_id in sample.annotation_entity_ids
        if str(entity_id) in entity_points
    ]
    return point_set_annotation_artifacts(points)
