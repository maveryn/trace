"""Annotation projection helpers for space-shooter scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
)

from .rendering import RenderedSpaceShooterScene
from .state import SpaceShooterSample


def entity_bbox_set(sample: SpaceShooterSample, rendered: RenderedSpaceShooterScene) -> AnnotationArtifacts:
    """Bind sampled annotation entity ids to an unordered bbox set."""

    entity_bboxes: Mapping[str, Any] = rendered.render_map["entity_bboxes_px"]
    bboxes = [list(entity_bboxes[str(entity_id)]) for entity_id in sample.annotation_entity_ids]
    return bbox_set_annotation_artifacts(bboxes)


def single_entity_bbox(sample: SpaceShooterSample, rendered: RenderedSpaceShooterScene) -> AnnotationArtifacts:
    """Bind the one selected annotation entity id to a scalar bbox."""

    if len(sample.annotation_entity_ids) != 1:
        raise ValueError("single bbox annotation requires exactly one annotation entity")
    entity_bboxes: Mapping[str, Any] = rendered.render_map["entity_bboxes_px"]
    return bbox_annotation_artifacts(entity_bboxes[str(sample.annotation_entity_ids[0])])
