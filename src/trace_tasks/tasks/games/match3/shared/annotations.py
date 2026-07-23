"""Annotation projection helpers for match-3 game tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .state import RenderedMatch3Scene


def match3_point_set_annotation(
    *,
    rendered: RenderedMatch3Scene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected gem/arrow entity centers into a public point-set annotation."""

    points = [
        list(rendered.render_map["entity_points_px"][str(entity_id)])
        for entity_id in entity_ids
        if str(entity_id) in rendered.render_map["entity_points_px"]
    ]
    return point_set_annotation_artifacts(points)


def match3_bbox_set_annotation(
    *,
    rendered: RenderedMatch3Scene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected gem entity ids into a public bbox-set annotation."""

    bboxes = [
        list(rendered.render_map["entity_bboxes_px"][str(entity_id)])
        for entity_id in entity_ids
        if str(entity_id) in rendered.render_map["entity_bboxes_px"]
    ]
    return bbox_set_annotation_artifacts(bboxes)


def match3_point_annotation(
    *,
    rendered: RenderedMatch3Scene,
    entity_id: str,
) -> AnnotationArtifacts:
    """Project one gem/arrow entity center into a public scalar point annotation."""

    return point_annotation_artifacts(rendered.render_map["entity_points_px"][str(entity_id)])


__all__ = ["match3_bbox_set_annotation", "match3_point_annotation", "match3_point_set_annotation"]
