"""Annotation projection helpers for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

from collections.abc import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_annotation_artifacts, bbox_set_annotation_artifacts

from .state import RenderedUltimateScene


def ultimate_bbox_set_annotation(
    rendered_scene: RenderedUltimateScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project task-owned entity ids into an unordered bbox-set annotation."""

    bboxes = [
        list(rendered_scene.render_map["entity_bboxes_px"][str(entity_id)])
        for entity_id in entity_ids
        if str(entity_id) in rendered_scene.render_map["entity_bboxes_px"]
    ]
    if len(bboxes) != len(tuple(entity_ids)):
        raise RuntimeError("missing rendered annotation bbox")
    return bbox_set_annotation_artifacts(bboxes)


def ultimate_bbox_annotation(
    rendered_scene: RenderedUltimateScene,
    entity_id: str,
) -> AnnotationArtifacts:
    """Project one task-owned entity id into a scalar bbox annotation."""

    bboxes = rendered_scene.render_map["entity_bboxes_px"]
    if str(entity_id) not in bboxes:
        raise RuntimeError("missing rendered annotation bbox")
    return bbox_annotation_artifacts(bboxes[str(entity_id)])
