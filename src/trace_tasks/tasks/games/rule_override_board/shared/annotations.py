"""Annotation projection helpers for rule-override board scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts

from .state import RenderedRuleOverrideScene


def board_bbox_set_annotation(rendered_scene: RenderedRuleOverrideScene, board_ids: Sequence[str]) -> AnnotationArtifacts:
    """Project selected mini-board ids to bbox-set annotations."""

    bboxes = rendered_scene.render_map["entity_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[str(board_id)] for board_id in board_ids])


__all__ = ["board_bbox_set_annotation"]
