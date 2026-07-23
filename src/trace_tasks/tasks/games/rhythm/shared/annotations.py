"""Annotation projection helpers for Rhythm scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_annotation_artifacts, bbox_set_annotation_artifacts

from .rendering import RenderedRhythmScene


def rhythm_note_bbox_set_annotation(rendered_scene: RenderedRhythmScene, note_ids: Sequence[str]) -> AnnotationArtifacts:
    """Project selected note ids to bbox-set annotations."""

    bboxes = rendered_scene.render_map["entity_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[str(note_id)] for note_id in note_ids])


def rhythm_note_bbox_annotation(rendered_scene: RenderedRhythmScene, note_ids: Sequence[str]) -> AnnotationArtifacts:
    """Project one selected note id to a scalar bbox annotation."""

    resolved = tuple(str(note_id) for note_id in note_ids)
    if len(resolved) != 1:
        raise ValueError("scalar rhythm bbox annotation requires exactly one note")
    bboxes = rendered_scene.render_map["entity_bboxes_px"]
    return bbox_annotation_artifacts(bboxes[str(resolved[0])])


__all__ = ["rhythm_note_bbox_annotation", "rhythm_note_bbox_set_annotation"]
