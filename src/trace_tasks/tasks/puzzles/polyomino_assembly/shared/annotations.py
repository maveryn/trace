"""Annotation projection helpers for polyomino assembly puzzles."""

from __future__ import annotations

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
)

from .state import RenderedPolyominoAssemblyScene


def selected_option_bbox_annotation(
    rendered_scene: RenderedPolyominoAssemblyScene,
    selected_option_choice_id: str,
) -> AnnotationArtifacts:
    """Return scalar bbox annotation for the selected option card."""

    option_id = str(selected_option_choice_id)
    if option_id not in rendered_scene.option_choice_bbox_map:
        raise ValueError(f"missing option bbox for {option_id}")
    return bbox_annotation_artifacts(rendered_scene.option_choice_bbox_map[option_id])


__all__ = ["selected_option_bbox_annotation"]
