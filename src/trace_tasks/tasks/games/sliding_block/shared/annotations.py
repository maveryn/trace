"""Annotation projection helpers for sliding-block task witnesses."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_map_annotation_artifacts,
    bbox_set_annotation_artifacts,
)

from .state import RenderedSlidingBlockScene


def block_bbox_set(
    rendered_scene: RenderedSlidingBlockScene,
    *,
    block_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build a bbox-set annotation from source-board block ids."""

    bboxes = [
        [round(float(value), 3) for value in rendered_scene.block_bbox_map[str(block_id)]]
        for block_id in block_ids
    ]
    return bbox_set_annotation_artifacts(bboxes)


def source_board_and_option_bbox_map(
    rendered_scene: RenderedSlidingBlockScene,
    *,
    option_id: str,
) -> AnnotationArtifacts:
    """Build role-keyed bbox annotation for source board and selected option."""

    return bbox_map_annotation_artifacts(
        {
            "source_board": rendered_scene.board_bbox_px,
            "selected_option": rendered_scene.option_panel_bbox_map[str(option_id)],
        }
    )


__all__ = ["block_bbox_set", "source_board_and_option_bbox_map"]
