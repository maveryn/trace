"""Annotation projection helpers for dominoes scene tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, segment_annotation_artifacts

from .rendering import RenderedDominoTaskContext


def domino_bbox_set_annotation(
    rendered_context: RenderedDominoTaskContext,
    tile_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected domino tile ids into an unordered bbox-set annotation."""

    render_map = rendered_context.rendered_scene.render_map
    bboxes = [
        list(render_map["domino_bboxes_px"][str(tile_id)])
        for tile_id in tile_ids
    ]
    return bbox_set_annotation_artifacts(bboxes)


def domino_join_segment_annotation(
    rendered_context: RenderedDominoTaskContext,
    join_label: str,
) -> AnnotationArtifacts:
    """Project one labeled domino join into a scalar segment annotation."""

    render_map = rendered_context.rendered_scene.render_map
    join_points = render_map["chain_join_endpoint_points_px"][str(join_label)]
    return segment_annotation_artifacts(join_points)


__all__ = ["domino_bbox_set_annotation", "domino_join_segment_annotation"]
