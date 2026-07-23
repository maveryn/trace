"""Annotation projection helpers for pipe-flow repair puzzles."""

from __future__ import annotations

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_map_annotation_artifacts,
)

from .state import PipeFlowDataset, PipeFlowMisrotatedDataset, RenderedPipeFlowScene


def pipe_flow_repair_annotation(
    *,
    dataset: PipeFlowDataset,
    rendered_scene: RenderedPipeFlowScene,
) -> AnnotationArtifacts:
    """Return role-keyed boxes for the selected option and missing gap."""

    item_bboxes = rendered_scene.item_bbox_map
    return bbox_map_annotation_artifacts(
        {
            "selected_option": item_bboxes[str(dataset.correct_option_panel_id)],
            "missing_gap": item_bboxes[str(dataset.missing_region_id)],
        }
    )


def pipe_flow_misrotated_annotation(
    *,
    dataset: PipeFlowMisrotatedDataset,
    rendered_scene: RenderedPipeFlowScene,
) -> AnnotationArtifacts:
    """Return the scalar box for the tile that should be rotated."""

    selected_tile = next(candidate for candidate in dataset.candidates if candidate.is_correct)
    return bbox_annotation_artifacts(rendered_scene.item_bbox_map[str(selected_tile.candidate_id)])
