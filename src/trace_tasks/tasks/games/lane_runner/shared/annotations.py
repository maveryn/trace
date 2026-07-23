"""Annotation projection helpers for lane-runner rendered scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .rendering import RenderedLaneRunnerScene


def lane_runner_coin_bbox_annotation(
    rendered_scene: RenderedLaneRunnerScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected coin ids to bbox-set annotations."""

    bboxes = [
        list(rendered_scene.render_map["coin_bboxes_px"][str(entity_id)])
        for entity_id in entity_ids
    ]
    return bbox_set_annotation_artifacts(bboxes)


def lane_runner_coin_point_annotation(
    rendered_scene: RenderedLaneRunnerScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project selected coin ids to point-set annotations."""

    points = [
        list(rendered_scene.render_map["entity_points_px"][str(entity_id)])
        for entity_id in entity_ids
    ]
    return point_set_annotation_artifacts(points)


def lane_runner_path_card_bbox_annotation(
    rendered_scene: RenderedLaneRunnerScene,
    label: str,
) -> AnnotationArtifacts:
    """Project one selected path card label to its bbox annotation."""

    card = dict(rendered_scene.render_map["path_options_px"][str(label)])
    return bbox_annotation_artifacts(list(card["card_bbox_px"]))


__all__ = [
    "lane_runner_coin_bbox_annotation",
    "lane_runner_coin_point_annotation",
    "lane_runner_path_card_bbox_annotation",
]
