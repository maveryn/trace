"""Annotation projection helpers for pool-table tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_set_annotation_artifacts

from .rendering import RenderedPoolScene


def _point_for_entity_id(rendered_scene: RenderedPoolScene, entity_id: str) -> list[float]:
    """Return a rendered pool entity center point by stable entity id."""

    point_maps = (
        rendered_scene.render_map.get("ball_points_px", {}),
        rendered_scene.render_map.get("pocket_points_px", {}),
    )
    for point_map in point_maps:
        if str(entity_id) in point_map:
            point = point_map[str(entity_id)]
            return [round(float(point[0]), 3), round(float(point[1]), 3)]
    raise ValueError(f"missing pool rendered entity point for {entity_id!r}")


def point_set_for_entity_ids(
    rendered_scene: RenderedPoolScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered point-set annotation for selected pool entities."""

    points = [_point_for_entity_id(rendered_scene, str(entity_id)) for entity_id in entity_ids]
    return point_set_annotation_artifacts(points)


def bbox_set_for_ball_ids(
    rendered_scene: RenderedPoolScene,
    ball_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Build unordered bbox-set annotation for selected pool balls."""

    ball_bboxes = rendered_scene.render_map.get("ball_bboxes_px", {})
    bboxes = []
    for ball_id in ball_ids:
        if str(ball_id) not in ball_bboxes:
            raise ValueError(f"missing pool rendered ball bbox for {ball_id!r}")
        bboxes.append([round(float(value), 3) for value in ball_bboxes[str(ball_id)][:4]])
    return bbox_set_annotation_artifacts(bboxes)


__all__ = ["bbox_set_for_ball_ids", "point_set_for_entity_ids"]
