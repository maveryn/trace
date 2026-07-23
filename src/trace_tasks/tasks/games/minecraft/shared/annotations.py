"""Annotation projection helpers for Minecraft-like block-world scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts, point_set_annotation_artifacts

from .state import RenderedMinecraftScene


def minecraft_point_set_annotation(
    *,
    rendered: RenderedMinecraftScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project entity ids to the public unordered point-set annotation."""

    points = [
        list(rendered.render_map["entity_points_px"][str(entity_id)])
        for entity_id in tuple(str(value) for value in entity_ids)
    ]
    return point_set_annotation_artifacts(points)


def minecraft_bbox_set_annotation(
    *,
    rendered: RenderedMinecraftScene,
    entity_ids: Sequence[str],
) -> AnnotationArtifacts:
    """Project entity ids to the public unordered bbox-set annotation."""

    bboxes = [
        list(rendered.render_map["entity_bboxes_px"][str(entity_id)])
        for entity_id in tuple(str(value) for value in entity_ids)
    ]
    return bbox_set_annotation_artifacts(bboxes)


__all__ = ["minecraft_bbox_set_annotation", "minecraft_point_set_annotation"]
