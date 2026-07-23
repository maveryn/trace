"""Annotation projection helpers for the Backgammon games scene."""

from __future__ import annotations

from .rendering import RenderedBackgammonScene
from .state import point_entity_id


def annotation_entity_ids_for_points(points: tuple[int, ...]) -> tuple[str, ...]:
    """Return point entity ids for target Backgammon point numbers."""

    return tuple(point_entity_id(int(point)) for point in points)


def annotation_bboxes_for_entity_ids(
    rendered_scene: RenderedBackgammonScene,
    entity_ids: tuple[str, ...],
) -> list[list[float]]:
    """Project point entity ids to rendered bboxes."""

    entity_bboxes = rendered_scene.render_map["entity_bboxes_px"]
    return [list(entity_bboxes[str(entity_id)]) for entity_id in entity_ids]


__all__ = ["annotation_bboxes_for_entity_ids", "annotation_entity_ids_for_points"]
