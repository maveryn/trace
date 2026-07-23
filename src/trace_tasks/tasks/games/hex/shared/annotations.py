"""Annotation projection helpers for Hex scene tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, point_annotation_artifacts, point_set_annotation_artifacts

from .rendering import RenderedHexScene
from .rules import Coord, coord_to_cell_id


def hex_cell_point_set_annotation(
    rendered_scene: RenderedHexScene,
    coords: Sequence[Coord],
) -> tuple[tuple[str, ...], AnnotationArtifacts]:
    """Project Hex board coordinates to point-set annotation artifacts."""

    entity_ids = tuple(coord_to_cell_id(coord) for coord in coords)
    points = [
        list(rendered_scene.render_map["cell_centers_px"][str(entity_id)])
        for entity_id in entity_ids
    ]
    return entity_ids, point_set_annotation_artifacts(points)


def hex_cell_point_annotation(
    rendered_scene: RenderedHexScene,
    coord: Coord,
) -> tuple[str, AnnotationArtifacts]:
    """Project one Hex board coordinate to a scalar point annotation."""

    entity_id = coord_to_cell_id(coord)
    point = list(rendered_scene.render_map["cell_centers_px"][str(entity_id)])
    return entity_id, point_annotation_artifacts(point)


__all__ = ["hex_cell_point_annotation", "hex_cell_point_set_annotation"]
