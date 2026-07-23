"""Annotation projection helpers for tower draughts board witnesses."""

from __future__ import annotations

from collections.abc import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts

from .rules import cell_id, stack_id
from .state import Coord, RenderedTowerDraughtsScene


def stack_bbox_set_annotation(
    rendered_scene: RenderedTowerDraughtsScene,
    coords: Sequence[Coord],
) -> AnnotationArtifacts:
    """Return bbox-set annotation artifacts for selected visible stacks."""

    bboxes = rendered_scene.render_map["stack_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[stack_id(coord)] for coord in coords])


def marked_or_stack_bbox_set_annotation(
    rendered_scene: RenderedTowerDraughtsScene,
    coords: Sequence[Coord],
    *,
    marked_coord: Coord | None,
) -> AnnotationArtifacts:
    """Return stack bboxes while respecting the renderer's marked-stack id."""

    bboxes = rendered_scene.render_map["stack_bboxes_px"]
    values = []
    for coord in coords:
        key = "stack_marked" if marked_coord is not None and tuple(coord) == tuple(marked_coord) else stack_id(coord)
        values.append(bboxes[key])
    return bbox_set_annotation_artifacts(values)


def cell_bbox_set_annotation(
    rendered_scene: RenderedTowerDraughtsScene,
    coords: Sequence[Coord],
) -> AnnotationArtifacts:
    """Return bbox-set annotation artifacts for selected playable board cells."""

    bboxes = rendered_scene.render_map["cell_bboxes_px"]
    return bbox_set_annotation_artifacts([bboxes[cell_id(coord)] for coord in coords])


__all__ = [
    "cell_bbox_set_annotation",
    "marked_or_stack_bbox_set_annotation",
    "stack_bbox_set_annotation",
]
