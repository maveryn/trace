"""Annotation projection helpers for Minesweeper board cells."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .state import Coord, coord_to_cell_id
from .rendering import RenderedMinesweeperScene


def _point_for_cell(rendered: RenderedMinesweeperScene, coord: Coord) -> list[float]:
    """Project one board coordinate to the center of its inset cell bbox."""

    entity_id = coord_to_cell_id(coord)
    left, top, right, bottom = [float(value) for value in rendered.render_map["cell_bboxes_px"][str(entity_id)]]
    return [0.5 * (left + right), 0.5 * (top + bottom)]


def minesweeper_point_set_annotation(
    *,
    rendered: RenderedMinesweeperScene,
    coords: Sequence[Coord],
) -> AnnotationArtifacts:
    """Project homogeneous cell witnesses to a public point-set annotation."""

    return point_set_annotation_artifacts([_point_for_cell(rendered, coord) for coord in coords])


def minesweeper_bbox_set_annotation(
    *,
    rendered: RenderedMinesweeperScene,
    coords: Sequence[Coord],
) -> AnnotationArtifacts:
    """Project homogeneous cell witnesses to a public bbox-set annotation."""

    bboxes = [
        list(rendered.render_map["cell_bboxes_px"][coord_to_cell_id(coord)])
        for coord in coords
    ]
    return bbox_set_annotation_artifacts(bboxes)


def minesweeper_point_annotation(
    *,
    rendered: RenderedMinesweeperScene,
    coord: Coord,
) -> AnnotationArtifacts:
    """Project one guaranteed cell witness to a scalar point annotation."""

    return point_annotation_artifacts(_point_for_cell(rendered, coord))


def minesweeper_point_set_map_annotation(
    *,
    rendered: RenderedMinesweeperScene,
    coords_by_role: Mapping[str, Sequence[Coord]],
) -> AnnotationArtifacts:
    """Project role-bound cell witness groups to point-set-map annotation."""

    value: dict[str, list[list[float]]] = {}
    for role, coords in sorted(coords_by_role.items()):
        value[str(role)] = [
            [round(float(v), 3) for v in _point_for_cell(rendered, coord)]
            for coord in coords
        ]
    return AnnotationArtifacts(
        annotation_type="point_set_map",
        value={str(role): [list(point) for point in points] for role, points in value.items()},
        annotation_gt=TypedValue(
            type="point_set_map",
            value={str(role): [list(point) for point in points] for role, points in value.items()},
        ),
        projected_annotation={
            "type": "point_set_map",
            "point_set_map": {str(role): [list(point) for point in points] for role, points in value.items()},
            "pixel_point_set_map": {str(role): [list(point) for point in points] for role, points in value.items()},
        },
    )


def cell_ids_for_coords(coords: Sequence[Coord]) -> tuple[str, ...]:
    """Return visible entity ids for board-coordinate witnesses."""

    return tuple(coord_to_cell_id(coord) for coord in coords)


def keyed_cell_ids_for_coords(coords_by_role: Mapping[str, Sequence[Coord]]) -> dict[str, list[str]]:
    """Return visible entity ids grouped by semantic annotation role."""

    return {str(role): [coord_to_cell_id(coord) for coord in coords] for role, coords in sorted(coords_by_role.items())}


__all__ = [
    "cell_ids_for_coords",
    "keyed_cell_ids_for_coords",
    "minesweeper_bbox_set_annotation",
    "minesweeper_point_set_map_annotation",
    "minesweeper_point_annotation",
    "minesweeper_point_set_annotation",
]
