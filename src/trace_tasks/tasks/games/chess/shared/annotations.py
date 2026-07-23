"""Annotation projection helpers for Chess games scenes."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.games.shared.piece_board_rules import Coord, coord_to_cell_id
from trace_tasks.tasks.games.shared.piece_board_renderer import RenderedChessScene


def _bbox_center(bbox: list[float] | tuple[float, ...]) -> list[float]:
    """Return the center point of one rendered bbox."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def point_set_for_entities(rendered_scene: RenderedChessScene, *, entity_ids: Tuple[str, ...], annotation_kind: str) -> list[list[float]]:
    """Project cell or piece entity ids into center-point annotation."""

    key = "cell_bboxes_px" if str(annotation_kind) == "cell" else "piece_bboxes_px"
    mapping = rendered_scene.render_map[str(key)]
    return [_bbox_center(mapping[str(entity_id)]) for entity_id in entity_ids]


def bbox_set_for_entities(rendered_scene: RenderedChessScene, *, entity_ids: Tuple[str, ...], annotation_kind: str) -> list[list[float]]:
    """Project cell or piece entity ids into bbox-set annotation."""

    key = "cell_bboxes_px" if str(annotation_kind) == "cell" else "piece_bboxes_px"
    mapping = rendered_scene.render_map[str(key)]
    return [list(mapping[str(entity_id)]) for entity_id in entity_ids]


def move_point_map(
    rendered_scene: RenderedChessScene,
    *,
    source: Coord,
    destination: Coord,
    king: Coord,
) -> dict[str, list[float]]:
    """Project checkmate source, destination, and king cells to point-map entries."""

    cells = rendered_scene.render_map["cell_bboxes_px"]
    return {
        "from": _bbox_center(cells[coord_to_cell_id(source)]),
        "to": _bbox_center(cells[coord_to_cell_id(destination)]),
        "king": _bbox_center(cells[coord_to_cell_id(king)]),
    }


def projected_point_payload(annotation_points: list[list[float]]) -> dict[str, Any]:
    """Return projected point-set trace payload."""

    return {
        "type": "point_set",
        "point_set": [list(point) for point in annotation_points],
        "pixel_point_set": [list(point) for point in annotation_points],
    }


def projected_bbox_payload(annotation_bboxes: list[list[float]]) -> dict[str, Any]:
    """Return projected bbox-set trace payload."""

    return {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in annotation_bboxes],
        "pixel_bbox_set": [list(bbox) for bbox in annotation_bboxes],
    }


def projected_point_map_payload(annotation_map: Mapping[str, list[float]]) -> dict[str, Any]:
    """Return projected point-map trace payload."""

    value = {str(key): list(point) for key, point in annotation_map.items()}
    return {"type": "point_map", "point_map": value, "pixel_point_map": value}

__all__ = [
    "bbox_set_for_entities",
    "move_point_map",
    "point_set_for_entities",
    "projected_bbox_payload",
    "projected_point_map_payload",
    "projected_point_payload",
]
