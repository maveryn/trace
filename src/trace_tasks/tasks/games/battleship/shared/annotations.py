"""Annotation projection helpers for the Battleship games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from .rendering import RenderedBattleshipScene
from .state import BattleshipShipPlacement, Coord, coord_to_cell_id, sorted_coords


@dataclass(frozen=True)
class BattleshipAnnotationProjection:
    """Projected annotation geometry and trace ids."""

    annotation_cell_ids: tuple[str, ...]
    annotation_points: list[list[float]]
    annotation_bboxes: list[list[float]]
    annotation_entity_ids: tuple[str, ...]
    annotation_point_set_map: Dict[str, list[list[float]]]
    annotation_bbox_set_map: Dict[str, list[list[float]]]
    annotation_key_to_ship_id: Dict[str, str]
    annotation_hit_cell_ids_by_key: Dict[str, list[str]]


def bbox_center(bbox: Sequence[float]) -> list[float]:
    """Return the rounded center of a pixel bbox."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def rounded_bbox(bbox: Sequence[float]) -> list[float]:
    """Return a rounded pixel bbox list."""

    return [round(float(value), 3) for value in bbox[:4]]


def cell_ids_for_coords(coords: tuple[Coord, ...]) -> tuple[str, ...]:
    """Return stable cell ids for sorted board coordinates."""

    return tuple(coord_to_cell_id(coord) for coord in sorted_coords(coords))


def project_point_set_annotation(
    *,
    annotation_coords: Sequence[Coord],
    rendered_scene: RenderedBattleshipScene,
) -> BattleshipAnnotationProjection:
    """Project counted board cells to homogeneous point-set annotation."""

    annotation_cell_ids = cell_ids_for_coords(tuple(annotation_coords))
    annotation_points = [
        bbox_center(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
        for cell_id in annotation_cell_ids
    ]
    return BattleshipAnnotationProjection(
        annotation_cell_ids=annotation_cell_ids,
        annotation_points=annotation_points,
        annotation_bboxes=[
            rounded_bbox(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
            for cell_id in annotation_cell_ids
        ],
        annotation_entity_ids=tuple(str(cell_id) for cell_id in annotation_cell_ids),
        annotation_point_set_map={},
        annotation_bbox_set_map={},
        annotation_key_to_ship_id={},
        annotation_hit_cell_ids_by_key={},
    )


def project_bbox_set_annotation(
    *,
    annotation_coords: Sequence[Coord],
    rendered_scene: RenderedBattleshipScene,
) -> BattleshipAnnotationProjection:
    """Project counted board cells to homogeneous bbox-set annotation."""

    annotation_cell_ids = cell_ids_for_coords(tuple(annotation_coords))
    annotation_bboxes = [
        rounded_bbox(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
        for cell_id in annotation_cell_ids
    ]
    return BattleshipAnnotationProjection(
        annotation_cell_ids=annotation_cell_ids,
        annotation_points=[
            bbox_center(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
            for cell_id in annotation_cell_ids
        ],
        annotation_bboxes=annotation_bboxes,
        annotation_entity_ids=tuple(str(cell_id) for cell_id in annotation_cell_ids),
        annotation_point_set_map={},
        annotation_bbox_set_map={},
        annotation_key_to_ship_id={},
        annotation_hit_cell_ids_by_key={},
    )


def project_ship_status_annotation(
    *,
    ship_placements: Sequence[BattleshipShipPlacement],
    annotation_ship_ids: Sequence[str],
    rendered_scene: RenderedBattleshipScene,
) -> BattleshipAnnotationProjection:
    """Project ship-status witness ships to hit-cell point-set maps."""

    ships_by_id = {str(ship.ship_id): ship for ship in ship_placements}
    annotation_point_set_map: Dict[str, list[list[float]]] = {}
    annotation_bbox_set_map: Dict[str, list[list[float]]] = {}
    annotation_key_to_ship_id: Dict[str, str] = {}
    annotation_hit_cell_ids_by_key: Dict[str, list[str]] = {}
    annotation_coords = tuple(
        coord
        for ship_id in annotation_ship_ids
        for coord in ships_by_id[str(ship_id)].hit_coords
    )
    annotation_cell_ids = cell_ids_for_coords(annotation_coords)
    annotation_points = [
        bbox_center(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
        for cell_id in annotation_cell_ids
    ]
    annotation_bboxes = [
        rounded_bbox(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
        for cell_id in annotation_cell_ids
    ]
    for ship_id in annotation_ship_ids:
        ship = ships_by_id[str(ship_id)]
        key = str(ship.display_name)
        hit_cell_ids = [
            coord_to_cell_id((int(row), int(col)))
            for row, col in sorted_coords(ship.hit_coords)
        ]
        if not hit_cell_ids:
            raise ValueError("cannot build annotation points for ship with no hit cells")
        annotation_point_set_map[str(key)] = [
            bbox_center(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
            for cell_id in hit_cell_ids
        ]
        annotation_bbox_set_map[str(key)] = [
            rounded_bbox(rendered_scene.render_map["cell_bboxes_px"][str(cell_id)])
            for cell_id in hit_cell_ids
        ]
        annotation_key_to_ship_id[str(key)] = str(ship.ship_id)
        annotation_hit_cell_ids_by_key[str(key)] = [str(cell_id) for cell_id in hit_cell_ids]
    return BattleshipAnnotationProjection(
        annotation_cell_ids=annotation_cell_ids,
        annotation_points=annotation_points,
        annotation_bboxes=annotation_bboxes,
        annotation_entity_ids=tuple(str(ship_id) for ship_id in annotation_ship_ids),
        annotation_point_set_map=annotation_point_set_map,
        annotation_bbox_set_map=annotation_bbox_set_map,
        annotation_key_to_ship_id=annotation_key_to_ship_id,
        annotation_hit_cell_ids_by_key=annotation_hit_cell_ids_by_key,
    )


def project_shape_option_annotation(
    *,
    option_label: str,
    rendered_scene: RenderedBattleshipScene,
) -> BattleshipAnnotationProjection:
    """Project one labeled shape answer option to a scalar bbox annotation."""

    option_bboxes = rendered_scene.render_map.get("shape_option_bboxes_px", {})
    if str(option_label) not in option_bboxes:
        raise ValueError(f"unknown Battleship shape option label: {option_label}")
    bbox = rounded_bbox(option_bboxes[str(option_label)])
    return BattleshipAnnotationProjection(
        annotation_cell_ids=tuple(),
        annotation_points=[bbox_center(bbox)],
        annotation_bboxes=[bbox],
        annotation_entity_ids=(f"shape_option_{str(option_label)}",),
        annotation_point_set_map={},
        annotation_bbox_set_map={},
        annotation_key_to_ship_id={},
        annotation_hit_cell_ids_by_key={},
    )


__all__ = [
    "BattleshipAnnotationProjection",
    "bbox_center",
    "cell_ids_for_coords",
    "project_bbox_set_annotation",
    "project_point_set_annotation",
    "project_shape_option_annotation",
    "project_ship_status_annotation",
    "rounded_bbox",
]
