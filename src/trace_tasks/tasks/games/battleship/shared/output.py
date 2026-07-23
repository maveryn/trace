"""Identity-free trace serialization helpers for Battleship scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.types import TypedValue

from .annotations import BattleshipAnnotationProjection
from .rendering import RenderedBattleshipTaskContext
from .sampling import ResolvedBattleshipSceneAxes
from .state import FLEET_SHAPES, BattleshipSample, coord_to_cell_id, sorted_coords


def ship_trace(sample: BattleshipSample) -> list[dict[str, Any]]:
    """Serialize Battleship ship placements for trace payloads."""

    return [
        {
            "ship_id": str(ship.ship_id),
            "shape_id": str(ship.shape_id),
            "display_name": str(ship.display_name),
            "coords": [[int(row), int(col)] for row, col in ship.coords],
            "hit_coords": [[int(row), int(col)] for row, col in ship.hit_coords],
            "is_sunk": bool(ship.is_sunk),
        }
        for ship in sample.ship_placements
    ]


def candidate_option_trace(sample: BattleshipSample) -> list[dict[str, Any]]:
    """Serialize in-board candidate options for trace payloads."""

    return [
        {
            "label": str(option.label),
            "coord": [int(option.coord[0]), int(option.coord[1])],
            "cell_id": coord_to_cell_id((int(option.coord[0]), int(option.coord[1]))),
            "is_answer": bool(option.is_answer),
        }
        for option in sample.candidate_options
    ]


def shape_option_trace(sample: BattleshipSample) -> list[dict[str, Any]]:
    """Serialize panel shape options for trace payloads."""

    return [
        {
            "label": str(option.label),
            "shape_id": str(option.shape_id),
            "display_name": str(option.display_name),
            "is_answer": bool(option.is_answer),
            "entity_id": f"shape_option_{str(option.label)}",
        }
        for option in sample.shape_options
    ]


def target_ship_cell_ids(sample: BattleshipSample, *, target_ship_id: str) -> list[str]:
    """Return target ship cell ids for one named ship."""

    target_ships = [
        ship
        for ship in sample.ship_placements
        if str(ship.ship_id) == str(target_ship_id)
    ]
    if len(target_ships) != 1:
        return []
    return [
        coord_to_cell_id(coord)
        for coord in sorted_coords(target_ships[0].coords)
    ]


def fleet_shape_trace() -> list[dict[str, Any]]:
    """Serialize the visible fleet-shape legend."""

    return [
        {
            "shape_id": str(shape.shape_id),
            "display_name": str(shape.display_name),
            "offsets": [[int(row), int(col)] for row, col in shape.offsets],
        }
        for shape in FLEET_SHAPES
    ]


def projected_annotation_payload(
    *,
    annotation_gt: TypedValue,
    annotation_projection: BattleshipAnnotationProjection,
) -> dict[str, Any]:
    """Return projected annotation metadata for an already-bound annotation."""

    if str(annotation_gt.type) == "point_set":
        return {
            "type": "point_set",
            "point_set": [list(point) for point in annotation_projection.annotation_points],
            "pixel_point_set": [list(point) for point in annotation_projection.annotation_points],
        }
    if str(annotation_gt.type) == "bbox_set":
        return {
            "type": "bbox_set",
            "bbox_set": [list(bbox) for bbox in annotation_projection.annotation_bboxes],
            "pixel_bbox_set": [list(bbox) for bbox in annotation_projection.annotation_bboxes],
        }
    if str(annotation_gt.type) == "point":
        return {
            "type": "point",
            "point": list(annotation_gt.value),
            "pixel_point": list(annotation_gt.value),
        }
    if str(annotation_gt.type) == "bbox":
        return {
            "type": "bbox",
            "bbox": list(annotation_gt.value),
            "pixel_bbox": list(annotation_gt.value),
        }
    if str(annotation_gt.type) == "bbox_set_map":
        return {
            "type": "bbox_set_map",
            "bbox_set_map": dict(annotation_projection.annotation_bbox_set_map),
            "pixel_bbox_set_map": dict(annotation_projection.annotation_bbox_set_map),
        }
    return {
        "type": "point_set_map",
        "point_set_map": dict(annotation_projection.annotation_point_set_map),
        "pixel_point_set_map": dict(annotation_projection.annotation_point_set_map),
    }


def witness_symbolic_payload(
    *,
    annotation_gt: TypedValue,
    annotation_projection: BattleshipAnnotationProjection,
) -> dict[str, Any]:
    """Return symbolic witness ids for an already-bound annotation."""

    if str(annotation_gt.type) in {"point", "point_set", "bbox_set"}:
        return {
            "type": "cell_set",
            "ids": [str(cell_id) for cell_id in annotation_projection.annotation_cell_ids],
        }
    if str(annotation_gt.type) == "bbox":
        return {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_projection.annotation_entity_ids],
        }
    return {
        "type": "object_map",
        "ids": dict(annotation_projection.annotation_key_to_ship_id),
    }


def common_trace_sections(
    *,
    axes: ResolvedBattleshipSceneAxes,
    sample: BattleshipSample,
    rendered_context: RenderedBattleshipTaskContext,
    annotation_projection: BattleshipAnnotationProjection,
) -> dict[str, Any]:
    """Return query-neutral trace sections shared by Battleship tasks."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": f"games_battleship_grid_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "board_size": int(sample.board_size),
                "candidate_options": candidate_option_trace(sample),
                "shape_options": shape_option_trace(sample),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_projection.annotation_entity_ids],
            },
        },
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "board_size": int(sample.board_size),
            "hit_coords": [[int(row), int(col)] for row, col in sample.hit_coords],
            "miss_coords": [[int(row), int(col)] for row, col in sample.miss_coords],
            "annotation_cell_ids": [str(cell_id) for cell_id in annotation_projection.annotation_cell_ids],
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_projection.annotation_entity_ids],
            "annotation_ship_name_to_ship_id": dict(annotation_projection.annotation_key_to_ship_id),
            "annotation_hit_cell_ids_by_ship_name": dict(annotation_projection.annotation_hit_cell_ids_by_key),
            "candidate_options": candidate_option_trace(sample),
            "shape_options": shape_option_trace(sample),
            "ship_placements": ship_trace(sample),
            "fleet_shapes": fleet_shape_trace(),
            "fleet_sunk_total": int(sample.sunk_ship_count),
            "fleet_partial_total": int(sample.partial_ship_count),
            "fleet_untouched_total": int(sample.untouched_ship_count),
            "construction_mode": str(sample.construction_mode),
        },
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = [
    "candidate_option_trace",
    "common_trace_sections",
    "fleet_shape_trace",
    "projected_annotation_payload",
    "shape_option_trace",
    "ship_trace",
    "target_ship_cell_ids",
    "witness_symbolic_payload",
]
