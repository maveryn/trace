"""Identity-free trace serialization helpers for circular-chess games tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.games.shared.piece_board_rules import color_name

from .defaults import RING_COUNT, SECTOR_COUNT
from .rules import serialize_board
from .state import CircularChessSample


def common_trace_sections(
    *,
    sample: CircularChessSample,
    image_size: tuple[int, int],
    render_map: dict[str, Any],
    scene_entities: tuple[dict[str, Any], ...],
    panel_style_meta: dict[str, Any],
    text_style_meta: dict[str, Any],
    background_meta: dict[str, Any],
    post_noise_meta: dict[str, Any],
) -> dict[str, Any]:
    """Return common trace payload sections for circular-chess tasks."""

    marked_piece = sample.evaluation.marked_piece
    marked_piece_payload = None if marked_piece is None else {"color": str(marked_piece.color), "kind": str(marked_piece.kind)}
    return {
        "scene_ir": {
            "scene_kind": f"games_circular_chess_{str(sample.scene_variant)}",
            "entities": [dict(entity) for entity in scene_entities],
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "style_variant": str(sample.style_variant),
                "ring_count": int(RING_COUNT),
                "sector_count": int(SECTOR_COUNT),
                "target_answer": int(sample.evaluation.answer),
                "target_color": str(sample.evaluation.target_color),
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids],
            },
        },
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(text_style_meta),
            "effective_ring_width_px": render_map.get("effective_ring_width_px"),
        },
        "render_map": dict(render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "board_rings": serialize_board(sample.board),
            "ring_count": int(RING_COUNT),
            "sector_count": int(SECTOR_COUNT),
            "occupied_count": int(sample.occupied_count),
            "construction_mode": str(sample.construction_mode),
            "target_answer": int(sample.evaluation.answer),
            "marked_coord": (
                None
                if sample.evaluation.marked_coord is None
                else [int(sample.evaluation.marked_coord[0]), int(sample.evaluation.marked_coord[1])]
            ),
            "target_coord": (
                None
                if sample.evaluation.target_coord is None
                else [int(sample.evaluation.target_coord[0]), int(sample.evaluation.target_coord[1])]
            ),
            "target_color": str(sample.evaluation.target_color),
            "target_color_name": color_name(sample.evaluation.target_color) if sample.evaluation.target_color else "",
            "marked_piece": marked_piece_payload,
            "annotation_kind": str(sample.evaluation.annotation_kind),
            "annotation_coords": [[int(coord[0]), int(coord[1])] for coord in sample.evaluation.annotation_coords],
            "annotation_entity_ids": [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids],
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids],
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = ["common_trace_sections"]
