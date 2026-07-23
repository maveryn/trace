"""Identity-free trace serialization helpers for chess-variant games tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.games.shared.piece_board_rules import BOARD_SIZE, coord_to_cell_id, serialize_board

from .state import ChessVariantSample


def common_trace_sections(
    *,
    sample: ChessVariantSample,
    image_size: tuple[int, int],
    render_map: dict[str, Any],
    scene_entities: tuple[dict[str, Any], ...],
    panel_style_meta: dict[str, Any],
    text_style_meta: dict[str, Any],
    background_meta: dict[str, Any],
    post_noise_meta: dict[str, Any],
    rule_family: str,
    range_k: int,
) -> dict[str, Any]:
    """Return common trace payload sections for chess-variant tasks."""

    marked_piece = sample.evaluation.marked_piece
    marked_piece_payload = None if marked_piece is None else {"color": str(marked_piece.color), "kind": str(marked_piece.kind)}
    return {
        "scene_ir": {
            "scene_kind": f"games_chess_variant_board_{str(sample.scene_variant)}",
            "entities": [dict(entity) for entity in scene_entities],
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "rule_family": str(rule_family),
                "range_k": int(range_k),
                "style_variant": str(sample.style_variant),
                "board_size": int(BOARD_SIZE),
                "target_answer": int(sample.evaluation.answer),
                "marked_cell_id": coord_to_cell_id(sample.evaluation.marked_coord),
                "marker_role": str(sample.evaluation.marker_role),
                "annotation_entity_ids": [str(v) for v in sample.evaluation.annotation_entity_ids],
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
            "effective_cell_size_px": render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "rule_family": str(rule_family),
            "range_k": int(range_k),
            "style_variant": str(sample.style_variant),
            "target_answer": int(sample.evaluation.answer),
            "board_size": int(BOARD_SIZE),
            "board_rows": serialize_board(sample.board),
            "construction_mode": str(sample.construction_mode),
            "occupied_count": int(sample.occupied_count),
            "marked_coord": [int(sample.evaluation.marked_coord[0]), int(sample.evaluation.marked_coord[1])],
            "marked_piece": marked_piece_payload,
            "marker_role": str(sample.evaluation.marker_role),
            "legal_destination_coords": [[int(r), int(c)] for r, c in sample.evaluation.legal_destinations],
            "capture_coords": [[int(r), int(c)] for r, c in sample.evaluation.capture_coords],
            "annotation_kind": str(sample.evaluation.annotation_kind),
            "annotation_coords": [[int(r), int(c)] for r, c in sample.evaluation.annotation_coords],
            "annotation_entity_ids": [str(v) for v in sample.evaluation.annotation_entity_ids],
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(v) for v in sample.evaluation.annotation_entity_ids],
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = ["common_trace_sections"]
