"""Identity-free trace serialization helpers for Chess games scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.games.shared.piece_board_rules import coord_to_cell_id, coord_to_square_name, serialize_board

from .rendering import RenderedChessTaskContext
from .state import ChessCheckmateSample, ChessSceneSample


def board_relations(sample: ChessSceneSample) -> dict[str, Any]:
    """Return scene relations that do not define the public objective."""

    return {
        "scene_variant": str(sample.scene_variant),
        "style_variant": str(sample.style_variant),
        "board_size": 8,
        "player_color": str(sample.player_color),
        "target_piece_kind": str(sample.target_piece_kind),
        "target_piece_color": str(sample.target_piece_color),
        "marked_cell_id": None if sample.marked_coord is None else coord_to_cell_id(sample.marked_coord),
        "target_cell_id": None if sample.target_coord is None else coord_to_cell_id(sample.target_coord),
        "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
    }


def common_trace_sections(*, sample: ChessSceneSample, rendered_context: RenderedChessTaskContext) -> dict[str, Any]:
    """Return common trace payload sections for non-option Chess tasks."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": f"games_chess_board_{str(sample.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": board_relations(sample),
        },
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_cell_size_px": rendered_scene.render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "board_size": 8,
            "board_rows": serialize_board(sample.board),
            "construction_mode": str(sample.construction_mode),
            "occupied_count": int(sample.occupied_count),
            "player_color": str(sample.player_color),
            "target_piece_kind": str(sample.target_piece_kind),
            "target_piece_color": str(sample.target_piece_color),
            "marked_coord": None if sample.marked_coord is None else [int(sample.marked_coord[0]), int(sample.marked_coord[1])],
            "target_coord": None if sample.target_coord is None else [int(sample.target_coord[0]), int(sample.target_coord[1])],
            "marked_piece": None if sample.marked_piece is None else {"color": sample.marked_piece.color, "kind": sample.marked_piece.kind},
            "destination_coords": [[int(row), int(col)] for row, col in sample.destination_coords],
            "capture_coords": [[int(row), int(col)] for row, col in sample.capture_coords],
            "attacker_coords": [[int(row), int(col)] for row, col in sample.attacker_coords],
            "blocker_coords": [[int(row), int(col)] for row, col in sample.blocker_coords],
            "annotation_kind": str(sample.annotation_kind),
            "annotation_coords": [[int(row), int(col)] for row, col in sample.annotation_coords],
            "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        },
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


def checkmate_move_options_trace(sample: ChessCheckmateSample) -> list[dict[str, Any]]:
    """Return visible checkmate option records for trace payloads."""

    from trace_tasks.tasks.games.shared.piece_board_rules import move_checkmates

    return [
        {
            "label": str(option.label),
            "text": str(option.text),
            "source_coord": [int(option.source[0]), int(option.source[1])],
            "destination_coord": [int(option.destination[0]), int(option.destination[1])],
            "source_square": coord_to_square_name(option.source),
            "destination_square": coord_to_square_name(option.destination),
            "piece": {"color": str(option.piece.color), "kind": str(option.piece.kind)},
            "is_checkmate": bool(move_checkmates(sample.board, option.source, option.destination)),
        }
        for option in sample.options
    ]


def common_checkmate_trace_sections(*, sample: ChessCheckmateSample, rendered_context: RenderedChessTaskContext) -> dict[str, Any]:
    """Return common trace sections for checkmate option scenes."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": "games_chess_board_checkmate_options",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "style_variant": str(sample.style_variant),
                "board_size": 8,
                "player_color": str(sample.player_color),
                "defender_color": str(sample.defender_color),
                "option_labels": [str(label) for label in sample.option_label_support],
            },
        },
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_cell_size_px": rendered_scene.render_map.get("effective_cell_size_px"),
            "show_coordinates": True,
            "option_count": int(sample.option_count),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "board_size": 8,
            "board_rows": serialize_board(sample.board),
            "construction_mode": str(sample.construction_mode),
            "occupied_count": int(sample.occupied_count),
            "player_color": str(sample.player_color),
            "defender_color": str(sample.defender_color),
            "defender_king_coord": [int(sample.defender_king_coord[0]), int(sample.defender_king_coord[1])],
            "correct_source_coord": [int(sample.correct_option.source[0]), int(sample.correct_option.source[1])],
            "correct_destination_coord": [int(sample.correct_option.destination[0]), int(sample.correct_option.destination[1])],
            "correct_source_square": coord_to_square_name(sample.correct_option.source),
            "correct_destination_square": coord_to_square_name(sample.correct_option.destination),
            "answer_option_label": str(sample.correct_option.label),
            "answer_support": [str(label) for label in sample.option_label_support],
            "move_options": checkmate_move_options_trace(sample),
            "annotation_kind": "point_map",
            "annotation_keys": ["from", "to", "king"],
            "annotation_cell_ids": {
                "from": coord_to_cell_id(sample.correct_option.source),
                "to": coord_to_cell_id(sample.correct_option.destination),
                "king": coord_to_cell_id(sample.defender_king_coord),
            },
            "mirror_columns": bool(sample.extra.get("mirror_columns", False)),
        },
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }

__all__ = ["common_checkmate_trace_sections", "common_trace_sections", "checkmate_move_options_trace"]
