"""Contract tests for Circular Chess games tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.circular_chess.marked_piece_destination_count import MARKED_CAPTURE_QUERY_ID, MARKED_MOVE_QUERY_ID
from trace_tasks.tasks.games.circular_chess.shared.rules import (
    capture_destinations,
    circular_coord_to_cell_id,
    circular_piece_to_entity_id,
    empty_board,
    freeze_board,
    legal_destinations,
    target_reachers,
)
from trace_tasks.tasks.games.circular_chess.target_cell_reacher_count import BLACK_REACHER_QUERY_ID, WHITE_REACHER_QUERY_ID
from trace_tasks.tasks.games.shared.piece_board_rules import BLACK, WHITE, ChessPiece, material_count, validate_circular_chess_material
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults


MARKED_DESTINATION_TASK_ID = "task_games__circular_chess__marked_piece_destination_count"
TARGET_REACHER_TASK_ID = "task_games__circular_chess__target_cell_reacher_count"


def _board_from_trace(out) -> tuple[tuple[ChessPiece | None, ...], ...]:
    rows = []
    for row in out.trace_payload["execution_trace"]["board_rings"]:
        parsed = []
        for value in row:
            if value is None:
                parsed.append(None)
            else:
                color, kind = str(value).split("_", 1)
                parsed.append(ChessPiece(color=str(color), kind=str(kind)))
        rows.append(parsed)
    return freeze_board(rows)


def test_games_circular_chess_defaults_expose_axes_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "circular_chess")
    generation, rendering, prompt = split_scene_generation_rendering_prompt_defaults(
        cfg,
        task_id=MARKED_DESTINATION_TASK_ID,
    )

    assert set(generation["marked_piece_kind_weights"].keys()) == {"king", "queen", "rook", "bishop", "knight"}
    assert list(generation["marked_piece_move_count_support"]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert int(rendering["max_board_size_px"]) > 0
    assert str(prompt["bundle_id"]) == "games_circular_chess_v1"


def test_games_circular_chess_prompt_bundle_has_queries() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/circular_chess/games_circular_chess_v1.json").read_text(encoding="utf-8")
    )
    assert set(bundle["templates"]["query"].keys()) == {
        MARKED_MOVE_QUERY_ID,
        MARKED_CAPTURE_QUERY_ID,
        WHITE_REACHER_QUERY_ID,
        BLACK_REACHER_QUERY_ID,
    }
    assert "Sectors wrap" in str(bundle["code_prompt_defaults"]["circular_board_rule_text"])
    assert "piece_rule_text" in bundle["required_slots_by_key"][f"query:{MARKED_MOVE_QUERY_ID}"]


def test_games_circular_chess_rook_wraps_around_ring_and_stops_at_blocker() -> None:
    board_rows = [list(row) for row in empty_board()]
    board_rows[1][0] = ChessPiece(color=WHITE, kind="rook")
    board_rows[1][15] = ChessPiece(color=BLACK, kind="knight")
    board_rows[1][2] = ChessPiece(color=WHITE, kind="bishop")
    board = freeze_board(board_rows)

    destinations = set(legal_destinations(board, (1, 0)))
    captures = set(capture_destinations(board, (1, 0)))

    assert (1, 15) in destinations
    assert (1, 15) in captures
    assert (1, 1) in destinations
    assert (1, 2) not in destinations


def test_games_circular_chess_marked_move_contract_matches_trace() -> None:
    out = create_task(MARKED_DESTINATION_TASK_ID).generate(
        92011,
        params={"query_id": MARKED_MOVE_QUERY_ID, "target_answer": 4, "marked_piece_kind": "rook"},
        max_attempts=80,
    )
    board = _board_from_trace(out)
    execution = out.trace_payload["execution_trace"]
    marked = tuple(execution["marked_coord"])
    expected_coords = tuple(sorted(legal_destinations(board, marked)))
    expected_ids = [circular_coord_to_cell_id(coord) for coord in expected_coords]
    expected_points = [out.trace_payload["render_map"]["cell_centers_px"][cell_id] for cell_id in expected_ids]

    assert out.scene_id == "circular_chess"
    assert out.query_id == MARKED_MOVE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.annotation_gt.type == "point_set"
    assert execution["annotation_entity_ids"] == expected_ids
    assert out.annotation_gt.value == expected_points
    assert out.trace_payload["projected_annotation"]["point_set"] == expected_points


def test_games_circular_chess_capture_contract_matches_trace() -> None:
    out = create_task(MARKED_DESTINATION_TASK_ID).generate(
        92012,
        params={"query_id": MARKED_CAPTURE_QUERY_ID, "target_answer": 3, "marked_piece_kind": "queen"},
        max_attempts=80,
    )
    board = _board_from_trace(out)
    execution = out.trace_payload["execution_trace"]
    marked = tuple(execution["marked_coord"])
    expected_coords = tuple(sorted(capture_destinations(board, marked)))
    expected_ids = [circular_coord_to_cell_id(coord) for coord in expected_coords]
    expected_points = [out.trace_payload["render_map"]["cell_centers_px"][cell_id] for cell_id in expected_ids]

    assert out.query_id == MARKED_CAPTURE_QUERY_ID
    assert int(out.answer_gt.value) == 3
    assert execution["annotation_entity_ids"] == expected_ids
    assert out.annotation_gt.value == expected_points


def test_games_circular_chess_target_reacher_contract_matches_trace() -> None:
    out = create_task(TARGET_REACHER_TASK_ID).generate(
        92013,
        params={"query_id": WHITE_REACHER_QUERY_ID, "target_answer": 3},
        max_attempts=80,
    )
    board = _board_from_trace(out)
    execution = out.trace_payload["execution_trace"]
    target = tuple(execution["target_coord"])
    expected_coords = tuple(sorted(target_reachers(board, target_coord=target, target_color=WHITE)))
    expected_ids = [
        circular_piece_to_entity_id(coord, board[int(coord[0])][int(coord[1])])
        for coord in expected_coords
        if board[int(coord[0])][int(coord[1])] is not None
    ]
    expected_points = [out.trace_payload["render_map"]["piece_centers_px"][piece_id] for piece_id in expected_ids]

    assert out.scene_id == "circular_chess"
    assert out.query_id == WHITE_REACHER_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 3
    assert out.annotation_gt.type == "point_set"
    assert execution["annotation_entity_ids"] == expected_ids
    assert out.annotation_gt.value == expected_points


def test_games_circular_chess_taxonomy_mapping() -> None:
    for task_id in (MARKED_DESTINATION_TASK_ID, TARGET_REACHER_TASK_ID):
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "circular_chess"
        assert taxonomy.source_domain == "games"


def test_games_circular_chess_boards_are_material_plausible() -> None:
    cases = (
        (
            MARKED_DESTINATION_TASK_ID,
            {"query_id": MARKED_MOVE_QUERY_ID, "target_answer": 4, "marked_piece_kind": "rook"},
        ),
        (
            MARKED_DESTINATION_TASK_ID,
            {"query_id": MARKED_CAPTURE_QUERY_ID, "target_answer": 3, "marked_piece_kind": "queen"},
        ),
        (
            TARGET_REACHER_TASK_ID,
            {"query_id": WHITE_REACHER_QUERY_ID, "target_answer": 3},
        ),
        (
            TARGET_REACHER_TASK_ID,
            {"query_id": BLACK_REACHER_QUERY_ID, "target_answer": 4},
        ),
    )
    for offset, (task_id, params) in enumerate(cases):
        out = create_task(task_id).generate(92100 + int(offset), params=params, max_attempts=160)
        board = _board_from_trace(out)
        assert validate_circular_chess_material(board)
        assert material_count(board, color=WHITE, kind="king") == 1
        assert material_count(board, color=BLACK, kind="king") == 1
