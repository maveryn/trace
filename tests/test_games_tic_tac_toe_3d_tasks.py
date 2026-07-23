"""Contract tests for 3D Tic-Tac-Toe game tasks."""

from __future__ import annotations

import json
from pathlib import Path

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.tic_tac_toe_3d.shared.rules import (
    WINNING_LINES,
    immediate_winning_cells,
)
from trace_tasks.tasks.registry import create_task, list_default_task_ids
from trace_tasks.tasks.shared.config_defaults import (
    split_generation_rendering_prompt_defaults,
)

WINNING_MOVE_TASK_ID = "task_games__tic_tac_toe_3d__winning_move_cell_label"
BLOCKING_MOVE_TASK_ID = "task_games__tic_tac_toe_3d__blocking_move_cell_label"
LAYER_COUNT_TASK_ID = "task_games__tic_tac_toe_3d__layer_piece_count"


def _trace_board_to_tuple(board_layers):
    return tuple(
        tuple(tuple(str(cell) for cell in row) for row in layer)
        for layer in board_layers
    )


def test_games_tic_tac_toe_3d_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("games", "tic_tac_toe_3d")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(cfg)

    assert "winning_move_query_id_weights" not in generation
    assert "blocking_move_query_id_weights" not in generation
    assert "layer_piece_count_query_id_weights" not in generation
    assert set(generation["layout_variant_weights"].keys()) == {
        "vertical_perspective_stack",
    }
    assert set(generation["style_variant_weights"].keys()) == {
        "classic_grid",
        "paper_board",
        "arcade_blue",
        "mint_table",
        "charcoal_lines",
    }
    assert int(rendering["canvas_width"]) == 760
    assert int(rendering["canvas_height"]) == 900
    assert (
        float(rendering["unit_size_scale_max"])
        / float(rendering["unit_size_scale_min"])
        >= 2.0
    )
    assert list(generation["layer_piece_count_support"]) == [0, 1, 2, 3, 4, 5, 6]
    assert list(generation["option_count_support"]) == [4]
    assert str(prompt["bundle_id"]) == "games_tic_tac_toe_3d_v1"


def test_games_tic_tac_toe_3d_prompt_bundle_has_queries() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/games/tic_tac_toe_3d/games_tic_tac_toe_3d_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(bundle["templates"]["query"].keys()) == {
        "x_winning_move_label",
        "o_winning_move_label",
        "x_blocking_move_label",
        "o_blocking_move_label",
        "piece_count_in_layer",
    }
    assert bool(bundle["allow_empty_task_templates"])


def test_games_tic_tac_toe_3d_registry_and_taxonomy() -> None:
    ids = set(list_default_task_ids())
    for task_id in (WINNING_MOVE_TASK_ID, BLOCKING_MOVE_TASK_ID, LAYER_COUNT_TASK_ID):
        assert task_id in ids
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "games"
        assert taxonomy.scene_id == "tic_tac_toe_3d"
        assert taxonomy.source_domain == "games"


def test_games_tic_tac_toe_3d_has_all_49_lines() -> None:
    assert len(WINNING_LINES) == 49
    assert len(set(WINNING_LINES)) == 49


def test_games_tic_tac_toe_3d_winning_move_has_unique_option() -> None:
    out = create_task(WINNING_MOVE_TASK_ID).generate(
        83031,
        params={
            "query_id": "x_winning_move_label",
            "option_count": 4,
            "answer_option_index": 2,
        },
        max_attempts=500,
    )
    trace = out.trace_payload["execution_trace"]
    board = _trace_board_to_tuple(trace["board_layers"])
    winning_cells = set(immediate_winning_cells(board, "X"))
    option_labels = trace["available_option_labels"][: len(trace["option_cells"])]
    option_cells = {
        label: tuple(coord)
        for label, coord in zip(option_labels, trace["option_cells"])
    }
    correct_labels = [
        label for label, coord in option_cells.items() if coord in winning_cells
    ]

    assert out.query_id == "x_winning_move_label"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "C"
    assert correct_labels == ["C"]
    assert tuple(trace["answer_cell"]) == option_cells["C"]
    assert len(trace["support_cells"]) == 2
    assert len(out.annotation_gt.value) == 3
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"
    assert out.trace_payload["render_spec"]["tic_tac_toe_3d_board_style"][
        "style_variant"
    ]
    assert out.trace_payload["render_spec"]["text_style"]["font_family"]


def test_games_tic_tac_toe_3d_blocking_move_has_unique_option() -> None:
    out = create_task(BLOCKING_MOVE_TASK_ID).generate(
        83051,
        params={
            "query_id": "x_blocking_move_label",
            "option_count": 4,
            "answer_option_index": 1,
        },
        max_attempts=500,
    )
    trace = out.trace_payload["execution_trace"]
    board = _trace_board_to_tuple(trace["board_layers"])
    target_player = str(trace["target_player"])
    threat_player = str(trace["threat_player"])
    opponent_wins = set(immediate_winning_cells(board, threat_player))
    target_wins = set(immediate_winning_cells(board, target_player))
    option_labels = trace["available_option_labels"][: len(trace["option_cells"])]
    option_cells = {
        label: tuple(coord)
        for label, coord in zip(option_labels, trace["option_cells"])
    }
    correct_labels = [
        label for label, coord in option_cells.items() if coord in opponent_wins
    ]

    assert out.query_id == "x_blocking_move_label"
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "B"
    assert target_player == "X"
    assert threat_player == "O"
    assert target_wins == set()
    assert opponent_wins == {tuple(trace["answer_cell"])}
    assert correct_labels == ["B"]
    assert tuple(trace["answer_cell"]) == option_cells["B"]
    assert len(trace["support_cells"]) == 2
    assert len(out.annotation_gt.value) == 3
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"


def test_games_tic_tac_toe_3d_layer_count_matches_trace() -> None:
    out = create_task(LAYER_COUNT_TASK_ID).generate(
        83041,
        params={
            "target_player": "O",
            "target_layer": "bottom",
            "target_answer": 4,
            "layout_variant": "vertical_perspective_stack",
        },
        max_attempts=500,
    )
    trace = out.trace_payload["execution_trace"]
    board = trace["board_layers"]
    target_layer_index = 2
    matching = [
        (target_layer_index, row, col)
        for row in range(3)
        for col in range(3)
        if board[target_layer_index][row][col] == "O"
    ]

    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert len(matching) == 4
    assert len(out.annotation_gt.value) == 4
    assert out.trace_payload["projected_annotation"]["type"] == "point_set"
    assert trace["target_player"] == "O"
    assert trace["prompt_query_key"] == "piece_count_in_layer"
