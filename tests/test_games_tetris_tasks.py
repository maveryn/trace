"""Contract tests for games Tetris tasks."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.tetris.active_piece_shape_label import GamesTetrisActivePieceShapeLabelTask
from trace_tasks.tasks.games.tetris.drop_collision_time_value import GamesTetrisDropCollisionTimeValueTask
from trace_tasks.tasks.games.tetris.drop_result_label import GamesTetrisDropResultLabelTask
from trace_tasks.tasks.games.tetris.line_clear_count import GamesTetrisLineClearCountTask
from trace_tasks.tasks.games.tetris.row_occupancy_status_count import GamesTetrisRowOccupancyStatusCountTask
from trace_tasks.tasks.games.tetris.shared.rules import (
    best_clear_outcomes,
    drop_collision,
    evaluate_outcome,
    freeze,
    is_supported_stack_board,
    piece_cells,
)
from trace_tasks.tasks.games.tetris.shared.sampling import construct_board_with_target_clear
from trace_tasks.tasks.games.tetris.shared.state import Placement


def _board_from_execution(execution: dict) -> tuple[tuple[str, ...], ...]:
    return freeze(execution["board_rows"])


def _assert_supported_stack_execution(execution: dict) -> None:
    board = _board_from_execution(execution)
    assert is_supported_stack_board(board)
    generation = execution.get("board_generation")
    assert isinstance(generation, dict)
    assert generation.get("mode") == "natural_supported_stack"


def _placement_from_execution(raw: dict) -> Placement:
    return Placement(
        piece=str(raw["piece"]),
        orientation_index=int(raw["orientation_index"]),
        col=int(raw["col"]),
        top=int(raw["top"]),
    )


def _qualifying_row_indices(board: tuple[tuple[str, ...], ...], *, query_id: str) -> tuple[int, ...]:
    if str(query_id) == "full_row_count":
        return tuple(
            int(row_index)
            for row_index, row in enumerate(board)
            if all(str(cell) != "." for cell in row)
        )
    if str(query_id) == "one_gap_row_count":
        return tuple(
            int(row_index)
            for row_index, row in enumerate(board)
            if sum(1 for cell in row if str(cell) == ".") == 1
        )
    raise AssertionError(f"unsupported test query_id: {query_id}")


def test_games_tetris_line_clear_contract_and_rule_match() -> None:
    out = GamesTetrisLineClearCountTask().generate(
        26052401,
        params={"query_id": "single", "target_clear_count": 4, "board_rows": 14, "board_cols": 9},
        max_attempts=240,
    )
    execution = out.trace_payload["execution_trace"]
    placement_raw = execution["placement"]
    assert placement_raw is not None
    placement = Placement(
        piece=str(placement_raw["piece"]),
        orientation_index=int(placement_raw["orientation_index"]),
        col=int(placement_raw["col"]),
        top=int(placement_raw["top"]),
    )
    board = _board_from_execution(execution)
    outcome = evaluate_outcome(board, placement)
    best_clear, _best_outcomes = best_clear_outcomes(board, piece=str(execution["piece"]))
    _assert_supported_stack_execution(execution)

    assert out.scene_id == "tetris"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_map"
    assert int(out.answer_gt.value) == int(outcome.clear_count) == int(best_clear) == 4
    assert set(out.annotation_gt.value) == {"board", "next_piece"}
    assert out.trace_payload["projected_annotation"]["type"] == "bbox_map"
    assert set(out.trace_payload["projected_annotation"]["bbox_map"]) == {"board", "next_piece"}
    render_spec = out.trace_payload["render_spec"]
    assert render_spec["tetris_board_style"]["style_variant"]
    assert render_spec["text_style"]["font_family"]


def test_games_tetris_positive_clear_constructor_uses_piece_variety() -> None:
    for target_clear_count in (1, 2, 3):
        pieces = set()
        for offset in range(18):
            rng = spawn_rng(26052600 + offset, f"tests.tetris.positive_clear.{target_clear_count}")
            _board, placement, outcome = construct_board_with_target_clear(
                rng,
                target_clear_count=int(target_clear_count),
                scene_variant="notched_stack",
                board_rows=14,
                board_cols=9,
            )
            assert int(outcome.clear_count) == int(target_clear_count)
            pieces.add(str(placement.piece))
        assert pieces - {"I"}

    rng = spawn_rng(26052700, "tests.tetris.positive_clear.four")
    _board, placement, outcome = construct_board_with_target_clear(
        rng,
        target_clear_count=4,
        scene_variant="notched_stack",
        board_rows=14,
        board_cols=9,
    )
    assert int(outcome.clear_count) == 4
    assert str(placement.piece) == "I"


def test_games_tetris_drop_result_contract() -> None:
    out = GamesTetrisDropResultLabelTask().generate(
        26052421,
        params={"query_id": "single", "target_clear_count": 1},
        max_attempts=240,
    )
    execution = out.trace_payload["execution_trace"]
    answer = str(out.answer_gt.value)
    options = {str(option["label"]): option for option in execution["options"]}
    falling = execution["falling_placement"]
    _assert_supported_stack_execution(execution)

    assert out.scene_id == "tetris"
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert answer in options
    assert bool(options[answer]["is_answer"])
    assert execution["target_clear_count"] == 1
    assert falling is not None
    assert int(falling["top"]) == 0
    assert all(option["placement"] is None for option in execution["options"])
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.trace_payload["projected_annotation"]["type"] == "bbox"


def test_games_tetris_drop_result_four_option_layout_stays_in_bounds() -> None:
    out = GamesTetrisDropResultLabelTask().generate(
        6227121074877783,
        params={"query_id": "single", "target_clear_count": 1, "board_rows": 15, "board_cols": 11},
        max_attempts=240,
    )
    width, height = out.image.size
    option_bboxes = out.trace_payload["render_map"]["option_bboxes_px"]
    panels = out.trace_payload["render_map"]["panels"]
    start_bbox = [float(v) for v in panels["start"]]

    assert len(option_bboxes) == 4
    assert sorted(option_bboxes) == ["option_a", "option_b", "option_c", "option_d"]
    assert out.trace_payload["query_spec"]["params"]["option_count"] == 4
    assert start_bbox[1] < min(float(bbox[1]) for bbox in option_bboxes.values())
    option_y0_values = sorted({round(float(bbox[1]), 3) for bbox in option_bboxes.values()})
    assert len(option_y0_values) == 2
    for y0 in option_y0_values:
        assert sum(1 for bbox in option_bboxes.values() if round(float(bbox[1]), 3) == y0) == 2
    for bbox in option_bboxes.values():
        x0, y0, x1, y1 = [float(v) for v in bbox]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
    x0, y0, x1, y1 = [float(v) for v in out.annotation_gt.value]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def test_games_tetris_active_piece_shape_label_contract() -> None:
    out = GamesTetrisActivePieceShapeLabelTask().generate(
        26052431,
        params={"target_piece": "T", "board_rows": 12, "board_cols": 8},
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    entity_ids = tuple(str(entity_id) for entity_id in execution["annotation_entity_ids"])
    cell_bboxes = [
        [float(v) for v in out.trace_payload["render_map"]["cell_bboxes_px"][str(entity_id)]]
        for entity_id in entity_ids
    ]
    expected_bbox = [
        min(float(bbox[0]) for bbox in cell_bboxes),
        min(float(bbox[1]) for bbox in cell_bboxes),
        max(float(bbox[2]) for bbox in cell_bboxes),
        max(float(bbox[3]) for bbox in cell_bboxes),
    ]

    assert out.scene_id == "tetris"
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) in {"A", "B", "C", "D"}
    assert execution["prompt_query_key"] == "active_piece_shape_label"
    assert execution["piece"] == "T"
    assert execution["target_piece"] == "T"
    assert len(execution["shape_options"]) == 4
    assert execution["shape_options"].count("T") == 1
    assert execution["correct_shape"] == "T"
    assert execution["correct_option_label"] == str(out.answer_gt.value)
    assert len(execution["shape_option_entries"]) == 4
    assert {str(entry["label"]) for entry in execution["shape_option_entries"]} == {"A", "B", "C", "D"}
    assert next(str(entry["piece"]) for entry in execution["shape_option_entries"] if str(entry["label"]) == str(out.answer_gt.value)) == "T"
    assert out.annotation_gt.type == "bbox"
    assert out.annotation_gt.value == expected_bbox
    assert out.trace_payload["projected_annotation"]["type"] == "bbox"
    assert out.trace_payload["render_map"]["shape_option_labels"] == ["A", "B", "C", "D"]
    assert out.trace_payload["render_map"]["shape_option_pieces"] == execution["shape_options"]


def test_games_tetris_row_occupancy_status_contract_and_rule_match() -> None:
    cases = (
        ("full_row_count", 3),
        ("one_gap_row_count", 4),
    )
    for query_id, target_row_count in cases:
        out = GamesTetrisRowOccupancyStatusCountTask().generate(
            26052441 + int(target_row_count),
            params={
                "query_id": str(query_id),
                "target_row_count": int(target_row_count),
                "board_rows": 12,
                "board_cols": 8,
            },
            max_attempts=64,
        )
        execution = out.trace_payload["execution_trace"]
        board = _board_from_execution(execution)
        _assert_supported_stack_execution(execution)
        qualifying_rows = _qualifying_row_indices(board, query_id=str(query_id))
        expected_entity_ids = tuple(f"main_row_{int(row)}" for row in qualifying_rows)
        expected_bboxes = [
            [float(v) for v in out.trace_payload["render_map"]["row_bboxes_px"][str(entity_id)]]
            for entity_id in expected_entity_ids
        ]

        assert out.scene_id == "tetris"
        assert out.query_id == str(query_id)
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == len(qualifying_rows) == int(target_row_count)
        assert execution["target_row_count"] == int(target_row_count)
        assert tuple(execution["qualifying_rows"]) == qualifying_rows
        assert tuple(execution["annotation_entity_ids"]) == expected_entity_ids
        assert out.annotation_gt.type == "bbox_set"
        assert out.annotation_gt.value == expected_bboxes
        assert out.trace_payload["projected_annotation"]["type"] == "bbox_set"
        assert out.trace_payload["projected_annotation"]["bbox_set"] == expected_bboxes


def test_games_tetris_drop_collision_time_contract_and_rule_match() -> None:
    cases = (
        ("no_shift_collision_time", 0, 1, 26052461),
        ("left_shift_collision_time", 3, 2, 26052471),
        ("right_shift_collision_time", 5, 2, 26052481),
    )
    for query_id, target_drop_steps, shift_magnitude, seed in cases:
        out = GamesTetrisDropCollisionTimeValueTask().generate(
            int(seed),
            params={
                "query_id": str(query_id),
                "target_drop_steps": int(target_drop_steps),
                "shift_magnitude": int(shift_magnitude),
                "board_rows": 14,
                "board_cols": 9,
            },
            max_attempts=128,
        )
        execution = out.trace_payload["execution_trace"]
        board = _board_from_execution(execution)
        _assert_supported_stack_execution(execution)
        falling = _placement_from_execution(execution["falling_placement"])
        collision = drop_collision(board, falling, shift_delta=int(execution["shift_delta"]))
        assert collision is not None

        assert out.scene_id == "tetris"
        assert out.query_id == str(query_id)
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == int(collision.drop_steps) == int(target_drop_steps)
        assert execution["target_drop_steps"] == int(target_drop_steps)
        assert execution["drop_steps"] == int(target_drop_steps)
        assert execution["collision_kind"] == "locked_block"
        assert execution["shifted_placement"] == {
            "piece": collision.shifted_placement.piece,
            "orientation_index": int(collision.shifted_placement.orientation_index),
            "col": int(collision.shifted_placement.col),
            "top": int(collision.shifted_placement.top),
            "cells": [[int(r), int(c)] for r, c in piece_cells(collision.shifted_placement)],
        }
        assert out.annotation_gt.type == "bbox_set_map"
        assert set(out.annotation_gt.value) == {"start_piece", "stop_witness"}
        expected_annotation = {
            key: [
                [float(v) for v in out.trace_payload["render_map"]["cell_bboxes_px"][str(entity_id)]]
                for entity_id in entity_ids
            ]
            for key, entity_ids in execution["annotation_entity_id_map"].items()
        }
        assert out.annotation_gt.value == expected_annotation
        assert out.trace_payload["projected_annotation"]["type"] == "bbox_set_map"
        assert out.trace_payload["projected_annotation"]["bbox_set_map"] == expected_annotation


def test_games_tetris_taxonomy() -> None:
    assert resolve_task_taxonomy("task_games__tetris__active_piece_shape_label").scene_id == "tetris"
    assert resolve_task_taxonomy("task_games__tetris__line_clear_count").scene_id == "tetris"
    assert resolve_task_taxonomy("task_games__tetris__drop_result_label").scene_id == "tetris"
    assert resolve_task_taxonomy("task_games__tetris__row_occupancy_status_count").scene_id == "tetris"
    assert resolve_task_taxonomy("task_games__tetris__drop_collision_time_value").scene_id == "tetris"
