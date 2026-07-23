"""Contract tests for games chess-variant tasks."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.chess_variant.marked_piece_capture_count import GamesChessVariantMarkedPieceCaptureCountTask
from trace_tasks.tasks.games.chess_variant.marked_piece_destination_count import GamesChessVariantMarkedPieceDestinationCountTask
from trace_tasks.tasks.games.chess_variant.shared.rules import (
    evaluate_marked_piece_board,
    with_destination_annotation,
)
from trace_tasks.tasks.games.chess_variant.shared.prompts import prompt_defaults
from trace_tasks.tasks.games.shared.piece_board_rules import ChessPiece, freeze_board, validate_square_chess_material


def _board_from_execution(execution: dict) -> tuple[tuple[ChessPiece | None, ...], ...]:
    rows = []
    for row in execution["board_rows"]:
        parsed = []
        for value in row:
            if value is None:
                parsed.append(None)
                continue
            color, kind = str(value).split("_", 1)
            parsed.append(ChessPiece(color=color, kind=kind))
        rows.append(parsed)
    return freeze_board(rows)


def _assert_bbox_annotation_matches_entity_ids(trace: dict, annotation: list[list[float]]) -> None:
    """Verify bbox annotations use cell boxes for cells and piece boxes for pieces."""

    execution = trace["execution_trace"]
    expected: list[list[float]] = []
    for entity_id in execution["annotation_entity_ids"]:
        entity_id = str(entity_id)
        if entity_id.startswith("cell_"):
            expected.append(trace["render_map"]["cell_bboxes_px"][entity_id])
        else:
            expected.append(trace["render_map"]["piece_bboxes_px"][entity_id])
    assert annotation == expected
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == annotation
    assert trace["projected_annotation"]["pixel_bbox_set"] == annotation


def test_games_chess_variant_destination_count_contract_and_rule_match() -> None:
    out = GamesChessVariantMarkedPieceDestinationCountTask().generate(
        26052401,
        params={
            "rule_family": "straight_range",
            "range_k": 3,
            "target_answer": 4,
        },
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(v) for v in execution["marked_coord"])
    evaluated = with_destination_annotation(
        evaluate_marked_piece_board(
            board,
            marked_coord=marked,
            rule_family=str(execution["rule_family"]),
            range_k=int(execution["range_k"]),
        ),
        destination_mode="empty",
    )

    assert out.scene_id == "chess_variant"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == int(evaluated.answer) == 4
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert execution["annotation_kind"] == "cell"
    assert set(execution["annotation_entity_ids"]) == set(evaluated.annotation_entity_ids)
    _assert_bbox_annotation_matches_entity_ids(trace, out.annotation_gt.value)
    assert execution["internal_query_id"] == "marked_piece_destination_count"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert "panel_scene_style" in trace["render_spec"]
    assert "text_style" in trace["render_spec"]
    assert "empty" in out.prompt.lower()
    assert "opponent-occupied capture squares" in out.prompt.lower()


def test_games_chess_variant_capture_count_contract_and_rule_match() -> None:
    out = GamesChessVariantMarkedPieceCaptureCountTask().generate(
        26052411,
        params={"rule_family": "leaper_2_1", "target_answer": 3},
        max_attempts=128,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    board = _board_from_execution(execution)
    marked = tuple(int(v) for v in execution["marked_coord"])
    evaluated = with_destination_annotation(
        evaluate_marked_piece_board(
            board,
            marked_coord=marked,
            rule_family=str(execution["rule_family"]),
            range_k=int(execution["range_k"]),
        ),
        destination_mode="capture",
    )

    assert out.scene_id == "chess_variant"
    assert out.query_id == "single"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == int(evaluated.answer) == 3
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert execution["annotation_kind"] == "cell"
    assert set(execution["annotation_entity_ids"]) == set(evaluated.annotation_entity_ids)
    _assert_bbox_annotation_matches_entity_ids(trace, out.annotation_gt.value)
    assert execution["internal_query_id"] == "marked_piece_capture_count"
    assert "opponent-occupied squares" in out.prompt.lower()
    assert "empty movement squares" in out.prompt.lower()


def test_games_chess_variant_taxonomy() -> None:
    assert resolve_task_taxonomy("task_games__chess_variant__marked_piece_capture_count").scene_id == "chess_variant"
    assert resolve_task_taxonomy("task_games__chess_variant__marked_piece_destination_count").scene_id == "chess_variant"


def test_games_chess_variant_prompt_is_marker_color_neutral() -> None:
    prompt = prompt_defaults()

    assert "outlined square" in str(prompt["marked_piece_rule_text"]).lower()
    assert "red outlined square" in str(prompt["marked_piece_rule_text"]).lower()
    assert "piece" in str(prompt["marked_piece_rule_text"]).lower()
    assert "token" not in str(prompt["marked_piece_rule_text"]).lower()
    assert "blue" not in str(prompt["marked_piece_rule_text"]).lower()
    annotation_hints = {key: str(value) for key, value in prompt.items() if str(key).startswith("annotation_hint_")}
    assert annotation_hints
    for hint in annotation_hints.values():
        assert "pixel-space" in hint


def test_games_chess_variant_boards_are_material_plausible() -> None:
    task_params = (
        (GamesChessVariantMarkedPieceDestinationCountTask, {"rule_family": "straight_range", "range_k": 3, "target_answer": 3}),
        (GamesChessVariantMarkedPieceCaptureCountTask, {"rule_family": "leaper_2_1", "target_answer": 2}),
    )
    for offset, (task_cls, params) in enumerate(task_params):
        out = task_cls().generate(26052500 + int(offset), params=params, max_attempts=256)
        board = _board_from_execution(out.trace_payload["execution_trace"])
        assert validate_square_chess_material(board)
