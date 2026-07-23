"""Contract tests for rule-override board games tasks."""

from __future__ import annotations

from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.tasks.games.rule_override_board.line_result_count import (
    LINE_LOSS_QUERY_ID,
    LINE_WIN_QUERY_ID,
    GamesRuleOverrideLineResultCountTask,
)
from trace_tasks.tasks.games.rule_override_board.piece_result_count import (
    PIECE_LOSS_QUERY_ID,
    PIECE_WIN_QUERY_ID,
    GamesRuleOverridePieceResultCountTask,
)
from trace_tasks.tasks.games.rule_override_board.shared.rendering import theme
from trace_tasks.tasks.games.rule_override_board.shared.state import SUPPORTED_BOARD_STYLES
from tests.helpers import read_jsonl


def _assert_count_contract(out, expected_query_id: str) -> None:
    trace = out.trace_payload
    execution = trace["execution_trace"]
    annotation_ids = [str(entity_id) for entity_id in execution["annotation_entity_ids"]]

    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert out.query_id == expected_query_id
    assert out.scene_id == "rule_override_board"
    assert trace["query_spec"]["query_id"] == expected_query_id
    assert execution["query_id"] == expected_query_id
    assert int(out.answer_gt.value) == len(annotation_ids) == len(out.annotation_gt.value)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert "rule_text" in trace["scene_ir"]["relations"]
    assert "rule_text" in execution
    assert "rule_card_text" not in trace["scene_ir"]["relations"]
    assert "rule_card_text" not in execution
    assert str(trace["scene_ir"]["relations"]["rule_text"]) in out.prompt
    assert all(str(entity.get("entity_type")) != "rule_card" for entity in trace["scene_ir"]["entities"])
    assert "panel_scene_style" in trace["render_spec"]
    assert trace["render_spec"]["font_asset"]["font_family"]


def test_games_rule_override_line_count_branches() -> None:
    task = GamesRuleOverrideLineResultCountTask()
    for query_id in (LINE_WIN_QUERY_ID, LINE_LOSS_QUERY_ID):
        out = task.generate(
            2026052901,
            params={"query_id": query_id, "target_answer": 3, "board_count": 5, "board_size": 3},
            max_attempts=256,
        )
        _assert_count_contract(out, query_id)
        boards = out.trace_payload["execution_trace"]["boards"]
        annotation_ids = set(out.trace_payload["execution_trace"]["annotation_entity_ids"])
        counted = [board for board in boards if str(board["board_id"]) in annotation_ids]
        expected_result = "win" if query_id == LINE_WIN_QUERY_ID else "loss"
        assert all(str(board["result_for_target_player"]) == expected_result for board in counted)


def test_games_rule_override_piece_count_branches() -> None:
    task = GamesRuleOverridePieceResultCountTask()
    for query_id in (PIECE_WIN_QUERY_ID, PIECE_LOSS_QUERY_ID):
        out = task.generate(
            2026052902,
            params={"query_id": query_id, "target_answer": 2, "board_count": 6, "board_size": 4},
            max_attempts=256,
        )
        _assert_count_contract(out, query_id)
        boards = out.trace_payload["execution_trace"]["boards"]
        annotation_ids = set(out.trace_payload["execution_trace"]["annotation_entity_ids"])
        counted = [board for board in boards if str(board["board_id"]) in annotation_ids]
        expected_result = "win" if query_id == PIECE_WIN_QUERY_ID else "loss"
        assert all(str(board["result_for_target_player"]) == expected_result for board in counted)


def test_games_rule_override_answer_range_can_be_empty_or_full() -> None:
    empty = GamesRuleOverrideLineResultCountTask().generate(
        2026052903,
        params={"query_id": LINE_WIN_QUERY_ID, "target_answer": 0, "board_count": 4, "board_size": 3},
        max_attempts=256,
    )
    full = GamesRuleOverridePieceResultCountTask().generate(
        2026052904,
        params={"query_id": PIECE_LOSS_QUERY_ID, "target_answer": 6, "board_count": 6, "board_size": 4},
        max_attempts=256,
    )

    assert int(empty.answer_gt.value) == 0
    assert empty.annotation_gt.value == []
    assert int(full.answer_gt.value) == 6
    assert len(full.annotation_gt.value) == 6


def test_games_rule_override_line_marks_share_xo_color() -> None:
    for style_name in SUPPORTED_BOARD_STYLES:
        colors = theme(str(style_name))
        assert colors["x"] == colors["o"]


def test_games_rule_override_board_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__rule_override_board"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__rule_override_board",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__rule_override_board__line_result_count", count=2, params={}),
            BuildTaskConfig(task_id="task_games__rule_override_board__piece_result_count", count=2, params={}),
        ],
        max_attempts_per_instance=256,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-rule-override-board-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 4
    assert all(row["domain"] == "games" for row in rows)
    assert all(row.get("scene_id") == "rule_override_board" for row in rows)
    assert {row["scene_id"] for row in rows} == {"rule_override_board"}
