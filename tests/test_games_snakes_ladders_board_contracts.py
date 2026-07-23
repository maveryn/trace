"""Contract tests for games Snakes and Ladders board tasks."""

from __future__ import annotations

from pathlib import Path

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.games.snakes_ladders.shared.state import (
    SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS,
    SnakesLaddersJump,
)
from trace_tasks.tasks.games.snakes_ladders.shared.rules import (
    apply_die_roll,
    square_to_cell_id,
)
from trace_tasks.tasks.games.snakes_ladders.move_outcome_value import (
    GamesSnakesLaddersMoveOutcomeValueTask,
)
from trace_tasks.tasks.games.snakes_ladders.remaining_to_finish_value import (
    GamesSnakesLaddersRemainingToFinishValueTask,
)
from trace_tasks.tasks.games.snakes_ladders.special_square_count import GamesSnakesLaddersSpecialSquareCountTask
from tests.helpers import read_jsonl


def _jumps_from_trace(execution: dict) -> tuple:
    """Return jump dataclasses from trace dictionaries."""

    return tuple(
        SnakesLaddersJump(
            jump_id=str(jump["jump_id"]),
            kind=str(jump["kind"]),
            start_square=int(jump["start_square"]),
            end_square=int(jump["end_square"]),
        )
        for jump in execution["jumps"]
    )


def _special_square_ids(execution: dict) -> tuple[str, ...]:
    kind = str(execution["special_square_kind"])
    starts = sorted(
        int(jump["start_square"])
        for jump in execution["jumps"]
        if str(jump["kind"]) == kind
    )
    return tuple(square_to_cell_id(square) for square in starts)


def test_games_snakes_ladders_move_outcome_matches_trace() -> None:
    out = GamesSnakesLaddersMoveOutcomeValueTask().generate(
        68101,
        params={"target_answer": 31, "die_value": 5},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    jumps = _jumps_from_trace(execution)
    move = apply_die_roll(int(execution["start_square"]), 5, jumps, board_side=int(execution["board_side"]))

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(move.final_square) == 31
    assert out.annotation_gt.type == "bbox_map"
    assert out.query_id == "single"
    assert out.scene_id == "snakes_ladders"
    assert execution["query_id"] == "single"
    assert execution["prompt_query_key"] == "move_outcome_value"
    assert execution["board_side"] in {5, 6, 7}
    assert trace_value(out, "query_spec", "params", "query_id") == "single"
    assert trace_value(out, "query_spec", "params", "prompt_query_key") == "move_outcome_value"
    assert trace_value(out, "projected_annotation", "type") == "bbox_map"
    assert trace_value(out, "projected_annotation", "bbox_map") == out.annotation_gt.value
    assert set(out.annotation_gt.value) == {"start_square", "end_square"}
    assert execution["annotation_entity_ids"] == [
        square_to_cell_id(int(execution["start_square"])),
        square_to_cell_id(int(move.final_square)),
    ]
    assert execution["annotation_role_entity_ids"]["end_square"] == square_to_cell_id(int(move.final_square))

def test_games_snakes_ladders_special_square_count_matches_trace() -> None:
    task = GamesSnakesLaddersSpecialSquareCountTask()
    cases = (
        ("ladder_count", 2),
        ("snake_count", 3),
    )

    for index, (query_id, target_answer) in enumerate(cases):
        out = task.generate(
            68600 + index,
            params={
                "query_id": query_id,
                "target_answer": target_answer,
                "board_side": 7,
            },
            max_attempts=512,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        expected_ids = _special_square_ids(execution)
        expected_bboxes = [
            list(trace["render_map"]["entity_bboxes_px"][entity_id])
            for entity_id in expected_ids
        ]

        assert out.scene_id == "snakes_ladders"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == int(target_answer) == len(expected_ids)
        assert out.annotation_gt.type == "bbox_set"
        assert out.annotation_gt.value == expected_bboxes
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == expected_bboxes
        assert execution["annotation_entity_ids"] == list(expected_ids)
        assert execution["count_scope"] == "all_visible_jumps_of_kind"


def test_games_snakes_ladders_remaining_to_finish_matches_trace() -> None:
    out = GamesSnakesLaddersRemainingToFinishValueTask().generate(
        68620,
        params={"target_answer": 25},
        max_attempts=512,
    )
    execution = out.trace_payload["execution_trace"]
    finish_square = int(execution["last_square"])
    token_square = int(execution["start_square"])

    assert int(out.answer_gt.value) == 25
    assert int(out.answer_gt.value) == finish_square - token_square
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == {"token_square", "finish_square"}
    assert execution["remaining_to_finish"] == 25
    assert execution["annotation_role_entity_ids"] == {
        "token_square": square_to_cell_id(token_square),
        "finish_square": square_to_cell_id(finish_square),
    }


def test_games_snakes_ladders_special_square_count_taxonomy() -> None:
    assert resolve_task_taxonomy("task_games__snakes_ladders__special_square_count").scene_id == "snakes_ladders"
    assert resolve_task_taxonomy("task_games__snakes_ladders__remaining_to_finish_value").scene_id == "snakes_ladders"


def test_games_snakes_ladders_sampling_cycles_cover_axes() -> None:
    move_task = GamesSnakesLaddersMoveOutcomeValueTask()
    remaining_task = GamesSnakesLaddersRemainingToFinishValueTask()
    move_answers: set[int] = set()
    remaining_answers: set[int] = set()
    board_sides: set[int] = set()
    styles: set[str] = set()

    for index in range(96):
        out = move_task.generate(68200 + index, params={}, max_attempts=512)
        move_answers.add(int(out.answer_gt.value))
        board_sides.add(int(out.trace_payload["execution_trace"]["board_side"]))
        styles.add(str(out.trace_payload["execution_trace"]["style_variant"]))

    for index in range(96):
        out = remaining_task.generate(68400 + index, params={}, max_attempts=512)
        remaining_answers.add(int(out.answer_gt.value))

    assert len(move_answers) >= 30
    assert len(remaining_answers) >= 20
    assert min(remaining_answers) >= 1
    assert max(remaining_answers) <= 25
    assert board_sides == {5, 6, 7}
    assert styles == set(SUPPORTED_SNAKES_LADDERS_STYLE_VARIANTS)


def test_games_snakes_ladders_generation_is_deterministic() -> None:
    params = {"target_answer": 25, "style_variant": "paper"}
    task = GamesSnakesLaddersRemainingToFinishValueTask()
    out_a = task.generate(68500, params=params, max_attempts=512)
    out_b = task.generate(68500, params=params, max_attempts=512)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_games_snakes_ladders_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_games__snakes_ladders"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_games__snakes_ladders",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(task_id="task_games__snakes_ladders__move_outcome_value", count=1, params={}),
            BuildTaskConfig(task_id="task_games__snakes_ladders__special_square_count", count=1, params={}),
            BuildTaskConfig(task_id="task_games__snakes_ladders__remaining_to_finish_value", count=1, params={}),
        ],
        max_attempts_per_instance=512,
        workers=1,
    )
    final_path = build_dataset(config, code_hash="games-snakes-ladders-smoke")
    rows = read_jsonl(final_path / "train_instances.jsonl")

    assert len(rows) == 3
    assert all(row["domain"] == "games" for row in rows)
    assert all(row["scene_id"] == "snakes_ladders" for row in rows)


def trace_value(out, *keys):
    """Read a nested trace value in tests."""

    value = out.trace_payload
    for key in keys:
        value = value[key]
    return value
