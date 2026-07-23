"""Contract tests for puzzle Sudoku-grid tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.builder import build_dataset
from trace_tasks.core.config import BuildConfig, BuildTaskConfig
from trace_tasks.core.trace_store import read_trace_shard
from trace_tasks.tasks import create_task
from trace_tasks.tasks.puzzles.sudoku.marked_cell_candidate_count import (
    PuzzlesSudokuMarkedCellCandidateCountTask,
)
from trace_tasks.tasks.puzzles.sudoku.marked_cell_value import (
    PuzzlesSudokuMarkedCellValueTask,
)
from trace_tasks.tasks.puzzles.sudoku.mistake_cell_label import (
    PuzzlesSudokuMistakeCellLabelTask,
)
from trace_tasks.tasks.puzzles.sudoku.shared.rules import (
    candidate_digits,
    coord_to_cell_id,
    unit_coords,
)
from tests.helpers import read_jsonl


@pytest.mark.parametrize(
    ("task_cls", "params", "expected_prompt_query", "expected_answer_type"),
    (
        (
            PuzzlesSudokuMarkedCellValueTask,
            {"target_answer": 7, "scene_variant": "sparse_grid"},
            "marked_cell_value",
            "integer",
        ),
        (
            PuzzlesSudokuMarkedCellCandidateCountTask,
            {"target_answer": 4, "scene_variant": "filled_grid"},
            "marked_cell_candidate_count",
            "integer",
        ),
        (
            PuzzlesSudokuMistakeCellLabelTask,
            {"answer_label": "B", "scene_variant": "filled_grid"},
            "mistake_cell_label",
            "option_letter",
        ),
    ),
)
def test_puzzles_sudoku_grid_emits_expected_contract(
    task_cls,
    params: dict[str, int | str],
    expected_prompt_query: str,
    expected_answer_type: str,
) -> None:
    out = task_cls().generate(40201, params=params, max_attempts=80)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.answer_gt.type == str(expected_answer_type)
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == "single"
    assert trace["query_spec"]["params"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["prompt_query_key"] == str(
        expected_prompt_query
    )
    assert execution["query_id"] == "single"
    assert execution["target_answer"] == out.answer_gt.value
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert len(out.annotation_gt.value) == 4
    assert isinstance(execution["annotation_entity_ids"], list)
    assert trace["render_spec"]["panel_scene_style"]["style_pack"]
    assert trace["render_spec"]["text_style"]["font_family"]


def test_puzzles_sudoku_marked_cell_value_has_unique_candidate() -> None:
    out = PuzzlesSudokuMarkedCellValueTask().generate(
        40211,
        params={"target_answer": 5},
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    board = tuple(tuple(int(value) for value in row) for row in execution["board_rows"])
    marked_cell = tuple(int(value) for value in execution["marked_cell"])

    assert int(out.answer_gt.value) == 5
    assert int(board[marked_cell[0]][marked_cell[1]]) == 0
    assert candidate_digits(board, marked_cell) == (5,)
    assert execution["annotation_entity_ids"] == [str(coord_to_cell_id(marked_cell))]
    assert (
        out.annotation_gt.value
        == out.trace_payload["render_map"]["cell_bboxes_px"][
            str(coord_to_cell_id(marked_cell))
        ]
    )


def test_puzzles_sudoku_marked_cell_candidate_count_matches_candidates() -> None:
    out = PuzzlesSudokuMarkedCellCandidateCountTask().generate(
        40215,
        params={"target_answer": 5},
        max_attempts=64,
    )
    execution = out.trace_payload["execution_trace"]
    board = tuple(tuple(int(value) for value in row) for row in execution["board_rows"])
    marked_cell = tuple(int(value) for value in execution["marked_cell"])
    candidates = candidate_digits(board, marked_cell)

    assert int(out.answer_gt.value) == len(candidates) == 5
    assert int(board[marked_cell[0]][marked_cell[1]]) == 0
    assert set(int(value) for value in execution["candidate_digit_values"]) == set(
        candidates
    )
    assert execution["annotation_entity_ids"] == [str(coord_to_cell_id(marked_cell))]
    assert (
        out.annotation_gt.value
        == out.trace_payload["render_map"]["cell_bboxes_px"][
            str(coord_to_cell_id(marked_cell))
        ]
    )


def test_puzzles_sudoku_mistake_cell_has_one_wrong_labeled_cell() -> None:
    out = PuzzlesSudokuMistakeCellLabelTask().generate(
        40231,
        params={"answer_label": "A"},
        max_attempts=80,
    )
    execution = out.trace_payload["execution_trace"]
    board = tuple(tuple(int(value) for value in row) for row in execution["board_rows"])
    solution = tuple(
        tuple(int(value) for value in row) for row in execution["solution_rows"]
    )
    option_specs = execution["option_specs"]
    option_coords = {(int(spec["row"]), int(spec["col"])) for spec in option_specs}
    wrong_specs = [spec for spec in option_specs if bool(spec["is_wrong_cell"])]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "A"
    assert len(option_specs) == 4
    assert len(wrong_specs) == 1
    assert wrong_specs[0]["label"] == out.answer_gt.value
    assert execution["correct_option_label"] == out.answer_gt.value
    assert execution["target_answer_support"] == ["A", "B", "C", "D"]

    wrong_coord = (int(wrong_specs[0]["row"]), int(wrong_specs[0]["col"]))
    wrong_value = int(board[wrong_coord[0]][wrong_coord[1]])
    assert wrong_value != int(solution[wrong_coord[0]][wrong_coord[1]])
    conflict_peer = tuple(int(value) for value in wrong_specs[0]["conflict_peer"])
    assert conflict_peer not in option_coords
    assert int(board[conflict_peer[0]][conflict_peer[1]]) == wrong_value
    assert conflict_peer in {
        tuple(coord)
        for coord in (
            [(wrong_coord[0], col) for col in range(9)]
            + [(row, wrong_coord[1]) for row in range(9)]
            + list(
                unit_coords(
                    "box",
                    int(wrong_coord[0] // 3) * 3 + int(wrong_coord[1] // 3),
                )
            )
        )
    }

    for spec in option_specs:
        coord = (int(spec["row"]), int(spec["col"]))
        assert int(board[coord[0]][coord[1]]) == int(spec["value"])
        if str(spec["label"]) == out.answer_gt.value:
            continue
        assert int(board[coord[0]][coord[1]]) == int(solution[coord[0]][coord[1]])

    assert execution["annotation_entity_ids"] == [str(coord_to_cell_id(wrong_coord))]
    assert (
        out.annotation_gt.value
        == out.trace_payload["render_map"]["cell_bboxes_px"][
            str(coord_to_cell_id(wrong_coord))
        ]
    )


@pytest.mark.parametrize(
    ("task_cls", "support"),
    (
        (PuzzlesSudokuMarkedCellValueTask, range(1, 10)),
        (PuzzlesSudokuMarkedCellCandidateCountTask, range(1, 6)),
    ),
)
def test_puzzles_sudoku_integer_tasks_accept_declared_answer_support(
    task_cls,
    support,
) -> None:
    task = task_cls()
    for answer_value in support:
        out = task.generate(
            40301 + int(answer_value),
            params={"target_answer": int(answer_value)},
            max_attempts=64,
        )
        assert out.query_id == "single"
        assert int(out.answer_gt.value) == int(answer_value)


@pytest.mark.parametrize("answer_label", ("A", "B", "C", "D"))
def test_puzzles_sudoku_option_tasks_accept_answer_labels(answer_label: str) -> None:
    mistake = PuzzlesSudokuMistakeCellLabelTask().generate(
        40341 + ord(answer_label),
        params={"answer_label": answer_label},
        max_attempts=80,
    )
    assert mistake.answer_gt.value == answer_label


def test_puzzles_sudoku_option_task_is_deterministic() -> None:
    params = {
        "query_id": "single",
        "answer_label": "D",
        "scene_variant": "filled_grid",
    }
    task = PuzzlesSudokuMistakeCellLabelTask()
    out_a = task.generate(40241, params=params, max_attempts=80)
    out_b = task.generate(40241, params=params, max_attempts=80)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert (
        out_a.trace_payload["query_spec"]["prompt_variant"]
        == out_b.trace_payload["query_spec"]["prompt_variant"]
    )
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_puzzles_sudoku_grid_prompt_bundle_matches_active_tasks() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/puzzles/sudoku/puzzles_sudoku_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert bundle["schema_version"] == "v1"
    required = bundle["required_slots_by_key"]
    assert required["scene:visible_sudoku_grid"] == ["object_description"]
    assert "query:hidden_single_cell_label" not in required
    assert "query:unit_missing_digits_count" not in required
    assert "query:repeated_digit_count" not in required
    static_slots = bundle["static_slots_by_key"]
    assert "marked cell" in static_slots["query:marked_cell_value"]["annotation_hint"]
    assert (
        "constraint_cells"
        not in static_slots["query:marked_cell_candidate_count"]["annotation_hint"]
    )
    assert (
        "selected lettered cell"
        in static_slots["query:mistake_cell_label"]["annotation_hint"]
    )
    assert (
        static_slots["query:mistake_cell_label"]["json_example_answer_only"]
        == '{"answer":"B"}'
    )


def test_puzzles_sudoku_retired_count_tasks_are_not_registered() -> None:
    for task_id in (
        "task_puzzles__sudoku__hidden_single_cell_label",
        "task_puzzles__sudoku__unit_missing_digits_count",
        "task_puzzles__sudoku__repeated_digit_count",
    ):
        with pytest.raises(KeyError):
            create_task(task_id)


def test_puzzles_sudoku_grid_build_smoke(tmp_path: Path) -> None:
    output_root = tmp_path / "task_puzzles__sudoku__mistake_cell_label"
    config = BuildConfig(
        output_root=str(output_root),
        dataset_name="build_smoke_task_puzzles__sudoku__mistake_cell_label",
        instance_version="v0",
        image_format="png",
        tasks=[
            BuildTaskConfig(
                task_id="task_puzzles__sudoku__mistake_cell_label",
                count=4,
                params={},
            )
        ],
        strict_repro=False,
        max_attempts_per_instance=80,
        sampling_seed=97,
    )
    final_path = build_dataset(config, code_hash="puzzles-sudoku-grid-smoke")
    assert final_path.exists()
    train_records = read_jsonl(final_path / "train_instances.jsonl")
    assert len(train_records) == 4
    assert all(record["domain"] == "puzzles" for record in train_records)
    assert all("scene_id" not in record for record in train_records)
    assert all(
        record["answer_gt"]["type"] == "option_letter" for record in train_records
    )

    curriculum_records = read_jsonl(final_path / "curriculum_index.jsonl")
    assert all("scene_id" not in record for record in curriculum_records)

    trace_records = read_trace_shard(
        final_path / "traces" / "trace_shard_0001.jsonl.zst"
    )
    assert all(record["taxonomy"]["scene_id"] == "sudoku" for record in trace_records)

    build_report = json.loads(
        (final_path / "build_report.json").read_text(encoding="utf-8")
    )
    assert (
        int(
            build_report["accepted_counts_by_task"][
                "task_puzzles__sudoku__mistake_cell_label"
            ]
        )
        == 4
    )

    validation = json.loads(
        (final_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["total_errors"] == 0
