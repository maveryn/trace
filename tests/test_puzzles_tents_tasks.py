"""Contracts for source-layoutd Tents puzzle tasks."""

from __future__ import annotations

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.tents.missing_tent_cell_label import (
    PuzzlesTentsMissingTentCellLabelTask,
)
from trace_tasks.tasks.puzzles.tents.violating_tent_label import (
    PuzzlesTentsViolatingTentLabelTask,
)
from trace_tasks.tasks.puzzles.tents.shared.rules import neighbors4
from trace_tasks.tasks.puzzles.tents.shared.state import SCENE_ID

TASK_CLASSES = (
    PuzzlesTentsMissingTentCellLabelTask,
    PuzzlesTentsViolatingTentLabelTask,
)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tents_tasks_emit_scene_package_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(72001, params={}, max_attempts=40)

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert "annotation" in out.prompt_variants["answer_and_annotation"].lower()
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["params"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["params"]["query_id"] == SINGLE_QUERY_ID
    assert trace["execution_trace"]["query_id"] == SINGLE_QUERY_ID
    assert trace["render_map"]["annotation_source"] == "item_bboxes_px"


def test_missing_tent_task_has_one_correct_labeled_cell() -> None:
    task = PuzzlesTentsMissingTentCellLabelTask()
    out = task.generate(72011, params={}, max_attempts=40)
    trace = out.trace_payload["execution_trace"]

    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    assert trace["option_count"] == 4
    assert trace["target_answer_support"] == ["A", "B", "C", "D"]
    assert len(trace["candidate_specs"]) == 4
    marked_row, marked_col = trace["marked_tree"]
    for spec in trace["candidate_specs"]:
        assert (
            abs(int(spec["row"]) - int(marked_row))
            + abs(int(spec["col"]) - int(marked_col))
            == 1
        )
    correct = [spec for spec in trace["candidate_specs"] if spec["is_correct"]]
    legal = [spec for spec in trace["candidate_specs"] if spec["is_legal"]]
    assert len(correct) == 1
    assert len(legal) == 1
    assert correct[0]["label"] == out.answer_gt.value
    assert out.trace_payload["projected_annotation"]["type"] == "bbox"


def test_violating_tent_task_has_one_invalid_labeled_tent() -> None:
    task = PuzzlesTentsViolatingTentLabelTask()
    for sampling_index in range(5):
        out = task.generate(72021 + sampling_index, params={}, max_attempts=40)
        trace = out.trace_payload["execution_trace"]
        labeled_tents = trace["labeled_tent_specs"]
        correct = [spec for spec in labeled_tents if spec["is_correct"]]

        assert out.answer_gt.type == "option_letter"
        assert out.annotation_gt.type == "bbox"
        assert out.answer_gt.value in {"A", "B", "C", "D"}
        assert trace["option_count"] == 4
        assert trace["target_answer_support"] == ["A", "B", "C", "D"]
        assert 6 <= int(trace["grid_rows"]) <= 8
        assert 6 <= int(trace["grid_cols"]) <= 8
        assert len(labeled_tents) == 4
        assert len(correct) == 1
        assert correct[0]["label"] == out.answer_gt.value
        assert trace["correct_tent_label"] == out.answer_gt.value
        assert trace["violation_type"] == "no_adjacent_tree"
        tree_cells = {tuple(cell) for cell in trace["tree_cells"]}
        rows = int(trace["grid_rows"])
        cols = int(trace["grid_cols"])
        for spec in labeled_tents:
            tent_cell = (int(spec["row"]), int(spec["col"]))
            has_tree = any(
                tuple(cell) in tree_cells
                for cell in neighbors4(tent_cell, rows, cols)
            )
            assert has_tree is (not bool(spec["is_correct"]))
        assert out.trace_payload["projected_annotation"]["type"] == "bbox"
        assert (
            f"labeled_tent_{out.answer_gt.value}"
            in out.trace_payload["render_map"]["item_bboxes_px"]
        )


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tents_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(72031, params=params, max_attempts=40)
    out_b = task.generate(72031, params=params, max_attempts=40)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
