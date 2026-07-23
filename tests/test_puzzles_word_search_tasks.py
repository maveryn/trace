"""Tests for word-search puzzle tasks."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.puzzles.word_search.search_location_label import (
    PuzzlesWordSearchLocationLabelTask,
    TASK_ID as LOCATION_TASK_ID,
)
from trace_tasks.tasks.puzzles.word_search.present_word_option_label import (
    PuzzlesWordSearchPresentWordOptionLabelTask,
    TASK_ID as PRESENT_WORD_OPTION_TASK_ID,
)
from trace_tasks.tasks.registry import TASK_REGISTRY


def test_word_search_tasks_are_registered() -> None:
    assert TASK_REGISTRY[LOCATION_TASK_ID] is PuzzlesWordSearchLocationLabelTask
    assert (
        TASK_REGISTRY[PRESENT_WORD_OPTION_TASK_ID]
        is PuzzlesWordSearchPresentWordOptionLabelTask
    )


def test_word_search_location_contract() -> None:
    out = PuzzlesWordSearchLocationLabelTask().generate(
        2026052201, params={}, max_attempts=80
    )
    trace = out.trace_payload["execution_trace"]

    assert out.scene_id == "word_search"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in "ABCDEF"
    assert trace["query_id"] == SINGLE_QUERY_ID
    assert 4 <= trace["grid_rows"] <= 6
    assert trace["grid_rows"] == trace["grid_cols"]
    assert len(trace["placements"]) == 1
    assert len(trace["option_specs"]) == 6
    correct = [spec for spec in trace["option_specs"] if spec["is_correct"]]
    assert len(correct) == 1
    assert correct[0]["label"] == out.answer_gt.value
    assert out.annotation_gt.type == "bbox_sequence"
    assert len(out.annotation_gt.value) == len(trace["placements"][0]["cells"])


def test_word_search_present_word_option_matches_trace() -> None:
    out = PuzzlesWordSearchPresentWordOptionLabelTask().generate(
        2026052203, params={}, max_attempts=80
    )
    trace = out.trace_payload["execution_trace"]
    correct = [spec for spec in trace["option_specs"] if spec["is_correct"]]

    assert out.scene_id == "word_search"
    assert out.query_id == SINGLE_QUERY_ID
    assert 4 <= trace["grid_rows"] <= 6
    assert trace["grid_rows"] == trace["grid_cols"]
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in "ABCD"
    assert len(trace["option_specs"]) == 4
    assert len(correct) == 1
    assert correct[0]["label"] == out.answer_gt.value
    assert correct[0]["word"] == trace["present_words"][0]
    assert len(trace["placements"]) == 1
    assert out.annotation_gt.type == "segment"
    assert len(out.annotation_gt.value) == 2


def test_word_search_generation_is_deterministic() -> None:
    task = PuzzlesWordSearchLocationLabelTask()
    a = task.generate(2026052204, params={}, max_attempts=80)
    b = task.generate(2026052204, params={}, max_attempts=80)

    assert a.answer_gt.to_dict() == b.answer_gt.to_dict()
    assert (
        a.trace_payload["execution_trace"]["grid"]
        == b.trace_payload["execution_trace"]["grid"]
    )
    assert a.annotation_gt.to_dict() == b.annotation_gt.to_dict()
