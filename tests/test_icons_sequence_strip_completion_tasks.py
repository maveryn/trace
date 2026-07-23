"""Behavior tests for icon sequence-strip completion tasks."""

from __future__ import annotations

import json
from typing import Any

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.icons.sequence_strip.count_progression_completion_label import (
    IconsSequenceStripCountProgressionCompletionTask,
)
from trace_tasks.tasks.icons.sequence_strip.rotation_progression_completion_label import (
    IconsSequenceStripRotationProgressionCompletionTask,
)
from trace_tasks.tasks.icons.sequence_strip.size_progression_completion_label import (
    IconsSequenceStripSizeProgressionCompletionTask,
)


def _extract_prompt_json_example(prompt: str) -> dict[str, Any]:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


@pytest.mark.parametrize(
    ("task_cls", "params", "attribute_id", "expected_sequence", "expected_answer_value"),
    (
        (
            IconsSequenceStripCountProgressionCompletionTask,
            {"missing_index": 1, "answer_count": 5, "count_step": 1},
            "count",
            [4, 5, 6, 7],
            5,
        ),
        (
            IconsSequenceStripRotationProgressionCompletionTask,
            {"missing_index": 2, "start_rotation_degrees": 0, "rotation_step_degrees": 90},
            "rotation",
            [0, 90, 180, 270],
            180,
        ),
        (
            IconsSequenceStripSizeProgressionCompletionTask,
            {"missing_index": 1, "answer_size_px": 56, "size_step_px": 12},
            "size",
            [44, 56, 68, 80],
            56,
        ),
    ),
)
def test_icons_sequence_completion_contract_matches_scene(
    task_cls,
    params: dict[str, Any],
    attribute_id: str,
    expected_sequence: list[int],
    expected_answer_value: int,
) -> None:
    task = task_cls()
    out = task.generate(15110, params=dict(params), max_attempts=200)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    option_entities = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if str(entity["entity_kind"]) == "option_cell"
    ]
    sequence_entities = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if str(entity["entity_kind"]) == "sequence_cell"
    ]
    assert out.scene_id == "sequence_strip"
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert str(out.answer_gt.value) in {"A", "B", "C", "D"}
    assert out.annotation_gt.type == "bbox"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["query_spec"]["query_id"] == "single"
    assert execution["scene_variant"] == "two_row_sequence_completion_options"
    assert execution["attribute_id"] == attribute_id
    assert execution["full_sequence_values"] == expected_sequence
    assert int(execution["correct_option_value"]) == int(expected_answer_value)
    assert len(sequence_entities) == 4
    assert len(option_entities) == 4
    assert sum(1 for entity in sequence_entities if bool(entity["is_missing"])) == 1
    correct = [entity for entity in option_entities if bool(entity["is_correct"])]
    assert len(correct) == 1
    assert correct[0]["option_label"] == out.answer_gt.value
    assert correct[0]["cell_bbox_xyxy"] == out.annotation_gt.value
    assert int(correct[0]["option_value"]) == int(expected_answer_value)


def test_icons_sequence_completion_prompt_example_matches_mcq_contract() -> None:
    task = IconsSequenceStripCountProgressionCompletionTask()
    out = task.generate(15112, params={"missing_index": 2, "answer_count": 4, "count_step": 1}, max_attempts=200)
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "C"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert isinstance(answer_and_annotation["annotation"], list)
    assert len(answer_and_annotation["annotation"]) == 4
    assert answer_and_annotation["answer"] == "C"


@pytest.mark.parametrize(
    "task_cls",
    (
        IconsSequenceStripCountProgressionCompletionTask,
        IconsSequenceStripRotationProgressionCompletionTask,
        IconsSequenceStripSizeProgressionCompletionTask,
    ),
)
def test_icons_sequence_completion_balances_option_labels_by_default(task_cls) -> None:
    task = task_cls()
    labels: set[str] = set()
    missing_indices: set[int] = set()
    for index in range(80):
        out = task.generate(hash64(15113, task.task_id, index), params={}, max_attempts=200)
        labels.add(str(out.answer_gt.value))
        missing_indices.add(int(out.trace_payload["execution_trace"]["missing_index"]))
    assert labels == {"A", "B", "C", "D"}
    assert missing_indices == {0, 1, 2, 3}
