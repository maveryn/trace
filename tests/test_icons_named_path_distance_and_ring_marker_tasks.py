"""Behavior tests for added named-path and named-ring icon tasks."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task


PATH_DISTANCE_TASK_ID = "task_icons__named_path__path_distance_value"
RING_NEAREST_TASK_ID = "task_icons__named_ring__nearest_marker_target_count"


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def _ring_distance(index_a: int, index_b: int, *, count: int) -> int:
    delta = abs(int(index_a) - int(index_b)) % int(count)
    return int(min(delta, int(count) - delta))


def test_icons_named_path_distance_contract_matches_scene() -> None:
    task = create_task(PATH_DISTANCE_TASK_ID)
    out = task.generate(
        hash64(20260701, "icons-named-path-distance", 0),
        params={
            "answer_count": 4,
            "extra_stop_count": 4,
            "first_shape_id": "star",
            "second_shape_id": "bell",
        },
        max_attempts=120,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    entities = trace["scene_ir"]["entities"]
    by_position = {int(entity["position_index"]): entity for entity in entities}
    first_position = int(execution["first_position_index"])
    second_position = int(execution["second_position_index"])
    between_positions = list(range(min(first_position, second_position) + 1, max(first_position, second_position)))

    assert out.scene_id == "named_path"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 4
    assert out.annotation_gt.type == "bbox_set"
    assert trace["scene_ir"]["scene_kind"] == "icons_named_path_distance"
    assert execution["question_format"] == "count_stops_between_two_named_icon_types_along_start_to_end_path"
    assert execution["between_positions"] == between_positions
    assert len(between_positions) == int(out.answer_gt.value)
    assert str(by_position[first_position]["shape_id"]) == "star"
    assert str(by_position[second_position]["shape_id"]) == "bell"
    assert sum(1 for entity in entities if str(entity["shape_id"]) == "star") == 1
    assert sum(1 for entity in entities if str(entity["shape_id"]) == "bell") == 1
    assert out.annotation_gt.value == [by_position[position]["bbox_xyxy"] for position in between_positions]
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert '"star"' in out.prompt
    assert '"bell"' in out.prompt
    assert "strictly between" in out.prompt


def test_icons_named_path_distance_prompt_example_matches_contract() -> None:
    out = create_task(PATH_DISTANCE_TASK_ID).generate(
        hash64(20260701, "icons-named-path-distance-prompt", 0),
        params={"answer_count": 2},
        max_attempts=120,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 4}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert isinstance(answer_and_annotation["annotation"], list)
    assert len(answer_and_annotation["annotation"]) == 4
    assert all(len(bbox) == 4 for bbox in answer_and_annotation["annotation"])
    assert answer_and_annotation["answer"] == 4


def test_icons_named_ring_nearest_marker_contract_matches_scene() -> None:
    task = create_task(RING_NEAREST_TASK_ID)
    out = task.generate(
        hash64(20260701, "icons-named-ring-nearest", 0),
        params={"target_shape_id": "star", "answer_count": 3, "ring_icon_count": 16, "off_arc_target_count": 2},
        max_attempts=160,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    entities = trace["scene_ir"]["entities"]
    entity_by_id = {str(entity["instance_id"]): entity for entity in entities}
    shape_by_index = {int(entity["ring_index"]): str(entity["shape_id"]) for entity in entities}
    start = int(execution["start_index"])
    end = int(execution["end_index"])
    count = int(execution["ring_icon_count"])
    counted_indices = [int(value) for value in execution["counted_indices"]]

    assert out.scene_id == "named_ring"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 3
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 3
    assert trace["scene_ir"]["scene_kind"] == "icons_named_ring"
    assert execution["question_format"] == "count_target_shape_icons_closer_to_marker_a_than_marker_b"
    assert set(trace["render_map"]["marker_label_bboxes_px"]) == {"A", "B"}
    assert all(shape_by_index[index] == "star" for index in counted_indices)
    assert all(
        _ring_distance(index, start, count=count) < _ring_distance(index, end, count=count)
        for index in counted_indices
    )
    realized = sum(
        1
        for index, shape_id in shape_by_index.items()
        if shape_id == "star"
        and _ring_distance(index, start, count=count) < _ring_distance(index, end, count=count)
    )
    assert realized == int(out.answer_gt.value)
    counted_ids = list(trace["render_map"]["counted_instance_ids"])
    assert len(counted_ids) == int(out.answer_gt.value)
    assert sorted(out.annotation_gt.value) == sorted(entity_by_id[instance_id]["bbox_xyxy"] for instance_id in counted_ids)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert '"star"' in out.prompt
    assert "closer to marker A" in out.prompt
    assert "equally far" in out.prompt


def test_icons_named_ring_nearest_marker_prompt_example_and_sampling() -> None:
    task = create_task(RING_NEAREST_TASK_ID)
    answer_counts: Counter[int] = Counter()
    ring_counts: Counter[int] = Counter()
    for index in range(80):
        out = task.generate(hash64(20260701, "icons-named-ring-nearest-sampling", index), params={}, max_attempts=160)
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == "single"
        assert 0 <= int(out.answer_gt.value) <= 6
        assert 12 <= int(execution["ring_icon_count"]) <= 22
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        answer_counts[int(out.answer_gt.value)] += 1
        ring_counts[int(execution["ring_icon_count"])] += 1
    assert set(answer_counts) == set(range(0, 7))
    assert len(ring_counts) >= 6

    out = task.generate(hash64(20260701, "icons-named-ring-nearest-prompt", 0), params={"answer_count": 2}, max_attempts=160)
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert isinstance(answer_and_annotation["annotation"], list)
    assert answer_and_annotation["answer"] == 2
