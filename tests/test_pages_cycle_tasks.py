"""Behavior tests for pages cycle diagram tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.cycle.offset_stage_label import PagesCycleOffsetStageLabelTask


def test_pages_cycle_offset_stage_contract_matches_trace() -> None:
    task = PagesCycleOffsetStageLabelTask()
    output = task.generate(56100, params={"query_id": "after_stage_offset_label"}, max_attempts=30)
    execution = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    answer_bbox_id = str(execution["answer_stage_bbox_id"])

    assert output.scene_id == "cycle"
    assert output.query_id == "after_stage_offset_label"
    assert output.answer_gt.type == "string"
    assert output.annotation_gt.type == "bbox"
    assert output.answer_gt.value == execution["answer_stage_label"]
    assert output.annotation_gt.value == render_map["stage_bboxes_px"][answer_bbox_id]
    assert execution["query_id"] == "after_stage_offset_label"
    assert execution["prompt_query_key"] == "offset_stage_label"
    assert execution["query_relationship"] == "after"
    assert output.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"


@pytest.mark.parametrize("relationship", ["after", "before"])
@pytest.mark.parametrize("direction", ["clockwise", "counterclockwise"])
def test_pages_cycle_offset_stage_direction_semantics(relationship: str, direction: str) -> None:
    task = PagesCycleOffsetStageLabelTask()
    query_id = f"{relationship}_stage_offset_label"
    output = task.generate(
        56140,
        params={
            "query_id": query_id,
            "cycle_direction": direction,
            "stage_count_min": 7,
            "stage_count_max": 7,
            "step_count_min": 3,
            "step_count_max": 3,
        },
        max_attempts=30,
    )
    execution = output.trace_payload["execution_trace"]
    stage_count = int(execution["stage_count"])
    query_index = int(execution["query_stage_index"])
    step_count = int(execution["step_count"])
    direction_delta = 1 if str(execution["direction"]) == "clockwise" else -1
    if str(execution["query_relationship"]) == "after":
        expected_index = (query_index + (direction_delta * step_count)) % stage_count
    else:
        expected_index = (query_index - (direction_delta * step_count)) % stage_count

    assert int(execution["answer_stage_index"]) == int(expected_index)
    assert str(execution["query_relationship"]) == relationship
    assert output.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert str(execution["direction"]) == direction
    assert output.answer_gt.value == execution["answer_stage_label"]


def test_pages_cycle_balanced_sampling_defaults_cover_internal_axes() -> None:
    task = PagesCycleOffsetStageLabelTask()
    query_ids: Counter[str] = Counter()
    relationships: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    for index in range(80):
        output = task.generate(hash64(56200, task.task_id, index), params={}, max_attempts=30)
        execution = output.trace_payload["execution_trace"]
        assert output.query_id in {"after_stage_offset_label", "before_stage_offset_label"}
        assert execution["query_id"] == output.query_id
        query_ids[str(output.query_id)] += 1
        relationships[str(execution["query_relationship"])] += 1
        directions[str(execution["direction"])] += 1
        variants[str(execution["scene_variant"])] += 1
        assert 5 <= int(execution["stage_count"]) <= 12
        assert 2 <= int(execution["step_count"]) <= int(execution["stage_count"]) - 1

    assert set(query_ids.keys()) == {"after_stage_offset_label", "before_stage_offset_label"}
    assert set(relationships.keys()) == {"after", "before"}
    assert set(directions.keys()) == {"clockwise", "counterclockwise"}
    assert set(variants.keys()) == {"cycle_ring"}
