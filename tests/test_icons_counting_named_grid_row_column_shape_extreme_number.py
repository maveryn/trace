"""Tests for named-grid row/column extremum icon counting."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task


TASK_ID = "task_icons__named_grid__row_column_shape_extreme_number"
QUERY_IDS = (
    "row_most_shape_number",
    "row_fewest_shape_number",
    "column_most_shape_number",
    "column_fewest_shape_number",
)


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def _line_counts(shape_grid: list[list[str]], *, axis: str, target_shape_id: str) -> list[int]:
    if axis == "row":
        return [sum(1 for value in row if str(value) == str(target_shape_id)) for row in shape_grid]
    return [
        sum(1 for row in shape_grid if str(row[col]) == str(target_shape_id))
        for col in range(len(shape_grid[0]))
    ]


def test_icons_counting_named_grid_row_most_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260528, "named-grid-extreme-row-most", 0),
        params={
            "query_id": "row_most_shape_number",
            "target_shape_id": "star",
            "grid_rows": 5,
            "grid_cols": 4,
            "answer_line_number": 3,
            "winning_target_count": 4,
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counts = _line_counts(
        execution["shape_ids_by_cell"],
        axis="row",
        target_shape_id=str(execution["target_shape_id"]),
    )

    assert out.scene_id == "named_grid"
    assert out.query_id == "row_most_shape_number"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 3
    entity_by_id = {str(entity["instance_id"]): entity for entity in trace["scene_ir"]["entities"]}
    counted_ids = set(trace["render_map"]["selected_line_target_instance_ids"])
    assert out.annotation_gt.type == "bbox_set"
    assert execution["question_format"] == "select_grid_line_number_by_extreme_named_shape_count"
    assert counts == execution["row_target_counts"]
    assert counts[2] == max(counts)
    assert counts.count(max(counts)) == 1
    assert len(out.annotation_gt.value) == counts[2]
    assert sorted(out.annotation_gt.value) == sorted((entity_by_id[instance_id]["bbox_xyxy"] for instance_id in counted_ids))
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert len(trace["projected_annotation"]["pixel_point_set"]) == len(out.annotation_gt.value)
    style = trace["render_spec"]["style"]
    assert "axis_label_stroke_rgb" in style
    assert style["text_legibility"]["required_role_count"] >= 2
    assert style["text_legibility"]["failure_count"] == 0


def test_icons_counting_named_grid_column_fewest_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260528, "named-grid-extreme-column-fewest", 0),
        params={
            "query_id": "column_fewest_shape_number",
            "target_shape_id": "guitar",
            "grid_rows": 5,
            "grid_cols": 6,
            "answer_line_number": 4,
            "winning_target_count": 1,
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counts = _line_counts(
        execution["shape_ids_by_cell"],
        axis="column",
        target_shape_id=str(execution["target_shape_id"]),
    )

    assert out.query_id == "column_fewest_shape_number"
    assert out.answer_gt.value == 4
    assert execution["column_target_counts"] == counts
    assert counts[3] == min(counts)
    assert counts.count(min(counts)) == 1
    assert len(out.annotation_gt.value) == counts[3]


def test_icons_counting_named_grid_extreme_prompt_example_matches_contract() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260528, "named-grid-extreme-prompt", 0),
        params={"query_id": "row_fewest_shape_number", "target_shape_id": "bell", "answer_line_number": 2},
        max_attempts=100,
    )
    assert '"bell"' in out.prompt
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 3}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert isinstance(answer_and_annotation["annotation"], list)
    assert all((len(bbox) == 4 for bbox in answer_and_annotation["annotation"]))
    assert answer_and_annotation["answer"] == 3


def test_icons_counting_named_grid_extreme_sampling_distribution() -> None:
    task = create_task(TASK_ID)
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    grid_sizes: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(
            hash64(20260528, "named-grid-extreme-sampling", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        query_id = str(out.query_id)
        axis = str(execution["queried_axis"])
        target_shape_id = str(execution["target_shape_id"])
        counts = _line_counts(execution["shape_ids_by_cell"], axis=axis, target_shape_id=target_shape_id)
        answer_line_index = int(out.answer_gt.value) - 1
        assert query_id in QUERY_IDS
        assert 1 <= int(out.answer_gt.value) <= 6
        if str(execution["extremum"]) == "most":
            assert counts[answer_line_index] == max(counts)
            assert counts.count(max(counts)) == 1
        else:
            assert counts[answer_line_index] == min(counts)
            assert counts.count(min(counts)) == 1
        assert len(out.annotation_gt.value) == int(execution["winning_target_count"])
        assert 4 <= int(execution["grid_rows"]) <= 6
        assert 4 <= int(execution["grid_cols"]) <= 6
        query_counts[query_id] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        grid_sizes[f"{int(execution['grid_rows'])}x{int(execution['grid_cols'])}"] += 1

    assert set(query_counts) == set(QUERY_IDS)
    assert set(answer_counts) == set(range(1, 7))
    assert len(grid_sizes) >= 6
