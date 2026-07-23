"""Tests for named-grid adjacent icon-pair counting."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task


TASK_ID = "task_icons__named_grid__line_adjacency_pair_count"
QUERY_IDS = (
    "row_unordered_adjacent_pair_count",
    "column_unordered_adjacent_pair_count",
)


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    return json.loads(str(prompt).split(marker, 1)[1].strip())


def _matching_pairs(
    shape_grid: list[list[str]],
    *,
    axis: str,
    line_index: int,
    first_shape_id: str,
    second_shape_id: str,
) -> list[list[list[int]]]:
    if axis == "row":
        cells = [[int(line_index), col] for col in range(len(shape_grid[0]))]
    else:
        cells = [[row, int(line_index)] for row in range(len(shape_grid))]
    target_pair = {str(first_shape_id), str(second_shape_id)}
    pairs: list[list[list[int]]] = []
    for index in range(max(0, len(cells) - 1)):
        left = cells[index]
        right = cells[index + 1]
        observed = {
            str(shape_grid[int(left[0])][int(left[1])]),
            str(shape_grid[int(right[0])][int(right[1])]),
        }
        if observed == target_pair:
            pairs.append([left, right])
    return pairs


def test_icons_named_grid_line_adjacency_row_counts_overlapping_pairs() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260701, "icons-named-grid-adjacent-row", 0),
        params={
            "query_id": "row_unordered_adjacent_pair_count",
            "first_shape_id": "star",
            "second_shape_id": "bell",
            "answer_count": 2,
            "grid_rows": 4,
            "grid_cols": 4,
            "target_row_number": 3,
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    shape_grid = execution["shape_ids_by_cell"]
    target_line = shape_grid[2]

    assert out.scene_id == "named_grid"
    assert out.query_id == "row_unordered_adjacent_pair_count"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 2
    assert out.annotation_gt.type == "segment_set"
    assert trace["scene_ir"]["scene_kind"] == "icons_named_grid_line_adjacency_pair_count"
    assert execution["question_format"] == "count_unordered_adjacent_named_shape_pairs_in_grid_row_or_column"
    assert str(execution["queried_axis"]) == "row"
    assert int(execution["queried_number"]) == 3
    assert any(
        target_line[index : index + 3] in (["star", "bell", "star"], ["bell", "star", "bell"])
        for index in range(2)
    )
    expected_pairs = _matching_pairs(
        shape_grid,
        axis="row",
        line_index=2,
        first_shape_id="star",
        second_shape_id="bell",
    )
    assert expected_pairs == execution["counted_pair_cells"]
    assert len(expected_pairs) == 2
    assert len(out.annotation_gt.value) == 2
    assert trace["projected_annotation"]["type"] == "segment_set"
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_segment_set"] == out.annotation_gt.value
    assert trace["render_map"]["counted_pair_segments_px"] == out.annotation_gt.value
    assert '"star"' in out.prompt
    assert '"bell"' in out.prompt
    assert "overlapping adjacent pairs separately" in out.prompt


def test_icons_named_grid_line_adjacency_column_contract_matches_scene() -> None:
    task = create_task(TASK_ID)
    out = task.generate(
        hash64(20260701, "icons-named-grid-adjacent-column", 0),
        params={
            "query_id": "column_unordered_adjacent_pair_count",
            "first_shape_id": "guitar",
            "second_shape_id": "anchor",
            "answer_count": 4,
            "grid_rows": 6,
            "grid_cols": 4,
            "target_column_number": 2,
        },
        max_attempts=100,
    )
    execution = out.trace_payload["execution_trace"]
    expected_pairs = _matching_pairs(
        execution["shape_ids_by_cell"],
        axis="column",
        line_index=1,
        first_shape_id="guitar",
        second_shape_id="anchor",
    )

    assert out.query_id == "column_unordered_adjacent_pair_count"
    assert out.answer_gt.value == 4
    assert str(execution["queried_axis"]) == "column"
    assert int(execution["queried_number"]) == 2
    assert expected_pairs == execution["counted_pair_cells"]
    assert len(out.annotation_gt.value) == 4
    assert '"guitar"' in out.prompt
    assert '"anchor"' in out.prompt
    assert "column 2" in out.prompt


def test_icons_named_grid_line_adjacency_prompt_example_matches_contract() -> None:
    out = create_task(TASK_ID).generate(
        hash64(20260701, "icons-named-grid-adjacent-prompt", 0),
        params={"query_id": "row_unordered_adjacent_pair_count", "answer_count": 2},
        max_attempts=100,
    )
    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": 2}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert isinstance(answer_and_annotation["annotation"], list)
    assert all(len(segment) == 2 for segment in answer_and_annotation["annotation"])
    assert all(len(point) == 2 for segment in answer_and_annotation["annotation"] for point in segment)
    assert answer_and_annotation["answer"] == 2


def test_icons_named_grid_line_adjacency_sampling_distribution() -> None:
    task = create_task(TASK_ID)
    query_counts: Counter[str] = Counter()
    answer_counts: Counter[int] = Counter()
    grid_sizes: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(hash64(20260701, "icons-named-grid-adjacent-sampling", index), params={}, max_attempts=100)
        execution = out.trace_payload["execution_trace"]
        expected_pairs = _matching_pairs(
            execution["shape_ids_by_cell"],
            axis=str(execution["queried_axis"]),
            line_index=int(execution["queried_index"]),
            first_shape_id=str(execution["first_shape_id"]),
            second_shape_id=str(execution["second_shape_id"]),
        )
        assert str(out.query_id) in QUERY_IDS
        assert 1 <= int(out.answer_gt.value) <= 5
        assert len(expected_pairs) == int(out.answer_gt.value)
        assert expected_pairs == execution["counted_pair_cells"]
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert 4 <= int(execution["grid_rows"]) <= 6
        assert 4 <= int(execution["grid_cols"]) <= 6
        query_counts[str(out.query_id)] += 1
        answer_counts[int(out.answer_gt.value)] += 1
        grid_sizes[f"{int(execution['grid_rows'])}x{int(execution['grid_cols'])}"] += 1

    assert set(query_counts) == set(QUERY_IDS)
    assert set(answer_counts) == set(range(1, 6))
    assert len(grid_sizes) >= 6
