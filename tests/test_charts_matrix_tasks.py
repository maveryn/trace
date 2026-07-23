"""Behavior tests for chart annotated-matrix tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import assert_counter_support_within, extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.charts.matrix.axis_extremum_label import ChartsMatrixAxisExtremumLabelTask
from trace_tasks.tasks.charts.matrix.off_diagonal_confusion_label import ChartsMatrixOffDiagonalConfusionLabelTask
from trace_tasks.tasks.charts.matrix.shared.defaults import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.charts.matrix.threshold_cell_count import ChartsMatrixThresholdCellCountTask


TASK_CASES = (
    ("axis", ChartsMatrixAxisExtremumLabelTask),
    ("off_diagonal", ChartsMatrixOffDiagonalConfusionLabelTask),
    ("threshold", ChartsMatrixThresholdCellCountTask),
)
TASK_KIND_INDEX = {kind: index for index, (kind, _task_cls) in enumerate(TASK_CASES)}


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _cell_value(execution: dict, row_index: int, column_index: int) -> int:
    value = execution["values"][int(row_index)][int(column_index)]
    assert value is not None
    return int(value)


def _line_cell_ids(execution: dict, *, query_axis: str, axis_index: int) -> list[str]:
    if str(query_axis) == "row":
        row = execution["values"][int(axis_index)]
        return [f"r{int(axis_index)}_c{c}" for c, value in enumerate(row) if value is not None]
    return [f"r{r}_c{int(axis_index)}" for r, row in enumerate(execution["values"]) if row[int(axis_index)] is not None]


def _expected_answer(task_kind: str, execution: dict, replay_params: dict) -> int | str:
    if str(execution.get("answerability", "answerable")) == "unanswerable":
        return "unanswerable"

    row_labels = [str(item) for item in execution["row_labels"]]
    column_labels = [str(item) for item in execution["column_labels"]]
    cells_by_id = {str(key): dict(value) for key, value in execution["cells_by_id"].items()}
    if task_kind == "axis":
        query_axis = str(execution["query_axis"])
        axis_index = int(execution["answer_row_index"] if query_axis == "row" else execution["answer_column_index"])
        cell_ids = _line_cell_ids(execution, query_axis=query_axis, axis_index=axis_index)
        values = [int(cells_by_id[cell_id]["value"]) for cell_id in cell_ids]
        rank = int(execution["extremum_rank"])
        ranked_distinct_values = sorted({int(value) for value in values}, reverse=str(execution["extremum_direction"]) == "highest")
        target = int(ranked_distinct_values[int(rank) - 1])
        winners = [cell_id for cell_id in cell_ids if int(cells_by_id[cell_id]["value"]) == int(target)]
        assert len(winners) == 1
        winner = cells_by_id[winners[0]]
        return column_labels[int(winner["column_index"])] if query_axis == "row" else row_labels[int(winner["row_index"])]
    if task_kind == "off_diagonal":
        row_index = int(execution["answer_row_index"])
        off_diagonal = [
            (column_index, _cell_value(execution, row_index, column_index))
            for column_index in range(int(execution["column_count"]))
            if int(column_index) != int(row_index)
        ]
        maximum = max(value for _, value in off_diagonal)
        winners = [column_index for column_index, value in off_diagonal if int(value) == int(maximum)]
        assert len(winners) == 1
        return column_labels[int(winners[0])]
    if task_kind == "threshold":
        query_axis = str(execution["query_axis"])
        axis_index = int(execution["answer_row_index"] if query_axis == "row" else execution["answer_column_index"])
        threshold = int(replay_params["threshold_value"])
        cell_ids = _line_cell_ids(execution, query_axis=query_axis, axis_index=axis_index)
        if str(execution["comparison"]) == "at_least":
            return sum(1 for cell_id in cell_ids if int(cells_by_id[cell_id]["value"]) >= int(threshold))
        return sum(1 for cell_id in cell_ids if int(cells_by_id[cell_id]["value"]) <= int(threshold))
    raise AssertionError(f"unsupported matrix task kind: {task_kind}")


def _expected_annotation_bboxes(trace: dict) -> list[list[float]]:
    render_map = trace["render_map"]
    execution = trace["execution_trace"]
    return [render_map["cell_bboxes_px"][str(cell_id)] for cell_id in execution["annotation_cell_ids"]]


@pytest.mark.parametrize(("task_kind", "task_cls"), TASK_CASES)
def test_chart_matrix_supported_queries_match_contract(task_kind: str, task_cls: type) -> None:
    task = task_cls()
    for query_index, query_id in enumerate(task.supported_query_ids):
        out = task.generate(
            hash64(20260503, f"charts_matrix_{task_kind}", TASK_KIND_INDEX[task_kind] * 10 + query_index),
            params={"query_id": str(query_id)},
            max_attempts=80,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        replay_params = trace["query_spec"]["params"]
        assert out.query_id == str(query_id)
        assert replay_params["query_id"] == str(query_id)
        assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
        assert out.annotation_gt.type == "bbox_set"
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert str(execution["question_format"]) == "matrix_cell_query"
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert 6 <= int(execution["row_count"]) <= 12
        assert 6 <= int(execution["column_count"]) <= 12
        assert len(execution["cells"]) == int(execution["row_count"]) * int(execution["column_count"])
        assert len(trace["render_map"]["cell_bboxes_px"]) == len(execution["cells"])
        assert out.answer_gt.type == ("integer" if task_kind == "threshold" else "string")
        expected_answer = _expected_answer(task_kind, execution, replay_params)
        assert out.answer_gt.value == expected_answer
        assert execution["answer_value"] == expected_answer
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert out.annotation_gt.value == _expected_annotation_bboxes(trace)
        assert trace["projected_annotation"]["cell_ids"] == execution["annotation_cell_ids"]
        assert "header_keys" not in trace["projected_annotation"]
        if str(execution.get("answerability", "answerable")) == "answerable":
            assert execution["annotation_cell_ids"]
            assert execution["support_header_keys"]
        else:
            assert execution["annotation_cell_ids"] == []
            assert execution["support_header_keys"] == []
        if task_kind == "threshold":
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas(
                [float(value) for value in bbox],
                width=int(render["canvas_width"]),
                height=int(render["canvas_height"]),
            )
        if task_kind == "off_diagonal":
            assert out.query_id == SINGLE_QUERY_ID
            assert str(execution["scene_variant"]) == "confusion_matrix_counts"
        if task_kind == "threshold":
            assert str(execution["comparison"]) in {"at_least", "at_most"}
            assert "threshold_value" in replay_params


def test_chart_matrix_prompt_examples_match_contract() -> None:
    expected = (
        (ChartsMatrixAxisExtremumLabelTask(), "row_highest_axis_extremum_label", "C7"),
        (ChartsMatrixOffDiagonalConfusionLabelTask(), SINGLE_QUERY_ID, "C5"),
        (ChartsMatrixThresholdCellCountTask(), "row_at_least_threshold_cell_count", 4),
    )
    for index, (task, query_id, answer) in enumerate(expected, start=90100):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=80)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation["answer"] == answer
        assert answer_only == {"answer": answer}
        assert isinstance(answer_and_annotation["annotation"], list)


def test_chart_matrix_sampling_covers_queries_and_scene_variants() -> None:
    axis_task = ChartsMatrixAxisExtremumLabelTask()
    threshold_task = ChartsMatrixThresholdCellCountTask()
    observed_queries: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    row_counts: Counter[int] = Counter()
    column_counts: Counter[int] = Counter()
    for index in range(120):
        task = axis_task if index % 2 == 0 else threshold_task
        out = task.generate(hash64(90200, "charts_matrix", index), params={}, max_attempts=80)
        execution = out.trace_payload["execution_trace"]
        observed_queries[str(out.query_id)] += 1
        scenes[str(execution["scene_variant"])] += 1
        row_counts[int(execution["row_count"])] += 1
        column_counts[int(execution["column_count"])] += 1
    assert set(axis_task.supported_query_ids).issubset(set(observed_queries.keys()))
    assert set(threshold_task.supported_query_ids).issubset(set(observed_queries.keys()))
    assert set(SUPPORTED_SCENE_VARIANTS).issubset(set(scenes.keys()))
    assert set(row_counts.keys()).issubset(set(range(6, 13)))
    assert set(column_counts.keys()).issubset(set(range(6, 13)))
    assert 6 in row_counts
    assert 12 in row_counts
    assert_counter_support_within(row_counts, tuple(range(6, 13)), expected_per_key=120 / 7, tolerance=18)


def test_chart_matrix_is_deterministic() -> None:
    task = ChartsMatrixThresholdCellCountTask()
    params = {
        "query_id": "row_at_least_threshold_cell_count",
        "scene_variant": "clustered_block_matrix",
        "palette_variant": "yellow_purple",
        "header_layout": "dual_headers",
        "grid_style": "heavy_block_lines",
    }
    out_a = task.generate(90300, params=params, max_attempts=80)
    out_b = task.generate(90300, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
