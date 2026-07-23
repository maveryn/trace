"""Behavior tests for table counting tasks."""

from __future__ import annotations

from trace_tasks.tasks.charts.table.categorical_value_count import ChartsTableCategoricalValueCountTask
from trace_tasks.tasks.charts.table.interval_value_count import ChartsTableIntervalValueCountTask
from trace_tasks.tasks.charts.table.threshold_count import ChartsTableThresholdCountTask
from tests.helpers import extract_prompt_json_example


def test_table_threshold_count_variants_match_contract() -> None:
    task = ChartsTableThresholdCountTask()
    for seed, query_id in enumerate(("above_threshold_count", "below_threshold_count"), start=18210):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": "spreadsheet"}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        values_by_row = execution["values_by_row"]
        row_labels = [str(label) for label in execution["row_labels"]]
        query_column = str(execution["query_column"])
        threshold = int(execution["threshold_value"])
        matching = [int(index) for index in execution["matching_row_indices"]]

        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert trace_cell_boxes(out) == out.annotation_gt.value
        if query_id == "above_threshold_count":
            expected = [i for i, label in enumerate(row_labels) if int(values_by_row[label][query_column]) > threshold]
        else:
            expected = [i for i, label in enumerate(row_labels) if int(values_by_row[label][query_column]) < threshold]
        assert matching == expected
        assert int(out.answer_gt.value) == len(expected)


def test_table_interval_and_categorical_count_contracts() -> None:
    cases = (
        (ChartsTableIntervalValueCountTask(), {}, "single"),
        (ChartsTableCategoricalValueCountTask(), {}, "single"),
    )
    for seed, (task, params, expected_query) in enumerate(cases, start=18240):
        out = task.generate(seed, params=params, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        row_labels = [str(label) for label in execution["row_labels"]]
        values_by_row = execution["values_by_row"]
        query_column = str(execution["query_column"])

        assert out.query_id == expected_query
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert trace_cell_boxes(out) == out.annotation_gt.value
        if task.task_id.endswith("__interval_value_count"):
            low = int(execution["interval_min"])
            high = int(execution["interval_max"])
            expected = [i for i, label in enumerate(row_labels) if low <= int(values_by_row[label][query_column]) <= high]
        else:
            target = str(execution["target_category"])
            expected = [i for i, label in enumerate(row_labels) if str(values_by_row[label][query_column]) == target]
        assert [int(value) for value in execution["matching_row_indices"]] == expected
        assert int(out.answer_gt.value) == len(expected)


def test_table_counting_prompt_examples_match_selected_variant() -> None:
    cases = (
        (ChartsTableThresholdCountTask(), "above_threshold_count", 2),
        (ChartsTableThresholdCountTask(), "below_threshold_count", 2),
        (ChartsTableIntervalValueCountTask(), "single", 3),
        (ChartsTableCategoricalValueCountTask(), "single", 2),
    )
    for index, (task, query_id, answer) in enumerate(cases, start=18270):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=10)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation["answer"] == answer
        assert isinstance(answer_and_annotation["annotation"], list)
        assert answer_only == {"answer": answer}


def test_table_counting_tasks_are_deterministic() -> None:
    task = ChartsTableThresholdCountTask()
    params = {"query_id": "below_threshold_count", "scene_variant": "ledger"}
    out_a = task.generate(18310, params=params, max_attempts=10)
    out_b = task.generate(18310, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def trace_cell_boxes(out) -> list[list[float]]:
    render_map = out.trace_payload["render_map"]
    cell_ids = [str(cell_id) for cell_id in out.trace_payload["execution_trace"]["supporting_cell_ids"]]
    return [[float(value) for value in render_map["cell_bboxes_px"][cell_id]] for cell_id in cell_ids]
