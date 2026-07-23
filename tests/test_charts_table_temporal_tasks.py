"""Behavior tests for chart table temporal tasks."""

from __future__ import annotations

from trace_tasks.tasks.charts.table.absolute_difference_between_rows_over_year_interval import (
    ChartsTableAbsoluteDifferenceBetweenRowsOverYearIntervalTask,
)
from trace_tasks.tasks.charts.table.sum_absolute_differences_between_rows_over_year_interval import (
    ChartsTableSumAbsoluteDifferencesBetweenRowsOverYearIntervalTask,
)
from tests.helpers import extract_prompt_json_example


def test_table_temporal_tasks_match_queried_year_cells() -> None:
    cases = (
        (ChartsTableAbsoluteDifferenceBetweenRowsOverYearIntervalTask(), "absolute_difference_between_rows_over_year_interval"),
        (ChartsTableSumAbsoluteDifferencesBetweenRowsOverYearIntervalTask(), "sum_absolute_differences_between_rows_over_year_interval"),
    )
    for seed, (task, operation) in enumerate(cases, start=19010):
        out = task.generate(seed, params={"scene_variant": "spreadsheet"}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        values_by_row = execution["values_by_row"]
        query_years = [str(year) for year in execution["query_years"]]
        query_cells = [dict(cell) for cell in execution["query_cells"]]
        cell_ids = [str(cell_id) for cell_id in execution["supporting_cell_ids"]]

        assert out.query_id == "single"
        assert execution["operation"] == operation
        assert out.answer_gt.type == "integer"
        if operation.startswith("absolute_difference"):
            assert out.annotation_gt.type == "bbox_map"
            expected_map = {}
            for row_label, row_cell_ids in execution["supporting_cell_ids_by_row"].items():
                boxes = [
                    [float(value) for value in out.trace_payload["render_map"]["cell_bboxes_px"][str(cell_id)]]
                    for cell_id in row_cell_ids
                ]
                expected_map[str(row_label)] = [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ]
            assert out.annotation_gt.value == expected_map
        else:
            assert out.annotation_gt.type == "bbox_set"
            assert out.annotation_gt.value == [
                [float(value) for value in out.trace_payload["render_map"]["cell_bboxes_px"][cell_id]]
                for cell_id in cell_ids
            ]
        assert len(query_cells) == len(cell_ids) == 2 * len(query_years)
        row_a = str(execution["query_row_label_a"])
        row_b = str(execution["query_row_label_b"])
        values_a = [int(values_by_row[row_a][str(cell["column"])]) for cell in query_cells[: len(query_years)]]
        values_b = [int(values_by_row[row_b][str(cell["column"])]) for cell in query_cells[len(query_years) :]]
        if operation.startswith("absolute_difference"):
            assert int(out.answer_gt.value) == abs(sum(values_a) - sum(values_b))
        else:
            diffs = [abs(a - b) for a, b in zip(values_a, values_b)]
            assert [int(value) for value in execution["paired_absolute_differences"]] == diffs
            assert int(out.answer_gt.value) == sum(diffs)


def test_table_temporal_prompt_examples_match_selected_variants() -> None:
    cases = (
        (ChartsTableAbsoluteDifferenceBetweenRowsOverYearIntervalTask(), 7),
        (ChartsTableSumAbsoluteDifferencesBetweenRowsOverYearIntervalTask(), 17),
    )
    for index, (task, answer) in enumerate(cases, start=19040):
        out = task.generate(index, params={}, max_attempts=10)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation["answer"] == answer
        if isinstance(task, ChartsTableAbsoluteDifferenceBetweenRowsOverYearIntervalTask):
            assert isinstance(answer_and_annotation["annotation"], dict)
            row_labels = out.trace_payload["execution_trace"]["query_row_labels"]
            assert sorted(answer_and_annotation["annotation"]) == sorted(row_labels)
        else:
            assert isinstance(answer_and_annotation["annotation"], list)
        assert answer_only == {"answer": answer}


def test_table_temporal_value_task_is_deterministic() -> None:
    task = ChartsTableSumAbsoluteDifferencesBetweenRowsOverYearIntervalTask()
    params = {"scene_variant": "ledger"}
    out_a = task.generate(19080, params=params, max_attempts=10)
    out_b = task.generate(19080, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
