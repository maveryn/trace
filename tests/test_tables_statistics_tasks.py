"""Behavior tests for table statistics tasks."""

from __future__ import annotations

from trace_tasks.core.type_registry import load_type_registry
from trace_tasks.tasks.charts.table.column_summary_value import ChartsTableColumnSummaryValueTask
from trace_tasks.tasks.charts.table.filtered_column_mean import ChartsTableFilteredColumnMeanTask
from tests.helpers import extract_prompt_json_example


def test_table_column_summary_variants_match_contract() -> None:
    task = ChartsTableColumnSummaryValueTask()
    for seed, query_id in enumerate(("column_sum", "column_mean", "column_median"), start=18110):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": "zebra"}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        values_by_row = execution["values_by_row"]
        row_labels = [str(label) for label in execution["row_labels"]]
        query_column = str(execution["query_column"])
        query_values = [int(values_by_row[label][query_column]) for label in row_labels]

        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox"
        assert out.annotation_gt.value == out.trace_payload["render_map"]["column_region_bboxes_px"][query_column]
        assert out.trace_payload["projected_annotation"] == {"type": "bbox", "bbox": out.annotation_gt.value}
        if query_id == "column_sum":
            assert int(out.answer_gt.value) == sum(query_values)
        elif query_id == "column_mean":
            assert int(out.answer_gt.value) == sum(query_values) // len(query_values)
            assert sum(query_values) % len(query_values) == 0
        else:
            assert len(query_values) % 2 == 1
            assert int(out.answer_gt.value) == sorted(query_values)[len(query_values) // 2]


def test_table_filtered_mean_variants_match_contract() -> None:
    task = ChartsTableFilteredColumnMeanTask()
    for seed, query_id in enumerate(
        ("above_threshold_filtered_mean", "below_threshold_filtered_mean", "interval_filtered_mean"),
        start=18160,
    ):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": "ledger"}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        values_by_row = execution["values_by_row"]
        row_labels = [str(label) for label in execution["row_labels"]]
        filter_column = str(execution["filter_column"])
        target_column = str(execution["target_column"])
        selected = [int(index) for index in execution["selected_row_indices"]]

        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set_map"
        assert sorted(out.annotation_gt.value) == ["filter_cells", "target_cells"]
        assert out.trace_payload["projected_annotation"]["bbox_set_map"] == out.annotation_gt.value
        assert len(out.annotation_gt.value["filter_cells"]) == len(selected)
        assert len(out.annotation_gt.value["target_cells"]) == len(selected)

        filter_values = [int(values_by_row[label][filter_column]) for label in row_labels]
        if execution["filter_variant"] == "above_threshold":
            expected = [i for i, value in enumerate(filter_values) if value > int(execution["threshold_value"])]
        elif execution["filter_variant"] == "below_threshold":
            expected = [i for i, value in enumerate(filter_values) if value < int(execution["threshold_value"])]
        else:
            expected = [
                i
                for i, value in enumerate(filter_values)
                if int(execution["interval_min"]) <= value <= int(execution["interval_max"])
            ]
        assert selected == expected
        target_values = [int(values_by_row[row_labels[index]][target_column]) for index in selected]
        assert int(out.answer_gt.value) == sum(target_values) // len(target_values)
        assert sum(target_values) % len(target_values) == 0


def test_table_statistics_prompt_examples_match_selected_variant() -> None:
    summary = ChartsTableColumnSummaryValueTask().generate(18220, params={"query_id": "column_mean"}, max_attempts=10)
    filtered = ChartsTableFilteredColumnMeanTask().generate(
        18221,
        params={"query_id": "interval_filtered_mean"},
        max_attempts=10,
    )
    assert extract_prompt_json_example(summary.prompt_variants["answer_and_annotation"]) == {
        "annotation": [260, 180, 372, 520],
        "answer": 14,
    }
    assert extract_prompt_json_example(filtered.prompt_variants["answer_and_annotation"]) == {
        "annotation": {
            "filter_cells": [[260, 180, 372, 236], [260, 236, 372, 292]],
            "target_cells": [[374, 180, 486, 236], [374, 236, 486, 292]],
        },
        "answer": 14,
    }


def test_table_statistics_task_is_deterministic() -> None:
    task = ChartsTableFilteredColumnMeanTask()
    params = {"query_id": "above_threshold_filtered_mean", "scene_variant": "spreadsheet"}
    out_a = task.generate(18260, params=params, max_attempts=10)
    out_b = task.generate(18260, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_table_statistics_registers_annotation_types() -> None:
    registry = load_type_registry()
    assert registry.validate_answer_type("integer") is True
    assert registry.validate_annotation_type("bbox") is True
    assert registry.validate_annotation_type("bbox_set_map") is True
