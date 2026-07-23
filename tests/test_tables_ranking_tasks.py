"""Behavior tests for table ranking tasks."""

from __future__ import annotations

from trace_tasks.tasks.charts.table.column_rank_label import ChartsTableKthRankInColumnLabelTask
from tests.helpers import extract_prompt_json_example


def test_table_ranking_label_variants_match_contract() -> None:
    task = ChartsTableKthRankInColumnLabelTask()
    cases = (("highest_rank_in_column", "highest"), ("lowest_rank_in_column", "lowest"))
    for seed, (query_id, direction) in enumerate(cases, start=18610):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": "card_table"}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        values_by_row = execution["values_by_row"]
        row_labels = [str(label) for label in execution["row_labels"]]
        query_column = str(execution["query_column"])
        query_rank = int(execution["query_rank"])

        assert out.query_id == query_id
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "bbox"
        assert out.annotation_gt.value == out.trace_payload["render_map"]["cell_bboxes_px"][execution["supporting_cell_id"]]
        ranked = sorted(
            (
                {"row_label": label, "value": int(values_by_row[label][query_column])}
                for label in row_labels
            ),
            key=lambda item: item["value"],
            reverse=(direction == "highest"),
        )
        assert str(out.answer_gt.value) == str(ranked[query_rank - 1]["row_label"])


def test_table_ranking_label_prompt_examples_match_selected_variant() -> None:
    task = ChartsTableKthRankInColumnLabelTask()
    out = task.generate(18640, params={"query_id": "highest_rank_in_column"}, max_attempts=10)
    assert extract_prompt_json_example(out.prompt_variants["answer_and_annotation"]) == {
        "annotation": [260, 180, 372, 236],
        "answer": "Ava",
    }
    assert extract_prompt_json_example(out.prompt_variants["answer_only"]) == {"answer": "Ava"}


def test_table_ranking_label_task_is_deterministic() -> None:
    task = ChartsTableKthRankInColumnLabelTask()
    params = {"query_id": "lowest_rank_in_column", "scene_variant": "spreadsheet"}
    out_a = task.generate(18670, params=params, max_attempts=10)
    out_b = task.generate(18670, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
