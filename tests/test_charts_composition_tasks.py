"""Behavior tests for chart composition-style part-whole tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.charts.part_whole.adjacent_transfer_gap_value import ChartsCompositionChartAdjacentTransferGapValueTask
from trace_tasks.tasks.charts.part_whole.contiguous_chart_order_sum import ChartsCompositionChartContiguousOrderSumTask
from trace_tasks.tasks.charts.part_whole.sector_share_to_angle import ChartsCompositionChartSectorShareToAngleTask
from trace_tasks.tasks.charts.part_whole.subset_denominator_share_value import ChartsCompositionSubsetDenominatorShareValueTask


PART_WHOLE_TASKS = (
    ("task_charts__part_whole__adjacent_transfer_gap_value", ChartsCompositionChartAdjacentTransferGapValueTask),
    ("task_charts__part_whole__contiguous_chart_order_sum", ChartsCompositionChartContiguousOrderSumTask),
    ("task_charts__part_whole__sector_share_to_angle", ChartsCompositionChartSectorShareToAngleTask),
    ("task_charts__part_whole__subset_denominator_share_value", ChartsCompositionSubsetDenominatorShareValueTask),
)


def _part_whole_cases() -> list[tuple[str, type, str, str]]:
    cases: list[tuple[str, type, str, str]] = []
    for task_id, task_cls in PART_WHOLE_TASKS:
        for branch in task_cls.supported_query_ids:
            for scene_variant in ("pie", "donut"):
                cases.append((task_id, task_cls, str(branch), str(scene_variant)))
    return cases


def _expected_part_whole_answer(task_id: str, trace: dict) -> int:
    values_by_label = {str(label): int(value) for label, value in trace["category_values"].items()}
    if task_id.endswith("__contiguous_chart_order_sum"):
        selected_share = sum(values_by_label[str(label)] for label in trace["category_list"])
        assert int(trace["selected_share_value"]) == int(selected_share)
        return int(selected_share)
    if task_id.endswith("__subset_denominator_share_value"):
        selected_share = sum(values_by_label[str(label)] for label in trace["category_list"])
        target_share = int(values_by_label[str(trace["target_category"])])
        assert int(trace["selected_share_value"]) == int(selected_share)
        assert int(trace["subset_share_total"]) == int(selected_share)
        assert int(trace["target_share_value"]) == int(target_share)
        assert str(trace["target_category"]) in {str(label) for label in trace["category_list"]}
        assert 100 * int(target_share) % int(selected_share) == 0
        return int(100 * int(target_share) // int(selected_share))
    if task_id.endswith("__sector_share_to_angle"):
        selected_share = sum(values_by_label[str(label)] for label in trace["category_list"])
        assert int(trace["selected_share_value"]) == int(selected_share)
        assert int(selected_share) % 5 == 0
        return int(int(selected_share) * 360 // 100)
    if task_id.endswith("__adjacent_transfer_gap_value"):
        source_new = int(values_by_label[str(trace["source_category"])]) - int(trace["transfer_delta"])
        target_new = int(values_by_label[str(trace["target_category"])]) + int(trace["transfer_delta"])
        assert int(trace["source_new_value"]) == int(source_new)
        assert int(trace["target_new_value"]) == int(target_new)
        return int(abs(int(target_new) - int(source_new)))
    raise AssertionError(f"unsupported part-whole task: {task_id}")


@pytest.mark.parametrize(("task_id", "task_cls", "query_id", "scene_variant"), _part_whole_cases())
def test_part_whole_tasks_match_contract(task_id: str, task_cls: type, query_id: str, scene_variant: str) -> None:
    task = task_cls()
    out = task.generate(
        41000 + len(query_id) * 17 + len(scene_variant),
        params={"query_id": query_id, "scene_variant": scene_variant},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    assert out.query_id == query_id
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_map"
    assert str(execution["scene_variant"]) == str(scene_variant)
    assert str(render["scene_variant"]) == str(scene_variant)
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert int(execution["category_count"]) == len(execution["categories"])
    assert sum(int(category["value"]) for category in execution["categories"]) == 100
    assert execution["table_order_labels"] == sorted(execution["chart_order_labels"])
    assert int(out.answer_gt.value) == int(execution["answer_value"])
    assert int(out.answer_gt.value) == _expected_part_whole_answer(task_id, execution)
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == dict(out.annotation_gt.value)
    assert trace["projected_annotation"]["pixel_point_map"] == dict(out.annotation_gt.value)
    assert set(out.annotation_gt.value) == set(execution["annotation_keys"])
    assert len(trace["projected_annotation"]["point_set"]) == len(execution["annotation_labels"])
    assert len(trace["projected_annotation"]["bbox_set"]) <= len(execution["annotation_labels"])
    assert len(execution["annotation_values"]) == len(execution["annotation_labels"])
    for label, key in zip(execution["annotation_labels"], execution["annotation_keys"]):
        point = out.annotation_gt.value[str(key)]
        assert len(point) == 2
        assert 0 <= float(point[0]) <= int(render["canvas_width"])
        assert 0 <= float(point[1]) <= int(render["canvas_height"])
        if not str(label).startswith("__"):
            matching_trace = next(item for item in trace["render_map"]["chart_traces"] if str(item["label"]) == str(label))
            assert [float(value) for value in point] == [float(value) for value in matching_trace["slice_center_px"]]
    assert str(trace["query_spec"]["query_id"]) == str(query_id)
    assert trace["query_spec"]["params"]["query_id"] == str(query_id)
    assert len(trace["scene_ir"]["entities"]) >= int(execution["category_count"]) * 2


def test_part_whole_single_query_task_uses_single_sentinel() -> None:
    task = ChartsCompositionSubsetDenominatorShareValueTask()
    assert task.supported_query_ids == ("single",)
    out = task.generate(41700, params={}, max_attempts=100)
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["query_id"] == "single"


def test_part_whole_tasks_reject_unsupported_query_and_scene_variant() -> None:
    for _task_id, task_cls in PART_WHOLE_TASKS:
        task = task_cls()
        with pytest.raises(ValueError, match="query_id"):
            task.generate(41800, params={"query_id": "__unsupported_query_id__"}, max_attempts=10)
        with pytest.raises(ValueError):
            task.generate(41801, params={"scene_variant": "composition_pie_panels"}, max_attempts=10)


def test_part_whole_tasks_are_registered_and_deterministic() -> None:
    for task_id, _task_cls in PART_WHOLE_TASKS:
        assert task_id in TASK_REGISTRY
    task = ChartsCompositionChartSectorShareToAngleTask()
    params = {"query_id": "counterclockwise_sector_angle", "scene_variant": "donut"}
    out_a = task.generate(41900, params=params, max_attempts=100)
    out_b = task.generate(41900, params=params, max_attempts=100)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
