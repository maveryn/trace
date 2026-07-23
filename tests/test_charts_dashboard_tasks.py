"""Behavior tests for mixed-dashboard chart tasks."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task
from trace_tasks.tasks.charts.dashboard.category_extremum_panel_label import ChartsDashboardCategoryExtremumPanelLabelTask
from trace_tasks.tasks.charts.dashboard.category_panel_condition_count import ChartsDashboardCategoryPanelConditionCountTask
from trace_tasks.tasks.charts.dashboard.category_total_extremum_label import ChartsDashboardCategoryTotalExtremumLabelTask
from trace_tasks.tasks.charts.dashboard.global_value_extremum_category_label import ChartsDashboardGlobalValueExtremumCategoryLabelTask
from trace_tasks.tasks.charts.dashboard.panel_total_extremum_label import ChartsDashboardPanelTotalExtremumLabelTask
from trace_tasks.tasks.charts.dashboard.panel_value_range_extremum_label import ChartsDashboardPanelValueRangeExtremumLabelTask
from trace_tasks.tasks.charts.dashboard.panel_value_range_value import ChartsDashboardPanelValueRangeValueTask
from trace_tasks.tasks.charts.dashboard.shared.state import SUPPORTED_PANEL_KINDS, SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.charts.dashboard.source_rank_target_value import ChartsDashboardSourceRankTargetValueTask
from trace_tasks.tasks.charts.dashboard.statement_option_selection_label import ChartsDashboardStatementOptionSelectionLabelTask

TASK_CASES = (
    ("task_charts__dashboard__category_extremum_panel_label", ChartsDashboardCategoryExtremumPanelLabelTask, "largest_category_panel_label", "string", "point"),
    ("task_charts__dashboard__category_panel_condition_count", ChartsDashboardCategoryPanelConditionCountTask, "category_panel_greater_than_threshold_count", "integer", "point_set"),
    ("task_charts__dashboard__category_total_extremum_label", ChartsDashboardCategoryTotalExtremumLabelTask, "largest_category_total_label", "string", "point_set"),
    ("task_charts__dashboard__global_value_extremum_category_label", ChartsDashboardGlobalValueExtremumCategoryLabelTask, "global_maximum_value_category_label", "string", "point"),
    ("task_charts__dashboard__panel_total_extremum_label", ChartsDashboardPanelTotalExtremumLabelTask, "largest_panel_total_label", "string", "point_set"),
    ("task_charts__dashboard__panel_value_range_extremum_label", ChartsDashboardPanelValueRangeExtremumLabelTask, "largest_panel_value_range_label", "string", "point_map"),
    ("task_charts__dashboard__panel_value_range_value", ChartsDashboardPanelValueRangeValueTask, "single", "integer", "point_map"),
    ("task_charts__dashboard__source_rank_target_value", ChartsDashboardSourceRankTargetValueTask, "largest_source_rank_target_value", "integer", "point_map"),
    ("task_charts__dashboard__statement_option_selection_label", ChartsDashboardStatementOptionSelectionLabelTask, "statement_option_selection_label", "option_letter", "point_set"),
)
TASK_IDS = tuple(case[0] for case in TASK_CASES)
SOURCE_RANK_TARGET_QUERY_IDS = {"largest_source_rank_target_value", "smallest_source_rank_target_value"}
CATEGORY_PANEL_CONDITION_QUERY_IDS = {
    "category_panel_greater_than_threshold_count",
    "category_panel_less_than_threshold_count",
}
CATEGORY_TOTAL_EXTREMUM_QUERY_IDS = {
    "largest_category_total_label",
    "smallest_category_total_label",
}
CATEGORY_EXTREMUM_PANEL_QUERY_IDS = {
    "largest_category_panel_label",
    "smallest_category_panel_label",
}
PANEL_TOTAL_EXTREMUM_QUERY_IDS = {
    "largest_panel_total_label",
    "smallest_panel_total_label",
}
PANEL_VALUE_RANGE_EXTREMUM_QUERY_IDS = {
    "largest_panel_value_range_label",
    "smallest_panel_value_range_label",
}
GLOBAL_VALUE_EXTREMUM_CATEGORY_QUERY_IDS = {
    "global_maximum_value_category_label",
    "global_minimum_value_category_label",
}
PANEL_TITLE_WORD_RE = re.compile(r"\bpanel\b", flags=re.IGNORECASE)


def _annotation_format_index(prompt: str) -> int:
    for marker in ('Annotation format:', 'Required annotation format:', 'Format for the "annotation" field'):
        if marker in prompt:
            return int(prompt.index(marker))
    raise AssertionError("prompt is missing an annotation format line")


def _task_params_for_query(task: Any, query_id: str) -> dict[str, str]:
    supported = set(str(value) for value in getattr(task, "supported_query_ids", ()))
    if str(query_id) in supported:
        return {"query_id": str(query_id)}
    return {}


def _semantic_variant(execution: dict[str, Any]) -> str:
    return str(execution.get("internal_query_id") or execution["query_id"])


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x < width
    assert 0 <= y < height


def _value(execution: dict[str, Any], panel_id: str, category_id: str) -> int:
    return int(execution["values_by_panel"][str(panel_id)]["values_by_category_id"][str(category_id)])


def _expected_answer(execution: dict[str, Any]) -> int | str:
    if str(execution.get("answerability", "answerable")) == "unanswerable":
        return "unanswerable"
    variant = _semantic_variant(execution)
    if variant in SOURCE_RANK_TARGET_QUERY_IDS:
        return _value(execution, execution["target_panel_id"], execution["selected_category_id"])
    if variant in CATEGORY_EXTREMUM_PANEL_QUERY_IDS:
        category_id = str(execution["target_category_id"])
        direction = str(execution["extremum_direction"])
        panel_values = {str(panel["panel_id"]): _value(execution, str(panel["panel_id"]), category_id) for panel in execution["panels"]}
        target_value = max(panel_values.values()) if direction == "largest" else min(panel_values.values())
        assert sum(1 for value in panel_values.values() if int(value) == int(target_value)) == 1
        answer_panel_id = next(panel_id for panel_id, value in panel_values.items() if int(value) == int(target_value))
        assert str(execution["answer_panel_id"]) == str(answer_panel_id)
        assert int(execution["answer_value"]) == int(target_value)
        panel_names = {str(panel["panel_id"]): str(panel["panel_name"]) for panel in execution["panels"]}
        return panel_names[str(answer_panel_id)]
    if variant in CATEGORY_TOTAL_EXTREMUM_QUERY_IDS:
        direction = str(execution["category_total_extremum_direction"])
        category_totals = {
            str(category["category_id"]): sum(
                _value(execution, str(panel["panel_id"]), str(category["category_id"]))
                for panel in execution["panels"]
            )
            for category in execution["categories"]
        }
        target_total = max(category_totals.values()) if direction == "largest" else min(category_totals.values())
        assert sum(1 for value in category_totals.values() if int(value) == int(target_total)) == 1
        answer_category_id = next(category_id for category_id, value in category_totals.items() if int(value) == int(target_total))
        assert str(execution["answer_category_id"]) == str(answer_category_id)
        assert int(execution["answer_category_total"]) == int(target_total)
        labels = {str(category["category_id"]): str(category["label"]) for category in execution["categories"]}
        return labels[str(answer_category_id)]
    if variant in PANEL_TOTAL_EXTREMUM_QUERY_IDS:
        direction = str(execution["panel_total_extremum_direction"])
        panel_totals = {
            str(panel["panel_id"]): sum(
                _value(execution, str(panel["panel_id"]), str(category["category_id"]))
                for category in execution["categories"]
            )
            for panel in execution["panels"]
        }
        target_total = max(panel_totals.values()) if direction == "largest" else min(panel_totals.values())
        assert sum(1 for value in panel_totals.values() if int(value) == int(target_total)) == 1
        answer_panel_id = next(panel_id for panel_id, value in panel_totals.items() if int(value) == int(target_total))
        assert str(execution["answer_panel_id"]) == str(answer_panel_id)
        assert int(execution["answer_panel_total"]) == int(target_total)
        panel_names = {str(panel["panel_id"]): str(panel["panel_name"]) for panel in execution["panels"]}
        return panel_names[str(answer_panel_id)]
    if variant in PANEL_VALUE_RANGE_EXTREMUM_QUERY_IDS:
        direction = str(execution["range_extremum_direction"])
        ranges = {
            str(panel["panel_id"]): max(_value(execution, str(panel["panel_id"]), str(category["category_id"])) for category in execution["categories"])
            - min(_value(execution, str(panel["panel_id"]), str(category["category_id"])) for category in execution["categories"])
            for panel in execution["panels"]
        }
        target_range = max(ranges.values()) if direction == "largest" else min(ranges.values())
        assert sum(1 for value in ranges.values() if int(value) == int(target_range)) == 1
        answer_panel_id = next(panel_id for panel_id, value in ranges.items() if int(value) == int(target_range))
        assert str(execution["answer_panel_id"]) == str(answer_panel_id)
        assert int(execution["answer_range_value"]) == int(target_range)
        panel_names = {str(panel["panel_id"]): str(panel["panel_name"]) for panel in execution["panels"]}
        return panel_names[str(answer_panel_id)]
    if variant == "single" and str(execution.get("range_operation")) == "panel_max_minus_min":
        panel_id = str(execution["selected_panel_id"])
        largest_category_id = str(execution["largest_category_id"])
        smallest_category_id = str(execution["smallest_category_id"])
        largest_value = _value(execution, panel_id, largest_category_id)
        smallest_value = _value(execution, panel_id, smallest_category_id)
        assert int(execution["largest_value"]) == int(largest_value)
        assert int(execution["smallest_value"]) == int(smallest_value)
        assert int(execution["range_value"]) == int(largest_value) - int(smallest_value)
        return int(execution["range_value"])
    if variant in GLOBAL_VALUE_EXTREMUM_CATEGORY_QUERY_IDS:
        direction = str(execution["global_extremum_direction"])
        values = {
            (str(panel["panel_id"]), str(category["category_id"])): _value(execution, str(panel["panel_id"]), str(category["category_id"]))
            for panel in execution["panels"]
            for category in execution["categories"]
        }
        target_value = max(values.values()) if direction == "maximum" else min(values.values())
        assert sum(1 for value in values.values() if int(value) == int(target_value)) == 1
        answer_panel_id, answer_category_id = next(ref for ref, value in values.items() if int(value) == int(target_value))
        assert str(execution["answer_panel_id"]) == str(answer_panel_id)
        assert str(execution["answer_category_id"]) == str(answer_category_id)
        assert int(execution["answer_value"]) == int(target_value)
        labels = {str(category["category_id"]): str(category["label"]) for category in execution["categories"]}
        return labels[str(answer_category_id)]
    if variant == "statement_option_selection_label":
        requested_truth = str(execution["requested_truth"]) == "true"
        matching_options: list[dict[str, Any]] = []
        for option in execution["statement_options"]:
            first_value = _value(execution, str(option["first_panel_id"]), str(option["first_category_id"]))
            second_value = _value(execution, str(option["second_panel_id"]), str(option["second_category_id"]))
            if str(option["comparison"]) == "greater_than":
                truth_value = first_value > second_value
            elif str(option["comparison"]) == "less_than":
                truth_value = first_value < second_value
            else:
                raise AssertionError(f"unsupported comparison: {option['comparison']}")
            assert bool(option["truth_value"]) is bool(truth_value)
            if bool(truth_value) is bool(requested_truth):
                matching_options.append(option)
        assert len(matching_options) == 1
        selected = matching_options[0]
        assert selected == execution["selected_statement"]
        assert str(selected["option_label"]) == str(execution["answer_option_label"])
        return str(selected["option_label"])
    if variant in CATEGORY_PANEL_CONDITION_QUERY_IDS:
        category_id = str(execution["condition_category_id"])
        threshold = int(execution["panel_threshold"])
        comparison = str(execution["panel_condition_comparison"])
        total = 0
        for panel in execution["panels"]:
            panel_id = str(panel["panel_id"])
            value = _value(execution, panel_id, category_id)
            match = value > threshold if comparison == "greater_than" else value < threshold
            total += int(bool(match))
        return int(total)
    raise AssertionError(f"unsupported variant: {variant}")


@pytest.mark.parametrize("case_index, case", tuple(enumerate(TASK_CASES)))
def test_chart_dashboard_tasks_match_contract(case_index: int, case: tuple[str, type, str, str, str]) -> None:
    task_id, task_cls, query_id, answer_type, annotation_type = case
    task = task_cls()
    out = task.generate(
        hash64(20260503, "charts_dashboard", case_index),
        params=_task_params_for_query(task, query_id),
        max_attempts=120,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    assert task.task_id == task_id
    if str(query_id) in set(str(value) for value in task.supported_query_ids):
        assert out.query_id == query_id
    else:
        assert out.query_id == "single"
    assert _semantic_variant(execution) == query_id
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert str(execution["question_format"]) == "dashboard_cross_panel_query"
    assert out.annotation_gt.type == annotation_type
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert "Styles may be" not in out.prompt
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 4 <= int(execution["category_count"]) <= 10
    assert all(len(str(category["label"])) <= 6 for category in execution["categories"])
    assert 4 <= int(execution["panel_count"]) <= 9
    assert len(execution["panel_order"]) == int(execution["panel_count"])
    assert len({str(item) for item in execution["panel_order"]}) == int(execution["panel_count"])
    assert all(str(kind) in SUPPORTED_PANEL_KINDS for kind in execution["panel_kinds"])
    assert len(execution["panels"]) == int(execution["panel_count"])
    assert all(PANEL_TITLE_WORD_RE.search(str(panel["panel_name"])) is None for panel in execution["panels"])
    expected_answer = _expected_answer(execution)
    assert out.answer_gt.type == answer_type
    assert out.answer_gt.value == expected_answer
    assert execution["answer"] == expected_answer
    expected_points = [list(trace["render_map"]["support_points_px"][str(panel_id)][str(category_id)]) for panel_id, category_id in execution["annotation_refs"]]
    assert trace["projected_annotation"]["type"] == out.annotation_gt.type
    assert len(trace["projected_annotation"]["annotation_refs"]) == len(expected_points)
    if out.annotation_gt.type == "point":
        assert out.annotation_gt.value == expected_points[0]
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        _assert_point_inside_canvas([float(value) for value in out.annotation_gt.value], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    elif out.annotation_gt.type == "point_set":
        assert out.annotation_gt.value == expected_points
        assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
        for point in out.annotation_gt.value:
            _assert_point_inside_canvas([float(value) for value in point], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    else:
        assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
        for point in out.annotation_gt.value.values():
            _assert_point_inside_canvas([float(value) for value in point], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
        if query_id in SOURCE_RANK_TARGET_QUERY_IDS:
            assert out.annotation_gt.value == {"source_panel": expected_points[0], "target_panel": expected_points[1]}
        if query_id in PANEL_VALUE_RANGE_EXTREMUM_QUERY_IDS:
            assert out.annotation_gt.value == {"largest_value": expected_points[0], "smallest_value": expected_points[1]}
            answer_panel_id = str(execution["answer_panel_id"])
            selected_values = execution["values_by_panel"][answer_panel_id]["values_by_category_id"]
            assert int(execution["largest_value"]) == max(int(value) for value in selected_values.values())
            assert int(execution["smallest_value"]) == min(int(value) for value in selected_values.values())
        if query_id == "single" and str(execution.get("range_operation")) == "panel_max_minus_min":
            assert out.annotation_gt.value == {"largest_value": expected_points[0], "smallest_value": expected_points[1]}
            selected_values = execution["values_by_panel"][str(execution["selected_panel_id"])]["values_by_category_id"]
            assert int(execution["largest_value"]) == max(int(value) for value in selected_values.values())
            assert int(execution["smallest_value"]) == min(int(value) for value in selected_values.values())
    if query_id in CATEGORY_PANEL_CONDITION_QUERY_IDS:
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    if query_id in CATEGORY_TOTAL_EXTREMUM_QUERY_IDS:
        assert len(out.annotation_gt.value) == int(execution["panel_count"])
    if query_id in PANEL_TOTAL_EXTREMUM_QUERY_IDS:
        assert len(out.annotation_gt.value) == int(execution["category_count"])
    if query_id == "statement_option_selection_label":
        assert int(execution["option_count"]) in {4, 6}
        assert len(execution["statement_options"]) == int(execution["option_count"])
        assert out.answer_gt.value in execution["option_labels"]
        for option in execution["statement_options"]:
            assert str(option["text"]) not in out.prompt
            assert f"option_{option['option_label']}" in trace["render_map"]["option_statement_bboxes_px"]


@pytest.mark.parametrize("task_id, task_cls, query_id, answer_type, annotation_type", TASK_CASES)
def test_chart_dashboard_prompt_examples_match_contract(task_id: str, task_cls: type, query_id: str, answer_type: str, annotation_type: str) -> None:
    del task_id, query_id, annotation_type
    out = task_cls().generate(hash64(93100, task_cls.__name__), params={}, max_attempts=120)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert "annotation" in answer_and_annotation
    if answer_type == "integer":
        assert isinstance(answer_and_annotation["answer"], int)
        assert isinstance(answer_only["answer"], int)
    else:
        assert isinstance(answer_and_annotation["answer"], str)
        assert isinstance(answer_only["answer"], str)


@pytest.mark.parametrize("option_count", [4, 6])
@pytest.mark.parametrize("requested_truth", ["true", "false"])
def test_chart_dashboard_statement_option_selection_contract(option_count: int, requested_truth: str) -> None:
    task = ChartsDashboardStatementOptionSelectionLabelTask()
    out = task.generate(hash64(20260604, "charts_dashboard_statement_option", option_count, requested_truth), params={"option_count": option_count, "requested_truth": requested_truth}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == "single"
    assert _semantic_variant(execution) == "statement_option_selection_label"
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "point_set"
    assert int(execution["option_count"]) == int(option_count)
    assert str(execution["requested_truth"]) == str(requested_truth)
    assert len(execution["statement_options"]) == int(option_count)
    assert len(trace["render_map"]["option_statement_bboxes_px"]) == int(option_count)
    expected_answer = _expected_answer(execution)
    assert out.answer_gt.value == expected_answer
    selected = execution["selected_statement"]
    expected_refs = [[str(selected["first_panel_id"]), str(selected["first_category_id"])], [str(selected["second_panel_id"]), str(selected["second_category_id"])]]
    assert execution["annotation_refs"] == expected_refs
    expected_points = [list(trace["render_map"]["support_points_px"][panel_id][category_id]) for panel_id, category_id in expected_refs]
    assert out.annotation_gt.value == expected_points
    for option in execution["statement_options"]:
        assert str(option["text"]) not in out.prompt


def test_chart_dashboard_panel_value_range_contract() -> None:
    task = ChartsDashboardPanelValueRangeValueTask()
    out = task.generate(hash64(20260615, "charts_dashboard_panel_value_range"), params={"target_answer": 37}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == "single"
    assert _semantic_variant(execution) == "single"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 37
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"largest_value", "smallest_value"}
    selected_panel_id = str(execution["selected_panel_id"])
    values = execution["values_by_panel"][selected_panel_id]["values_by_category_id"]
    assert int(execution["largest_value"]) == max(int(value) for value in values.values())
    assert int(execution["smallest_value"]) == min(int(value) for value in values.values())
    assert int(execution["largest_value"]) - int(execution["smallest_value"]) == 37
    refs = execution["annotation_refs"]
    expected_points = [list(trace["render_map"]["support_points_px"][panel_id][category_id]) for panel_id, category_id in refs]
    assert out.annotation_gt.value == {"largest_value": expected_points[0], "smallest_value": expected_points[1]}


@pytest.mark.parametrize("query_index,query_id,direction", [(0, "largest_panel_value_range_label", "largest"), (1, "smallest_panel_value_range_label", "smallest")])
def test_chart_dashboard_panel_value_range_extremum_contract(query_index: int, query_id: str, direction: str) -> None:
    task = ChartsDashboardPanelValueRangeExtremumLabelTask()
    out = task.generate(hash64(20260615, "charts_dashboard_panel_value_range_extremum", query_index), params={"query_id": query_id}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == query_id
    assert _semantic_variant(execution) == query_id
    assert str(execution["range_extremum_direction"]) == str(direction)
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point_map"
    assert out.answer_gt.value == _expected_answer(execution)
    answer_panel_id = str(execution["answer_panel_id"])
    values = execution["values_by_panel"][answer_panel_id]["values_by_category_id"]
    assert int(execution["largest_value"]) == max(int(value) for value in values.values())
    assert int(execution["smallest_value"]) == min(int(value) for value in values.values())
    assert int(execution["answer_range_value"]) == int(execution["largest_value"]) - int(execution["smallest_value"])
    refs = execution["annotation_refs"]
    expected_points = [list(trace["render_map"]["support_points_px"][panel_id][category_id]) for panel_id, category_id in refs]
    assert out.annotation_gt.value == {"largest_value": expected_points[0], "smallest_value": expected_points[1]}


@pytest.mark.parametrize("query_index,query_id,direction", [(0, "largest_category_total_label", "largest"), (1, "smallest_category_total_label", "smallest")])
def test_chart_dashboard_category_total_extremum_contract(query_index: int, query_id: str, direction: str) -> None:
    task = ChartsDashboardCategoryTotalExtremumLabelTask()
    out = task.generate(hash64(20260615, "charts_dashboard_category_total_extremum", query_index), params={"query_id": query_id}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == query_id
    assert _semantic_variant(execution) == query_id
    assert str(execution["category_total_extremum_direction"]) == str(direction)
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point_set"
    assert out.answer_gt.value == _expected_answer(execution)
    category_id = str(execution["answer_category_id"])
    expected_refs = [[str(panel["panel_id"]), category_id] for panel in execution["panels"]]
    assert execution["annotation_refs"] == expected_refs
    expected_points = [list(trace["render_map"]["support_points_px"][panel_id][category_id]) for panel_id, category_id in expected_refs]
    assert out.annotation_gt.value == expected_points
    assert int(execution["answer_category_total"]) == sum(
        _value(execution, str(panel["panel_id"]), category_id)
        for panel in execution["panels"]
    )


@pytest.mark.parametrize("query_id,direction", [("largest_category_total_label", "largest"), ("smallest_category_total_label", "smallest")])
def test_chart_dashboard_category_total_extremum_supports_unanswerable_missing_category(query_id: str, direction: str) -> None:
    task = ChartsDashboardCategoryTotalExtremumLabelTask()
    query_index = 0 if str(query_id) == "largest_category_total_label" else 1
    out = task.generate(
        hash64(20260616, "charts_dashboard_category_total_unanswerable", query_index),
        params={"query_id": query_id, "force_unanswerable": True},
        max_attempts=160,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scene_relations = trace["scene_ir"]["relations"]
    witness = trace["witness_symbolic"]
    projected = trace["projected_annotation"]

    assert out.query_id == query_id
    assert _semantic_variant(execution) == query_id
    assert str(execution["category_total_extremum_direction"]) == str(direction)
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "unanswerable"
    assert execution["answer"] == "unanswerable"
    assert out.annotation_gt.type == "point_set"
    assert out.annotation_gt.value == []
    assert execution["annotation_refs"] == []
    assert projected["type"] == "point_set"
    assert projected["point_set"] == []
    assert projected["pixel_point_set"] == []
    assert projected["annotation_refs"] == []
    assert execution["answerability"] == "unanswerable"
    assert scene_relations["answerability"] == "unanswerable"
    assert witness["answerability"] == "unanswerable"

    missing_category_id = str(execution["missing_category_id"])
    missing_category_label = str(execution["missing_category_label"])
    missing_panel_id = str(execution["missing_category_panel_id"])
    assert missing_category_label in {str(category["label"]) for category in execution["categories"]}
    assert missing_category_id not in execution["values_by_panel"][missing_panel_id]["values_by_category_id"]
    assert missing_category_id not in trace["render_map"]["support_points_px"][missing_panel_id]
    for panel in execution["panels"]:
        panel_id = str(panel["panel_id"])
        presence = bool(execution["category_presence_by_panel_id"][panel_id])
        assert presence is (panel_id != missing_panel_id)
        if panel_id != missing_panel_id:
            assert missing_category_id in execution["values_by_panel"][panel_id]["values_by_category_id"]
            assert missing_category_id in trace["render_map"]["support_points_px"][panel_id]
    assert execution["absence_proof"]["requested_item"] == (
        f"{missing_category_label} in every dashboard panel"
    )
    assert "unanswerable" in out.prompt.lower()
    assert "not shown in every dashboard panel" in out.prompt.lower()
    assert out.prompt.index("not shown in every dashboard panel") < _annotation_format_index(out.prompt)
    assert "All panels share the same" not in out.prompt


@pytest.mark.parametrize("query_index,query_id,direction", [(0, "largest_panel_total_label", "largest"), (1, "smallest_panel_total_label", "smallest")])
def test_chart_dashboard_panel_total_extremum_contract(query_index: int, query_id: str, direction: str) -> None:
    task = ChartsDashboardPanelTotalExtremumLabelTask()
    out = task.generate(hash64(20260615, "charts_dashboard_panel_total_extremum", query_index), params={"query_id": query_id}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == query_id
    assert _semantic_variant(execution) == query_id
    assert str(execution["panel_total_extremum_direction"]) == str(direction)
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point_set"
    assert out.answer_gt.value == _expected_answer(execution)
    panel_id = str(execution["answer_panel_id"])
    expected_refs = [[panel_id, str(category["category_id"])] for category in execution["categories"]]
    assert execution["annotation_refs"] == expected_refs
    expected_points = [list(trace["render_map"]["support_points_px"][panel_id][category_id]) for panel_id, category_id in expected_refs]
    assert out.annotation_gt.value == expected_points
    assert int(execution["answer_panel_total"]) == sum(
        _value(execution, panel_id, str(category["category_id"]))
        for category in execution["categories"]
    )


@pytest.mark.parametrize("query_index,query_id,direction", [(0, "global_maximum_value_category_label", "maximum"), (1, "global_minimum_value_category_label", "minimum")])
def test_chart_dashboard_global_value_extremum_category_contract(query_index: int, query_id: str, direction: str) -> None:
    task = ChartsDashboardGlobalValueExtremumCategoryLabelTask()
    out = task.generate(hash64(20260615, "charts_dashboard_global_value_extremum_category", query_index), params={"query_id": query_id}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == query_id
    assert _semantic_variant(execution) == query_id
    assert str(execution["global_extremum_direction"]) == str(direction)
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point"
    assert out.answer_gt.value == _expected_answer(execution)
    expected_point = list(trace["render_map"]["support_points_px"][str(execution["answer_panel_id"])][str(execution["answer_category_id"])])
    assert out.annotation_gt.value == expected_point
    assert trace["projected_annotation"]["point"] == expected_point


@pytest.mark.parametrize("query_index,query_id,direction", [(0, "largest_category_panel_label", "largest"), (1, "smallest_category_panel_label", "smallest")])
def test_chart_dashboard_category_extremum_panel_contract(query_index: int, query_id: str, direction: str) -> None:
    task = ChartsDashboardCategoryExtremumPanelLabelTask()
    out = task.generate(hash64(20260615, "charts_dashboard_category_extremum_panel", query_index), params={"query_id": query_id}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    assert out.query_id == query_id
    assert _semantic_variant(execution) == query_id
    assert str(execution["extremum_direction"]) == str(direction)
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point"
    assert out.answer_gt.value == _expected_answer(execution)
    category_id = str(execution["target_category_id"])
    values = {
        str(panel["panel_id"]): _value(execution, str(panel["panel_id"]), category_id)
        for panel in execution["panels"]
    }
    target_value = max(values.values()) if str(direction) == "largest" else min(values.values())
    assert int(execution["answer_value"]) == int(target_value)
    assert sum(1 for value in values.values() if int(value) == int(target_value)) == 1
    expected_point = list(trace["render_map"]["support_points_px"][str(execution["answer_panel_id"])][category_id])
    assert out.annotation_gt.value == expected_point
    assert trace["projected_annotation"]["point"] == expected_point


def test_chart_dashboard_all_tasks_are_deterministic() -> None:
    for task_index, (_task_id, task_cls, query_id, _answer_type, _annotation_type) in enumerate(TASK_CASES):
        task = task_cls()
        params = _task_params_for_query(task, query_id)
        out_a = task.generate(hash64(93300, task_index), params=params, max_attempts=120)
        out_b = task.generate(hash64(93300, task_index), params=params, max_attempts=120)
        assert out_a.prompt == out_b.prompt
        assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
        assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
        assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


@pytest.mark.parametrize("case_index, case", tuple(enumerate(TASK_CASES)))
def test_chart_dashboard_tasks_generate_every_query_branch(case_index: int, case: tuple[str, type, str, str, str]) -> None:
    task_id, task_cls, _default_query_id, answer_type, _annotation_type = case
    task = task_cls()
    assert create_task(task_id).task_id == task_id
    for query_index, query_id in enumerate(task.supported_query_ids):
        out = task.generate(hash64(20260613, "charts_dashboard_query_branch", case_index, query_index), params={"query_id": query_id}, max_attempts=180)
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == query_id
        assert str(execution["query_id"]) == str(query_id)
        assert str(out.trace_payload["query_spec"]["params"]["query_id"]) == str(query_id)
        assert out.answer_gt.type == answer_type
        assert out.answer_gt.value == _expected_answer(execution)


def test_chart_dashboard_registered_and_scene_config_loaded() -> None:
    for task_id in TASK_IDS:
        assert create_task(task_id).task_id == task_id
    cfg = get_scene_defaults("charts", "dashboard")
    assert isinstance(cfg.get("generation"), dict)
    assert isinstance(cfg.get("rendering"), dict)
    assert isinstance(cfg.get("prompt"), dict)
    generation = cfg["generation"]["shared"]
    assert "query_id_weights" not in generation
    assert "rank_direction_weights" not in generation
    assert "gap_extremum_weights" not in generation
    assert "condition_comparison_weights" not in generation
    assert int(generation["panel_count_min"]) == 4
    assert int(generation["panel_count_max"]) == 9
    assert int(generation["category_count_min"]) == 4
    assert int(generation["category_count_max"]) == 10
    assert int(generation["category_label_min_chars"]) == 2
    assert int(generation["category_label_max_chars"]) == 6
    assert sorted(generation["panel_kind_weights"].keys()) == sorted(SUPPORTED_PANEL_KINDS)
    prompt = cfg["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_dashboard_v1"
    assert str(prompt["scene_key"]) == "dashboard_mixed_chart"
    assert str(prompt["task_key"]) == "dashboard_cross_panel_query"


def test_chart_dashboard_sampling_covers_scene_ranges() -> None:
    category_counts: Counter[int] = Counter()
    panel_counts: Counter[int] = Counter()
    answer_types: Counter[str] = Counter()
    for task_index, (_task_id, task_cls, query_id, _answer_type, _annotation_type) in enumerate(TASK_CASES):
        task = task_cls()
        for sample_index in range(16):
            out = task.generate(
                hash64(93200, task_index, sample_index),
                params=_task_params_for_query(task, query_id),
                max_attempts=160,
            )
            execution = out.trace_payload["execution_trace"]
            category_counts[int(execution["category_count"])] += 1
            panel_counts[int(execution["panel_count"])] += 1
            answer_types[str(out.answer_gt.type)] += 1
    assert set(category_counts).issubset({4, 5, 6, 7, 8, 9, 10})
    assert set(panel_counts).issubset({4, 5, 6, 7, 8, 9})
    assert {"integer", "option_letter"}.issubset(set(answer_types))
