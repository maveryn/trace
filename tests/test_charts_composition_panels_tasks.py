"""Behavior tests for chart composition-panel chart tasks."""

from __future__ import annotations

from typing import Any

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.charts.composition_panels.composition_shift_l1_distance import (
    ChartsCompositionPanelsCompositionShiftL1DistanceTask,
)
from trace_tasks.tasks.charts.composition_panels.conditioned_panel_sum_from_percent import (
    ChartsCompositionPanelsConditionedPanelSumFromPercentTask,
)
from trace_tasks.tasks.charts.composition_panels.segment_count_extremum_panel_label import (
    LARGEST_QUERY_ID,
    SMALLEST_QUERY_ID,
    ChartsCompositionPanelsSegmentCountExtremumPanelLabelTask,
)
from trace_tasks.tasks.charts.composition_panels.segment_count_nearest_target_panel_label import (
    ChartsCompositionPanelsSegmentCountNearestTargetPanelLabelTask,
)
from trace_tasks.tasks.charts.composition_panels.segment_pair_count_gap_extremum_panel_label import (
    LARGEST_QUERY_ID as LARGEST_COUNT_GAP_QUERY_ID,
    SMALLEST_QUERY_ID as SMALLEST_COUNT_GAP_QUERY_ID,
    ChartsCompositionPanelsSegmentPairCountGapExtremumPanelLabelTask,
)
from trace_tasks.tasks.charts.composition_panels.shared.state import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.charts.composition_panels.top_k_by_segment_then_sum_other_segment_count import (
    ChartsCompositionPanelsTopKBySegmentThenSumOtherSegmentCountTask,
)


TASK_CASES = (
    (
        "task_charts__composition_panels__top_k_by_segment_then_sum_other_segment_count",
        ChartsCompositionPanelsTopKBySegmentThenSumOtherSegmentCountTask,
    ),
    (
        "task_charts__composition_panels__conditioned_panel_sum_from_percent",
        ChartsCompositionPanelsConditionedPanelSumFromPercentTask,
    ),
    (
        "task_charts__composition_panels__composition_shift_l1_distance",
        ChartsCompositionPanelsCompositionShiftL1DistanceTask,
    ),
    (
        "task_charts__composition_panels__segment_count_extremum_panel_label",
        ChartsCompositionPanelsSegmentCountExtremumPanelLabelTask,
    ),
    (
        "task_charts__composition_panels__segment_count_nearest_target_panel_label",
        ChartsCompositionPanelsSegmentCountNearestTargetPanelLabelTask,
    ),
    (
        "task_charts__composition_panels__segment_pair_count_gap_extremum_panel_label",
        ChartsCompositionPanelsSegmentPairCountGapExtremumPanelLabelTask,
    ),
)


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    assert 0 <= float(point[0]) <= int(width)
    assert 0 <= float(point[1]) <= int(height)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    assert 0 <= float(bbox[0]) < float(bbox[2]) <= int(width)
    assert 0 <= float(bbox[1]) < float(bbox[3]) <= int(height)


def _round_bbox(bbox: list[float]) -> list[float]:
    return [round(float(value), 3) for value in bbox]


def _expected_answer(task_id: str, execution: dict[str, Any]) -> int | str:
    if task_id.endswith("__top_k_by_segment_then_sum_other_segment_count"):
        return int(sum(int(value) for value in execution["selected_target_counts"]))
    if task_id.endswith("__conditioned_panel_sum_from_percent"):
        return int(sum(int(value) for value in execution["selected_target_counts"]))
    if task_id.endswith("__composition_shift_l1_distance"):
        return int(sum(int(value) for value in execution["segment_changes"]))
    if task_id.endswith("__segment_count_extremum_panel_label"):
        return str(execution["answer_panel"])
    if task_id.endswith("__segment_count_nearest_target_panel_label"):
        return str(execution["answer_panel"])
    if task_id.endswith("__segment_pair_count_gap_extremum_panel_label"):
        return str(execution["answer_panel"])
    raise AssertionError(f"unsupported task: {task_id}")


@pytest.mark.parametrize(("task_id", "task_cls"), TASK_CASES)
@pytest.mark.parametrize("scene_variant", SUPPORTED_SCENE_VARIANTS)
def test_chart_composition_panels_tasks_match_contract(task_id: str, task_cls: type, scene_variant: str) -> None:
    task = task_cls()
    out = task.generate(79200 + len(task_id) + len(scene_variant), params={"scene_variant": scene_variant}, max_attempts=160)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    projected = trace["projected_annotation"]

    assert out.query_id in task.supported_query_ids
    expected_answer = _expected_answer(str(task_id), execution)
    assert out.answer_gt.type == ("integer" if isinstance(expected_answer, int) else "string")
    assert str(execution["scene_variant"]) == str(scene_variant)
    assert str(render["scene_variant"]) == str(scene_variant)
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert int(execution["panel_count"]) == len(execution["panels"])
    assert int(execution["segment_count"]) == len(execution["segment_labels"])
    assert out.answer_gt.value == expected_answer
    assert out.answer_gt.value == execution["answer_value"]
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["query_spec"]["params"]["query_id"] == out.query_id
    assert str(execution["program_code"])

    if (
        task_id.endswith("__composition_shift_l1_distance")
        or task_id.endswith("__conditioned_panel_sum_from_percent")
        or task_id.endswith("__top_k_by_segment_then_sum_other_segment_count")
    ):
        assert out.annotation_gt.type == "bbox_set"
        assert execution["annotation_type"] == "bbox_set"
        assert projected["type"] == "bbox_set"
        assert projected["bbox_set"] == list(out.annotation_gt.value)
        assert projected["pixel_bbox_set"] == list(out.annotation_gt.value)
        if task_id.endswith("__composition_shift_l1_distance"):
            expected_panels = [str(execution["start_panel"]), str(execution["end_panel"])]
        else:
            expected_panels = [str(label) for label in execution["selected_panels"]]
        panel_bboxes = {
            str(panel["panel_label"]): _round_bbox(list(panel["panel_bbox_px"]))
            for panel in trace["render_map"]["panel_traces"]
        }
        assert out.annotation_gt.value == [panel_bboxes[label] for label in expected_panels]
        assert len(out.annotation_gt.value) == len(expected_panels)
        for bbox in out.annotation_gt.value:
            _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    elif (
        task_id.endswith("__segment_count_nearest_target_panel_label")
        or task_id.endswith("__segment_pair_count_gap_extremum_panel_label")
    ):
        assert out.annotation_gt.type == "bbox"
        assert execution["annotation_type"] == "bbox"
        assert projected["type"] == "bbox"
        assert projected["bbox"] == list(out.annotation_gt.value)
        expected_panel = str(execution["answer_panel"])
        panel_bboxes = {
            str(panel["panel_label"]): _round_bbox(list(panel["panel_bbox_px"]))
            for panel in trace["render_map"]["panel_traces"]
        }
        assert out.annotation_gt.value == panel_bboxes[expected_panel]
        _assert_bbox_inside_canvas([float(value) for value in out.annotation_gt.value], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    else:
        assert out.annotation_gt.type == "point_map"
        assert execution["annotation_type"] == "point_map"
        assert projected["type"] == "point_map"
        assert projected["point_map"] == dict(out.annotation_gt.value)
        assert projected["pixel_point_map"] == dict(out.annotation_gt.value)
        assert set(out.annotation_gt.value) == set(execution["annotation_point_keys"])
        assert len(projected["point_set"]) == len(out.annotation_gt.value)
        assert len(execution["annotation_roles"]) == len(out.annotation_gt.value)
        for point in out.annotation_gt.value.values():
            _assert_point_inside_canvas([float(value) for value in point], width=int(render["canvas_width"]), height=int(render["canvas_height"]))


def test_chart_composition_panels_prompt_examples_match_contract() -> None:
    for _task_id, task_cls in TASK_CASES:
        task = task_cls()
        out = task.generate(79300 + len(task.task_id), params={}, max_attempts=160)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if (
            task.task_id.endswith("__composition_shift_l1_distance")
            or task.task_id.endswith("__conditioned_panel_sum_from_percent")
            or task.task_id.endswith("__top_k_by_segment_then_sum_other_segment_count")
        ):
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) >= 2
        elif (
            task.task_id.endswith("__segment_count_nearest_target_panel_label")
            or task.task_id.endswith("__segment_pair_count_gap_extremum_panel_label")
        ):
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) == 4
        else:
            assert isinstance(answer_and_annotation["annotation"], dict)
        expected_type = str if (
            task.task_id.endswith("__segment_count_extremum_panel_label")
            or task.task_id.endswith("__segment_count_nearest_target_panel_label")
            or task.task_id.endswith("__segment_pair_count_gap_extremum_panel_label")
        ) else int
        assert isinstance(answer_and_annotation["answer"], expected_type)
        assert isinstance(answer_only["answer"], expected_type)


def test_chart_composition_panels_tasks_are_registered_and_reject_unsupported_query() -> None:
    for task_id, task_cls in TASK_CASES:
        assert task_id in TASK_REGISTRY
        task = task_cls()
        with pytest.raises(ValueError, match="query_id"):
            task.generate(79400, params={"query_id": "__unsupported_query_id__"}, max_attempts=10)


def test_chart_composition_panels_is_deterministic() -> None:
    task = ChartsCompositionPanelsCompositionShiftL1DistanceTask()
    params = {"scene_variant": "composition_donut_panels"}
    out_a = task.generate(79500, params=params, max_attempts=160)
    out_b = task.generate(79500, params=params, max_attempts=160)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("query_id", (LARGEST_QUERY_ID, SMALLEST_QUERY_ID))
def test_chart_composition_panels_segment_count_extremum_queries(query_id: str) -> None:
    task = ChartsCompositionPanelsSegmentCountExtremumPanelLabelTask()
    out = task.generate(79600 + len(query_id), params={"query_id": query_id}, max_attempts=200)
    execution = out.trace_payload["execution_trace"]
    panels = execution["panels"]
    target_segment = str(execution["target_segment"])
    counts = {
        str(panel["panel_label"]): int(panel["counts_by_segment"][target_segment])
        for panel in panels
    }
    expected = max(counts, key=counts.get) if str(query_id) == LARGEST_QUERY_ID else min(counts, key=counts.get)
    assert out.query_id == query_id
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == str(expected)
    assert set(out.annotation_gt.value) == {"segment_percent", "panel_total"}
    assert int(execution["answer_count"]) == int(counts[str(expected)])


def test_chart_composition_panels_segment_count_nearest_target() -> None:
    task = ChartsCompositionPanelsSegmentCountNearestTargetPanelLabelTask()
    out = task.generate(79650, params={}, max_attempts=200)
    execution = out.trace_payload["execution_trace"]
    target_segment = str(execution["target_segment"])
    target_count = int(execution["target_count"])
    counts = {
        str(panel["panel_label"]): int(panel["counts_by_segment"][target_segment])
        for panel in execution["panels"]
    }
    expected = min(counts, key=lambda label: abs(int(counts[label]) - int(target_count)))
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox"
    assert out.answer_gt.value == str(expected)
    assert int(execution["answer_count"]) == int(counts[str(expected)])
    assert int(execution["answer_distance_from_target"]) == abs(int(counts[str(expected)]) - int(target_count))


@pytest.mark.parametrize("query_id", (LARGEST_COUNT_GAP_QUERY_ID, SMALLEST_COUNT_GAP_QUERY_ID))
def test_chart_composition_panels_segment_pair_count_gap_extremum_queries(query_id: str) -> None:
    task = ChartsCompositionPanelsSegmentPairCountGapExtremumPanelLabelTask()
    out = task.generate(79700 + len(query_id), params={"query_id": query_id}, max_attempts=200)
    execution = out.trace_payload["execution_trace"]
    panels = execution["panels"]
    segment_a = str(execution["segment_a"])
    segment_b = str(execution["segment_b"])
    gaps = {
        str(panel["panel_label"]): abs(int(panel["counts_by_segment"][segment_a]) - int(panel["counts_by_segment"][segment_b]))
        for panel in panels
    }
    expected = max(gaps, key=gaps.get) if str(query_id) == LARGEST_COUNT_GAP_QUERY_ID else min(gaps, key=gaps.get)
    assert out.query_id == query_id
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox"
    assert out.answer_gt.value == str(expected)
    assert int(execution["answer_count_gap"]) == int(gaps[str(expected)])
