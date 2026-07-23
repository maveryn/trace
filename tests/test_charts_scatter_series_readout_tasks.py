"""Behavior tests for multi-series scatter readout chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task
from trace_tasks.tasks.charts.scatter_readout.series_pair_value_gap_at_x import (
    ChartsScatterSeriesPairValueGapAtXTask,
)
from trace_tasks.tasks.charts.scatter_readout.series_value_at_x_value import (
    ChartsScatterSeriesValueAtXValueTask,
)
from trace_tasks.tasks.charts.scatter_readout.series_x_extremum_label import (
    ChartsScatterSeriesExtremumXLabelTask,
)
from trace_tasks.tasks.charts.scatter_readout.series_y_anchor_other_series_value import (
    ChartsScatterSeriesYAnchorOtherSeriesValueTask,
)
from trace_tasks.tasks.charts.scatter_readout.x_value_rank_series_label import (
    ChartsScatterXValueRankSeriesLabelTask,
)


CASES = (
    (ChartsScatterSeriesExtremumXLabelTask, "series_highest_x_label", "string", "point"),
    (ChartsScatterSeriesExtremumXLabelTask, "series_lowest_x_label", "string", "point"),
    (ChartsScatterSeriesPairValueGapAtXTask, SINGLE_QUERY_ID, "integer", "segment"),
    (ChartsScatterSeriesValueAtXValueTask, SINGLE_QUERY_ID, "integer", "point"),
    (ChartsScatterSeriesYAnchorOtherSeriesValueTask, SINGLE_QUERY_ID, "integer", "point"),
    (ChartsScatterXValueRankSeriesLabelTask, "x_highest_series_label", "string", "point"),
    (ChartsScatterXValueRankSeriesLabelTask, "x_lowest_series_label", "string", "point"),
)


def _bbox_center(bbox: list[float]) -> list[float]:
    assert len(bbox) == 4
    return [
        (float(bbox[0]) + float(bbox[2])) / 2.0,
        (float(bbox[1]) + float(bbox[3])) / 2.0,
    ]


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _series_points(execution: dict, label: str) -> list[dict]:
    points = execution["values_by_series"][str(label)]
    assert isinstance(points, list)
    return [dict(point) for point in points]


def _point_at_x(execution: dict, label: str, x_label: str) -> dict:
    matches = [point for point in _series_points(execution, label) if str(point["x_label"]) == str(x_label)]
    assert len(matches) == 1
    return dict(matches[0])


def _expected_answer(execution: dict, answer_type: str) -> int | str:
    if str(execution.get("answerability", "answerable")) == "unanswerable":
        return "unanswerable"

    target_series = str(execution["target_series_label"])
    target_x = str(execution["target_x_label"])
    operation = str(execution.get("operation", ""))

    if operation == "direct_value_readout":
        return int(_point_at_x(execution, target_series, target_x)["y_value"])

    if operation == "x_value_rank_selection":
        candidates = [
            {
                "series_label": str(label),
                "y_value": int(_point_at_x(execution, str(label), target_x)["y_value"]),
            }
            for label in execution["values_by_series"]
        ]
        if str(execution["extremum"]) == "highest":
            return str(max(candidates, key=lambda item: (int(item["y_value"]), str(item["series_label"])))["series_label"])
        return str(min(candidates, key=lambda item: (int(item["y_value"]), str(item["series_label"])))["series_label"])

    points = _series_points(execution, target_series)
    if str(answer_type) == "string":
        if str(execution["extremum"]) == "highest":
            return str(max(points, key=lambda point: (int(point["y_value"]), str(point["x_label"])))["x_label"])
        return str(min(points, key=lambda point: (int(point["y_value"]), str(point["x_label"])))["x_label"])

    comparison_series = str(execution["comparison_series_label"])
    source = _point_at_x(execution, target_series, target_x)
    comparison = _point_at_x(execution, comparison_series, target_x)
    if operation == "absolute_difference":
        return abs(int(source["y_value"]) - int(comparison["y_value"]))
    if operation == "same_x_transfer_value":
        return int(comparison["y_value"])
    raise AssertionError(f"unsupported scatter-readout operation: {operation!r}")


@pytest.mark.parametrize(("task_cls", "query_id", "answer_type", "annotation_type"), CASES)
def test_chart_scatter_series_readout_queries_match_contract(
    task_cls,
    query_id: str,
    answer_type: str,
    annotation_type: str,
) -> None:
    task = task_cls()
    out = task.generate(
        93100 + CASES.index((task_cls, query_id, answer_type, annotation_type)),
        params={"query_id": query_id, "_enable_unanswerable": False},
        max_attempts=80,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert out.query_id == query_id
    assert out.scene_id == "scatter_readout"
    assert out.answer_gt.type == answer_type
    assert out.annotation_gt.type == annotation_type
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(execution["question_format"]) == "scatter_series_readout_query"
    assert str(execution["scene_variant"]) == "marker_scatter"
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert int(execution["series_count"]) == 5
    assert 9 <= int(execution["x_count"]) <= 10
    assert int(execution["total_point_count"]) == int(execution["series_count"]) * int(execution["x_count"])
    expected_answer = _expected_answer(execution, answer_type)
    assert out.answer_gt.value == expected_answer
    assert execution["answer"] == expected_answer
    assert trace["projected_annotation"]["type"] == annotation_type
    target_point_id = str(execution["target_point_id"])
    if annotation_type == "point":
        expected_point_id = (
            str(execution["comparison_point_id"])
            if str(execution.get("operation", "")) == "same_x_transfer_value"
            else target_point_id
        )
        expected_point = _bbox_center(render_map["point_bboxes_px"][expected_point_id])
        _assert_point_inside_canvas(
            [float(value) for value in out.annotation_gt.value],
            width=int(render["canvas_width"]),
            height=int(render["canvas_height"]),
        )
        assert out.annotation_gt.value == pytest.approx(expected_point)
        assert trace["projected_annotation"]["point"] == pytest.approx(expected_point)
        assert trace["projected_annotation"]["pixel_point"] == pytest.approx(expected_point)
    else:
        assert annotation_type == "segment"
        expected_segment = [
            _bbox_center(render_map["point_bboxes_px"][target_point_id]),
            _bbox_center(render_map["point_bboxes_px"][str(execution["comparison_point_id"])]),
        ]
        assert len(out.annotation_gt.value) == 2
        for actual_point, expected_point in zip(out.annotation_gt.value, expected_segment):
            _assert_point_inside_canvas(
                [float(value) for value in actual_point],
                width=int(render["canvas_width"]),
                height=int(render["canvas_height"]),
            )
            assert actual_point == pytest.approx(expected_point)
        for actual_point, expected_point in zip(trace["projected_annotation"]["segment"], expected_segment):
            assert actual_point == pytest.approx(expected_point)
        for actual_point, expected_point in zip(trace["projected_annotation"]["pixel_segment"], expected_segment):
            assert actual_point == pytest.approx(expected_point)


def test_chart_scatter_series_readout_prompt_examples_match_contract() -> None:
    for index, (task_cls, query_id, answer_type, annotation_type) in enumerate(CASES, start=93200):
        out = task_cls().generate(index, params={"query_id": query_id, "_enable_unanswerable": False}, max_attempts=80)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if answer_type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert isinstance(answer_only["answer"], str)
        annotation = answer_and_annotation["annotation"]
        if annotation_type == "segment":
            assert isinstance(annotation, list)
            assert len(annotation) == 2
            assert all(isinstance(point, list) and len(point) == 2 for point in annotation)
        else:
            assert isinstance(annotation, list)
            assert len(annotation) == 2
        assert "multi-series scatter plot" in out.prompt
        assert "Annotation" in out.prompt or "annotation" in out.prompt


def test_chart_scatter_series_readout_balanced_sampling_covers_queries() -> None:
    counts: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    for index in range(140):
        task_cls, _query_id, _answer_type, _annotation_type = CASES[int(index) % len(CASES)]
        task = task_cls()
        out = task.generate(hash64(93300, "charts_scatter_series", index), params={}, max_attempts=120)
        counts[str(out.query_id)] += 1
        answer_types[str(out.answer_gt.type)] += 1
    assert set(counts) == {query_id for _cls, query_id, _answer_type, _annotation_type in CASES}
    assert counts[SINGLE_QUERY_ID] >= 55
    assert counts["series_highest_x_label"] >= 15
    assert counts["series_lowest_x_label"] >= 15
    assert counts["x_highest_series_label"] >= 15
    assert counts["x_lowest_series_label"] >= 15
    assert answer_types["string"] > 0
    assert answer_types["integer"] > 0


def test_chart_scatter_series_readout_registered_and_scene_config_loaded() -> None:
    assert (
        create_task("task_charts__scatter_readout__series_x_extremum_label").task_id
        == "task_charts__scatter_readout__series_x_extremum_label"
    )
    assert (
        create_task("task_charts__scatter_readout__series_pair_value_gap_at_x").task_id
        == "task_charts__scatter_readout__series_pair_value_gap_at_x"
    )
    assert (
        create_task("task_charts__scatter_readout__series_value_at_x_value").task_id
        == "task_charts__scatter_readout__series_value_at_x_value"
    )
    assert (
        create_task("task_charts__scatter_readout__series_y_anchor_other_series_value").task_id
        == "task_charts__scatter_readout__series_y_anchor_other_series_value"
    )
    assert (
        create_task("task_charts__scatter_readout__x_value_rank_series_label").task_id
        == "task_charts__scatter_readout__x_value_rank_series_label"
    )
    cfg = get_scene_defaults("charts", "scatter_readout")
    assert isinstance(cfg.get("generation"), dict)
    assert isinstance(cfg.get("rendering"), dict)
    assert isinstance(cfg.get("prompt"), dict)
    assert str(cfg["prompt"]["shared"]["bundle_id"]) == "charts_scatter_readout_v1"
