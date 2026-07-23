"""Behavior tests for scatter point chart tasks."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task
from trace_tasks.tasks.charts.scatter_points.axis_threshold_point_count import (
    ChartsScatterPointsAxisThresholdPointCountTask,
)
from trace_tasks.tasks.charts.scatter_points.category_axis_mean_extremum_label import (
    ChartsScatterPointsCategoryAxisMeanExtremumLabelTask,
)
from trace_tasks.tasks.charts.scatter_points.category_threshold_point_count import (
    ChartsScatterPointsCategoryThresholdPointCountTask,
)


AXIS_THRESHOLD_QUERY_IDS = (
    "x_above_threshold_count",
    "x_below_threshold_count",
    "y_above_threshold_count",
    "y_below_threshold_count",
)
MEAN_EXTREMUM_QUERY_IDS = (
    "largest_mean_x_category_label",
    "smallest_mean_x_category_label",
    "largest_mean_y_category_label",
    "smallest_mean_y_category_label",
)
CATEGORY_THRESHOLD_QUERY_IDS = (
    "category_x_above_threshold_count",
    "category_x_below_threshold_count",
    "category_y_above_threshold_count",
    "category_y_below_threshold_count",
)
CASES = (
    *((ChartsScatterPointsAxisThresholdPointCountTask, query_id, "integer", "point_set") for query_id in AXIS_THRESHOLD_QUERY_IDS),
    *((ChartsScatterPointsCategoryAxisMeanExtremumLabelTask, query_id, "string", "bbox") for query_id in MEAN_EXTREMUM_QUERY_IDS),
    *((ChartsScatterPointsCategoryThresholdPointCountTask, query_id, "integer", "point_set") for query_id in CATEGORY_THRESHOLD_QUERY_IDS),
)


def _points(execution: dict) -> list[dict]:
    return [dict(point) for point in execution["points"]]


def _matching_threshold(point: dict, *, axis: str, direction: str, threshold: int) -> bool:
    value = float(point[f"{axis}_value"])
    if direction == "above":
        return value > float(threshold)
    return value < float(threshold)


def _expected_answer(execution: dict) -> int | str:
    points = _points(execution)
    question_format = str(execution["question_format"])
    if question_format == "scatter_points_threshold_count":
        axis = str(execution["threshold_axis"])
        direction = str(execution["threshold_direction"])
        threshold = int(execution["threshold_value"])
        return sum(1 for point in points if _matching_threshold(point, axis=axis, direction=direction, threshold=threshold))
    if question_format == "scatter_points_category_mean_extremum":
        axis = str(execution["mean_axis"])
        means: dict[str, list[float]] = defaultdict(list)
        for point in points:
            means[str(point["category_label"])].append(float(point[f"{axis}_value"]))
        mean_by_category = {label: sum(values) / len(values) for label, values in means.items()}
        if str(execution["mean_extremum"]) == "largest":
            return max(sorted(mean_by_category), key=lambda label: (mean_by_category[label], label))
        return min(sorted(mean_by_category), key=lambda label: (mean_by_category[label], label))
    if question_format == "scatter_points_category_threshold_count":
        axis = str(execution["threshold_axis"])
        direction = str(execution["threshold_direction"])
        threshold = int(execution["threshold_value"])
        target = str(execution["target_category_label"])
        return sum(
            1
            for point in points
            if str(point["category_label"]) == target
            and _matching_threshold(point, axis=axis, direction=direction, threshold=threshold)
        )
    raise AssertionError(f"unsupported question format: {question_format}")


def _expected_annotation_point_ids(execution: dict) -> list[str]:
    points = _points(execution)
    question_format = str(execution["question_format"])
    if question_format == "scatter_points_threshold_count":
        axis = str(execution["threshold_axis"])
        direction = str(execution["threshold_direction"])
        threshold = int(execution["threshold_value"])
        return [str(point["point_id"]) for point in points if _matching_threshold(point, axis=axis, direction=direction, threshold=threshold)]
    if question_format == "scatter_points_category_mean_extremum":
        answer = str(_expected_answer(execution))
        return [str(point["point_id"]) for point in points if str(point["category_label"]) == answer]
    if question_format == "scatter_points_category_threshold_count":
        axis = str(execution["threshold_axis"])
        direction = str(execution["threshold_direction"])
        threshold = int(execution["threshold_value"])
        target = str(execution["target_category_label"])
        return [
            str(point["point_id"])
            for point in points
            if str(point["category_label"]) == target
            and _matching_threshold(point, axis=axis, direction=direction, threshold=threshold)
        ]
    raise AssertionError(f"unsupported question format: {question_format}")


def _bbox_union(boxes: list[list[float]]) -> list[float]:
    return [
        round(min(float(box[0]) for box in boxes), 3),
        round(min(float(box[1]) for box in boxes), 3),
        round(max(float(box[2]) for box in boxes), 3),
        round(max(float(box[3]) for box in boxes), 3),
    ]


@pytest.mark.parametrize(("task_cls", "query_id", "answer_type", "annotation_type"), CASES)
def test_chart_scatter_points_queries_match_contract(task_cls, query_id: str, answer_type: str, annotation_type: str) -> None:
    task = task_cls()
    out = task.generate(142100 + CASES.index((task_cls, query_id, answer_type, annotation_type)), params={"query_id": query_id}, max_attempts=80)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert out.scene_id == "scatter_points"
    assert out.query_id == query_id
    assert out.answer_gt.type == answer_type
    assert out.annotation_gt.type == annotation_type
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(execution["question_format"]).startswith("scatter_points_")
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert str(render["font_asset_version"])
    assert str(render["chart_font_family"])
    assert str(render["font_assets"]["chart_font_family"]) == str(render["chart_font_family"])
    expected_answer = _expected_answer(execution)
    assert out.answer_gt.value == expected_answer
    assert execution["answer"] == expected_answer
    expected_point_ids = _expected_annotation_point_ids(execution)
    assert execution["annotation_point_ids"] == expected_point_ids
    expected_points = [render_map["point_centers_px"][point_id] for point_id in expected_point_ids]
    assert trace["projected_annotation"]["type"] == annotation_type
    assert trace["projected_annotation"]["point_ids"] == expected_point_ids
    if annotation_type == "point_set":
        assert out.annotation_gt.value == expected_points
        assert trace["projected_annotation"]["point_set"] == expected_points
        assert trace["projected_annotation"]["pixel_point_set"] == expected_points
        for point in out.annotation_gt.value:
            assert len(point) == 2
            x, y = [float(value) for value in point]
            assert 0 <= x <= int(render["canvas_width"])
            assert 0 <= y <= int(render["canvas_height"])
    else:
        expected_bbox = _bbox_union([render_map["point_bboxes_px"][point_id] for point_id in expected_point_ids])
        assert out.annotation_gt.value == expected_bbox
        assert trace["projected_annotation"]["bbox"] == expected_bbox
        assert trace["projected_annotation"]["pixel_bbox"] == expected_bbox
        x0, y0, x1, y1 = [float(value) for value in out.annotation_gt.value]
        assert 0 <= x0 < x1 <= int(render["canvas_width"])
        assert 0 <= y0 < y1 <= int(render["canvas_height"])
    if str(execution["question_format"]) == "scatter_points_threshold_count":
        assert 28 <= int(execution["point_count"]) <= 54
        assert 4 <= int(out.answer_gt.value) <= 12
        assert not execution["categories"]
        assert render_map["threshold_guide_bbox_px"]
    elif str(execution["question_format"]) == "scatter_points_category_mean_extremum":
        assert 4 <= int(execution["category_count"]) <= 6
        assert len(execution["categories"]) == int(execution["category_count"])
        assert str(out.answer_gt.value) in {str(category["label"]) for category in execution["categories"]}
        assert float(execution["mean_margin"]) > 0.0
        assert not render_map["threshold_guide_bbox_px"]
    elif str(execution["question_format"]) == "scatter_points_category_threshold_count":
        assert 4 <= int(execution["category_count"]) <= 6
        assert 2 <= int(out.answer_gt.value) <= 9
        assert str(execution["target_category_label"]) in {str(category["label"]) for category in execution["categories"]}
        assert render_map["threshold_guide_bbox_px"]


def test_chart_scatter_points_prompt_examples_match_contract() -> None:
    for index, (task_cls, query_id, answer_type, annotation_type) in enumerate(CASES, start=142200):
        out = task_cls().generate(index, params={"query_id": query_id}, max_attempts=80)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert isinstance(answer_and_annotation["annotation"], list)
        if annotation_type == "bbox":
            assert len(answer_and_annotation["annotation"]) == 4
            assert all(isinstance(value, int | float) for value in answer_and_annotation["annotation"])
        else:
            assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])
        if answer_type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert len(str(answer_and_annotation["answer"])) > 1
            assert isinstance(answer_only["answer"], str)
        assert "Displayed is" not in out.prompt
        assert "Shown is" not in out.prompt
        assert "scatter" in out.prompt


def test_chart_scatter_points_balanced_sampling_covers_query_ids() -> None:
    axis_task = ChartsScatterPointsAxisThresholdPointCountTask()
    mean_task = ChartsScatterPointsCategoryAxisMeanExtremumLabelTask()
    category_task = ChartsScatterPointsCategoryThresholdPointCountTask()
    observed_axis_queries: Counter[str] = Counter()
    observed_mean_queries: Counter[str] = Counter()
    observed_category_queries: Counter[str] = Counter()
    axis_answers: Counter[int] = Counter()
    category_answers: Counter[int] = Counter()
    category_counts: Counter[int] = Counter()
    for index in range(96):
        out = axis_task.generate(hash64(142300, "scatter_points_axis", index), params={}, max_attempts=80)
        execution = out.trace_payload["execution_trace"]
        observed_axis_queries[str(out.query_id)] += 1
        axis_answers[int(out.answer_gt.value)] += 1
        assert execution["threshold_axis"] in {"x", "y"}
        assert execution["threshold_direction"] in {"above", "below"}
        assert int(out.answer_gt.value) <= 12

        out = mean_task.generate(hash64(142400, "scatter_points_mean", index), params={}, max_attempts=80)
        execution = out.trace_payload["execution_trace"]
        observed_mean_queries[str(out.query_id)] += 1
        category_counts[int(execution["category_count"])] += 1
        assert execution["mean_axis"] in {"x", "y"}
        assert execution["mean_extremum"] in {"largest", "smallest"}

        out = category_task.generate(hash64(142500, "scatter_points_category_threshold", index), params={}, max_attempts=80)
        execution = out.trace_payload["execution_trace"]
        observed_category_queries[str(out.query_id)] += 1
        category_answers[int(out.answer_gt.value)] += 1
        category_counts[int(execution["category_count"])] += 1
        assert execution["threshold_axis"] in {"x", "y"}
        assert execution["threshold_direction"] in {"above", "below"}
    assert set(observed_axis_queries) == set(AXIS_THRESHOLD_QUERY_IDS)
    assert set(observed_mean_queries) == set(MEAN_EXTREMUM_QUERY_IDS)
    assert set(observed_category_queries) == set(CATEGORY_THRESHOLD_QUERY_IDS)
    assert len(axis_answers) >= 8
    assert len(category_answers) >= 5
    assert min(category_counts) >= 4
    assert max(category_counts) <= 6


def test_chart_scatter_points_is_deterministic() -> None:
    params = {"query_id": "category_y_below_threshold_count"}
    task = ChartsScatterPointsCategoryThresholdPointCountTask()
    out_a = task.generate(142600, params=params, max_attempts=80)
    out_b = task.generate(142600, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_chart_scatter_points_registered_and_scene_config_loaded() -> None:
    assert create_task("task_charts__scatter_points__axis_threshold_point_count").task_id == "task_charts__scatter_points__axis_threshold_point_count"
    assert create_task("task_charts__scatter_points__category_axis_mean_extremum_label").task_id == "task_charts__scatter_points__category_axis_mean_extremum_label"
    assert create_task("task_charts__scatter_points__category_threshold_point_count").task_id == "task_charts__scatter_points__category_threshold_point_count"
    cfg = get_scene_defaults("charts", "scatter_points")
    generation = cfg["generation"]["shared"]
    assert int(generation["scatter_points_count_min"]) == 28
    assert int(generation["scatter_points_count_max"]) == 54
    assert int(generation["axis_threshold_answer_max"]) == 12
    prompt = cfg["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_scatter_points_v1"
