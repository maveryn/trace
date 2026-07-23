"""Behavior tests for scientific error-bar series chart tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.errorbar_series.bound_extremum_x_label import (
    ChartsErrorbarSeriesBoundExtremumXLabelTask,
    QUERY_IDS as BOUND_EXTREMUM_QUERY_IDS,
)
from trace_tasks.tasks.charts.errorbar_series.same_x_interval_overlap_count import (
    ChartsErrorbarSeriesSameXIntervalOverlapCountTask,
)
from trace_tasks.tasks.charts.errorbar_series.shared.state import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.registry import TASK_REGISTRY


OVERLAP_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_CASES = (
    (ChartsErrorbarSeriesBoundExtremumXLabelTask, BOUND_EXTREMUM_QUERY_IDS, "string", "point"),
    (ChartsErrorbarSeriesSameXIntervalOverlapCountTask, OVERLAP_QUERY_IDS, "integer", "segment_set"),
)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _assert_segment_inside_canvas(segment: list[list[float]], *, width: int, height: int) -> None:
    assert len(segment) == 2
    for point in segment:
        _assert_point_inside_canvas(list(point), width=width, height=height)


def _series_by_id(execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(series["series_id"]): dict(series) for series in execution["series"]}


def _semantic_query_id(execution: dict[str, Any], query_id: str) -> str:
    query_params = execution.get("query_params", {})
    if isinstance(query_params, dict) and "internal_query_id" in query_params:
        return str(query_params["internal_query_id"])
    return str(query_id)


def _expected_answer(execution: dict[str, Any], query_id: str) -> int | str:
    semantic_query_id = _semantic_query_id(execution, query_id)
    target = _series_by_id(execution)[str(execution["target_series_id"])]
    if semantic_query_id == "highest_upper_bound_x_label":
        index = max(range(int(execution["x_count"])), key=lambda idx: int(target["upper_values"][idx]))
        return str(execution["x_labels"][int(index)])
    if semantic_query_id == "lowest_lower_bound_x_label":
        index = min(range(int(execution["x_count"])), key=lambda idx: int(target["lower_values"][idx]))
        return str(execution["x_labels"][int(index)])
    if semantic_query_id == "overlap_target_errorbar_at_x_count":
        target_x = int(execution["target_x_index"])
        target_low = int(target["lower_values"][target_x])
        target_high = int(target["upper_values"][target_x])
        count = 0
        for series in execution["series"]:
            if str(series["series_id"]) == str(execution["target_series_id"]):
                continue
            low = int(series["lower_values"][target_x])
            high = int(series["upper_values"][target_x])
            if max(target_low, low) <= min(target_high, high):
                count += 1
        return int(count)
    raise AssertionError(f"unsupported query_id: {semantic_query_id}")


@pytest.mark.parametrize(("task_cls", "query_ids", "answer_type", "annotation_type"), TASK_CASES)
def test_charts_errorbar_series_tasks_match_contract(
    task_cls: type,
    query_ids: tuple[str, ...],
    answer_type: str,
    annotation_type: str,
) -> None:
    task = task_cls()
    for index, query_id in enumerate(query_ids):
        out = task.generate(
            hash64(20260605, f"{task_cls.task_id}.{query_id}", index),
            params={"query_id": str(query_id)},
            max_attempts=80,
        )
        execution = out.trace_payload["execution_trace"]
        render_spec = out.trace_payload["render_spec"]
        assert out.query_id == str(query_id)
        assert out.trace_payload["query_spec"]["query_id"] == str(query_id)
        assert out.trace_payload["query_spec"]["params"]["query_id"] == str(query_id)
        assert out.answer_gt.type == answer_type
        assert out.answer_gt.value == _expected_answer(execution, str(query_id))
        assert out.annotation_gt.type == annotation_type
        width = int(render_spec["canvas_width"])
        height = int(render_spec["canvas_height"])
        if annotation_type == "bbox_set":
            assert isinstance(out.annotation_gt.value, list)
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            for bbox in out.annotation_gt.value:
                _assert_bbox_inside_canvas(bbox, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["bbox_set"] == out.annotation_gt.value
            assert out.trace_payload["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
        elif annotation_type == "point_set":
            assert isinstance(out.annotation_gt.value, list)
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            for point in out.annotation_gt.value:
                _assert_point_inside_canvas(point, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value
            assert out.trace_payload["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
        elif annotation_type == "segment_set":
            assert isinstance(out.annotation_gt.value, list)
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            for segment in out.annotation_gt.value:
                _assert_segment_inside_canvas([list(point) for point in segment], width=width, height=height)
            assert out.trace_payload["projected_annotation"]["segment_set"] == out.annotation_gt.value
            assert out.trace_payload["projected_annotation"]["pixel_segment_set"] == out.annotation_gt.value
        elif annotation_type == "point":
            assert isinstance(out.annotation_gt.value, list)
            _assert_point_inside_canvas(out.annotation_gt.value, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["point"] == out.annotation_gt.value
            assert out.trace_payload["projected_annotation"]["pixel_point"] == out.annotation_gt.value
        else:
            raise AssertionError(f"unsupported annotation type in test: {annotation_type}")


def test_charts_errorbar_series_prompt_examples_match_contract() -> None:
    for task_cls, _query_ids, answer_type, annotation_type in TASK_CASES:
        out = task_cls().generate(hash64(20260605, f"{task_cls.task_id}.prompt", len(task_cls.task_id)), params={}, max_attempts=80)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if answer_type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert isinstance(answer_only["answer"], str)
            assert len(answer_and_annotation["answer"]) > 1
        if annotation_type == "bbox_set":
            assert isinstance(answer_and_annotation["annotation"], list)
        elif annotation_type == "point_set":
            assert isinstance(answer_and_annotation["annotation"], list)
            assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])
        elif annotation_type == "segment_set":
            assert isinstance(answer_and_annotation["annotation"], list)
            assert all(
                isinstance(segment, list)
                and len(segment) == 2
                and all(isinstance(point, list) and len(point) == 2 for point in segment)
                for segment in answer_and_annotation["annotation"]
            )
        elif annotation_type == "point":
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) == 2
        else:
            assert isinstance(answer_and_annotation["annotation"], dict)


def test_charts_errorbar_series_balanced_sampling_covers_axes() -> None:
    query_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    task = ChartsErrorbarSeriesBoundExtremumXLabelTask()
    for index in range(72):
        out = task.generate(
            hash64(20260605, "charts_errorbar_series_axes", index),
            params={"_sample_cursor": index},
            max_attempts=80,
        )
        query_counts[str(out.query_id)] += 1
        scene_counts[str(out.trace_payload["execution_trace"]["scene_variant"])] += 1
    assert set(query_counts) == set(BOUND_EXTREMUM_QUERY_IDS)
    assert set(scene_counts) == set(SUPPORTED_SCENE_VARIANTS)


def test_charts_errorbar_series_is_deterministic() -> None:
    params = {"query_id": SINGLE_QUERY_ID, "scene_variant": "grouped_errorbar"}
    task = ChartsErrorbarSeriesSameXIntervalOverlapCountTask()
    out_a = task.generate(2026060501, params=params, max_attempts=80)
    out_b = task.generate(2026060501, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_charts_errorbar_series_registry_and_config_are_wired() -> None:
    defaults = get_scene_defaults("charts", "errorbar_series")
    prompt = defaults["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_errorbar_series_v1"
    assert str(prompt["scene_key"]) == "errorbar_series_scene"
    assert str(prompt["task_key"]) == "errorbar_series_query"
    assert ChartsErrorbarSeriesBoundExtremumXLabelTask.task_id in TASK_REGISTRY
    assert ChartsErrorbarSeriesSameXIntervalOverlapCountTask.task_id in TASK_REGISTRY
