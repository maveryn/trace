"""Behavior tests for uncertainty-band chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.uncertainty_band.band_overlap_count import (
    ChartsUncertaintyBandOverlapCountTask,
    SUPPORTED_QUERY_IDS as OVERLAP_QUERY_IDS,
)
from trace_tasks.tasks.charts.uncertainty_band.band_width_extremum_x_label import (
    ChartsUncertaintyBandWidthExtremumXLabelTask,
    SUPPORTED_QUERY_IDS as WIDTH_EXTREMUM_QUERY_IDS,
)
from trace_tasks.tasks.registry import list_default_task_ids


TASK_CASES = (
    (ChartsUncertaintyBandOverlapCountTask, OVERLAP_QUERY_IDS, "integer", "point_set"),
    (ChartsUncertaintyBandWidthExtremumXLabelTask, WIDTH_EXTREMUM_QUERY_IDS, "string", "segment"),
)


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _series_by_id(execution: dict) -> dict[str, dict]:
    return {str(series["series_id"]): dict(series) for series in execution["series"]}


def _expected_answer(execution: dict, query_id: str) -> int | str:
    series = list(execution["series"])
    if query_id == SINGLE_QUERY_ID:
        count = 0
        for index in range(int(execution["x_count"])):
            low = max(int(series[0]["lower_values"][index]), int(series[1]["lower_values"][index]))
            high = min(int(series[0]["upper_values"][index]), int(series[1]["upper_values"][index]))
            if int(low) <= int(high):
                count += 1
        return int(count)
    target = _series_by_id(execution)[str(execution["target_series_id"])]
    widths = [
        int(high) - int(low)
        for low, high in zip(target["lower_values"], target["upper_values"])
    ]
    if query_id == "widest_band_x_label":
        index = max(range(len(widths)), key=lambda idx: widths[idx])
    elif query_id == "narrowest_band_x_label":
        index = min(range(len(widths)), key=lambda idx: widths[idx])
    else:
        raise AssertionError(f"unsupported query_id: {query_id}")
    return str(execution["x_labels"][int(index)])


@pytest.mark.parametrize(("task_cls", "query_ids", "answer_type", "annotation_type"), TASK_CASES)
def test_charts_uncertainty_band_tasks_match_contract(
    task_cls: type,
    query_ids: tuple[str, ...],
    answer_type: str,
    annotation_type: str,
) -> None:
    assert tuple(task_cls.supported_query_ids) == tuple(query_ids)
    for query_id in query_ids:
        out = task_cls().generate(242000 + len(task_cls.task_id), params={"query_id": query_id}, max_attempts=60)
        assert out.query_id == query_id
        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == annotation_type
        assert out.answer_gt.value == _expected_answer(out.trace_payload["execution_trace"], query_id)
        width = int(out.trace_payload["render_spec"]["canvas_width"])
        height = int(out.trace_payload["render_spec"]["canvas_height"])
        if annotation_type == "point_set":
            assert isinstance(out.annotation_gt.value, list)
            assert len(out.annotation_gt.value) == int(out.answer_gt.value)
            for point in out.annotation_gt.value:
                _assert_point_inside_canvas(point, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value
            assert out.trace_payload["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
        else:
            assert isinstance(out.annotation_gt.value, list)
            assert len(out.annotation_gt.value) == 2
            for point in out.annotation_gt.value:
                _assert_point_inside_canvas(point, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["segment"] == out.annotation_gt.value
            assert out.trace_payload["projected_annotation"]["pixel_segment"] == out.annotation_gt.value


def test_charts_uncertainty_band_prompt_examples_match_contract() -> None:
    for task_cls, _query_ids, answer_type, annotation_type in TASK_CASES:
        out = task_cls().generate(242000 + len(task_cls.task_id), params={}, max_attempts=60)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if answer_type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert isinstance(answer_only["answer"], str)
            assert len(answer_and_annotation["answer"]) > 1
        if annotation_type == "point_set":
            assert isinstance(answer_and_annotation["annotation"], list)
            assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])
        else:
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) == 2
            assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])


def test_charts_uncertainty_band_sampling_covers_width_queries() -> None:
    queries: Counter[str] = Counter()
    for index in range(8):
        out = ChartsUncertaintyBandWidthExtremumXLabelTask().generate(
            hash64(243000, "charts_uncertainty_band", index),
            params={"_sample_cursor": index},
            max_attempts=60,
        )
        queries[str(out.query_id)] += 1
    assert set(queries) == set(WIDTH_EXTREMUM_QUERY_IDS)


def test_charts_uncertainty_band_overlap_uses_single_query_id() -> None:
    out = ChartsUncertaintyBandOverlapCountTask().generate(244001, params={}, max_attempts=60)
    assert out.query_id == SINGLE_QUERY_ID
    assert tuple(ChartsUncertaintyBandOverlapCountTask.supported_query_ids) == (SINGLE_QUERY_ID,)


def test_charts_uncertainty_band_is_deterministic() -> None:
    params = {"query_id": "widest_band_x_label"}
    out_a = ChartsUncertaintyBandWidthExtremumXLabelTask().generate(244000, params=params, max_attempts=60)
    out_b = ChartsUncertaintyBandWidthExtremumXLabelTask().generate(244000, params=params, max_attempts=60)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_charts_uncertainty_band_tasks_are_default_enabled() -> None:
    task_ids = set(list_default_task_ids())
    assert ChartsUncertaintyBandOverlapCountTask.task_id in task_ids
    assert ChartsUncertaintyBandWidthExtremumXLabelTask.task_id in task_ids
