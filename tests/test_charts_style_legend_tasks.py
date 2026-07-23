"""Behavior tests for style-legend chart tasks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.style_legend.series_extremum_x_label import ChartsStyleLegendSeriesExtremumXLabelTask
from trace_tasks.tasks.charts.style_legend.shared.state import SUPPORTED_LEGEND_POSITIONS, SUPPORTED_STYLE_PALETTE_MODES
from trace_tasks.tasks.charts.style_legend.threshold_series_count import ChartsStyleLegendThresholdSeriesCountTask
from trace_tasks.tasks.charts.style_legend.x_position_extremum_series_label import ChartsStyleLegendXPositionExtremumSeriesLabelTask
from trace_tasks.tasks.registry import list_default_task_ids


TASK_CASES = (
    (
        ChartsStyleLegendSeriesExtremumXLabelTask,
        ("series_highest_x_label", "series_lowest_x_label"),
        "string",
        "point",
    ),
    (
        ChartsStyleLegendThresholdSeriesCountTask,
        ("above_threshold_series_count", "below_threshold_series_count"),
        "integer",
        "point_set",
    ),
    (
        ChartsStyleLegendXPositionExtremumSeriesLabelTask,
        ("x_position_highest_series_label", "x_position_lowest_series_label"),
        "string",
        "point",
    ),
)


def _assert_point_inside_canvas(point: Sequence[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _expected_answer(task_id: str, execution: dict[str, Any]) -> int | str:
    target_x = int(execution["target_x_index"])
    series = list(execution["series"])
    if task_id == ChartsStyleLegendXPositionExtremumSeriesLabelTask.task_id:
        direction = str(execution["extremum_direction"])
        if direction == "highest":
            winner = max(series, key=lambda item: int(item["values"][target_x]))
        elif direction == "lowest":
            winner = min(series, key=lambda item: int(item["values"][target_x]))
        else:
            raise AssertionError(f"unsupported extremum direction: {direction}")
        return str(winner["label"])
    if task_id == ChartsStyleLegendSeriesExtremumXLabelTask.task_id:
        direction = str(execution["extremum_direction"])
        target_series_id = str(execution["target_series_id"])
        target_series = next(item for item in series if str(item["series_id"]) == target_series_id)
        values = [int(value) for value in target_series["values"]]
        if direction == "highest":
            winner_index = max(range(len(values)), key=lambda index: values[int(index)])
        elif direction == "lowest":
            winner_index = min(range(len(values)), key=lambda index: values[int(index)])
        else:
            raise AssertionError(f"unsupported extremum direction: {direction}")
        return str(execution["x_labels"][int(winner_index)])
    if task_id == ChartsStyleLegendThresholdSeriesCountTask.task_id:
        threshold = int(execution["threshold_value"])
        comparator = str(execution["threshold_comparator"])
        if comparator == "above":
            return sum(1 for item in series if int(item["values"][target_x]) > threshold)
        if comparator == "below":
            return sum(1 for item in series if int(item["values"][target_x]) < threshold)
        raise AssertionError(f"unsupported comparator: {comparator}")
    raise AssertionError(f"unsupported task: {task_id}")


@pytest.mark.parametrize(("task_cls", "query_ids", "answer_type", "annotation_type"), TASK_CASES)
def test_charts_style_legend_tasks_match_contract(
    task_cls: type,
    query_ids: tuple[str, ...],
    answer_type: str,
    annotation_type: str,
) -> None:
    task = task_cls()
    assert tuple(task.supported_query_ids) == tuple(query_ids)
    for index, query_id in enumerate(query_ids):
        out = task.generate(
            hash64(20260605, f"{task_cls.task_id}.{query_id}", index),
            params={"query_id": query_id},
            max_attempts=80,
        )
        execution = out.trace_payload["execution_trace"]
        width = int(out.trace_payload["render_spec"]["canvas_width"])
        height = int(out.trace_payload["render_spec"]["canvas_height"])
        assert out.query_id == query_id
        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == annotation_type
        assert out.answer_gt.value == _expected_answer(task_cls.task_id, execution)
        if annotation_type == "point":
            _assert_point_inside_canvas(out.annotation_gt.value, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["type"] == "point"
        else:
            assert isinstance(out.annotation_gt.value, list)
            for point in out.annotation_gt.value:
                _assert_point_inside_canvas(point, width=width, height=height)
            assert out.trace_payload["projected_annotation"]["type"] == "point_set"


def test_charts_style_legend_prompt_examples_match_contract() -> None:
    for task_cls, _query_ids, answer_type, annotation_type in TASK_CASES:
        out = task_cls().generate(
            hash64(20260605, f"{task_cls.task_id}.prompt", len(task_cls.task_id)),
            params={},
            max_attempts=80,
        )
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if annotation_type == "point":
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) == 2
            assert all(isinstance(value, int) for value in answer_and_annotation["annotation"])
        else:
            assert isinstance(answer_and_annotation["annotation"], list)
            assert all(isinstance(point, list) and len(point) == 2 for point in answer_and_annotation["annotation"])
        if answer_type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert isinstance(answer_only["answer"], str)
            assert len(answer_and_annotation["answer"]) > 1


def test_charts_style_legend_sampling_covers_style_axes() -> None:
    query_counts: Counter[str] = Counter()
    palette_counts: Counter[str] = Counter()
    legend_counts: Counter[str] = Counter()
    for index in range(72):
        out = ChartsStyleLegendThresholdSeriesCountTask().generate(
            hash64(20260605, "charts_style_legend_axes", index),
            params={},
            max_attempts=80,
        )
        execution = out.trace_payload["execution_trace"]
        query_counts[str(out.query_id)] += 1
        palette_counts[str(execution["style_palette_mode"])] += 1
        legend_counts[str(execution["legend_position"])] += 1
    assert set(query_counts) == set(ChartsStyleLegendThresholdSeriesCountTask.supported_query_ids)
    assert set(palette_counts).issubset(set(SUPPORTED_STYLE_PALETTE_MODES))
    assert set(legend_counts).issubset(set(SUPPORTED_LEGEND_POSITIONS))
    assert {"right", "top"}.issubset(set(legend_counts))


def test_charts_style_legend_render_trace_records_series_styles() -> None:
    out = ChartsStyleLegendXPositionExtremumSeriesLabelTask().generate(
        2026060502,
        params={"query_id": "x_position_highest_series_label", "style_palette_mode": "grayscale", "legend_position": "right"},
        max_attempts=80,
    )
    execution = out.trace_payload["execution_trace"]
    render_map = out.trace_payload["render_map"]
    legend_items = render_map["legend_item_bboxes_px"]
    for series in execution["series"]:
        assert str(series["series_id"]) in legend_items
        style = dict(series["style"])
        assert style["line_style"]
        assert style["marker_shape"]
        assert style["marker_fill"]
        assert len(style["color_rgb"]) == 3


def test_charts_style_legend_is_deterministic() -> None:
    params = {"query_id": "series_highest_x_label", "style_palette_mode": "grayscale", "legend_position": "right"}
    out_a = ChartsStyleLegendSeriesExtremumXLabelTask().generate(2026060501, params=params, max_attempts=80)
    out_b = ChartsStyleLegendSeriesExtremumXLabelTask().generate(2026060501, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_charts_style_legend_registry_and_config_are_wired() -> None:
    defaults = get_scene_defaults("charts", "style_legend")
    prompt = defaults["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_style_legend_v1"
    assert ChartsStyleLegendXPositionExtremumSeriesLabelTask.task_id in list_default_task_ids()
    assert ChartsStyleLegendSeriesExtremumXLabelTask.task_id in list_default_task_ids()
    assert ChartsStyleLegendThresholdSeriesCountTask.task_id in list_default_task_ids()
