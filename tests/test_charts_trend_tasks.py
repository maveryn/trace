"""Behavior tests for single-series chart trend tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.tasks.charts.single_series.endpoint_change_value import ChartsTrendEndpointChangeValueTask
from trace_tasks.tasks.charts.single_series.interval_rate_value import ChartsTrendIntervalRateValueTask
from trace_tasks.tasks.charts.single_series.monotone_streak_length import ChartsTrendMonotoneStreakLengthTask
from trace_tasks.tasks.charts.single_series.observed_threshold_crossing_label import ChartsTrendObservedThresholdCrossingLabelTask
from trace_tasks.tasks.charts.single_series.turning_point_count import ChartsTrendTurningPointCountTask
from trace_tasks.tasks.registry import list_default_task_ids


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _assert_value_axis_covers_values(render: dict, values: list[int]) -> None:
    assert int(render["value_axis_min"]) <= min(int(value) for value in values)
    assert max(int(value) for value in values) <= int(render["value_axis_max"])
    assert int(render["value_axis_span"]) == int(render["value_axis_max"]) - int(render["value_axis_min"])
    assert set(int(value) for value in render["y_ticks"]).issubset(
        set(int(value) for value in render["value_axis_minor_ticks"])
    )


def _step_signs(values: list[int]) -> list[int]:
    return [1 if int(right) > int(left) else -1 for left, right in zip(values[:-1], values[1:])]


def _peak_labels(labels: list[str], values: list[int]) -> list[str]:
    signs = _step_signs(values)
    return sorted(str(labels[index + 1]) for index in range(len(signs) - 1) if signs[index] > 0 and signs[index + 1] < 0)


def _trough_labels(labels: list[str], values: list[int]) -> list[str]:
    signs = _step_signs(values)
    return sorted(str(labels[index + 1]) for index in range(len(signs) - 1) if signs[index] < 0 and signs[index + 1] > 0)


def _longest_run_labels(labels: list[str], values: list[int], *, increasing: bool) -> list[str]:
    signs = _step_signs(values)
    target = 1 if increasing else -1
    runs: list[list[str]] = []
    index = 0
    while index < len(signs):
        if signs[index] != target:
            index += 1
            continue
        start = index
        while index + 1 < len(signs) and signs[index + 1] == target:
            index += 1
        end = index + 1
        runs.append([str(labels[position]) for position in range(start, end + 1)])
        index += 1
    winners = [list(run) for run in runs if len(run) == max(len(candidate) for candidate in runs)]
    assert len(winners) == 1
    return sorted(str(label) for label in winners[0])


def _first_crossing_index(values: list[int], *, threshold: int, comparison: str, start_index: int = 0) -> int:
    for index in range(int(start_index), len(values)):
        value = int(values[index])
        if str(comparison) == "greater_than" and value > int(threshold):
            return int(index)
        if str(comparison) == "less_than" and value < int(threshold):
            return int(index)
    raise AssertionError("no threshold crossing found")


def test_chart_trend_turning_and_streak_tasks_match_contract() -> None:
    cases = (
        (ChartsTrendTurningPointCountTask(), "peak_turning_point_count", "line"),
        (ChartsTrendTurningPointCountTask(), "trough_turning_point_count", "area"),
        (ChartsTrendMonotoneStreakLengthTask(), "longest_increasing_streak_length", "bar"),
        (ChartsTrendMonotoneStreakLengthTask(), "longest_decreasing_streak_length", "dot_plot"),
    )
    for seed, (task, query_id, scene_variant) in enumerate(cases, start=16010):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": scene_variant}, max_attempts=10)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        labels = [str(label) for label in execution["labels"]]
        values = [int(value) for value in execution["values"]]
        annotation_labels = [str(label) for label in execution["annotation_labels"]]
        annotation_points = [list(point) for point in out.annotation_gt.value]

        assert str(out.query_id) == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "point_set"
        assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
        assert str(execution["scene_variant"]) == scene_variant
        assert str(render["scene_variant"]) == scene_variant
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert annotation_labels == sorted(annotation_labels)
        assert trace["projected_annotation"]["point_set"] == annotation_points
        assert "label_set" not in trace["projected_annotation"]
        assert len(trace["projected_annotation"]["bbox_set"]) == len(annotation_labels)
        assert str(trace["query_spec"]["query_id"]) == query_id
        assert len(trace["scene_ir"]["entities"]) == int(execution["mark_count"])
        assert {str(entity["attrs"]["label"]) for entity in trace["scene_ir"]["entities"]} == set(labels)
        assert set(trace["render_map"]["label_centers_px"]) == set(labels)
        _assert_value_axis_covers_values(render, values)
        assert all(int(left) != int(right) for left, right in zip(values[:-1], values[1:]))

        if query_id == "peak_turning_point_count":
            expected = _peak_labels(labels, values)
        elif query_id == "trough_turning_point_count":
            expected = _trough_labels(labels, values)
        elif query_id == "longest_increasing_streak_length":
            expected = _longest_run_labels(labels, values, increasing=True)
        else:
            expected = _longest_run_labels(labels, values, increasing=False)
        assert annotation_labels == expected
        assert int(out.answer_gt.value) == len(expected)
        assert int(out.answer_gt.value) == int(execution["answer_value"])


def test_chart_trend_prompts_describe_chart_order() -> None:
    task = ChartsTrendTurningPointCountTask()
    line = task.generate(16021, params={"query_id": "peak_turning_point_count", "scene_variant": "line"}, max_attempts=10)
    horizontal = task.generate(
        16022,
        params={"query_id": "peak_turning_point_count", "scene_variant": "horizontal_bar"},
        max_attempts=10,
    )
    dot_plot = task.generate(
        16023,
        params={"query_id": "peak_turning_point_count", "scene_variant": "dot_plot"},
        max_attempts=10,
    )
    assert "displayed order" in str(line.prompt)
    assert "top to bottom" in str(horizontal.prompt)
    assert "left to right" in str(dot_plot.prompt)


def test_chart_trend_supports_all_ordered_scene_variants() -> None:
    task = ChartsTrendTurningPointCountTask()
    prompts = {}
    for seed, scene_variant in enumerate(("area", "bar", "horizontal_bar", "line", "dot_plot", "lollipop"), start=16024):
        out = task.generate(
            seed,
            params={"query_id": "peak_turning_point_count", "scene_variant": scene_variant},
            max_attempts=10,
        )
        prompts[str(scene_variant)] = str(out.prompt)
        assert str(out.trace_payload["execution_trace"]["scene_variant"]) == str(scene_variant)
        assert str(out.trace_payload["render_spec"]["scene_variant"]) == str(scene_variant)
    assert "left to right" in prompts["area"]
    assert "left to right" in prompts["bar"]
    assert "top to bottom" in prompts["horizontal_bar"]
    assert "left to right" in prompts["line"]
    assert "left to right" in prompts["dot_plot"]
    assert "left to right" in prompts["lollipop"]
    for scene_variant in ("scatter", "pie", "radar"):
        with pytest.raises((ValueError, RuntimeError)):
            task.generate(
                16040,
                params={"query_id": "peak_turning_point_count", "scene_variant": scene_variant},
                max_attempts=10,
            )


def test_chart_trend_prompt_examples_match_selected_variant() -> None:
    cases = (
        (ChartsTrendTurningPointCountTask(), "peak_turning_point_count", {"annotation": [[220, 180], [520, 210]], "answer": 2}),
        (
            ChartsTrendMonotoneStreakLengthTask(),
            "longest_increasing_streak_length",
            {"annotation": [[180, 420], [300, 350], [420, 260], [540, 170]], "answer": 4},
        ),
    )
    for index, (task, query_id, expected) in enumerate(cases, start=16050):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected
        assert answer_only == {"answer": expected["answer"]}


def test_chart_trend_task_is_deterministic() -> None:
    task = ChartsTrendMonotoneStreakLengthTask()
    params = {"query_id": "longest_increasing_streak_length", "scene_variant": "line"}
    out_a = task.generate(16061, params=params, max_attempts=10)
    out_b = task.generate(16061, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_chart_trend_supports_zero_turning_point_answers() -> None:
    task = ChartsTrendTurningPointCountTask()
    for seed, query_id in enumerate(("peak_turning_point_count", "trough_turning_point_count"), start=16070):
        out = task.generate(
            seed,
            params={"query_id": query_id, "target_answer_min": 0, "target_answer_max": 0},
            max_attempts=10,
        )
        assert int(out.answer_gt.value) == 0
        assert list(out.annotation_gt.value) == []


def test_chart_trend_supports_explicit_mark_count_10() -> None:
    cases = (
        (ChartsTrendTurningPointCountTask(), "peak_turning_point_count"),
        (ChartsTrendTurningPointCountTask(), "trough_turning_point_count"),
        (ChartsTrendMonotoneStreakLengthTask(), "longest_increasing_streak_length"),
        (ChartsTrendMonotoneStreakLengthTask(), "longest_decreasing_streak_length"),
    )
    for seed, (task, query_id) in enumerate(cases, start=16080):
        out = task.generate(
            seed,
            params={"query_id": query_id, "scene_variant": "bar", "mark_count": 10},
            max_attempts=10,
        )
        assert int(out.trace_payload["execution_trace"]["mark_count"]) == 10


def test_chart_trend_threshold_crossing_tasks_match_contract() -> None:
    cases = (
        (ChartsTrendObservedThresholdCrossingLabelTask(), "observed_above_threshold_crossing_label", "observed", "above", "line"),
        (ChartsTrendObservedThresholdCrossingLabelTask(), "observed_below_threshold_crossing_label", "observed", "below", "bar"),
    )
    for seed, (task, query_id, crossing_mode, crossing_direction, scene_variant) in enumerate(cases, start=16100):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": scene_variant}, max_attempts=10)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        values = [int(value) for value in execution["values"]]
        labels = [str(label) for label in execution["labels"]]
        threshold = int(execution["threshold"])
        comparison = str(execution["comparison"])
        answer_index = int(execution["answer_index"])
        annotation_labels = [str(label) for label in execution["annotation_labels"]]
        ordered_annotation_labels = [str(label) for label in execution["ordered_annotation_labels"]]
        annotation_point = [float(value) for value in out.annotation_gt.value]

        assert str(out.query_id) == query_id
        assert str(execution["crossing_mode"]) == crossing_mode
        assert str(execution["crossing_direction"]) == crossing_direction
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "point"
        assert str(out.answer_gt.value) == str(labels[answer_index])
        assert str(execution["scene_variant"]) == scene_variant
        assert str(trace["render_spec"]["scene_variant"]) == scene_variant
        assert str(threshold) in str(out.prompt)
        assert "red dashed" not in str(out.prompt).lower()
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
        assert "point_sequence" not in trace["projected_annotation"]
        assert "pixel_point_sequence" not in trace["projected_annotation"]
        assert "label_set" not in trace["projected_annotation"]
        assert "ordered_label_set" not in trace["projected_annotation"]
        assert len(annotation_point) == 2
        assert len(trace["projected_annotation"]["bbox_set"]) == 1
        assert set(trace["render_map"]["label_centers_px"]) == set(labels)
        assert {str(entity["attrs"]["label"]) for entity in trace["scene_ir"]["entities"]} == set(labels)

        expected_index = _first_crossing_index(values, threshold=threshold, comparison=comparison)
        assert answer_index == expected_index
        assert not execution["projected_labels"]
        expected_annotation = str(labels[answer_index])
        assert annotation_labels == [expected_annotation]
        assert ordered_annotation_labels == [expected_annotation]


@pytest.mark.parametrize(
    ("task_cls", "task_id", "expected_mode", "query_id"),
    (
        (
            ChartsTrendObservedThresholdCrossingLabelTask,
            "task_charts__single_series__observed_threshold_crossing_label",
            "observed",
            "observed_above_threshold_crossing_label",
        ),
    ),
)
def test_chart_trend_threshold_crossing_public_tasks_fix_crossing_mode(
    task_cls: type,
    task_id: str,
    expected_mode: str,
    query_id: str,
) -> None:
    task = task_cls()
    assert task_id in list_default_task_ids()
    out = task.generate(16110, params={"query_id": query_id, "scene_variant": "line"}, max_attempts=10)
    trace = out.trace_payload
    assert str(out.query_id) == query_id
    assert str(trace["execution_trace"]["crossing_mode"]) == expected_mode
    assert str(trace["query_spec"]["params"]["crossing_mode"]) == expected_mode
    with pytest.raises(ValueError, match="query_id"):
        task.generate(16111, params={"query_id": "__unsupported_query_id__"}, max_attempts=10)


def test_chart_trend_threshold_crossing_supports_scene_variants_and_rejects_incompatible_ones() -> None:
    task = ChartsTrendObservedThresholdCrossingLabelTask()
    for seed, scene_variant in enumerate(("area", "bar", "line", "dot_plot", "lollipop"), start=16120):
        out = task.generate(
            seed,
            params={"query_id": "observed_above_threshold_crossing_label", "scene_variant": scene_variant},
            max_attempts=10,
        )
        assert str(out.trace_payload["execution_trace"]["scene_variant"]) == str(scene_variant)
    for scene_variant in ("horizontal_bar", "scatter", "radar", "pie", "donut"):
        with pytest.raises((ValueError, RuntimeError)):
            task.generate(
                16140,
                params={"query_id": "observed_above_threshold_crossing_label", "scene_variant": scene_variant},
                max_attempts=10,
            )


def test_chart_trend_threshold_crossing_prompt_examples_match_selected_variant() -> None:
    cases = (
        (ChartsTrendObservedThresholdCrossingLabelTask(), "observed_above_threshold_crossing_label"),
        (ChartsTrendObservedThresholdCrossingLabelTask(), "observed_below_threshold_crossing_label"),
    )
    for index, (task, query_id) in enumerate(cases, start=16150):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert isinstance(answer_and_annotation["annotation"], list)
        assert len(answer_and_annotation["annotation"]) == 2
        assert all(isinstance(value, int) for value in answer_and_annotation["annotation"])
        assert isinstance(answer_and_annotation["answer"], str)
        assert answer_only == {"answer": answer_and_annotation["answer"]}

def test_chart_trend_threshold_crossing_uses_bounded_value_axis() -> None:
    task = ChartsTrendObservedThresholdCrossingLabelTask()
    out = task.generate(
        16190,
        params={"query_id": "observed_below_threshold_crossing_label", "scene_variant": "bar"},
        max_attempts=10,
    )
    render_spec = out.trace_payload["render_spec"]
    execution = out.trace_payload["execution_trace"]
    _assert_value_axis_covers_values(render_spec, [int(value) for value in execution["values"]])
    assert 25 <= int(render_spec["value_axis_span"]) <= 100
    assert len(render_spec["y_ticks"]) <= 13


def test_chart_trend_interval_change_tasks_match_contract() -> None:
    cases = (
        (ChartsTrendEndpointChangeValueTask(), "absolute_endpoint_change_value", "line"),
        (ChartsTrendEndpointChangeValueTask(), "signed_endpoint_change_value", "bar"),
        (ChartsTrendEndpointChangeValueTask(), "percent_endpoint_change_value", "dot_plot"),
        (ChartsTrendIntervalRateValueTask(), "single", "horizontal_bar"),
    )
    for seed, (task, query_id, scene_variant) in enumerate(cases, start=16200):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": scene_variant}, max_attempts=10)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        labels = [str(label) for label in execution["labels"]]
        values = [int(value) for value in execution["values"]]
        start_index = int(execution["start_index"])
        end_index = int(execution["end_index"])
        start_value = int(execution["start_value"])
        end_value = int(execution["end_value"])
        delta = int(execution["delta"])
        gap = int(execution["interval_gap"])

        assert str(out.query_id) == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "point_map"
        assert sorted(out.annotation_gt.value) == ["end_mark", "start_mark"]
        assert str(execution["scene_variant"]) == scene_variant
        assert str(render["scene_variant"]) == scene_variant
        assert labels[start_index] == str(execution["start_label"])
        assert labels[end_index] == str(execution["end_label"])
        assert values[start_index] == start_value
        assert values[end_index] == end_value
        assert end_index - start_index == gap
        assert end_value - start_value == delta
        if query_id == "absolute_endpoint_change_value":
            assert int(out.answer_gt.value) == abs(delta)
        elif query_id == "signed_endpoint_change_value":
            assert int(out.answer_gt.value) == delta
        elif query_id == "percent_endpoint_change_value":
            assert int(out.answer_gt.value) == int(round(100 * delta / start_value))
            assert "negative" in str(out.prompt).lower()
            assert "percent sign" in str(out.prompt).lower()
        else:
            assert query_id == "single"
            assert int(out.answer_gt.value) == abs(int(delta // gap))
            assert int(out.answer_gt.value) >= 0
            assert "absolute" in str(out.prompt).lower()
        assert int(out.answer_gt.value) == int(execution["answer_value"])
        _assert_value_axis_covers_values(render, values)


def test_chart_trend_interval_change_task_is_deterministic() -> None:
    task = ChartsTrendEndpointChangeValueTask()
    params = {"query_id": "signed_endpoint_change_value", "scene_variant": "line"}
    out_a = task.generate(16250, params=params, max_attempts=10)
    out_b = task.generate(16250, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()
