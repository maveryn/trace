"""Behavior tests for single-series chart counting tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.tasks.charts.single_series.interval_value_count import ChartsCountingIntervalValueCountTask
from trace_tasks.tasks.charts.single_series.threshold_value_count import ChartsCountingThresholdValueCountTask


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _expected_count(query_id: str, values: list[int], trace: dict) -> int:
    if str(query_id) == "above_threshold_count":
        threshold = int(trace["threshold"])
        return sum(1 for value in values if int(value) > threshold)
    if str(query_id) == "below_threshold_count":
        threshold = int(trace["threshold"])
        return sum(1 for value in values if int(value) < threshold)
    if str(query_id) == "single":
        interval_min = int(trace["interval_min"])
        interval_max = int(trace["interval_max"])
        return sum(1 for value in values if interval_min <= int(value) <= interval_max)
    raise AssertionError(f"unsupported variant: {query_id}")


def _assert_value_axis_covers_values(render: dict, values: list[int]) -> None:
    assert int(render["value_axis_min"]) <= min(int(value) for value in values)
    assert max(int(value) for value in values) <= int(render["value_axis_max"])
    assert int(render["value_axis_span"]) == int(render["value_axis_max"]) - int(render["value_axis_min"])
    assert set(int(value) for value in render["y_ticks"]).issubset(
        set(int(value) for value in render["value_axis_minor_ticks"])
    )


def test_chart_counting_tasks_match_contract() -> None:
    cases = (
        (ChartsCountingThresholdValueCountTask(), "above_threshold_count", "bar"),
        (ChartsCountingThresholdValueCountTask(), "below_threshold_count", "line"),
        (ChartsCountingIntervalValueCountTask(), "single", "scatter"),
    )
    for seed, (task, query_id, scene_variant) in enumerate(cases, start=9910):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": scene_variant}, max_attempts=10)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        labels = [str(label) for label in execution["labels"]]
        values = [int(value) for value in execution["values"]]
        annotation_points = [list(point) for point in out.annotation_gt.value]
        annotation_labels = [str(label) for label in execution["annotation_labels"]]
        values_by_label = {str(label): int(value) for label, value in execution["values_by_label"].items()}

        assert str(out.query_id) == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "point_set"
        assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
        assert str(execution["scene_variant"]) == scene_variant
        assert str(render["scene_variant"]) == scene_variant
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert annotation_labels == sorted(annotation_labels)
        assert trace["projected_annotation"]["point_set"] == annotation_points
        assert trace["projected_annotation"]["pixel_point_set"] == annotation_points
        assert len(trace["projected_annotation"]["bbox_set"]) == len(annotation_labels)
        assert int(out.answer_gt.value) == int(execution["answer_value"])
        assert int(out.answer_gt.value) == _expected_count(query_id, values, execution)
        assert len(annotation_labels) == int(out.answer_gt.value)
        assert len(annotation_points) == int(out.answer_gt.value)
        assert str(trace["query_spec"]["query_id"]) == query_id
        assert str(trace["query_spec"]["params"]["scene_variant"]) == scene_variant
        assert len(trace["scene_ir"]["entities"]) == int(execution["mark_count"])
        assert {str(entity["attrs"]["label"]) for entity in trace["scene_ir"]["entities"]} == set(labels)
        assert set(trace["render_map"]["label_centers_px"]) == set(labels)
        _assert_value_axis_covers_values(render, values)
        if query_id == "above_threshold_count":
            threshold = int(execution["threshold"])
            assert all(int(values_by_label[label]) > threshold for label in annotation_labels)
        elif query_id == "below_threshold_count":
            threshold = int(execution["threshold"])
            assert all(int(values_by_label[label]) < threshold for label in annotation_labels)
        else:
            interval_min = int(execution["interval_min"])
            interval_max = int(execution["interval_max"])
            assert bool(execution["interval_inclusive"]) is True
            assert all(interval_min <= int(values_by_label[label]) <= interval_max for label in annotation_labels)


def test_chart_counting_prompts_describe_value_axis() -> None:
    line = ChartsCountingThresholdValueCountTask().generate(
        9921,
        params={"query_id": "above_threshold_count", "scene_variant": "line"},
        max_attempts=10,
    )
    scatter = ChartsCountingIntervalValueCountTask().generate(
        9922,
        params={"query_id": "single", "scene_variant": "scatter"},
        max_attempts=10,
    )
    area = ChartsCountingThresholdValueCountTask().generate(
        9923,
        params={"query_id": "below_threshold_count", "scene_variant": "area"},
        max_attempts=10,
    )
    assert "y-values" in str(line.prompt)
    assert "y-values" in str(scatter.prompt)
    assert "y-values" in str(area.prompt)


def test_chart_counting_supports_additional_scene_variants() -> None:
    task = ChartsCountingThresholdValueCountTask()
    prompts = {}
    for seed, scene_variant in enumerate(("area", "horizontal_bar", "dot_plot", "lollipop"), start=9924):
        out = task.generate(
            seed,
            params={"query_id": "above_threshold_count", "scene_variant": scene_variant},
            max_attempts=10,
        )
        prompts[str(scene_variant)] = str(out.prompt)
        assert str(out.trace_payload["execution_trace"]["scene_variant"]) == str(scene_variant)
        assert str(out.trace_payload["render_spec"]["scene_variant"]) == str(scene_variant)
    assert "y-values" in prompts["area"]
    assert "horizontal axis" in prompts["horizontal_bar"]
    assert "y-values" in prompts["dot_plot"]
    assert "y-values" in prompts["lollipop"]


def test_chart_counting_bar_annotation_uses_value_endpoint() -> None:
    task = ChartsCountingThresholdValueCountTask()
    cases = (("bar", 9927), ("horizontal_bar", 9928))
    for scene_variant, seed in cases:
        out = task.generate(
            seed,
            params={
                "query_id": "above_threshold_count",
                "scene_variant": scene_variant,
                "target_answer_min": 3,
                "target_answer_max": 3,
                "guide_line_mode": "always",
            },
            max_attempts=10,
        )
        render = out.trace_payload["render_spec"]
        assert render["guide_line_style"] in {"dashed", "dotted", "solid"}
        assert len(render["guide_lines"]) == int(out.trace_payload["execution_trace"]["mark_count"])
        entities_by_label = {
            str(entity["attrs"]["label"]): entity["attrs"] for entity in out.trace_payload["scene_ir"]["entities"]
        }
        for label, point in out.trace_payload["projected_annotation"]["pixel_point_map"].items():
            bbox = [float(value) for value in entities_by_label[str(label)]["mark_bbox_px"]]
            point_x, point_y = [float(value) for value in point]
            if str(scene_variant) == "bar":
                assert point_x == pytest.approx(0.5 * (bbox[0] + bbox[2]))
                assert point_y == pytest.approx(bbox[1])
            else:
                assert point_x == pytest.approx(bbox[2])
                assert point_y == pytest.approx(0.5 * (bbox[1] + bbox[3]))


def test_chart_counting_excludes_pie_donut_and_radar_scene_variants() -> None:
    task = ChartsCountingThresholdValueCountTask()
    for scene_variant in ("pie", "donut", "radar"):
        with pytest.raises((ValueError, RuntimeError), match="unsupported scene_variant"):
            task.generate(
                9929,
                params={"query_id": "above_threshold_count", "scene_variant": scene_variant},
                max_attempts=10,
            )


def test_chart_counting_prompt_examples_match_selected_variant() -> None:
    cases = (
        (ChartsCountingThresholdValueCountTask(), "above_threshold_count", {"annotation": [[180, 260], [390, 180], [610, 320]], "answer": 3}),
        (ChartsCountingIntervalValueCountTask(), "single", {"annotation": [[165, 310], [380, 250], [590, 285]], "answer": 3}),
    )
    for index, (task, query_id, expected) in enumerate(cases, start=9930):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected
        assert answer_only == {"answer": expected["answer"]}


def test_chart_counting_task_is_deterministic() -> None:
    task = ChartsCountingIntervalValueCountTask()
    params = {"query_id": "single", "scene_variant": "scatter"}
    out_a = task.generate(9941, params=params, max_attempts=10)
    out_b = task.generate(9941, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_chart_counting_supports_zero_answer_with_empty_annotation() -> None:
    cases = (
        (ChartsCountingThresholdValueCountTask(), "above_threshold_count"),
        (ChartsCountingThresholdValueCountTask(), "below_threshold_count"),
        (ChartsCountingIntervalValueCountTask(), "single"),
    )
    for seed, (task, query_id) in enumerate(cases, start=9950):
        out = task.generate(
            seed,
            params={"query_id": query_id, "target_answer_min": 0, "target_answer_max": 0},
            max_attempts=10,
        )
        assert int(out.answer_gt.value) == 0
        assert list(out.annotation_gt.value) == []


def test_chart_counting_supports_explicit_mark_count_20() -> None:
    cases = (
        (ChartsCountingThresholdValueCountTask(), "above_threshold_count"),
        (ChartsCountingThresholdValueCountTask(), "below_threshold_count"),
        (ChartsCountingIntervalValueCountTask(), "single"),
    )
    for seed, (task, query_id) in enumerate(cases, start=9960):
        out = task.generate(
            seed,
            params={"query_id": query_id, "scene_variant": "bar", "mark_count": 20},
            max_attempts=10,
        )
        assert int(out.trace_payload["execution_trace"]["mark_count"]) == 20
