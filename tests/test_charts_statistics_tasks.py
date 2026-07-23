"""Behavior tests for single-series chart statistics tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.core.type_registry import load_type_registry
from trace_tasks.tasks.charts.single_series.order_statistic_label import ChartsSingleSeriesOrderStatisticLabelTask
from trace_tasks.tasks.charts.single_series.order_statistic_value import ChartsSingleSeriesOrderStatisticValueTask
from trace_tasks.tasks.shared.color_distance import color_distance
from trace_tasks.tasks.shared.named_colors import darken_color


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def _expected_answer(statistic_kind: str, values: list[int], *, rank_n: int | None = None) -> int:
    if str(statistic_kind) == "median":
        ordered = sorted(int(value) for value in values)
        return int(ordered[len(ordered) // 2])
    if str(statistic_kind) == "nth_highest":
        assert rank_n is not None
        return int(sorted(set(int(value) for value in values), reverse=True)[int(rank_n) - 1])
    if str(statistic_kind) == "nth_lowest":
        assert rank_n is not None
        return int(sorted(set(int(value) for value in values))[int(rank_n) - 1])
    raise AssertionError(f"unsupported statistic: {statistic_kind}")


def _expected_label(statistic_kind: str, labels: list[str], values: list[int], *, rank_n: int | None = None) -> str:
    by_label = {str(label): int(value) for label, value in zip(labels, values)}
    if str(statistic_kind) == "median":
        ordered = sorted((int(value), str(label)) for label, value in by_label.items())
        return str(ordered[len(ordered) // 2][1])
    if str(statistic_kind) == "nth_highest":
        assert rank_n is not None
        winning_value = sorted(set(int(value) for value in values), reverse=True)[int(rank_n) - 1]
    elif str(statistic_kind) == "nth_lowest":
        assert rank_n is not None
        winning_value = sorted(set(int(value) for value in values))[int(rank_n) - 1]
    else:
        raise AssertionError(f"unsupported label-answer statistic: {statistic_kind}")
    winners = [str(label) for label, value in by_label.items() if int(value) == int(winning_value)]
    assert len(winners) == 1
    return str(winners[0])


def _assert_value_axis_covers_values(render: dict, values: list[int]) -> None:
    assert bool(render["value_axis_window_enabled"]) is True
    assert int(render["value_axis_min"]) <= min(int(value) for value in values)
    assert max(int(value) for value in values) <= int(render["value_axis_max"])
    assert 1 <= int(render["value_axis_span"]) <= 25
    assert set(int(value) for value in render["y_ticks"]).issubset(
        set(int(value) for value in render["value_axis_minor_ticks"])
    )
    assert render["guide_line_style"] in {"dashed", "dotted", "solid"}
    assert len(render["guide_lines"]) == len(values)


def test_chart_statistics_order_statistic_value_variants_match_contract() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    cases = (
        ("median_order_statistic_value", "median", "line"),
        ("nth_highest_order_statistic_value", "nth_highest", "bar"),
        ("nth_lowest_order_statistic_value", "nth_lowest", "dot_plot"),
    )
    for seed, (query_id, statistic_kind, scene_variant) in enumerate(cases, start=9100):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": scene_variant}, max_attempts=10)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        assert str(out.query_id) == query_id
        assert str(execution["statistic_kind"]) == statistic_kind
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "point"
        assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
        assert str(execution["scene_variant"]) == scene_variant
        assert str(render["scene_variant"]) == scene_variant
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        labels = [str(label) for label in execution["labels"]]
        values = [int(value) for value in execution["values"]]
        annotation_label = str(trace["projected_annotation"]["label"])
        assert len(labels) == int(execution["mark_count"])
        assert trace["projected_annotation"]["point"] == list(out.annotation_gt.value)
        assert trace["projected_annotation"]["pixel_point"] == list(out.annotation_gt.value)
        assert len(trace["projected_annotation"]["bbox_set"]) == 1
        assert int(out.answer_gt.value) == _expected_answer(
            statistic_kind,
            values,
            rank_n=int(execution["rank_n"]) if execution.get("rank_n") is not None else None,
        )
        assert int(out.answer_gt.value) == int(execution["answer_value"])
        assert str(trace["query_spec"]["query_id"]) == query_id
        assert str(trace["query_spec"]["params"]["scene_variant"]) == scene_variant
        assert len(trace["scene_ir"]["entities"]) == int(execution["mark_count"])
        assert {str(entity["attrs"]["label"]) for entity in trace["scene_ir"]["entities"]} == set(labels)
        assert set(trace["render_map"]["label_centers_px"]) == set(labels)
        _assert_value_axis_covers_values(render, values)
        assert int(execution["values_by_label"][annotation_label]) == int(out.answer_gt.value)
        if statistic_kind == "nth_highest":
            assert int(execution["rank_n"]) >= 3
            assert str(execution["rank_direction"]) == "highest"
        elif statistic_kind == "nth_lowest":
            assert int(execution["rank_n"]) >= 3
            assert str(execution["rank_direction"]) == "lowest"


def test_chart_statistics_prompts_mention_value_axis() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    params = {"query_id": "median_order_statistic_value"}
    line = task.generate(9201, params={**params, "scene_variant": "line"}, max_attempts=10)
    dot_plot = task.generate(9202, params={**params, "scene_variant": "dot_plot"}, max_attempts=10)
    bar = task.generate(9203, params={**params, "scene_variant": "bar"}, max_attempts=10)
    assert "y-values" in str(line.prompt)
    assert "y-values" in str(dot_plot.prompt)
    assert "height" in str(bar.prompt)


def test_chart_statistics_supports_allowed_scene_variants() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    prompts = {}
    for seed, scene_variant in enumerate(("bar", "line", "dot_plot", "area", "horizontal_bar"), start=9205):
        out = task.generate(
            seed,
            params={"query_id": "nth_highest_order_statistic_value", "scene_variant": scene_variant},
            max_attempts=10,
        )
        prompts[str(scene_variant)] = str(out.prompt)
        assert str(out.trace_payload["execution_trace"]["scene_variant"]) == str(scene_variant)
        assert str(out.trace_payload["render_spec"]["scene_variant"]) == str(scene_variant)
        _assert_value_axis_covers_values(
            out.trace_payload["render_spec"],
            [int(value) for value in out.trace_payload["execution_trace"]["values"]],
        )
    assert "height" in prompts["bar"]
    assert "y-values" in prompts["line"]
    assert "y-values" in prompts["dot_plot"]
    assert "y-values" in prompts["area"]
    assert "horizontal axis" in prompts["horizontal_bar"]
    for scene_variant in ("pie", "donut", "radar"):
        with pytest.raises((ValueError, RuntimeError)):
            task.generate(
                9215,
                params={"query_id": "nth_highest_order_statistic_value", "scene_variant": scene_variant},
                max_attempts=10,
            )


def test_chart_statistics_value_prompt_examples_match_selected_variant() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    expected = {
        "median_order_statistic_value": {"annotation": [300, 240], "answer": 6},
        "nth_highest_order_statistic_value": {"annotation": [300, 240], "answer": 6},
        "nth_lowest_order_statistic_value": {"annotation": [300, 240], "answer": 6},
    }
    for index, (query_id, expected_json) in enumerate(expected.items(), start=9300):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_json
        assert answer_only == {"answer": expected_json["answer"]}


def test_chart_statistics_task_is_deterministic() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    params = {"query_id": "nth_lowest_order_statistic_value", "scene_variant": "dot_plot"}
    out_a = task.generate(9401, params=params, max_attempts=10)
    out_b = task.generate(9401, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_chart_statistics_order_statistic_label_variants_match_contract() -> None:
    task = ChartsSingleSeriesOrderStatisticLabelTask()
    cases = (
        ("nth_highest_order_statistic_label", "nth_highest", "bar"),
        ("nth_lowest_order_statistic_label", "nth_lowest", "line"),
        ("median_order_statistic_label", "median", "dot_plot"),
    )
    for seed, (query_id, statistic_kind, scene_variant) in enumerate(cases, start=9450):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": scene_variant}, max_attempts=10)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        labels = [str(label) for label in execution["labels"]]
        values = [int(value) for value in execution["values"]]
        assert str(out.query_id) == query_id
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "point"
        assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
        assert str(execution["scene_variant"]) == scene_variant
        assert str(render["scene_variant"]) == scene_variant
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        rank_n = execution.get("rank_n")
        assert str(out.answer_gt.value) == _expected_label(
            statistic_kind,
            labels,
            values,
            rank_n=int(rank_n) if rank_n is not None else None,
        )
        assert trace["projected_annotation"]["point"] == list(out.annotation_gt.value)
        assert trace["projected_annotation"]["pixel_point"] == list(out.annotation_gt.value)
        assert len(trace["projected_annotation"]["bbox_set"]) == 1
        assert str(trace["query_spec"]["query_id"]) == query_id
        assert str(trace["query_spec"]["params"]["scene_variant"]) == scene_variant
        assert len(trace["scene_ir"]["entities"]) == int(execution["mark_count"])
        assert {str(entity["attrs"]["label"]) for entity in trace["scene_ir"]["entities"]} == set(labels)
        assert set(trace["render_map"]["label_centers_px"]) == set(labels)
        _assert_value_axis_covers_values(render, values)
        annotation_label = str(trace["projected_annotation"]["label"])
        if statistic_kind == "median":
            ordered_values = sorted(int(value) for value in values)
            assert int(execution["values_by_label"][annotation_label]) == int(ordered_values[len(ordered_values) // 2])
        elif statistic_kind == "nth_highest":
            unique_values = sorted(set(int(value) for value in values), reverse=True)
            assert int(execution["rank_n"]) >= 3
            assert int(execution["values_by_label"][annotation_label]) == int(unique_values[int(execution["rank_n"]) - 1])
            assert str(execution["rank_direction"]) == "highest"
        else:
            unique_values = sorted(set(int(value) for value in values))
            assert int(execution["rank_n"]) >= 3
            assert int(execution["values_by_label"][annotation_label]) == int(unique_values[int(execution["rank_n"]) - 1])
            assert str(execution["rank_direction"]) == "lowest"


def test_chart_statistics_label_prompt_examples_match_selected_variant() -> None:
    task = ChartsSingleSeriesOrderStatisticLabelTask()
    expected = {
        "median_order_statistic_label": {"annotation": [420, 180], "answer": "K7P2"},
        "nth_highest_order_statistic_label": {"annotation": [420, 180], "answer": "K7P2"},
        "nth_lowest_order_statistic_label": {"annotation": [420, 180], "answer": "K7P2"},
    }
    for index, (query_id, expected_json) in enumerate(expected.items(), start=9550):
        out = task.generate(index, params={"query_id": query_id}, max_attempts=10)
        answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert answer_and_annotation == expected_json
        assert answer_only == {"answer": expected_json["answer"]}


def test_chart_statistics_label_task_is_deterministic() -> None:
    task = ChartsSingleSeriesOrderStatisticLabelTask()
    params = {"query_id": "median_order_statistic_label", "scene_variant": "dot_plot"}
    out_a = task.generate(9651, params=params, max_attempts=10)
    out_b = task.generate(9651, params=params, max_attempts=10)
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_chart_statistics_label_supports_allowed_scene_variants() -> None:
    task = ChartsSingleSeriesOrderStatisticLabelTask()
    prompts = {}
    for seed, scene_variant in enumerate(("bar", "line", "dot_plot", "area", "horizontal_bar"), start=9660):
        out = task.generate(
            seed,
            params={"query_id": "nth_highest_order_statistic_label", "scene_variant": scene_variant},
            max_attempts=10,
        )
        prompts[str(scene_variant)] = str(out.prompt)
        assert str(out.trace_payload["execution_trace"]["scene_variant"]) == str(scene_variant)
        assert str(out.trace_payload["render_spec"]["scene_variant"]) == str(scene_variant)
    assert "height" in prompts["bar"]
    assert "y-values" in prompts["line"]
    assert "y-values" in prompts["dot_plot"]
    assert "y-values" in prompts["area"]
    assert "horizontal axis" in prompts["horizontal_bar"]
    for scene_variant in ("pie", "donut", "radar"):
        with pytest.raises((ValueError, RuntimeError)):
            task.generate(
                9672,
                params={"query_id": "nth_highest_order_statistic_label", "scene_variant": scene_variant},
                max_attempts=10,
            )


def test_chart_statistics_summary_query_label_default_mark_count_range_is_15_to_17() -> None:
    task = ChartsSingleSeriesOrderStatisticLabelTask()
    out = task.generate(
        9675,
        params={"query_id": "nth_lowest_order_statistic_label", "scene_variant": "bar"},
        max_attempts=10,
    )
    assert 15 <= int(out.trace_payload["execution_trace"]["mark_count"]) <= 17


def test_chart_statistics_summary_query_label_supports_explicit_mark_count_17() -> None:
    task = ChartsSingleSeriesOrderStatisticLabelTask()
    out = task.generate(
        9676,
        params={"query_id": "nth_highest_order_statistic_label", "scene_variant": "bar", "mark_count": 17},
        max_attempts=10,
    )
    assert int(out.trace_payload["execution_trace"]["mark_count"]) == 17
    assert int(out.trace_payload["execution_trace"]["rank_n"]) == 3


def test_chart_statistics_rejects_removed_extremum_variants() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    for query_id in ("argmax", "argmin", "__unsupported_query_id__"):
        with pytest.raises(ValueError, match="query_id"):
            task.generate(9690, params={"query_id": query_id, "scene_variant": "bar"}, max_attempts=10)


def test_point_annotation_type_is_registered_for_chart_label_tasks() -> None:
    registry = load_type_registry()
    assert registry.validate_annotation_type("point") is True


def test_chart_statistics_labels_use_compact_dense_axis_ids() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    out = task.generate(
        9100,
        params={"query_id": "median_order_statistic_value", "scene_variant": "bar"},
        max_attempts=10,
    )
    labels = [str(label) for label in out.trace_payload["execution_trace"]["labels"]]
    assert len(labels) >= 10
    assert len(labels) == len(set(labels))
    assert all(2 <= len(label) <= 4 for label in labels)
    assert all(label.isascii() and label.isalnum() for label in labels)
    assert set(labels) != {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}


def test_chart_statistics_rank_supports_explicit_mark_count_15() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    for seed, query_id in enumerate(("nth_highest_order_statistic_value", "nth_lowest_order_statistic_value"), start=9500):
        out = task.generate(seed, params={"query_id": query_id, "scene_variant": "bar", "mark_count": 15}, max_attempts=10)
        assert int(out.trace_payload["execution_trace"]["mark_count"]) == 15


def test_chart_statistics_median_supports_mark_count_17() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    out = task.generate(
        9601,
        params={"query_id": "median_order_statistic_value", "scene_variant": "line", "mark_count": 17},
        max_attempts=10,
    )
    assert int(out.trace_payload["execution_trace"]["mark_count"]) == 17


def test_chart_statistics_mark_color_is_randomized_and_traced() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    observed_colors = set()
    for seed in range(9700, 9708):
        out = task.generate(
            seed,
            params={"query_id": "median_order_statistic_value", "scene_variant": "bar"},
            max_attempts=10,
        )
        render_spec = out.trace_payload["render_spec"]
        execution = out.trace_payload["execution_trace"]
        fill_rgb = tuple(int(channel) for channel in render_spec["mark_style"]["mark_fill_rgb"])
        outline_rgb = tuple(int(channel) for channel in render_spec["mark_style"]["mark_outline_rgb"])
        assert render_spec["mark_style"]["sampling_policy"] == "random_rgb"
        assert outline_rgb == darken_color(fill_rgb, factor=0.55)
        assert float(render_spec["mark_style"]["mark_color_min_distance"]) == 40.0
        assert str(render_spec["mark_style"]["mark_color_distance_space"]) == "lab"
        assert float(color_distance(fill_rgb, (255, 255, 255), distance_space="lab")) >= 40.0
        assert float(color_distance(fill_rgb, (248, 248, 248), distance_space="lab")) >= 40.0
        assert execution["mark_color_sampling_policy"] == "random_rgb"
        assert tuple(int(channel) for channel in execution["mark_fill_rgb"]) == fill_rgb
        assert tuple(int(channel) for channel in execution["mark_outline_rgb"]) == outline_rgb
        observed_colors.add(fill_rgb)
    assert len(observed_colors) >= 2


def test_chart_statistics_render_uses_traced_mark_fill_color() -> None:
    task = ChartsSingleSeriesOrderStatisticValueTask()
    out = task.generate(
        9801,
        params={"query_id": "median_order_statistic_value", "scene_variant": "bar"},
        max_attempts=10,
    )
    fill_rgb = tuple(int(channel) for channel in out.trace_payload["render_spec"]["mark_style"]["mark_fill_rgb"])
    first_entity = out.trace_payload["scene_ir"]["entities"][0]
    left, top, right, bottom = first_entity["attrs"]["mark_bbox_px"]
    sample_x = int(round((float(left) + float(right)) / 2.0))
    sample_y = int(round(float(top) + 0.7 * (float(bottom) - float(top))))
    pixel = tuple(int(channel) for channel in out.image.convert("RGB").getpixel((sample_x, sample_y)))
    assert pixel == fill_rgb
