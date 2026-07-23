"""Behavior tests for scientific multi-panel chart tasks."""

from __future__ import annotations
from collections import Counter
import pytest
from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task
from trace_tasks.tasks.charts.scientific_axis_frame.axis_span_value import AXIS_SPAN_QUERY_IDS
from trace_tasks.tasks.charts.scientific_axis_frame.tick_spacing_value import TICK_SPACING_QUERY_IDS

SUPPORTED_SCENE_VARIANTS = ("multipanel_line_grid",)
CURVE_PANEL_QUERY_TO_TASK_ID = {
    "cross_panel_delta_extremum_label": "task_charts__curve_panels__cross_panel_delta_extremum_label",
    "cross_panel_upward_threshold_earliest_label": "task_charts__curve_panels__cross_panel_threshold_earliest_label",
    "cross_panel_downward_threshold_earliest_label": "task_charts__curve_panels__cross_panel_threshold_earliest_label",
    "curve_at_x_extremum_label": "task_charts__curve_panels__curve_at_x_extremum_label",
    "curve_intersection_count": "task_charts__curve_panels__curve_intersection_count",
    "end_highest_panel_label": "task_charts__curve_panels__endpoint_rank_panel_label",
    "end_lowest_panel_label": "task_charts__curve_panels__endpoint_rank_panel_label",
    "earliest_maximum_panel_label": "task_charts__curve_panels__earliest_maximum_panel_label",
    "largest_panel_spread_label": "task_charts__curve_panels__panel_spread_extremum_label",
    "overall_maximum_value_panel_label": "task_charts__curve_panels__global_value_extremum_panel_label",
    "overall_minimum_value_panel_label": "task_charts__curve_panels__global_value_extremum_panel_label",
    "panel_curve_upward_threshold_crossing_count": "task_charts__curve_panels__panel_curve_threshold_crossing_count",
    "panel_curve_downward_threshold_crossing_count": "task_charts__curve_panels__panel_curve_threshold_crossing_count",
    "smallest_panel_spread_label": "task_charts__curve_panels__panel_spread_extremum_label",
    "start_highest_panel_label": "task_charts__curve_panels__endpoint_rank_panel_label",
    "start_lowest_panel_label": "task_charts__curve_panels__endpoint_rank_panel_label",
    "above_threshold_series_count": "task_charts__curve_panels__threshold_series_count",
    "below_threshold_series_count": "task_charts__curve_panels__threshold_series_count",
}
SUPPORTED_QUERY_IDS = tuple(CURVE_PANEL_QUERY_TO_TASK_ID)
AXIS_FRAME_QUERY_IDS = tuple(TICK_SPACING_QUERY_IDS) + tuple(AXIS_SPAN_QUERY_IDS)
KEYED_POINT_QUERIES = {
    "cross_panel_delta_extremum_label",
    "largest_panel_spread_label",
    "smallest_panel_spread_label",
}
SCALAR_POINT_QUERIES = {
    "cross_panel_upward_threshold_earliest_label",
    "cross_panel_downward_threshold_earliest_label",
    "curve_at_x_extremum_label",
    "earliest_maximum_panel_label",
    "end_highest_panel_label",
    "end_lowest_panel_label",
    "overall_maximum_value_panel_label",
    "overall_minimum_value_panel_label",
    "start_highest_panel_label",
    "start_lowest_panel_label",
}


def _curve_panel_task_for_query(query_id: str):
    return create_task(CURVE_PANEL_QUERY_TO_TASK_ID[str(query_id)])


def _curve_panel_params_for_query(query_id: str) -> dict[str, str]:
    task = _curve_panel_task_for_query(str(query_id))
    supported = set(str(value) for value in getattr(task, "supported_query_ids", ()))
    if str(query_id) in supported:
        return {"query_id": str(query_id)}
    return {}


def _semantic_variant(execution: dict) -> str:
    return str(execution.get("internal_query_id") or execution["query_id"])


def _annotation_format_index(prompt: str) -> int:
    for marker in ('Annotation format:', 'Format for the "annotation" field'):
        if marker in prompt:
            return int(prompt.index(marker))
    raise AssertionError("prompt is missing an annotation format line")


def _axis_frame_task_for_query(query_id: str):
    if query_id in TICK_SPACING_QUERY_IDS:
        return create_task("task_charts__scientific_axis_frame__tick_spacing_value")
    return create_task("task_charts__scientific_axis_frame__axis_span_value")


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


def _assert_segment_inside_canvas(
    segment: list[list[float]], *, width: int, height: int
) -> None:
    assert len(segment) == 2
    for point in segment:
        _assert_point_inside_canvas(
            [float(value) for value in point], width=int(width), height=int(height)
        )


def _bbox_center(bbox: list[float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [round((x0 + x1) * 0.5, 3), round((y0 + y1) * 0.5, 3)]


def _assert_bbox_map_inside_canvas(
    annotation: dict, *, width: int, height: int
) -> None:
    assert annotation
    for bbox in annotation.values():
        _assert_bbox_inside_canvas(
            [float(value) for value in bbox], width=int(width), height=int(height)
        )


def _assert_point_map_inside_canvas(
    annotation: dict, *, width: int, height: int
) -> None:
    assert annotation
    for point in annotation.values():
        _assert_point_inside_canvas(
            [float(value) for value in point], width=int(width), height=int(height)
        )


def _expected_answer(execution: dict) -> str | int:
    variant = _semantic_variant(execution)
    if variant == "curve_at_x_extremum_label":
        values = {
            str(key): int(value)
            for key, value in execution["values_at_query_x"].items()
        }
        return max(values, key=lambda label: (values[label], label))
    if variant == "above_threshold_series_count":
        threshold = int(execution["threshold_value"])
        values = {
            str(key): int(value)
            for key, value in execution["values_at_query_x"].items()
        }
        return sum((1 for value in values.values() if int(value) > int(threshold)))
    if variant == "below_threshold_series_count":
        threshold = int(execution["threshold_value"])
        values = {
            str(key): int(value)
            for key, value in execution["values_at_query_x"].items()
        }
        return sum((1 for value in values.values() if int(value) < int(threshold)))
    if variant in {
        "panel_curve_upward_threshold_crossing_count",
        "panel_curve_downward_threshold_crossing_count",
    }:
        return len(execution["threshold_crossing_points"])
    if variant == "cross_panel_delta_extremum_label":
        deltas = {
            str(key): int(value) for key, value in execution["deltas_by_panel"].items()
        }
        return max(deltas, key=lambda label: (deltas[label], label))
    if variant in {
        "cross_panel_upward_threshold_earliest_label",
        "cross_panel_downward_threshold_earliest_label",
    }:
        crossing_x = {
            str(key): float(value)
            for key, value in execution["threshold_crossing_x_by_panel"].items()
        }
        return min(crossing_x, key=lambda label: (crossing_x[label], label))
    if variant == "curve_intersection_count":
        return int(execution["intersection_count"])
    if variant == "earliest_maximum_panel_label":
        peak_x = {
            str(key): int(value) for key, value in execution["peak_x_by_panel"].items()
        }
        return min(peak_x, key=lambda label: (peak_x[label], label))
    if variant == "overall_maximum_value_panel_label":
        extrema = {
            str(key): int(value) for key, value in execution["panel_extrema"].items()
        }
        return max(extrema, key=lambda label: (extrema[label], label))
    if variant == "overall_minimum_value_panel_label":
        extrema = {
            str(key): int(value) for key, value in execution["panel_extrema"].items()
        }
        return min(extrema, key=lambda label: (extrema[label], label))
    if variant in {
        "start_highest_panel_label",
        "end_highest_panel_label",
    }:
        values = {
            str(key): int(value)
            for key, value in execution["endpoint_values_by_panel"].items()
        }
        return max(values, key=lambda label: (values[label], label))
    if variant in {
        "start_lowest_panel_label",
        "end_lowest_panel_label",
    }:
        values = {
            str(key): int(value)
            for key, value in execution["endpoint_values_by_panel"].items()
        }
        return min(values, key=lambda label: (values[label], label))
    if variant == "largest_panel_spread_label":
        spreads = {
            str(key): int(value) for key, value in execution["panel_spreads"].items()
        }
        return max(spreads, key=lambda label: (spreads[label], label))
    if variant == "smallest_panel_spread_label":
        spreads = {
            str(key): int(value) for key, value in execution["panel_spreads"].items()
        }
        return min(spreads, key=lambda label: (spreads[label], label))
    raise AssertionError(f"unsupported variant: {variant}")


@pytest.mark.parametrize("query_id", SUPPORTED_QUERY_IDS)
def test_charts_scientific_variants_match_contract(query_id: str) -> None:
    task = _curve_panel_task_for_query(query_id)
    out = task.generate(
        93100 + SUPPORTED_QUERY_IDS.index(query_id),
        params=_curve_panel_params_for_query(query_id),
        max_attempts=80,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    if str(query_id) in set(str(value) for value in task.supported_query_ids):
        assert out.query_id == query_id
    else:
        assert out.query_id == "single"
    assert _semantic_variant(execution) == query_id
    assert out.scene_id == "curve_panels"
    assert sorted(out.prompt_variants.keys()) == [
        "answer_and_annotation",
        "answer_only",
    ]
    assert str(execution["question_format"]) == "curve_panels_subplot_query"
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 4 <= int(execution["panel_count"]) <= 9
    assert 3 <= int(execution["method_count"]) <= 6
    assert 4 <= int(len(execution["x_values"])) <= 10
    x_values = [int(value) for value in execution["x_values"]]
    assert x_values[0] == 0
    assert (
        len(
            {
                x_values[index + 1] - x_values[index]
                for index in range(len(x_values) - 1)
            }
        )
        == 1
    )
    expected_answer = _expected_answer(execution)
    assert out.answer_gt.value == expected_answer
    assert execution["answer"] == expected_answer
    if query_id in KEYED_POINT_QUERIES:
        assert out.annotation_gt.type == "point_map"
        assert trace["projected_annotation"]["type"] == "point_map"
        assert (
            trace["projected_annotation"]["point_map"] == out.annotation_gt.value
        )
        assert (
            trace["projected_annotation"]["pixel_point_map"]
            == out.annotation_gt.value
        )
    elif query_id in SCALAR_POINT_QUERIES:
        assert out.annotation_gt.type == "point"
        assert trace["projected_annotation"]["type"] == "point"
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
    else:
        assert out.annotation_gt.type == "point_set"
        assert trace["projected_annotation"]["type"] == "point_set"
        assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
        assert (
            trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
        )
    assert trace["render_spec"]["font_assets"]["chart_font_family"]
    if query_id in KEYED_POINT_QUERIES:
        _assert_point_map_inside_canvas(
            out.annotation_gt.value,
            width=int(render["canvas_width"]),
            height=int(render["canvas_height"]),
        )
    elif query_id in SCALAR_POINT_QUERIES:
        _assert_point_inside_canvas(
            [float(value) for value in out.annotation_gt.value],
            width=int(render["canvas_width"]),
            height=int(render["canvas_height"]),
        )
    else:
        for point in out.annotation_gt.value:
            _assert_point_inside_canvas(
                [float(value) for value in point],
                width=int(render["canvas_width"]),
                height=int(render["canvas_height"]),
            )
    for panel_label in trace["projected_annotation"]["panel_labels"]:
        _assert_bbox_inside_canvas(
            render_map["panel_bboxes_px"][str(panel_label)],
            width=int(render["canvas_width"]),
            height=int(render["canvas_height"]),
        )
    if query_id in KEYED_POINT_QUERIES:
        expected_keyed_points = {
            str(role): _bbox_center(render_map["point_bboxes_px"][str(point_id)])
            for role, point_id in trace["projected_annotation"][
                "keyed_point_ids"
            ].items()
        }
        assert out.annotation_gt.value == expected_keyed_points
    else:
        expected_points = []
        for point_id in trace["projected_annotation"]["point_ids"]:
            expected_points.append(
                _bbox_center(render_map["point_bboxes_px"][str(point_id)])
            )
        for intersection_id in trace["projected_annotation"]["intersection_ids"]:
            expected_points.append(
                _bbox_center(render_map["intersection_bboxes_px"][str(intersection_id)])
            )
        for crossing_id in trace["projected_annotation"]["threshold_crossing_ids"]:
            expected_points.append(
                _bbox_center(
                    render_map["threshold_crossing_bboxes_px"][str(crossing_id)]
                )
            )
        if query_id in SCALAR_POINT_QUERIES:
            assert len(expected_points) == 1
            assert out.annotation_gt.value == expected_points[0]
        else:
            assert out.annotation_gt.value == expected_points
    if query_id in {
        "above_threshold_series_count",
        "below_threshold_series_count",
    }:
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == len(
            trace["projected_annotation"]["point_ids"]
        )
    elif query_id in {
        "panel_curve_upward_threshold_crossing_count",
        "panel_curve_downward_threshold_crossing_count",
    }:
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == len(
            trace["projected_annotation"]["threshold_crossing_ids"]
        )
    elif query_id == "curve_intersection_count":
        assert out.answer_gt.type == "integer"
        assert int(out.answer_gt.value) == len(
            trace["projected_annotation"]["intersection_ids"]
        )
    else:
        assert out.answer_gt.type == "string"


def test_charts_scientific_prompt_examples_match_contract() -> None:
    for index, query_id in enumerate(SUPPORTED_QUERY_IDS, start=93200):
        task = _curve_panel_task_for_query(query_id)
        out = task.generate(
            index,
            params=_curve_panel_params_for_query(query_id),
            max_attempts=80,
        )
        answer_and_annotation = extract_prompt_json_example(
            out.prompt_variants["answer_and_annotation"]
        )
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        if query_id in KEYED_POINT_QUERIES:
            assert isinstance(answer_and_annotation["annotation"], dict)
            annotation_keys = set(
                (str(key) for key in answer_and_annotation["annotation"])
            )
            if query_id == "cross_panel_delta_extremum_label":
                assert annotation_keys == {"start_point", "end_point"}
            else:
                assert annotation_keys == {"min_point", "max_point"}
        elif query_id in SCALAR_POINT_QUERIES:
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) == 2
            assert all(
                isinstance(value, (int, float))
                for value in answer_and_annotation["annotation"]
            )
        else:
            assert isinstance(answer_and_annotation["annotation"], list)
        if out.answer_gt.type == "integer":
            assert isinstance(answer_and_annotation["answer"], int)
            assert isinstance(answer_only["answer"], int)
        else:
            assert isinstance(answer_and_annotation["answer"], str)
            assert isinstance(answer_only["answer"], str)


def test_charts_scientific_cross_panel_delta_supports_unanswerable_missing_panel_method() -> None:
    task = create_task(
        "task_charts__curve_panels__cross_panel_delta_extremum_label"
    )
    out = task.generate(
        123456,
        params={"force_unanswerable": True},
        max_attempts=80,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scene_relations = trace["scene_ir"]["relations"]
    witness = trace["witness_symbolic"]
    projected = trace["projected_annotation"]

    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "unanswerable"
    assert execution["answer"] == "unanswerable"
    assert out.annotation_gt.type == "point_map"
    assert out.annotation_gt.value == {}
    assert projected["type"] == "point_map"
    assert projected["point_map"] == {}
    assert projected["pixel_point_map"] == {}
    assert projected["keyed_point_ids"] == {}
    assert execution["annotation_keyed_point_ids"] == {}
    assert execution["annotation_panel_labels"] == []

    requested_method = str(execution["requested_method_label"])
    visible_methods = [str(label) for label in execution["method_labels"]]
    assert requested_method == str(execution["method_label"])
    assert requested_method in set(visible_methods)
    missing_panels = [str(label) for label in execution["missing_method_panel_labels"]]
    present_panels = [str(label) for label in execution["present_method_panel_labels"]]
    assert missing_panels
    assert present_panels
    values_by_panel = execution["values_by_panel_method"]
    for panel_label in missing_panels:
        assert requested_method not in values_by_panel[str(panel_label)]
    for panel_label in present_panels:
        assert requested_method in values_by_panel[str(panel_label)]
    assert execution["answerability"] == "unanswerable"
    assert scene_relations["answerability"] == "unanswerable"
    assert witness["answerability"] == "unanswerable"
    assert execution["absence_proof"]["requested_item"] == (
        f"{requested_method} in every subplot"
    )
    assert execution["absence_proof"]["visible_candidate_set"] == [
        f"{requested_method} in {panel_label}" for panel_label in present_panels
    ]
    assert "unanswerable" in out.prompt.lower()
    assert "not plotted in every subplot" in out.prompt.lower()
    assert out.prompt.index("not plotted in every subplot") < _annotation_format_index(
        out.prompt
    )


def test_charts_scientific_cross_panel_delta_defines_unanswerable_before_formats() -> None:
    task = create_task(
        "task_charts__curve_panels__cross_panel_delta_extremum_label"
    )
    out = task.generate(
        123457,
        params={"force_unanswerable": False},
        max_attempts=80,
    )

    assert out.answer_gt.value != "unanswerable"
    assert "not plotted in every subplot" in out.prompt.lower()
    assert out.prompt.index("not plotted in every subplot") < _annotation_format_index(
        out.prompt
    )


@pytest.mark.parametrize("query_id", AXIS_FRAME_QUERY_IDS)
def test_charts_scientific_axis_frame_variants_match_contract(query_id: str) -> None:
    task = _axis_frame_task_for_query(query_id)
    out = task.generate(
        93600 + AXIS_FRAME_QUERY_IDS.index(query_id),
        params={"query_id": query_id},
        max_attempts=80,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert out.query_id == query_id
    assert out.scene_id == "scientific_axis_frame"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "segment"
    assert sorted(out.prompt_variants.keys()) == [
        "answer_and_annotation",
        "answer_only",
    ]
    assert str(execution["question_format"]) == "scientific_axis_frame"
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 4 <= len(execution["x_tick_values"]) <= 8
    assert 4 <= len(execution["y_tick_values"]) <= 8
    assert trace["render_spec"]["font_assets"]["chart_font_family"]
    query_params = execution["query_params"]
    if query_id in TICK_SPACING_QUERY_IDS:
        expected_answer = int(query_params["next_tick_value"]) - int(
            query_params["first_tick_value"]
        )
        axis = str(query_params["axis"])
        deltas = execution[f"{axis}_tick_deltas"]
        assert len(set(int(value) for value in deltas)) >= 2
        if query_params["tick_pair_position"] == "first":
            assert int(query_params["tick_pair_index"]) == 0
        else:
            assert int(query_params["tick_pair_index"]) == len(deltas) - 1
    else:
        expected_answer = int(query_params["max_tick_value"]) - int(
            query_params["min_tick_value"]
        )
    expected_annotation = [
        render_map["tick_points_px"][execution["annotation_tick_keys"][0]],
        render_map["tick_points_px"][execution["annotation_tick_keys"][1]],
    ]
    assert int(out.answer_gt.value) == int(expected_answer)
    assert execution["answer_value"] == int(expected_answer)
    assert out.annotation_gt.value == expected_annotation
    assert trace["projected_annotation"]["type"] == "segment"
    assert trace["projected_annotation"]["segment"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_segment"] == out.annotation_gt.value
    _assert_segment_inside_canvas(
        out.annotation_gt.value,
        width=int(render["canvas_width"]),
        height=int(render["canvas_height"]),
    )


def test_charts_scientific_axis_frame_prompt_examples_match_contract() -> None:
    for index, query_id in enumerate(AXIS_FRAME_QUERY_IDS, start=93650):
        task = _axis_frame_task_for_query(query_id)
        out = task.generate(index, params={"query_id": query_id}, max_attempts=80)
        answer_and_annotation = extract_prompt_json_example(
            out.prompt_variants["answer_and_annotation"]
        )
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert isinstance(answer_and_annotation["annotation"], list)
        assert len(answer_and_annotation["annotation"]) == 2
        assert isinstance(answer_and_annotation["answer"], int)
        assert isinstance(answer_only["answer"], int)
        for point in answer_and_annotation["annotation"]:
            assert isinstance(point, list)
            assert len(point) == 2


def test_charts_scientific_balanced_sampling_covers_axes() -> None:
    variants: Counter[str] = Counter()
    curve_answers: Counter[str] = Counter()
    threshold_answers: Counter[int] = Counter()
    panel_crossing_answers: Counter[int] = Counter()
    delta_answers: Counter[str] = Counter()
    threshold_earliest_answers: Counter[str] = Counter()
    intersection_answers: Counter[int] = Counter()
    earliest_answers: Counter[str] = Counter()
    global_extremum_answers: Counter[str] = Counter()
    endpoint_rank_answers: Counter[str] = Counter()
    spread_answers: Counter[str] = Counter()
    for index, query_id in enumerate(SUPPORTED_QUERY_IDS):
        task = _curve_panel_task_for_query(query_id)
        for sample_index in range(24):
            out = task.generate(
                hash64(93300, query_id, sample_index),
                params=_curve_panel_params_for_query(query_id),
                max_attempts=120,
            )
            execution = out.trace_payload["execution_trace"]
            variant = _semantic_variant(execution)
            variants[variant] += 1
            if variant == "curve_at_x_extremum_label":
                curve_answers[str(execution["answer"])] += 1
            elif variant in {
                "above_threshold_series_count",
                "below_threshold_series_count",
            }:
                threshold_answers[int(execution["answer"])] += 1
            elif variant in {
                "panel_curve_upward_threshold_crossing_count",
                "panel_curve_downward_threshold_crossing_count",
            }:
                panel_crossing_answers[int(execution["answer"])] += 1
            elif variant == "cross_panel_delta_extremum_label":
                delta_answers[str(execution["answer"])] += 1
            elif variant in {
                "cross_panel_upward_threshold_earliest_label",
                "cross_panel_downward_threshold_earliest_label",
            }:
                threshold_earliest_answers[str(execution["answer"])] += 1
            elif variant == "curve_intersection_count":
                intersection_answers[int(execution["answer"])] += 1
            elif variant == "earliest_maximum_panel_label":
                earliest_answers[str(execution["answer"])] += 1
            elif variant in {
                "overall_maximum_value_panel_label",
                "overall_minimum_value_panel_label",
            }:
                global_extremum_answers[str(execution["answer"])] += 1
            elif variant in {
                "start_highest_panel_label",
                "start_lowest_panel_label",
                "end_highest_panel_label",
                "end_lowest_panel_label",
            }:
                endpoint_rank_answers[str(execution["answer"])] += 1
            elif variant in {
                "largest_panel_spread_label",
                "smallest_panel_spread_label",
            }:
                spread_answers[str(execution["answer"])] += 1
    assert set(variants) == set(SUPPORTED_QUERY_IDS)
    assert len(curve_answers) >= 20
    assert set(threshold_answers).issubset({1, 2, 3, 4, 5, 6})
    assert {1, 2, 3, 4}.issubset(set(threshold_answers))
    assert set(panel_crossing_answers).issubset({1, 2, 3, 4, 5})
    assert {1, 2, 3, 4}.issubset(set(panel_crossing_answers))
    assert len(delta_answers) >= 4
    assert len(threshold_earliest_answers) >= 4
    assert set(intersection_answers) == {0, 1, 2, 3, 4}
    assert len(earliest_answers) >= 4
    assert len(global_extremum_answers) >= 4
    assert len(endpoint_rank_answers) >= 4
    assert len(spread_answers) >= 4


def test_charts_scientific_intersection_count_review_distribution() -> None:
    task = create_task("task_charts__curve_panels__curve_intersection_count")
    answers: Counter[int] = Counter()
    for index in range(100):
        out = task.generate(
            hash64(20260507, task.task_id, index), params={}, max_attempts=120
        )
        answers[int(out.answer_gt.value)] += 1
    assert set(answers) == {0, 1, 2, 3, 4}
    assert max(answers.values()) <= 25


@pytest.mark.parametrize(
    "task_id,allowed_queries",
    [
        (
            "task_charts__scientific_axis_frame__tick_spacing_value",
            set(TICK_SPACING_QUERY_IDS),
        ),
        (
            "task_charts__scientific_axis_frame__axis_span_value",
            set(AXIS_SPAN_QUERY_IDS),
        ),
    ],
)
def test_charts_scientific_axis_frame_public_task_distribution(
    task_id: str, allowed_queries: set[str]
) -> None:
    task = create_task(task_id)
    answers: Counter[int] = Counter()
    query_ids: Counter[str] = Counter()
    for index in range(100):
        out = task.generate(
            hash64(20260605, task.task_id, index), params={}, max_attempts=120
        )
        answers[int(out.answer_gt.value)] += 1
        query_ids[str(out.query_id)] += 1
    assert set(query_ids).issubset(set(allowed_queries))
    assert set(query_ids) == set(allowed_queries)
    assert max(answers.values()) <= 25


def test_charts_scientific_is_deterministic() -> None:
    task = create_task("task_charts__curve_panels__cross_panel_delta_extremum_label")
    params = {}
    out_a = task.generate(93400, params=params, max_attempts=80)
    out_b = task.generate(93400, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )


def test_charts_scientific_axis_frame_is_deterministic() -> None:
    params = {"query_id": "x_axis_span_value"}
    task = _axis_frame_task_for_query(str(params["query_id"]))
    out_a = task.generate(93690, params=params, max_attempts=80)
    out_b = task.generate(93690, params=params, max_attempts=80)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )


def test_charts_scientific_registered_and_group_config_loaded() -> None:
    assert (
        create_task("task_charts__curve_panels__curve_at_x_extremum_label").task_id
        == "task_charts__curve_panels__curve_at_x_extremum_label"
    )
    cfg = get_scene_defaults("charts", "curve_panels")
    assert isinstance(cfg.get("generation"), dict)
    assert isinstance(cfg.get("rendering"), dict)
    assert isinstance(cfg.get("prompt"), dict)
    generation = cfg["generation"]["shared"]
    assert int(generation["panel_count_min"]) == 4
    assert int(generation["method_count_min"]) == 3
    assert int(generation["method_count_max"]) == 6
    assert int(generation["x_tick_count_min"]) == 4
    assert int(generation["x_tick_count_max"]) == 10
    assert int(generation["x_step_min"]) == 5
    assert int(generation["x_step_max"]) == 20
    prompt = cfg["prompt"]["shared"]
    assert str(prompt["bundle_id"]) == "charts_curve_panels_v1"


def test_scientific_curve_at_x_public_task_uses_calibrated_density() -> None:
    task = create_task("task_charts__curve_panels__curve_at_x_extremum_label")
    out = task.generate(2026052301, params={}, max_attempts=120)
    execution = out.trace_payload["execution_trace"]
    assert out.query_id == "single"
    assert _semantic_variant(execution) == "curve_at_x_extremum_label"
    assert 4 <= int(execution["panel_count"]) <= 9
    assert int(execution["method_count"]) == 6
    assert 8 <= len(execution["x_values"]) <= 10
    values = {
        str(key): int(value) for key, value in execution["values_at_query_x"].items()
    }
    answer = str(out.answer_gt.value)
    assert answer in values
    assert values[answer] == max(values.values())
    assert (
        min(
            (
                values[answer] - value
                for key, value in values.items()
                if str(key) != answer
            )
        )
        >= 4
    )
