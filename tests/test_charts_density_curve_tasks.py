"""Behavior tests for smooth density-curve chart tasks."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.charts.density_curve.density_at_x_extremum_label import (
    ChartsDistributionDensityCurveDensityAtXExtremumLabelTask,
)
from trace_tasks.tasks.charts.density_curve.interval_mass_extremum_label import (
    ChartsDistributionDensityCurveIntervalMassExtremumLabelTask,
)
from trace_tasks.tasks.charts.density_curve.mean_extremum_label import (
    ChartsDistributionDensityCurveMeanExtremumLabelTask,
)
from trace_tasks.tasks.charts.density_curve.mode_location_extremum_label import (
    ChartsDistributionDensityCurveModeLocationExtremumLabelTask,
)
from trace_tasks.tasks.charts.density_curve.shared.defaults import (
    DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E,
    SCENE_ID,
    SUPPORTED_CURVE_LINE_STYLES,
    SUPPORTED_DENSITY_FAMILIES,
)
from trace_tasks.tasks.registry import create_task
from trace_tasks.tasks.shared.color_distance import color_distance


TASK_CASES = (
    (
        ChartsDistributionDensityCurveMeanExtremumLabelTask,
        "highest_mean_label",
        "answer_mean_marker",
        "mean_marker_bboxes_px",
        "mean_x_by_label",
        "max",
        "point",
    ),
    (
        ChartsDistributionDensityCurveMeanExtremumLabelTask,
        "lowest_mean_label",
        "answer_mean_marker",
        "mean_marker_bboxes_px",
        "mean_x_by_label",
        "min",
        "point",
    ),
    (
        ChartsDistributionDensityCurveModeLocationExtremumLabelTask,
        "leftmost_mode_label",
        "answer_mode_marker",
        "mode_marker_bboxes_px",
        "mode_x_by_label",
        "min",
        "point",
    ),
    (
        ChartsDistributionDensityCurveModeLocationExtremumLabelTask,
        "rightmost_mode_label",
        "answer_mode_marker",
        "mode_marker_bboxes_px",
        "mode_x_by_label",
        "max",
        "point",
    ),
    (
        ChartsDistributionDensityCurveIntervalMassExtremumLabelTask,
        "greatest_interval_mass_label",
        "answer_interval_mass",
        "interval_mass_points_px",
        "interval_mass_by_label",
        "max",
        "point",
    ),
    (
        ChartsDistributionDensityCurveIntervalMassExtremumLabelTask,
        "least_interval_mass_label",
        "answer_interval_mass",
        "interval_mass_points_px",
        "interval_mass_by_label",
        "min",
        "point",
    ),
    (
        ChartsDistributionDensityCurveDensityAtXExtremumLabelTask,
        "highest_density_at_x_label",
        "answer_density_at_x",
        "density_at_x_points_px",
        "density_at_x_by_label",
        "max",
        "point",
    ),
    (
        ChartsDistributionDensityCurveDensityAtXExtremumLabelTask,
        "lowest_density_at_x_label",
        "answer_density_at_x",
        "density_at_x_points_px",
        "density_at_x_by_label",
        "min",
        "point",
    ),
)

TASK_CLASSES = (
    ChartsDistributionDensityCurveMeanExtremumLabelTask,
    ChartsDistributionDensityCurveModeLocationExtremumLabelTask,
    ChartsDistributionDensityCurveIntervalMassExtremumLabelTask,
    ChartsDistributionDensityCurveDensityAtXExtremumLabelTask,
)
ALL_QUERY_IDS = tuple(query_id for _task_cls, query_id, *_rest in TASK_CASES)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _bbox_contains(outer: list[float], inner: list[float]) -> bool:
    return (
        float(outer[0]) <= float(inner[0])
        and float(outer[1]) <= float(inner[1])
        and float(outer[2]) >= float(inner[2])
        and float(outer[3]) >= float(inner[3])
    )


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x_value, y_value = [float(value) for value in point]
    assert 0 <= x_value <= width
    assert 0 <= y_value <= height


def _bbox_contains_point(outer: list[float], point: list[float]) -> bool:
    return float(outer[0]) <= float(point[0]) <= float(outer[2]) and float(outer[1]) <= float(point[1]) <= float(outer[3])


def _bbox_center(bbox: list[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _expected_label(values_by_label: dict[str, float], mode: str) -> str:
    if str(mode) == "max":
        return max(sorted(values_by_label), key=lambda label: (float(values_by_label[label]), label))
    return min(sorted(values_by_label), key=lambda label: (float(values_by_label[label]), label))


@pytest.mark.parametrize(
    (
        "task_cls",
        "query_id",
        "annotation_key",
        "render_map_key",
        "metric_key",
        "mode",
        "annotation_type",
    ),
    TASK_CASES,
)
def test_charts_density_curve_tasks_match_contract(
    task_cls: type,
    query_id: str,
    annotation_key: str,
    render_map_key: str,
    metric_key: str,
    mode: str,
    annotation_type: str,
) -> None:
    out = task_cls().generate(
        hash64(20260606, "density_curve_contract", ALL_QUERY_IDS.index(query_id)),
        params={"query_id": query_id},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    width, height = (int(render["canvas_width"]), int(render["canvas_height"]))
    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert str(execution["question_format"]) == "density_curve_label_selection"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == annotation_type
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert out.image.size == (width, height)
    assert str(render["font_asset_version"])
    assert str(render["chart_font_family"])
    assert str(render["font_assets"]["chart_font_family"]) == str(render["chart_font_family"])
    assert 4 <= int(execution["curve_count"]) <= 6
    assert len(execution["labels"]) == int(execution["curve_count"])
    assert len(set(execution["labels"])) == int(execution["curve_count"])
    assert set(execution["families_by_label"].values()).issubset(set(SUPPORTED_DENSITY_FAMILIES))
    assert set(execution["line_style_by_label"].values()).issubset(set(SUPPORTED_CURVE_LINE_STYLES))
    colors_by_label = {
        str(record["label"]): tuple(int(channel) for channel in record["color_rgb"])
        for record in execution["curve_records"]
    }
    assert set(colors_by_label) == set(execution["labels"])
    for first_label, second_label in combinations(sorted(colors_by_label), 2):
        assert (
            color_distance(colors_by_label[first_label], colors_by_label[second_label], distance_space="lab")
            >= DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E
        )
    assert float(execution["min_curve_pairwise_lab_distance"]) >= DEFAULT_DENSITY_CURVE_PAIRWISE_DELTA_E
    line_style_rendering = render["line_style_rendering"]
    assert float(line_style_rendering["dash_off_px"]) <= 5.0
    assert float(line_style_rendering["dash_off_px"]) < float(line_style_rendering["dash_on_px"])
    assert float(line_style_rendering["dot_spacing_px"]) <= 10.0
    assert 0.0 <= float(execution["interval_start"]) < float(execution["interval_end"]) <= 100.0
    assert float(execution["winner_gap"]) >= 0.0
    expected = _expected_label({str(label): float(value) for label, value in execution[metric_key].items()}, mode)
    assert str(out.answer_gt.value) == expected
    assert str(execution["answer"]) == expected
    assert trace["projected_annotation"]["type"] == annotation_type
    if annotation_type == "point":
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point"] == out.annotation_gt.value
    else:
        assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert str(trace["projected_annotation"]["annotation_key"]) == annotation_key
    annotation_witness = list(out.annotation_gt.value)
    source_witness = list(render_map[render_map_key][expected])
    expected_witness = _bbox_center(source_witness) if annotation_type == "point" and len(source_witness) == 4 else source_witness
    assert annotation_witness == expected_witness
    if annotation_type == "point":
        _assert_point_inside_canvas(annotation_witness, width=width, height=height)
        if annotation_key == "answer_mean_marker":
            plot_bbox = render_map["plot_bbox_px"]
            assert float(plot_bbox[0]) <= float(annotation_witness[0]) <= float(plot_bbox[2])
            assert float(annotation_witness[1]) >= float(plot_bbox[3])
        else:
            assert _bbox_contains_point(render_map["plot_bbox_px"], annotation_witness)
    else:
        _assert_bbox_inside_canvas(annotation_witness, width=width, height=height)
        if annotation_key == "answer_mean_marker":
            plot_bbox = render_map["plot_bbox_px"]
            assert float(plot_bbox[0]) <= float(annotation_witness[0]) < float(annotation_witness[2]) <= float(plot_bbox[2])
            assert float(annotation_witness[1]) >= float(plot_bbox[3])
        else:
            assert _bbox_contains(render_map["plot_bbox_px"], annotation_witness)


def test_charts_density_curve_prompt_examples_match_contract() -> None:
    for index, (task_cls, query_id, annotation_key, _render_map_key, _metric_key, _mode, _annotation_type) in enumerate(TASK_CASES):
        out = task_cls().generate(
            hash64(20260606, "density_curve_prompt", index),
            params={"query_id": query_id},
            max_attempts=100,
        )
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert set(answer_only) == {"answer"}
        assert isinstance(answer_only["answer"], str)
        assert set(answer_and_annotation) == {"annotation", "answer"}
        assert isinstance(answer_and_annotation["answer"], str)
        assert isinstance(answer_and_annotation["annotation"], list)
        expected_length = 2 if "point" in _annotation_type else 4
        assert len(answer_and_annotation["annotation"]) == expected_length


def test_charts_density_curve_balanced_sampling_covers_branches_counts_and_families() -> None:
    queries: Counter[str] = Counter()
    counts: Counter[int] = Counter()
    families: Counter[str] = Counter()
    for index in range(96):
        task_cls = TASK_CLASSES[index % len(TASK_CLASSES)]
        task = task_cls()
        task_cursor = index // len(TASK_CLASSES)
        out = task.generate(
            hash64(20260606, "density_curve_balanced", index),
            params={"_sample_cursor": task_cursor},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        queries[str(out.query_id)] += 1
        counts[int(execution["curve_count"])] += 1
        families.update((str(value) for value in execution["families_by_label"].values()))
    assert set(queries) == set(ALL_QUERY_IDS)
    assert min(counts) >= 4
    assert max(counts) <= 6
    assert len(counts) >= 3
    assert set(families).issubset(set(SUPPORTED_DENSITY_FAMILIES))
    assert len(families) >= 5


def test_charts_density_curve_is_deterministic() -> None:
    task = ChartsDistributionDensityCurveMeanExtremumLabelTask()
    seed = hash64(20260606, "density_curve_deterministic")
    first = task.generate(seed, params={}, max_attempts=100)
    second = task.generate(seed, params={}, max_attempts=100)
    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_charts_density_curve_registered_and_group_config_loaded() -> None:
    expected_task_ids = {
        "task_charts__density_curve__density_at_x_extremum_label",
        "task_charts__density_curve__mean_extremum_label",
        "task_charts__density_curve__mode_location_extremum_label",
        "task_charts__density_curve__interval_mass_extremum_label",
    }
    for task_id in expected_task_ids:
        out = create_task(task_id).generate(hash64(20260606, task_id), params={}, max_attempts=100)
        assert out.scene_id == SCENE_ID
        assert out.query_id in ALL_QUERY_IDS
    cfg = get_scene_defaults("charts", "density_curve")
    generation = cfg["generation"]["shared"]
    assert int(generation["density_curve_count_min"]) == 4
    assert int(generation["density_curve_count_max"]) == 6
    assert dict(generation["density_curve_count_weights"]) == {"4": 1.0, "5": 1.0, "6": 1.0}
    prompt = cfg["prompt"]["shared"]
    assert str(prompt["scene_key"]) == "density_curve"
    assert str(prompt["task_key"]) == "density_curve_query"
