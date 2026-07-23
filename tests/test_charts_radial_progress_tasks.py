"""Behavior tests for radial progress chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.charts.radial_progress.extremum_remaining_label import (
    SUPPORTED_QUERY_IDS as REMAINING_EXTREMUM_QUERY_IDS,
    ChartsRadialProgressExtremumRemainingLabelTask,
)
from trace_tasks.tasks.charts.radial_progress.progress_interval_count import (
    ChartsRadialProgressIntervalCountTask,
)
from trace_tasks.tasks.charts.radial_progress.progress_threshold_count import (
    SUPPORTED_QUERY_IDS as PROGRESS_THRESHOLD_QUERY_IDS,
    ChartsRadialProgressThresholdCountTask,
)
from trace_tasks.tasks.charts.radial_progress.shared.state import SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.registry import list_default_task_ids


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _expected_answer(execution: dict) -> int | str:
    items = list(execution["items"])
    condition = str(execution.get("count_condition", ""))
    if condition == "progress_at_least_threshold":
        threshold = int(execution["threshold_value"])
        return sum(1 for item in items if int(item["value"]) >= threshold)
    if condition == "progress_below_threshold":
        threshold = int(execution["threshold_value"])
        return sum(1 for item in items if int(item["value"]) < threshold)
    if condition == "progress_within_range":
        lower = int(execution["range_lower"])
        upper = int(execution["range_upper"])
        return sum(1 for item in items if lower <= int(item["value"]) <= upper)
    extremum = str(execution.get("remaining_extremum", ""))
    if extremum == "highest_remaining":
        return str(min(items, key=lambda item: int(item["value"]))["label"])
    if extremum == "lowest_remaining":
        return str(max(items, key=lambda item: int(item["value"]))["label"])
    raise AssertionError(f"unsupported radial progress execution: {execution}")


RADIAL_COUNT_CASES = (
    (ChartsRadialProgressThresholdCountTask, "at_least_threshold_count"),
    (ChartsRadialProgressThresholdCountTask, "below_threshold_count"),
    (ChartsRadialProgressIntervalCountTask, SINGLE_QUERY_ID),
)


@pytest.mark.parametrize(("task_cls", "query_id"), RADIAL_COUNT_CASES)
def test_charts_radial_progress_task_matches_contract(task_cls: type, query_id: str) -> None:
    task = task_cls()
    out = task.generate(126000 + len(query_id) + len(task.task_id), params={"query_id": query_id}, max_attempts=60)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert task.task_id in list_default_task_ids()
    assert out.scene_id == "radial_progress"
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(execution["question_format"]) == "radial_progress_condition_count"
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 6 <= int(execution["item_count"]) <= 10
    expected = _expected_answer(execution)
    assert out.answer_gt.value == expected
    assert int(execution["answer_value"]) == expected
    assert 1 <= int(out.answer_gt.value) <= 5
    assert int(out.answer_gt.value) == len(out.annotation_gt.value)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert str(render["font_assets"]["font_asset_version"])
    assert str(render["font_assets"]["chart_font_family"])
    annotation_item_ids = [str(value) for value in trace["projected_annotation"]["item_ids"]]
    expected_boxes = [render_map["item_bboxes_px"][item_id] for item_id in annotation_item_ids]
    assert out.annotation_gt.value == expected_boxes
    for bbox in out.annotation_gt.value:
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render["canvas_width"]), height=int(render["canvas_height"]))


@pytest.mark.parametrize("query_id", REMAINING_EXTREMUM_QUERY_IDS)
def test_charts_radial_progress_extremum_remaining_task_matches_contract(query_id: str) -> None:
    task = ChartsRadialProgressExtremumRemainingLabelTask()
    out = task.generate(126400 + len(query_id), params={"query_id": query_id}, max_attempts=60)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert task.task_id in list_default_task_ids()
    assert out.scene_id == "radial_progress"
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox"
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(execution["question_format"]) == "radial_progress_extremum_remaining_label"
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 6 <= int(execution["item_count"]) <= 10
    expected = _expected_answer(execution)
    assert out.answer_gt.value == expected
    assert execution["answer_value"] == expected
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert str(render["font_assets"]["font_asset_version"])
    assert str(render["font_assets"]["chart_font_family"])
    annotation_item_ids = [str(value) for value in trace["projected_annotation"]["item_ids"]]
    assert len(annotation_item_ids) == 1
    expected_box = render_map["item_bboxes_px"][annotation_item_ids[0]]
    assert out.annotation_gt.value == expected_box
    assert trace["projected_annotation"]["item_labels"] == [str(expected)]
    _assert_bbox_inside_canvas([float(value) for value in out.annotation_gt.value], width=int(render["canvas_width"]), height=int(render["canvas_height"]))


def test_charts_radial_progress_prompt_examples_match_contract() -> None:
    out = ChartsRadialProgressThresholdCountTask().generate(127000, params={}, max_attempts=60)
    answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
    assert isinstance(answer_and_annotation["answer"], int)
    assert isinstance(answer_only["answer"], int)
    assert isinstance(answer_and_annotation["annotation"], list)
    assert answer_and_annotation["annotation"] and all(isinstance(box, list) for box in answer_and_annotation["annotation"])
    label_out = ChartsRadialProgressExtremumRemainingLabelTask().generate(127100, params={}, max_attempts=60)
    label_answer_and_annotation = extract_prompt_json_example(label_out.prompt_variants["answer_and_annotation"])
    label_answer_only = extract_prompt_json_example(label_out.prompt_variants["answer_only"])
    assert isinstance(label_answer_and_annotation["answer"], str)
    assert isinstance(label_answer_only["answer"], str)
    assert isinstance(label_answer_and_annotation["annotation"], list)
    assert len(label_answer_and_annotation["annotation"]) == 4


def test_charts_radial_progress_balanced_sampling_covers_axes() -> None:
    scenes: Counter[str] = Counter()
    observed: set[tuple[str, str]] = set()
    for index in range(96):
        task_cls, query_id = RADIAL_COUNT_CASES[int(index) % len(RADIAL_COUNT_CASES)]
        out = task_cls().generate(
            hash64(128000, "charts_radial_progress", index),
            params={"query_id": query_id},
            max_attempts=60,
        )
        scenes[str(out.trace_payload["execution_trace"]["scene_variant"])] += 1
        observed.add((str(out.task_versions.domain if hasattr(out.task_versions, "domain") else task_cls().task_id), str(out.query_id)))
    assert set(scenes) == set(SUPPORTED_SCENE_VARIANTS)
    assert set(PROGRESS_THRESHOLD_QUERY_IDS) == {"at_least_threshold_count", "below_threshold_count"}


def test_charts_radial_progress_remaining_balanced_sampling_covers_axes() -> None:
    scenes: Counter[str] = Counter()
    queries: Counter[str] = Counter()
    for index in range(72):
        out = ChartsRadialProgressExtremumRemainingLabelTask().generate(hash64(128500, "charts_radial_progress_remaining", index), params={}, max_attempts=60)
        scenes[str(out.trace_payload["execution_trace"]["scene_variant"])] += 1
        queries[str(out.query_id)] += 1
    assert set(scenes) == set(SUPPORTED_SCENE_VARIANTS)
    assert set(queries) == set(REMAINING_EXTREMUM_QUERY_IDS)


def test_charts_radial_progress_dark_theme_uses_inactive_track() -> None:
    params = {
        "query_id": "highest_remaining_label",
        "scene_variant": "segmented_radial_bars",
        "information_scene_treatments": ["dark_report_card"],
        "information_scene_palettes": ["dark_mint"],
        "information_scene_chrome_modes": ["none"],
    }
    out = ChartsRadialProgressExtremumRemainingLabelTask().generate(1516849462425006, params=params, max_attempts=60)
    style = out.trace_payload["render_spec"]["render_meta"]["style"]
    track_style = out.trace_payload["render_spec"]["render_meta"]["progress_track_style"]
    card_fill = style["card_fill_rgb"]
    track = style["track_rgb"]
    base_track = track_style["base_track_rgb"]

    def luminance(rgb: list[int]) -> float:
        return (0.2126 * float(rgb[0])) + (0.7152 * float(rgb[1])) + (0.0722 * float(rgb[2]))

    assert track_style["progress_track_policy"] == "dark_inactive_track"
    assert luminance(track) < luminance(card_fill)
    assert luminance(base_track) > luminance(card_fill)


def test_charts_radial_progress_is_deterministic() -> None:
    params = {"scene_variant": "semicircle_gauges", "query_id": SINGLE_QUERY_ID}
    out_a = ChartsRadialProgressIntervalCountTask().generate(129000, params=params, max_attempts=60)
    out_b = ChartsRadialProgressIntervalCountTask().generate(129000, params=params, max_attempts=60)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
