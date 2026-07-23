"""Behavior tests for pictogram/waffle chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.charts.pictogram.category_total_extremum_label import (
    CATEGORY_TOTAL_EXTREMUM_QUERY_IDS,
    LARGEST_TOTAL_QUERY_ID,
    SMALLEST_TOTAL_QUERY_ID,
    ChartsPictogramCategoryTotalExtremumLabelTask,
)
from trace_tasks.tasks.charts.pictogram.category_total_value import ChartsPictogramCategoryTotalValueTask
from trace_tasks.tasks.charts.pictogram.group_difference_value import ChartsPictogramGroupDifferenceValueTask
from trace_tasks.tasks.charts.pictogram.shared.state import SUPPORTED_GLYPHS, SUPPORTED_SCENE_VARIANTS
from trace_tasks.tasks.charts.pictogram.target_value_nearest_category_label import ChartsPictogramTargetValueNearestCategoryLabelTask
from trace_tasks.tasks.charts.pictogram.threshold_count import (
    GREATER_THAN_QUERY_ID,
    LESS_THAN_QUERY_ID,
    ChartsPictogramThresholdCountTask,
)
from trace_tasks.tasks.registry import create_task


TASK_CASES = (
    (ChartsPictogramCategoryTotalExtremumLabelTask, LARGEST_TOTAL_QUERY_ID, "bbox", "string"),
    (ChartsPictogramCategoryTotalExtremumLabelTask, SMALLEST_TOTAL_QUERY_ID, "bbox", "string"),
    (ChartsPictogramCategoryTotalValueTask, "single", "bbox", "integer"),
    (ChartsPictogramGroupDifferenceValueTask, "single", "bbox_map", "integer"),
    (ChartsPictogramTargetValueNearestCategoryLabelTask, "single", "bbox", "string"),
    (ChartsPictogramThresholdCountTask, GREATER_THAN_QUERY_ID, "bbox_set", "integer"),
    (ChartsPictogramThresholdCountTask, LESS_THAN_QUERY_ID, "bbox_set", "integer"),
)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _expected_answer(execution: dict, query_params: dict, query_id: str) -> int | str:
    totals_by_category = {str(label): int(value) for label, value in execution["totals_by_category"].items()}
    if query_id in CATEGORY_TOTAL_EXTREMUM_QUERY_IDS:
        target_total = max(totals_by_category.values()) if query_id == LARGEST_TOTAL_QUERY_ID else min(totals_by_category.values())
        winners = [label for label, value in totals_by_category.items() if int(value) == int(target_total)]
        assert winners == [str(query_params["answer_category_label"])]
        return str(query_params["answer_category_label"])
    if "answer_category_label" in query_params:
        target_value = int(query_params["target_value"])
        distances = {label: abs(int(value) - target_value) for label, value in totals_by_category.items()}
        best_distance = min(distances.values())
        winners = [label for label, distance in distances.items() if int(distance) == int(best_distance)]
        assert winners == [str(query_params["answer_category_label"])]
        return str(query_params["answer_category_label"])
    if query_id == "single" and "target_category_label" in query_params:
        return int(totals_by_category[str(query_params["target_category_label"])])
    if query_id == "single":
        a = int(totals_by_category[str(query_params["category_label_a"])])
        b = int(totals_by_category[str(query_params["category_label_b"])])
        return abs(a - b)
    threshold = int(query_params["threshold_value"])
    if str(query_params["threshold_direction"]) == "greater_than":
        return sum(1 for value in totals_by_category.values() if int(value) > threshold)
    return sum(1 for value in totals_by_category.values() if int(value) < threshold)


@pytest.mark.parametrize(("task_cls", "query_id", "expected_annotation_type", "expected_answer_type"), TASK_CASES)
def test_charts_pictogram_tasks_match_contract(
    task_cls: type,
    query_id: str,
    expected_annotation_type: str,
    expected_answer_type: str,
) -> None:
    task = task_cls()
    out = task.generate(91200 + len(task_cls.task_id) + len(query_id), params={"query_id": query_id}, max_attempts=60)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    query_params = trace["query_spec"]["params"]

    assert isinstance(create_task(task_cls.task_id), task_cls)
    assert out.scene_id == "pictogram"
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert str(query_params["query_id"]) == query_id
    assert out.answer_gt.type == expected_answer_type
    assert out.annotation_gt.type == expected_annotation_type
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
    assert str(execution["question_format"]) == "pictogram_quantity"
    assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
    assert str(render["glyph_name"]) in SUPPORTED_GLYPHS
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert int(render["canvas_width"]) * int(render["canvas_height"]) <= 1_280_000
    assert 6 <= int(execution["category_count"]) <= 10
    assert 1 <= int(execution["unit_scale"]) <= 5

    expected = _expected_answer(execution, query_params, query_id)
    assert out.answer_gt.value == expected
    assert execution["answer_value"] == expected
    assert trace["projected_annotation"]["type"] == expected_annotation_type

    annotation_category_ids = [str(value) for value in trace["projected_annotation"]["category_ids"]]
    annotation_category_labels = [str(value) for value in trace["projected_annotation"]["category_labels"]]
    expected_boxes = [render_map["category_bboxes_px"][category_id] for category_id in annotation_category_ids]
    if expected_annotation_type == "bbox":
        assert out.annotation_gt.value == expected_boxes[0]
        assert trace["projected_annotation"]["bbox"] == expected_boxes[0]
        boxes_to_check = [out.annotation_gt.value]
    elif expected_annotation_type == "bbox_map":
        expected_keyed = {
            str(execution["category_id_to_label"][category_id]): box
            for category_id, box in zip(annotation_category_ids, expected_boxes, strict=True)
        }
        assert out.annotation_gt.value == expected_keyed
        assert trace["projected_annotation"]["bbox_map"] == expected_keyed
        assert trace["projected_annotation"]["pixel_bbox_map"] == expected_keyed
        assert trace["projected_annotation"]["bbox_set"] == list(expected_keyed.values())
        boxes_to_check = list(out.annotation_gt.value.values())
    else:
        assert out.annotation_gt.value == expected_boxes
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        boxes_to_check = list(out.annotation_gt.value)

    assert annotation_category_labels == [
        str(execution["category_id_to_label"][category_id]) for category_id in annotation_category_ids
    ]
    for bbox in boxes_to_check:
        _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
    if task_cls in {
        ChartsPictogramCategoryTotalExtremumLabelTask,
        ChartsPictogramCategoryTotalValueTask,
        ChartsPictogramTargetValueNearestCategoryLabelTask,
    }:
        assert len(boxes_to_check) == 1
    elif task_cls is ChartsPictogramGroupDifferenceValueTask:
        assert len(boxes_to_check) == 2
    else:
        assert int(out.answer_gt.value) == len(out.annotation_gt.value)
        assert 1 <= int(out.answer_gt.value) <= 5


def test_charts_pictogram_prompt_examples_match_contract() -> None:
    for task_cls, query_id, expected_annotation_type, expected_answer_type in TASK_CASES:
        out = task_cls().generate(
            91300 + len(task_cls.task_id) + len(query_id),
            params={"query_id": query_id},
            max_attempts=60,
        )
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert isinstance(answer_and_annotation["answer"], str if expected_answer_type == "string" else int)
        if expected_annotation_type == "bbox":
            assert isinstance(answer_and_annotation["annotation"], list)
            assert len(answer_and_annotation["annotation"]) == 4
        elif expected_annotation_type == "bbox_map":
            assert isinstance(answer_and_annotation["annotation"], dict)
        else:
            assert isinstance(answer_and_annotation["annotation"], list)
            assert all(isinstance(item, list) and len(item) == 4 for item in answer_and_annotation["annotation"])
        assert isinstance(answer_only["answer"], str if expected_answer_type == "string" else int)


def test_charts_pictogram_balanced_sampling_covers_scene_and_glyph_axes() -> None:
    task = ChartsPictogramThresholdCountTask()
    scenes: Counter[str] = Counter()
    glyphs: Counter[str] = Counter()
    for index in range(48):
        out = task.generate(
            hash64(91400, "charts_pictogram", index),
            params={"query_id": GREATER_THAN_QUERY_ID},
            max_attempts=60,
        )
        execution = out.trace_payload["execution_trace"]
        render = out.trace_payload["render_spec"]
        scenes[str(execution["scene_variant"])] += 1
        glyphs[str(render["glyph_name"])] += 1
    assert set(scenes) == set(SUPPORTED_SCENE_VARIANTS)
    assert set(glyphs) == set(SUPPORTED_GLYPHS)


def test_charts_pictogram_is_deterministic() -> None:
    params = {"query_id": "single", "scene_variant": "pictogram_rows", "glyph_name": "star"}
    out_a = ChartsPictogramGroupDifferenceValueTask().generate(91500, params=params, max_attempts=60)
    out_b = ChartsPictogramGroupDifferenceValueTask().generate(91500, params=params, max_attempts=60)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
