"""Behavior tests for chart size-encoding tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from tests.helpers import extract_prompt_json_example
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.charts.size_encoding.category_relative_size_count import (
    ChartsSizeEncodingCategoryRelativeSizeCountTask,
)
from trace_tasks.tasks.charts.size_encoding.filtered_item_extremum_label import (
    ChartsSizeEncodingFilteredItemExtremumLabelTask,
)
from trace_tasks.tasks.charts.size_encoding.global_item_extremum_category_label import (
    ChartsSizeEncodingGlobalItemExtremumCategoryLabelTask,
)
from trace_tasks.tasks.charts.size_encoding.panel_category_extremum_panel_label import (
    ChartsSizeEncodingPanelCategoryExtremumPanelLabelTask,
)
from trace_tasks.tasks.charts.size_encoding.shared.rendering import render_size_encoding_scene
from trace_tasks.tasks.charts.size_encoding.shared.state import (
    PANEL_SCENE_VARIANTS,
    SINGLE_PANEL_SCENE_VARIANTS,
    SUPPORTED_SCENE_VARIANTS,
    SizeEncodingDataset,
    SizeEncodingItem,
)


TASK_CASES = (
    (
        ChartsSizeEncodingCategoryRelativeSizeCountTask,
        ("larger_than_reference_in_category_count", "smaller_than_reference_in_category_count"),
        "bbox_set_map",
        "integer",
    ),
    (
        ChartsSizeEncodingFilteredItemExtremumLabelTask,
        ("largest_size_item_in_category_label", "smallest_size_item_in_category_label"),
        "bbox",
        "string",
    ),
    (
        ChartsSizeEncodingGlobalItemExtremumCategoryLabelTask,
        ("largest_overall_size_category_label", "smallest_overall_size_category_label"),
        "bbox",
        "string",
    ),
    (
        ChartsSizeEncodingPanelCategoryExtremumPanelLabelTask,
        ("largest_category_item_panel_label", "smallest_category_item_panel_label"),
        "bbox",
        "string",
    ),
)


def _assert_bbox_inside_canvas(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _assert_bbox_min_side(bbox: list[float], *, min_side_px: float = 24.0) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert min(x1 - x0, y1 - y0) >= float(min_side_px)


def _expected_answer(execution: dict[str, Any], query_params: dict[str, Any]) -> str | int:
    values_by_label = {str(label): int(value) for label, value in execution["values_by_label"].items()}
    category_by_label = {str(label): str(value) for label, value in execution["category_by_label"].items()}
    question_format = str(execution["question_format"])
    assert question_format in {"size_encoded_label_comparison", "size_encoded_panel_label_comparison", "size_encoded_relative_count"}

    if str(query_params["query_id"]) in {"larger_than_reference_in_category_count", "smaller_than_reference_in_category_count"}:
        category = str(query_params["category_label"])
        reference = str(query_params["reference_label"])
        reference_value = int(values_by_label[reference])
        candidates = [label for label, value in category_by_label.items() if value == category and label != reference]
        if str(query_params["query_id"]) == "larger_than_reference_in_category_count":
            labels = [label for label in candidates if int(values_by_label[label]) > reference_value]
        else:
            labels = [label for label in candidates if int(values_by_label[label]) < reference_value]
        assert sorted(str(label) for label in execution["counted_labels"]) == sorted(labels)
        return int(len(labels))

    if str(query_params["query_id"]) in {"largest_size_item_in_category_label", "smallest_size_item_in_category_label"}:
        category = str(query_params["category_label"])
        candidates = [label for label, value in category_by_label.items() if value == category]
        reverse = str(execution["extremum_direction"]) == "largest"
        return str(sorted(candidates, key=lambda label: (values_by_label[label], label), reverse=reverse)[0])

    if str(query_params["query_id"]) in {"largest_overall_size_category_label", "smallest_overall_size_category_label"}:
        reverse = str(execution["extremum_direction"]) == "largest"
        winner = str(sorted(values_by_label, key=lambda label: (values_by_label[label], label), reverse=reverse)[0])
        assert str(execution["winner_item_label"]) == winner
        return str(category_by_label[winner])

    if str(query_params["query_id"]) in {"largest_category_item_panel_label", "smallest_category_item_panel_label"}:
        panel_by_label = {str(label): str(value) for label, value in execution["panel_by_label"].items()}
        category = str(query_params["category_label"])
        candidates = [label for label, value in category_by_label.items() if value == category]
        reverse = str(execution["extremum_direction"]) == "largest"
        winner = str(sorted(candidates, key=lambda label: (values_by_label[label], label), reverse=reverse)[0])
        assert str(execution["winner_item_label"]) == winner
        return str(panel_by_label[winner])

    raise AssertionError(f"unsupported branch: {query_params['query_id']}")


@pytest.mark.parametrize(("task_cls", "branches", "annotation_type", "answer_type"), TASK_CASES)
def test_chart_size_encoding_tasks_match_contract(task_cls: type, branches: tuple[str, ...], annotation_type: str, answer_type: str) -> None:
    task = task_cls()
    for branch_index, branch in enumerate(branches):
        out = task.generate(78300 + branch_index, params={"query_id": branch}, max_attempts=160)
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        render_map = trace["render_map"]
        query_params = trace["query_spec"]["params"]

        assert out.query_id == branch
        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == annotation_type
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert str(execution["scene_variant"]) in SUPPORTED_SCENE_VARIANTS
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert 3 <= int(execution["category_count"]) <= 5
        assert 1 <= int(execution["panel_count"]) <= 4
        assert int(execution["item_count"]) == len(execution["values_by_label"])

        expected_answer = _expected_answer(execution, query_params)
        assert out.answer_gt.value == expected_answer
        if answer_type == "integer":
            assert int(execution["answer_value"]) == expected_answer
            assert str(execution["answer_label"]) == str(expected_answer)
        else:
            assert str(execution["answer_label"]) == expected_answer
        assert trace["projected_annotation"]["type"] == annotation_type

        annotation_item_ids = [str(item_id) for item_id in trace["projected_annotation"]["annotation_item_ids"]]
        expected_item_boxes = [render_map["item_bboxes_px"][item_id] for item_id in annotation_item_ids]
        if annotation_type == "bbox":
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            assert out.annotation_gt.value == expected_item_boxes[0]
            annotation_boxes = [list(out.annotation_gt.value)]
            assert len(annotation_item_ids) == 1
            assert 24 <= int(execution["winner_gap"]) <= 36
            if str(query_params["query_id"]) in {"largest_size_item_in_category_label", "smallest_size_item_in_category_label"}:
                assert int(execution["outside_extreme_count"]) >= 2
            elif str(query_params["query_id"]) in {"largest_category_item_panel_label", "smallest_category_item_panel_label"}:
                assert str(execution["winner_panel_label"]) == str(out.answer_gt.value)
                assert int(execution["panel_count"]) == 4
                assert str(execution["scene_variant"]) in PANEL_SCENE_VARIANTS
            else:
                assert str(execution["winner_item_category"]) == str(out.answer_gt.value)
        elif annotation_type == "bbox_set_map":
            assert trace["projected_annotation"]["bbox_set_map"] == out.annotation_gt.value
            ref_id = str(execution["reference_item_id"])
            counted_ids = [str(item_id) for item_id in execution["counted_item_ids"]]
            assert out.annotation_gt.value == {
                "reference_item": [render_map["item_bboxes_px"][ref_id]],
                "counted_items": [render_map["item_bboxes_px"][item_id] for item_id in counted_ids],
            }
            annotation_boxes = [bbox for bboxes in out.annotation_gt.value.values() for bbox in bboxes]
            assert int(out.answer_gt.value) == len(counted_ids)
            assert 1 <= int(out.answer_gt.value) <= 5
            assert int(execution["closest_counted_gap"]) >= int(execution["relative_size_reference_gap_min"])
            assert str(execution["scene_variant"]) in SINGLE_PANEL_SCENE_VARIANTS
        else:
            assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
            assert out.annotation_gt.value == expected_item_boxes
            annotation_boxes = [list(bbox) for bbox in out.annotation_gt.value]
            assert str(out.answer_gt.value) in execution["categories"]
            assert len(out.annotation_gt.value) >= 2

        assert str(render["font_assets"]["font_asset_version"])
        assert str(render["font_assets"]["chart_font_family"])
        for bbox in annotation_boxes:
            _assert_bbox_inside_canvas([float(value) for value in bbox], width=int(render["canvas_width"]), height=int(render["canvas_height"]))
            _assert_bbox_min_side([float(value) for value in bbox])


def test_chart_size_encoding_prompt_examples_match_contract() -> None:
    for task_cls, branches, annotation_type, answer_type in TASK_CASES:
        task = task_cls()
        for index, branch in enumerate(branches, start=78400):
            out = task.generate(index, params={"query_id": branch}, max_attempts=160)
            answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
            answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
            assert isinstance(answer_and_annotation["answer"], int if answer_type == "integer" else str)
            if annotation_type == "bbox":
                assert isinstance(answer_and_annotation["annotation"], list)
                assert len(answer_and_annotation["annotation"]) == 4
            elif annotation_type == "bbox_set_map":
                assert isinstance(answer_and_annotation["annotation"], dict)
                assert sorted(answer_and_annotation["annotation"]) == ["counted_items", "reference_item"]
                assert len(answer_and_annotation["annotation"]["reference_item"]) == 1
                assert all(isinstance(value, list) for value in answer_and_annotation["annotation"]["counted_items"])
            else:
                assert isinstance(answer_and_annotation["annotation"], list)
                assert all(isinstance(value, list) for value in answer_and_annotation["annotation"])
            assert isinstance(answer_only["answer"], int if answer_type == "integer" else str)


def test_chart_size_encoding_sampling_covers_scene_variants_and_branches() -> None:
    variants: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    for task_cls, branches, _annotation_type, _answer_type in TASK_CASES:
        task = task_cls()
        for index, branch in enumerate(branches):
            out = task.generate(hash64(78500, task.task_id, index), params={"query_id": branch}, max_attempts=300)
            execution = out.trace_payload["execution_trace"]
            variants[str(out.query_id)] += 1
            scenes[str(execution["scene_variant"])] += 1
    assert set(variants) == {
        "larger_than_reference_in_category_count",
        "smaller_than_reference_in_category_count",
        "largest_size_item_in_category_label",
        "smallest_size_item_in_category_label",
        "largest_overall_size_category_label",
        "smallest_overall_size_category_label",
        "largest_category_item_panel_label",
        "smallest_category_item_panel_label",
    }
    assert set(scenes).issubset(set(SINGLE_PANEL_SCENE_VARIANTS) | set(PANEL_SCENE_VARIANTS))
    assert any(scene in scenes for scene in PANEL_SCENE_VARIANTS)
    assert len(scenes) >= 3


def test_chart_size_encoding_small_multiple_bubbles_use_global_size_scale() -> None:
    dataset = SizeEncodingDataset(
        items=(
            SizeEncodingItem(item_id="item_00", label="A1", category="Alpha", panel="PanelA", value=99),
            SizeEncodingItem(item_id="item_01", label="A2", category="Beta", panel="PanelA", value=12),
            SizeEncodingItem(item_id="item_02", label="B1", category="Alpha", panel="PanelB", value=50),
            SizeEncodingItem(item_id="item_03", label="B2", category="Beta", panel="PanelB", value=12),
        ),
        categories=("Alpha", "Beta"),
        panels=("PanelA", "PanelB"),
        trace={},
    )
    rendered = render_size_encoding_scene(
        dataset,
        scene_variant="small_multiple_bubble_cloud",
        params={"canvas_width": 900, "canvas_height": 620},
        instance_seed=78700,
    )
    radii = {
        str(entity["entity_id"]): float(entity["attrs"]["radius_px"])
        for entity in rendered.entities
        if entity.get("entity_type") == "chart_size_bubble_item"
    }
    assert radii["item_00"] > radii["item_02"]
    assert rendered.render_meta["size_value_scale_scope"] == "global"


def test_chart_size_encoding_is_deterministic() -> None:
    task = ChartsSizeEncodingFilteredItemExtremumLabelTask()
    params = {"query_id": "largest_size_item_in_category_label", "scene_variant": "packed_bubble_cloud"}
    out_a = task.generate(78600, params=params, max_attempts=160)
    out_b = task.generate(78600, params=params, max_attempts=160)
    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
