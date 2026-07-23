"""Tests for illustration environment-object tasks."""

from __future__ import annotations

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.illustrations.environment.shared.rendering import (
    BRIDGE_STYLE_IDS,
    BUILDING_STYLE_IDS,
    RIVER_STYLE_IDS,
    ROAD_STYLE_IDS,
)


def _bbox_center(box: list[float]) -> list[float]:
    return [
        round((float(box[0]) + float(box[2])) / 2.0, 3),
        round((float(box[1]) + float(box[3])) / 2.0, 3),
    ]


def _assert_bbox_min_side(box: list[float], *, min_side_px: float = 24.0) -> None:
    width = abs(float(box[2]) - float(box[0]))
    height = abs(float(box[3]) - float(box[1]))
    assert width >= float(min_side_px), box
    assert height >= float(min_side_px), box


def _assert_bbox_set_min_side(boxes: list[list[float]], *, min_side_px: float = 24.0) -> None:
    for box in boxes:
        _assert_bbox_min_side(box, min_side_px=float(min_side_px))


def test_feature_relation_object_count_contracts() -> None:
    scenarios = (
        ("park_road", "road", "above_feature", "above"),
        ("river_meadow", "river", "below_feature", "below"),
        ("road_and_river", "road", "below_feature", "below"),
        ("road_and_river", "river", "above_feature", "above"),
        ("canal_city", "river", "below_feature", "below"),
        ("skyline_street", "road", "above_feature", "above"),
        ("road_and_river", "road", "on_feature", "on"),
        ("road_and_river", "river", "on_feature", "on"),
    )
    for index, (theme_id, feature_type, query_id, relation) in enumerate(scenarios):
        out = create_task("task_illustrations__environment__feature_relation_object_count").generate(
            hash64(2026052101, f"{theme_id}:{feature_type}:{query_id}", index),
            params={
                "query_id": query_id,
                "theme_id": theme_id,
                "feature_type": feature_type,
                "object_count": 14,
                "target_count_min": 1,
            },
            max_attempts=400,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render_map = trace["render_map"]
        assert out.scene_id == "environment"
        assert out.query_id == query_id
        assert trace["query_spec"]["query_id"] == query_id
        assert execution["theme_id"] == theme_id
        assert execution["feature_type"] == feature_type
        assert execution["relation"] == relation
        if relation == "on":
            assert all(execution["object_zones"][object_id] == feature_type for object_id in execution["counted_object_ids"])
        layout = trace["render_spec"]["style"]["layout"]
        assert layout["road_style_id"] in ROAD_STYLE_IDS
        assert layout["river_style_id"] in RIVER_STYLE_IDS
        assert layout["bridge_style_id"] in BRIDGE_STYLE_IDS
        target_feature = next(
            entity
            for entity in trace["scene_ir"]["entities"]
            if entity["entity_type"] == "environment_feature" and entity["entity_id"] == execution["feature_id"]
        )
        if feature_type == "road":
            assert target_feature["attributes"]["road_style_id"] in ROAD_STYLE_IDS
        else:
            assert target_feature["attributes"]["river_style_id"] in RIVER_STYLE_IDS
        assert int(out.answer_gt.value) == len(execution["counted_object_ids"])
        assert out.annotation_gt.type == "bbox_set"
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        _assert_bbox_set_min_side(out.annotation_gt.value)
        assert execution["feature_id"] in render_map["feature_bboxes_px"]
        assert execution["feature_id"] in render_map["feature_paths_px"]
        assert render_map["counted_object_ids"] == execution["counted_object_ids"]
        expected = [render_map["object_bboxes_px"][object_id] for object_id in execution["counted_object_ids"]]
        assert sorted(out.annotation_gt.value) == sorted(expected)
        assert trace["projected_annotation"]["type"] == "bbox_set"
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value


def test_crossing_feature_count_contract() -> None:
    out = create_task("task_illustrations__environment__crossing_feature_count").generate(
        hash64(2026052302, "crossing-feature", 0),
        params={"crossing_type": "bridge", "theme_id": "canal_city", "object_count": 12},
        max_attempts=300,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    assert out.scene_id == "environment"
    assert out.query_id == "single"
    assert execution["crossing_type"] == "bridge"
    bridge_features = [
        entity
        for entity in trace["scene_ir"]["entities"]
        if entity["entity_type"] == "environment_feature" and entity["entity_id"] in set(execution["counted_feature_ids"])
    ]
    assert bridge_features
    assert all(feature["attributes"]["bridge_style_id"] in BRIDGE_STYLE_IDS for feature in bridge_features)
    assert int(out.answer_gt.value) == len(execution["counted_feature_ids"])
    assert out.annotation_gt.type == "bbox_set"
    expected = [render_map["feature_bboxes_px"][feature_id] for feature_id in execution["counted_feature_ids"]]
    assert sorted(out.annotation_gt.value) == sorted(expected)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value


def test_building_window_count_contract() -> None:
    out = create_task("task_illustrations__environment__lit_window_count").generate(
        hash64(2026052302, "building-window", 0),
        params={"theme_id": "skyline_street", "target_count": 6, "object_count": 10},
        max_attempts=300,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    assert out.scene_id == "environment"
    assert out.query_id == "single"
    assert execution["window_mode"] == "lit"
    assert int(out.answer_gt.value) == 6
    assert int(out.answer_gt.value) == len(execution["counted_window_ids"])
    buildings = [entity for entity in trace["scene_ir"]["entities"] if entity["entity_type"] == "environment_building"]
    assert buildings
    assert all(building["attributes"]["building_style_id"] in BUILDING_STYLE_IDS for building in buildings)
    assert out.annotation_gt.type == "bbox_set"
    expected = [render_map["window_bboxes_px"][window_id] for window_id in execution["counted_window_ids"]]
    assert sorted(out.annotation_gt.value) == sorted(expected)
    _assert_bbox_set_min_side(out.annotation_gt.value)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value


def test_environment_count_bbox_annotations_respect_min_side_floor() -> None:
    cases = (
        (
            "task_illustrations__environment__feature_relation_object_count",
            {"query_id": "above_feature", "theme_id": "road_and_river", "feature_type": "river"},
            10,
        ),
        (
            "task_illustrations__environment__feature_relation_object_count",
            {"query_id": "below_feature", "theme_id": "road_and_river", "feature_type": "road"},
            10,
        ),
        (
            "task_illustrations__environment__feature_relation_object_count",
            {"query_id": "on_feature", "theme_id": "road_and_river", "feature_type": "river"},
            10,
        ),
        (
            "task_illustrations__environment__lit_window_count",
            {"theme_id": "skyline_street", "target_count": 6},
            12,
        ),
        (
            "task_illustrations__environment__lit_window_count",
            {"theme_id": "canal_city", "target_count": 6},
            12,
        ),
    )
    for task_id, params, sample_count in cases:
        for index in range(sample_count):
            out = create_task(task_id).generate(
                hash64(2026061901, f"{task_id}:min-bbox-side", index),
                params=dict(params),
                max_attempts=500,
            )
            assert out.annotation_gt.type == "bbox_set"
            _assert_bbox_set_min_side(out.annotation_gt.value)


def test_building_window_default_answer_range_is_capped_at_six() -> None:
    generation_defaults, _rendering_defaults, _prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        "illustrations",
        "environment",
        task_id="task_illustrations__environment__lit_window_count",
    )
    assert int(generation_defaults["target_count_min"]) == 1
    assert int(generation_defaults["target_count_max"]) == 6
    for index in range(12):
        out = create_task("task_illustrations__environment__lit_window_count").generate(
            hash64(2026061404, "building-window-range", index),
            params={},
            max_attempts=300,
        )
        assert 1 <= int(out.answer_gt.value) <= 6


def test_feature_relation_default_answer_range_is_one_to_six() -> None:
    generation_defaults, _rendering_defaults, _prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        "illustrations",
        "environment",
        task_id="task_illustrations__environment__feature_relation_object_count",
    )
    assert int(generation_defaults["feature_side_target_count_min"]) == 1
    assert int(generation_defaults["feature_side_target_count_max"]) == 6
    assert int(generation_defaults["on_feature_target_count_min"]) == 1
    assert int(generation_defaults["on_feature_target_count_max"]) == 6
    scenarios = (
        ("above_feature", 1, 6),
        ("below_feature", 1, 6),
        ("on_feature", 1, 6),
    )
    for query_id, low, high in scenarios:
        for index in range(8):
            out = create_task("task_illustrations__environment__feature_relation_object_count").generate(
                hash64(2026070401, f"feature-relation-range:{query_id}", index),
                params={"query_id": query_id},
                max_attempts=400,
            )
            assert int(low) <= int(out.answer_gt.value) <= int(high)


def test_missing_patch_label_contract() -> None:
    out = create_task("task_illustrations__environment__missing_patch_label").generate(
        hash64(2026061401, "environment-missing-patch", 0),
        params={"theme_id": "road_and_river", "source_object_count": 14, "option_count": 4, "correct_index": 2},
        max_attempts=300,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value
    assert out.scene_id == "environment"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert execution["patch_mode"] == "plain"
    assert execution["selected_transform"] == "none"
    assert execution["candidate_crop_count"] == 0
    assert set(annotation) == {"missing_region", "selected_option"}
    assert annotation["missing_region"] == render_map["missing_region_bbox_px"]
    assert annotation["selected_option"] == render_map["selected_option_bbox_px"]
    assert annotation["selected_option"] == render_map["option_bboxes_px_by_label"][out.answer_gt.value]
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert len(render_map["option_bboxes_px_by_label"]) == 4
    assert len(render_map["option_source_crop_boxes_px"]) == 4
    assert render_map["option_source_crop_boxes_px"][2] == render_map["source_crop_box_px"]
    assert trace["query_spec"]["params"]["option_labels"] == ["A", "B", "C", "D"]
    assert trace["query_spec"]["params"]["candidate_crop_count"] == render_map["candidate_crop_count"]


def test_rotated_tile_label_contract() -> None:
    out = create_task("task_illustrations__environment__rotated_tile_label").generate(
        hash64(2026061502, "environment-rotated-tile", 0),
        params={"theme_id": "road_and_river", "source_object_count": 14, "rotation_degrees": 90, "correct_index": 2},
        max_attempts=300,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "environment"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert execution["query_id"] == "single"
    assert execution["answer_label"] == out.answer_gt.value
    assert execution["rotation_degrees"] == 90
    assert execution["grid_shape"] == [2, 3]
    assert len(render_map["tile_bboxes_px_by_label"]) == 6
    assert set(render_map["tile_bboxes_px_by_label"]) == {"A", "B", "C", "D", "E", "F"}
    assert trace["render_spec"]["canvas_size"] == trace["query_spec"]["params"]["source_size"]
    assert render_map["tile_bboxes_px_by_label"]["A"][:2] == [0.0, 0.0]
    assert render_map["tile_bboxes_px_by_label"]["F"][2:] == [
        float(trace["render_spec"]["canvas_size"][0]),
        float(trace["render_spec"]["canvas_size"][1]),
    ]
    assert annotation == render_map["rotated_tile_bbox_px"]
    assert annotation == render_map["tile_bboxes_px_by_label"][out.answer_gt.value]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == annotation
    assert execution["rotated_tile_index"] in execution["usable_tile_indices"]
