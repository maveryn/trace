"""Tests for indoor-room illustration tasks."""

from __future__ import annotations

from collections import Counter
from inspect import getsourcefile
from pathlib import Path

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.illustrations.indoor_room.shared.rendering import ANNOTATION_BBOX_MIN_SIDE_PX
from trace_tasks.tasks.illustrations.indoor_room.shared.state import INDOOR_OBJECT_TYPES


SURFACE_TASK_ID = "task_illustrations__indoor_room__surface_object_count"
FURNITURE_TASK_ID = "task_illustrations__indoor_room__furniture_side_count"
ROTATED_TILE_TASK_ID = "task_illustrations__indoor_room__rotated_tile_label"
MISSING_PATCH_TASK_ID = "task_illustrations__indoor_room__missing_patch_label"
SWAPPED_TILE_PAIR_TASK_ID = "task_illustrations__indoor_room__swapped_tile_pair_label"
TASK_SOURCE_STEMS = {
    SURFACE_TASK_ID: "surface_object_count.py",
    FURNITURE_TASK_ID: "furniture_side_count.py",
    ROTATED_TILE_TASK_ID: "rotated_tile_label.py",
    MISSING_PATCH_TASK_ID: "missing_patch_label.py",
    SWAPPED_TILE_PAIR_TASK_ID: "swapped_tile_pair_label.py",
}


def _assert_scene_packaged_task(task_id: str) -> None:
    task = create_task(task_id)
    assert not hasattr(task, "scene_id")
    source = Path(getsourcefile(task.__class__) or "").as_posix()
    assert source.endswith(
        f"src/trace_tasks/tasks/illustrations/indoor_room/{TASK_SOURCE_STEMS[task_id]}"
    )


def _assert_scene_prompt_metadata(trace: dict) -> None:
    prompt_variant = trace["query_spec"]["prompt_variant"]
    assert prompt_variant["prompt_bundle_id"] == "illustrations_indoor_room_v0"
    assert prompt_variant["prompt_scene_id"] == "indoor_room"


def _expected_bboxes(trace: dict, ids: list[str]) -> list[list[float]]:
    boxes = trace["render_map"]["object_bboxes_px"]
    return [boxes[object_id] for object_id in ids]


def _expected_points(trace: dict, ids: list[str]) -> list[list[float]]:
    return [
        [
            round((float(box[0]) + float(box[2])) / 2.0, 3),
            round((float(box[1]) + float(box[3])) / 2.0, 3),
        ]
        for box in _expected_bboxes(trace, ids)
    ]


def _assert_objects_rest_on_surface(trace: dict, ids: list[str], surface_type: str) -> None:
    placements = trace["render_map"]["placements"]
    for object_id in ids:
        bbox_bottom = trace["render_map"]["object_bboxes_px"][object_id][3]
        contact = placements[object_id]["surface_contact_px"]
        assert placements[object_id]["surface_type"] == surface_type
        assert contact is not None
        assert abs(float(bbox_bottom) - float(contact[1])) <= 8.0


def _assert_hash_balanced_counts(counts: Counter, expected_keys) -> None:
    assert sorted(counts) == sorted(expected_keys)
    expected = sum(counts.values()) / max(1, len(counts))
    assert min(counts.values()) >= max(1, int(expected * 0.4))
    assert max(counts.values()) <= int(expected * 1.7) + 1


def _assert_min_bbox_sides(boxes: list[list[float]]) -> None:
    for box in boxes:
        width = float(box[2]) - float(box[0])
        height = float(box[3]) - float(box[1])
        assert min(width, height) >= ANNOTATION_BBOX_MIN_SIDE_PX


def test_object_type_on_surface_count_contract() -> None:
    _assert_scene_packaged_task(SURFACE_TASK_ID)
    out = create_task(SURFACE_TASK_ID).generate(
        hash64(2026052401, "type-on-surface", 0),
        params={"object_type": "mug", "surface_type": "shelf", "target_count": 2, "object_count": 12, "theme_id": "study"},
        max_attempts=80,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    execution = trace["execution_trace"]
    placements = trace["render_map"]["placements"]
    assert out.scene_id == "indoor_room"
    assert out.query_id == "single"
    assert execution["object_type"] == "mug"
    assert execution["surface_type"] == "shelf"
    assert int(out.answer_gt.value) == 2
    assert all(
        placements[object_id]["surface_type"] == "shelf" and placements[object_id]["object_type"] == "mug"
        for object_id in execution["counted_object_ids"]
    )
    _assert_objects_rest_on_surface(trace, execution["counted_object_ids"], "shelf")
    assert out.annotation_gt.type == "bbox_set"
    assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_object_bboxes_px"])
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value


def test_indoor_count_annotations_keep_minimum_bbox_side() -> None:
    for task_id in (SURFACE_TASK_ID, FURNITURE_TASK_ID):
        task = create_task(task_id)
        for index in range(60):
            out = task.generate(
                hash64(2026061905, f"indoor-count-min-bbox-side:{task_id}", index),
                params={},
                max_attempts=120,
            )
            assert out.annotation_gt.type == "bbox_set"
            _assert_min_bbox_sides(out.annotation_gt.value)


def test_object_type_on_surface_count_answer_range() -> None:
    task = create_task(SURFACE_TASK_ID)
    answer_counts: Counter[int] = Counter()
    for index in range(60):
        out = task.generate(
            hash64(2026052401, "type-on-surface-answer-range", index),
            params={},
            max_attempts=80,
        )
        answer_counts[int(out.answer_gt.value)] += 1

    assert sorted(answer_counts) == [1, 2, 3, 4, 5]

    try:
        task.generate(
            hash64(2026052401, "type-on-surface-answer-range-explicit", 0),
            params={"target_count": 6},
            max_attempts=1,
        )
    except RuntimeError as exc:
        assert "target_count is outside configured support" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("target_count=6 should be outside the configured answer range")


def test_counter_objects_rest_on_surface_baseline() -> None:
    out = create_task(SURFACE_TASK_ID).generate(
        hash64(2026052401, "type-on-counter", 0),
        params={"object_type": "mug", "surface_type": "counter", "target_count": 3, "object_count": 13, "theme_id": "kitchen"},
        max_attempts=80,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    placements = trace["render_map"]["placements"]
    surface_ids = [
        object_id
        for object_id, placement in placements.items()
        if placement["surface_type"] == "counter"
    ]
    assert int(out.answer_gt.value) == 3
    assert execution["surface_type"] == "counter"
    _assert_objects_rest_on_surface(trace, surface_ids, "counter")


def test_indoor_object_pool_uses_small_tabletop_items() -> None:
    assert "umbrella" not in INDOOR_OBJECT_TYPES
    assert "banana" not in INDOOR_OBJECT_TYPES
    assert "trash_bin" not in INDOOR_OBJECT_TYPES
    assert "apple" in INDOOR_OBJECT_TYPES
    assert "egg" in INDOOR_OBJECT_TYPES
    assert "spoon" in INDOOR_OBJECT_TYPES
    assert "plate" in INDOOR_OBJECT_TYPES
    assert "book" in INDOOR_OBJECT_TYPES
    assert "remote" in INDOOR_OBJECT_TYPES
    assert "pencil" in INDOOR_OBJECT_TYPES
    assert "ruler" in INDOOR_OBJECT_TYPES
    assert "clock" in INDOOR_OBJECT_TYPES
    assert "vase" in INDOOR_OBJECT_TYPES
    assert "bowl" in INDOOR_OBJECT_TYPES
    assert "candle" in INDOOR_OBJECT_TYPES


def test_furniture_side_count_contract() -> None:
    _assert_scene_packaged_task(FURNITURE_TASK_ID)
    out = create_task(FURNITURE_TASK_ID).generate(
        hash64(2026052401, "furniture-side", 0),
        params={
            "object_type": "mug",
            "furniture_type": "table",
            "relation": "left",
            "target_count": 3,
            "object_count": 10,
            "theme_id": "bedroom",
        },
        max_attempts=80,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    execution = trace["execution_trace"]
    placements = trace["render_map"]["placements"]
    furniture_id = execution["furniture_id"]
    assert out.scene_id == "indoor_room"
    assert out.query_id == "left_side"
    assert execution["object_type"] == "mug"
    assert execution["furniture_type"] == "table"
    assert execution["relation"] == "left"
    assert int(out.answer_gt.value) == 3
    assert all(
        placements[object_id]["relations"][furniture_id]["left"] and placements[object_id]["object_type"] == "mug"
        for object_id in execution["counted_object_ids"]
    )
    assert out.annotation_gt.type == "bbox_set"
    assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_object_bboxes_px"])
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value


def test_furniture_side_count_calibration_sampling_is_decoupled() -> None:
    task = create_task(FURNITURE_TASK_ID)
    answer_counts: Counter[int] = Counter()
    object_type_counts: Counter[str] = Counter()
    furniture_relation_counts: Counter[tuple[str, str]] = Counter()

    for index in range(100):
        out = task.generate(
            hash64(2026052401, "furniture-side-sampling", index),
            params={},
            max_attempts=80,
        )
        execution = out.trace_payload["execution_trace"]
        answer_counts[int(out.answer_gt.value)] += 1
        object_type_counts[str(execution["object_type"])] += 1
        furniture_relation_counts[(str(execution["furniture_type"]), str(execution["relation"]))] += 1

    _assert_hash_balanced_counts(answer_counts, range(1, 7))
    assert len(object_type_counts) >= 18
    assert set(furniture_relation_counts) == {
        ("table", "left"),
        ("table", "right"),
    }


def test_furniture_side_count_query_id_selects_relation() -> None:
    task = create_task(FURNITURE_TASK_ID)
    for index, (query_id, relation) in enumerate(
        (
            ("left_side", "left"),
            ("right_side", "right"),
        )
    ):
        out = task.generate(
            hash64(2026052401, "furniture-query", index),
            params={
                "query_id": query_id,
                "object_type": "mug",
                "target_count": 1,
                "object_count": 8,
                "theme_id": "living_room",
            },
            max_attempts=80,
        )
        execution = out.trace_payload["execution_trace"]
        assert out.query_id == query_id
        assert execution["query_id"] == query_id
        assert execution["relation"] == relation


def test_rotated_tile_label_contract() -> None:
    _assert_scene_packaged_task(ROTATED_TILE_TASK_ID)
    out = create_task(ROTATED_TILE_TASK_ID).generate(
        hash64(2026061407, "indoor-rotated-tile", 0),
        params={"theme_id": "living_room", "source_object_count": 16, "rotation_degrees": 90, "canvas_profile": "landscape"},
        max_attempts=120,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "indoor_room"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D", "E", "F"}
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
    assert out.annotation_gt.type == "bbox"
    assert annotation == render_map["rotated_tile_bbox_px"]
    assert annotation == render_map["tile_bboxes_px_by_label"][out.answer_gt.value]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == annotation
    assert execution["rotated_tile_index"] in execution["usable_tile_indices"]


def test_missing_patch_label_contract() -> None:
    _assert_scene_packaged_task(MISSING_PATCH_TASK_ID)
    out = create_task(MISSING_PATCH_TASK_ID).generate(
        hash64(2026061503, "indoor-missing-patch", 0),
        params={"theme_id": "living_room", "source_object_count": 16, "option_count": 4, "correct_index": 2},
        max_attempts=120,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "indoor_room"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert execution["query_id"] == "single"
    assert execution["prompt_query_key"] == "missing_patch_label"
    assert execution["patch_mode"] == "plain"
    assert execution["selected_transform"] == "none"
    assert set(annotation) == {"missing_region", "selected_option"}
    assert annotation["missing_region"] == render_map["missing_region_bbox_px"]
    assert annotation["selected_option"] == render_map["selected_option_bbox_px"]
    assert annotation["selected_option"] == render_map["option_bboxes_px_by_label"][out.answer_gt.value]
    assert trace["projected_annotation"]["bbox_map"] == annotation
    assert len(render_map["option_bboxes_px_by_label"]) == 4
    assert len(render_map["option_source_crop_boxes_px"]) == 4
    assert render_map["option_source_crop_boxes_px"][2] == render_map["source_crop_box_px"]


def test_swapped_tile_pair_label_contract() -> None:
    _assert_scene_packaged_task(SWAPPED_TILE_PAIR_TASK_ID)
    out = create_task(SWAPPED_TILE_PAIR_TASK_ID).generate(
        hash64(2026063001, "indoor-swapped-tile-pair", 0),
        params={"theme_id": "living_room", "source_object_count": 16, "correct_index": 2, "canvas_profile": "landscape"},
        max_attempts=160,
    )
    trace = out.trace_payload
    _assert_scene_prompt_metadata(trace)
    execution = trace["execution_trace"]
    render_map = trace["render_map"]

    assert out.scene_id == "indoor_room"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 2
    assert execution["query_id"] == "single"
    assert execution["prompt_query_key"] == "swapped_tile_pair_label"
    assert execution["answer_label"] == out.answer_gt.value
    assert execution["grid_shape"] == [3, 3]
    assert render_map["grid_shape"] == [3, 3]
    assert len(render_map["tile_bboxes_px_by_number"]) == 9
    assert len(render_map["option_bboxes_px_by_label"]) == 4
    assert sorted(render_map["option_pairs_by_label"]) == ["A", "B", "C", "D"]
    assert render_map["option_pairs_by_label"][out.answer_gt.value] == execution["swapped_cell_numbers"]
    assert render_map["option_pair_indices_by_label"][out.answer_gt.value] == execution["swapped_pair_indices"]
    assert sorted(out.annotation_gt.value) == sorted(render_map["swapped_cell_bboxes_px"])
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert sorted(trace["projected_annotation"]["bbox_set"]) == sorted(out.annotation_gt.value)
    assert int(trace["query_spec"]["params"]["source_size"][0]) % 3 == 0
    assert int(trace["query_spec"]["params"]["source_size"][1]) % 3 == 0
