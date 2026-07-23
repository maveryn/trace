"""Contract tests for illustration library tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.illustrations.library.books_in_section_count import _sample_spec as _sample_books_in_section_spec
from trace_tasks.tasks.illustrations.library.filtered_book_in_section_count import _sample_spec as _sample_filtered_book_spec


SWAPPED_TILE_PAIR_TASK_ID = "task_illustrations__library__swapped_tile_pair_label"


def _assert_hash_balanced_counts(counts: Counter, expected_keys) -> None:
    assert sorted(counts) == sorted(expected_keys)
    expected = sum(counts.values()) / max(1, len(counts))
    assert min(counts.values()) >= max(1, int(expected * 0.4))
    assert max(counts.values()) <= int(expected * 1.7) + 1


def _assert_points_inside_book_bboxes(trace: dict, book_ids: list[str]) -> None:
    bboxes = trace["render_map"]["book_bboxes_px"]
    points = trace["render_map"]["book_points_px"]
    for book_id in book_ids:
        x0, y0, x1, y1 = [float(value) for value in bboxes[book_id]]
        x, y = [float(value) for value in points[book_id]]
        assert x0 <= x <= x1
        assert y0 <= y <= y1


def test_books_in_section_count_contract() -> None:
    out = create_task("task_illustrations__library__books_in_section_count").generate(
        hash64(2026052404, "library-books-section", 0),
        params={"query_id": SINGLE_QUERY_ID, "section_key": "science", "target_count": 7, "section_count": 4},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counted_book_ids = execution["counted_book_ids"]
    book_points = trace["render_map"]["book_points_px"]

    assert out.scene_id == "library"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["query_id"] == SINGLE_QUERY_ID
    assert trace["query_spec"]["params"]["prompt_query_key"] == "books_in_section_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == 7
    assert len(counted_book_ids) == 7
    assert execution["target_section_key"] == "science"
    assert sorted(out.annotation_gt.value) == sorted(book_points[book_id] for book_id in counted_book_ids)
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    _assert_points_inside_book_bboxes(trace, counted_book_ids)
    for book in execution["books"]:
        if book["book_id"] in set(counted_book_ids):
            assert book["section_key"] == "science"
        else:
            assert book["section_key"] != "science"


def test_books_in_section_countseeded_sampler_covers_answer_counts() -> None:
    answers = [
        _sample_books_in_section_spec(
            instance_seed=hash64(2026052404, "library-books-section-sampling", index),
            params={},
            attempt_index=0,
        ).target_count
        for index in range(100)
    ]
    counts = Counter(answers)
    _assert_hash_balanced_counts(counts, range(3, 9))


def test_book_color_count_contract() -> None:
    out = create_task("task_illustrations__library__filtered_book_in_section_count").generate(
        hash64(2026052405, "library-book-color", 0),
        params={
            "query_id": "book_color_in_section_count",
            "section_key": "history",
            "color_name": "red",
            "target_count": 4,
            "section_count": 5,
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counted_book_ids = execution["counted_book_ids"]
    book_points = trace["render_map"]["book_points_px"]

    assert out.scene_id == "library"
    assert out.query_id == "book_color_in_section_count"
    assert int(out.answer_gt.value) == 4
    assert len(counted_book_ids) == 4
    assert execution["target_section_key"] == "history"
    assert execution["target_color_name"] == "red"
    assert execution["target_color_label"] == "red [#E63232]"
    assert out.annotation_gt.type == "point_set"
    assert sorted(out.annotation_gt.value) == sorted(book_points[book_id] for book_id in counted_book_ids)
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    _assert_points_inside_book_bboxes(trace, counted_book_ids)
    for book in execution["books"]:
        is_target = book["section_key"] == "history" and book["color_name"] == "red"
        assert (book["book_id"] in set(counted_book_ids)) == is_target


def test_book_color_countseeded_sampler_covers_answer_counts() -> None:
    samples = [
        _sample_filtered_book_spec(
            instance_seed=hash64(2026052405, "library-book-color-sampling", index),
            params={"query_id": "book_color_in_section_count"},
            attempt_index=0,
        )
        for index in range(100)
    ]
    counts = Counter(sample.target_count for sample in samples)
    _assert_hash_balanced_counts(counts, range(1, 7))
    assert {sample.color_name for sample in samples} >= {"red", "blue", "green", "orange", "purple"}


def test_book_orientation_count_contract() -> None:
    out = create_task("task_illustrations__library__filtered_book_in_section_count").generate(
        hash64(2026052406, "library-book-orientation", 0),
        params={"query_id": "horizontal_book_in_section_count", "section_key": "art", "target_count": 3, "section_count": 4},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counted_book_ids = execution["counted_book_ids"]
    book_points = trace["render_map"]["book_points_px"]

    assert out.scene_id == "library"
    assert out.query_id == "horizontal_book_in_section_count"
    assert trace["query_spec"]["query_id"] == "horizontal_book_in_section_count"
    assert int(out.answer_gt.value) == 3
    assert len(counted_book_ids) == 3
    assert execution["target_section_key"] == "art"
    assert execution["target_orientation"] == "horizontal"
    assert out.annotation_gt.type == "point_set"
    assert sorted(out.annotation_gt.value) == sorted(book_points[book_id] for book_id in counted_book_ids)
    assert trace["projected_annotation"]["type"] == "point_set"
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    _assert_points_inside_book_bboxes(trace, counted_book_ids)
    for book in execution["books"]:
        is_target = book["section_key"] == "art" and book["orientation"] == "horizontal"
        assert (book["book_id"] in set(counted_book_ids)) == is_target


def test_book_orientation_countseeded_sampler_covers_answer_counts_and_variants() -> None:
    samples = [
        _sample_filtered_book_spec(
            instance_seed=hash64(2026052406, "library-book-orientation-sampling", index),
            params={"query_id": "upright_book_in_section_count" if index % 2 == 0 else "horizontal_book_in_section_count"},
            attempt_index=0,
        )
        for index in range(100)
    ]
    counts = Counter(sample.target_count for sample in samples)
    _assert_hash_balanced_counts(counts, range(1, 7))
    assert {sample.query_id for sample in samples} == {"upright_book_in_section_count", "horizontal_book_in_section_count"}


def test_rotated_tile_label_contract() -> None:
    out = create_task("task_illustrations__library__rotated_tile_label").generate(
        hash64(2026061501, "library-rotated-tile", 0),
        params={"section_count": 5, "rotation_degrees": 90},
        max_attempts=120,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "library"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert out.answer_gt.value in {"A", "B", "C", "D", "E", "F"}
    assert execution["query_id"] == SINGLE_QUERY_ID
    assert execution["prompt_query_key"] == "rotated_tile_label"
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


def test_missing_patch_label_contract() -> None:
    out = create_task("task_illustrations__library__missing_patch_label").generate(
        hash64(2026061503, "library-missing-patch", 0),
        params={"section_count": 5, "option_count": 4, "correct_index": 2},
        max_attempts=120,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]
    annotation = out.annotation_gt.value

    assert out.scene_id == "library"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_map"
    assert execution["query_id"] == SINGLE_QUERY_ID
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
    out = create_task(SWAPPED_TILE_PAIR_TASK_ID).generate(
        hash64(2026063002, "library-swapped-tile-pair", 0),
        params={"section_count": 5, "correct_index": 1, "canvas_profile": "landscape"},
        max_attempts=160,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render_map = trace["render_map"]

    assert out.scene_id == "library"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "B"
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 2
    assert execution["query_id"] == SINGLE_QUERY_ID
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
