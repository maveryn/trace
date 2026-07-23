"""Contract tests for illustration park/playground tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks import create_task
from trace_tasks.tasks.illustrations.park_playground.jigsaw_arrangement_label import (
    _sample_spec as _sample_jigsaw_arrangement_spec,
)
from trace_tasks.tasks.illustrations.shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS
from trace_tasks.tasks.illustrations.park_playground.missing_patch_label import (
    PLAIN_QUERY_ID,
    _sample_spec as _sample_missing_patch_spec,
)
from trace_tasks.tasks.illustrations.park_playground.person_count import _sample_spec as _sample_person_count_spec
from trace_tasks.tasks.illustrations.park_playground.playground_equipment_count import (
    _sample_spec as _sample_playground_equipment_spec,
)
from trace_tasks.tasks.illustrations.park_playground.rotated_tile_label import (
    _sample_spec as _sample_rotated_tile_spec,
)
from trace_tasks.tasks.illustrations.park_playground.swapped_tile_pair_label import (
    _sample_spec as _sample_swapped_tile_pair_spec,
)


def _assert_hash_balanced_counts(counts: Counter, expected_keys) -> None:
    assert sorted(counts) == sorted(expected_keys)
    expected = sum(counts.values()) / max(1, len(counts))
    assert min(counts.values()) >= max(1, int(expected * 0.4))
    assert max(counts.values()) <= int(expected * 1.7) + 1


def _assert_annotation_inside_canvas(out) -> None:
    width, height = out.trace_payload["render_spec"]["canvas_size"]
    for x0, y0, x1, y1 in out.annotation_gt.value:
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height


def _assert_point_annotation_inside_canvas(out) -> None:
    width, height = out.trace_payload["render_spec"]["canvas_size"]
    for x, y in out.annotation_gt.value:
        assert 0 <= float(x) <= width
        assert 0 <= float(y) <= height


def _assert_keyed_annotation_inside_canvas(out) -> None:
    width, height = out.trace_payload["render_spec"]["canvas_size"]
    for x0, y0, x1, y1 in out.annotation_gt.value.values():
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height


def _assert_bbox_inside_canvas(out) -> None:
    width, height = out.trace_payload["render_spec"]["canvas_size"]
    x0, y0, x1, y1 = out.annotation_gt.value
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def test_person_count_contract() -> None:
    out = create_task("task_illustrations__park_playground__person_count").generate(
        hash64(2026061503, "park-person-total", 0),
        params={"person_count": 8},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counted_person_ids = execution["counted_person_ids"]
    person_bboxes = trace["render_map"]["person_bboxes_px"]

    assert out.scene_id == "park_playground"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["query_id"] == SINGLE_QUERY_ID
    assert trace["query_spec"]["task_id"] == "task_illustrations__park_playground__person_count"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == 8
    assert len(counted_person_ids) == 8
    assert len(execution["persons"]) == 8
    assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_person_bboxes_px"])
    assert len(trace["render_map"]["counted_person_bboxes_px"]) == len(counted_person_ids)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert "8 people" not in out.prompt
    _assert_annotation_inside_canvas(out)


def test_person_count_seeded_sampler_covers_answer_range() -> None:
    samples = [
        _sample_person_count_spec(
            instance_seed=hash64(2026061503, "park-person-total-sampling", index),
            params={},
            attempt_index=0,
        )
        for index in range(100)
    ]
    answer_counts = Counter(sample.person_count for sample in samples)
    query_counts = Counter(sample.branch_id for sample in samples)

    _assert_hash_balanced_counts(answer_counts, set(range(5, 13)))
    assert query_counts == Counter({SINGLE_QUERY_ID: 100})


def test_playground_equipment_count_contract() -> None:
    out = create_task("task_illustrations__park_playground__playground_equipment_count").generate(
        hash64(2026052407, "park-equipment", 0),
        params={"target_equipment_type": "slide", "target_count": 3, "equipment_count": 6, "person_count": 6},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    counted_equipment_ids = execution["counted_equipment_ids"]
    decor_bboxes = trace["render_map"]["decor_bboxes_px"]

    assert out.scene_id == "park_playground"
    assert out.query_id == SINGLE_QUERY_ID
    assert trace["query_spec"]["query_id"] == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert int(out.answer_gt.value) == 3
    assert len(counted_equipment_ids) == 3
    assert execution["target_equipment_type"] == "slide"
    assert sorted(out.annotation_gt.value) == sorted(trace["render_map"]["counted_equipment_bboxes_px"])
    assert len(trace["render_map"]["counted_equipment_bboxes_px"]) == len(counted_equipment_ids)
    assert all(min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1])) >= 24.0 for box in trace["render_map"]["counted_equipment_bboxes_px"])
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert "6 people" not in out.prompt
    for decor in execution["decor"]:
        if str(decor["decor_id"]).startswith("equipment_"):
            is_target = decor["decor_type"] == "slide"
            assert (decor["decor_id"] in set(counted_equipment_ids)) == is_target


def test_playground_equipment_countseeded_sampler_covers_answers_and_variants() -> None:
    samples = [
        _sample_playground_equipment_spec(
            instance_seed=hash64(2026052407, "park-equipment-sampling", index),
            params={},
            attempt_index=0,
        )
        for index in range(100)
    ]
    answer_counts = Counter(sample.target_count for sample in samples)
    query_counts = Counter(sample.branch_id for sample in samples)
    equipment_type_counts = Counter(sample.target_equipment_type for sample in samples)

    answer_support = set(range(1, 6))
    _assert_hash_balanced_counts(answer_counts, answer_support)
    assert query_counts == Counter({SINGLE_QUERY_ID: 100})
    _assert_hash_balanced_counts(equipment_type_counts, {"slide", "swing_set", "seesaw", "climbing_frame"})
    for equipment_type in equipment_type_counts:
        per_query_answers = Counter(sample.target_count for sample in samples if sample.target_equipment_type == equipment_type)
        assert set(per_query_answers) <= answer_support
        assert per_query_answers


def test_jigsaw_arrangement_label_contract() -> None:
    out = create_task("task_illustrations__park_playground__jigsaw_arrangement_label").generate(
        hash64(2026061502, "park-jigsaw-arrangement", 0),
        params={"correct_index": 2, "person_count": 9, "equipment_count": 5},
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    answer_label = str(out.answer_gt.value)
    option_bboxes = trace["render_map"]["option_bboxes_px_by_label"]
    option_permutations = trace["render_map"]["option_permutations_by_label"]

    assert out.scene_id == "park_playground"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert answer_label == "C"
    assert sorted(option_bboxes) == ["A", "B", "C", "D"]
    assert out.annotation_gt.value == option_bboxes[answer_label]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["render_map"]["selected_option_bbox_px"] == option_bboxes[answer_label]
    correct_permutation = trace["render_map"]["correct_permutation"]
    assert execution["option_permutations_by_label"][answer_label] == correct_permutation
    assert option_permutations[answer_label] == correct_permutation
    assert sum(perm == correct_permutation for perm in option_permutations.values()) == 1
    assert trace["query_spec"]["params"]["grid_shape"] in ([3, 3], [2, 3], [3, 2])
    assert trace["render_map"]["option_layout_shape"] == [2, 2]
    assert min(trace["query_spec"]["params"]["tile_detail_scores"]) >= 600
    assert trace["render_spec"]["canvas_size"][0] * trace["render_spec"]["canvas_size"][1] <= MAX_RECONSTRUCTION_OUTPUT_PIXELS
    assert "9 people" not in out.prompt
    _assert_bbox_inside_canvas(out)


def test_jigsaw_arrangement_seeded_sampler_covers_answer_labels() -> None:
    samples = [
        _sample_jigsaw_arrangement_spec(
            instance_seed=hash64(2026061502, "park-jigsaw-arrangement-sampling", index),
            params={"_sample_cursor": index},
            attempt_index=0,
        )
        for index in range(100)
    ]
    answer_counts = Counter(sample.correct_index for sample in samples)
    person_counts = Counter(sample.person_count for sample in samples)
    equipment_counts = Counter(sample.equipment_count for sample in samples)

    assert answer_counts == Counter({0: 25, 1: 25, 2: 25, 3: 25})
    assert set(person_counts) <= set(range(8, 14))
    assert set(equipment_counts) <= set(range(4, 8))
    assert person_counts
    assert equipment_counts


def test_missing_patch_label_contract() -> None:
    out = create_task("task_illustrations__park_playground__missing_patch_label").generate(
        hash64(2026061503, "park-missing-patch", 0),
        params={
            "option_count": 4,
            "correct_index": 2,
            "source_person_count": 9,
            "source_equipment_count": 5,
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    answer_label = str(out.answer_gt.value)

    assert out.scene_id == "park_playground"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox_map"
    assert answer_label == "C"
    assert set(out.annotation_gt.value) == {"missing_region", "selected_option"}
    assert out.annotation_gt.value["missing_region"] == trace["render_map"]["missing_region_bbox_px"]
    assert out.annotation_gt.value["selected_option"] == trace["render_map"]["selected_option_bbox_px"]
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["query_spec"]["params"]["patch_mode"] == "plain"
    assert trace["query_spec"]["params"]["option_labels"] == ["A", "B", "C", "D"]
    assert trace["query_spec"]["params"]["correct_index"] == 2
    _assert_keyed_annotation_inside_canvas(out)


def test_missing_patch_seeded_sampler_covers_answer_labels() -> None:
    samples = [
        _sample_missing_patch_spec(
            instance_seed=hash64(2026061503, "park-missing-patch-sampling", index),
            params={"_sample_cursor": index},
            attempt_index=0,
        )
        for index in range(100)
    ]
    query_counts = Counter(sample.query_id for sample in samples)
    option_counts = Counter(sample.option_count for sample in samples)
    answer_counts = Counter(sample.correct_index for sample in samples)

    assert query_counts == Counter({PLAIN_QUERY_ID: 100})
    _assert_hash_balanced_counts(option_counts, {4, 6})
    assert set(answer_counts) <= set(range(6))
    assert {0, 1, 2, 3} <= set(answer_counts)


def test_rotated_tile_label_contract() -> None:
    out = create_task("task_illustrations__park_playground__rotated_tile_label").generate(
        hash64(2026061503, "park-rotated-tile", 0),
        params={"source_person_count": 10, "source_equipment_count": 6},
        max_attempts=200,
    )
    trace = out.trace_payload
    answer_label = str(out.answer_gt.value)
    tile_bboxes = trace["render_map"]["tile_bboxes_px_by_label"]

    assert out.scene_id == "park_playground"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox"
    assert answer_label in {"A", "B", "C", "D", "E", "F"}
    assert sorted(tile_bboxes) == ["A", "B", "C", "D", "E", "F"]
    assert trace["render_spec"]["canvas_size"] == trace["query_spec"]["params"]["source_size"]
    assert tile_bboxes["A"][:2] == [0.0, 0.0]
    assert tile_bboxes["F"][2:] == [
        float(trace["render_spec"]["canvas_size"][0]),
        float(trace["render_spec"]["canvas_size"][1]),
    ]
    assert out.annotation_gt.value == tile_bboxes[answer_label]
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["render_map"]["rotated_tile_bbox_px"] == tile_bboxes[answer_label]
    assert trace["query_spec"]["params"]["grid_shape"] == [2, 3]
    assert trace["query_spec"]["params"]["rotation_degrees"] in {90, 270}
    assert trace["query_spec"]["params"]["answer_label"] == answer_label
    _assert_bbox_inside_canvas(out)


def test_rotated_tile_seeded_sampler_covers_rotation_support() -> None:
    samples = [
        _sample_rotated_tile_spec(
            instance_seed=hash64(2026061503, "park-rotated-tile-sampling", index),
            params={},
            attempt_index=0,
        )
        for index in range(100)
    ]
    rotation_counts = Counter(sample.rotation_degrees for sample in samples)
    person_counts = Counter(sample.source_person_count for sample in samples)
    equipment_counts = Counter(sample.source_equipment_count for sample in samples)

    _assert_hash_balanced_counts(rotation_counts, {90, 270})
    assert set(person_counts) <= set(range(8, 14))
    assert set(equipment_counts) <= set(range(4, 8))
    assert person_counts
    assert equipment_counts


def test_swapped_tile_pair_label_contract() -> None:
    out = create_task("task_illustrations__park_playground__swapped_tile_pair_label").generate(
        hash64(2026061603, "park-swapped-tile-pair", 0),
        params={"correct_index": 2, "source_person_count": 10, "source_equipment_count": 6},
        max_attempts=300,
    )
    trace = out.trace_payload
    answer_label = str(out.answer_gt.value)
    render_map = trace["render_map"]
    option_pairs = render_map["option_pairs_by_label"]
    option_pair_indices = render_map["option_pair_indices_by_label"]
    swapped_indices = render_map["swapped_pair_indices"]
    swapped_cell_bboxes = render_map["swapped_cell_bboxes_px"]

    assert out.scene_id == "park_playground"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "option_letter"
    assert out.annotation_gt.type == "bbox_set"
    assert answer_label == "C"
    assert sorted(option_pairs) == ["A", "B", "C", "D"]
    assert len({tuple(value) for value in option_pair_indices.values()}) == 4
    assert option_pair_indices[answer_label] == swapped_indices
    assert option_pairs[answer_label] == [swapped_indices[0] + 1, swapped_indices[1] + 1]
    assert len(out.annotation_gt.value) == 2
    assert sorted(out.annotation_gt.value) == sorted(swapped_cell_bboxes)
    assert trace["projected_annotation"]["type"] == "bbox_set"
    assert sorted(trace["projected_annotation"]["bbox_set"]) == sorted(out.annotation_gt.value)
    assert trace["query_spec"]["params"]["grid_shape"] == [3, 3]
    assert trace["query_spec"]["params"]["source_size"][0] % 3 == 0
    assert trace["query_spec"]["params"]["source_size"][1] % 3 == 0
    assert trace["query_spec"]["params"]["candidate_pair_count"] >= 4
    assert trace["render_spec"]["canvas_size"][0] * trace["render_spec"]["canvas_size"][1] <= MAX_RECONSTRUCTION_OUTPUT_PIXELS
    assert "10 people" not in out.prompt
    _assert_annotation_inside_canvas(out)


def test_swapped_tile_pair_seeded_sampler_covers_answer_labels_and_profiles() -> None:
    samples = [
        _sample_swapped_tile_pair_spec(
            instance_seed=hash64(2026061603, "park-swapped-tile-pair-sampling", index),
            params={"_sample_cursor": index},
            attempt_index=0,
        )
        for index in range(100)
    ]
    answer_counts = Counter(sample.correct_index for sample in samples)
    profile_counts = Counter(sample.source_profile_trace["canvas_profile"] for sample in samples)
    person_counts = Counter(sample.source_person_count for sample in samples)
    equipment_counts = Counter(sample.source_equipment_count for sample in samples)

    assert answer_counts == Counter({0: 25, 1: 25, 2: 25, 3: 25})
    assert set(profile_counts) == {"landscape", "square", "portrait"}
    assert all(sample.source_size[0] % 3 == 0 and sample.source_size[1] % 3 == 0 for sample in samples)
    assert set(person_counts) <= set(range(8, 14))
    assert set(equipment_counts) <= set(range(4, 8))
    assert person_counts
    assert equipment_counts
