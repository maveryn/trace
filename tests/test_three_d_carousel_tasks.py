"""Tests for synthetic 3D carousel tasks."""

from __future__ import annotations

from pathlib import Path

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import create_task
from trace_tasks.tasks.three_d.carousel.between_object_type_anchors_count import TASK_ID as BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID
from trace_tasks.tasks.three_d.carousel.color_ordered_adjacent_pair_count import TASK_ID as COLOR_ORDERED_PAIR_TASK_ID
from trace_tasks.tasks.three_d.carousel.color_transfer_total_count import TASK_ID as COLOR_TRANSFER_TASK_ID
from trace_tasks.tasks.three_d.carousel.belt_total_object_count import (
    QUERY_ID as TOTAL_QUERY_ID,
    TASK_ID as TOTAL_TASK_ID,
)
from trace_tasks.tasks.three_d.carousel.belt_object_type_count_arithmetic_value import (
    DIFFERENCE_QUERY_ID as ARITH_DIFFERENCE_QUERY_ID,
    TASK_ID as OBJECT_TYPE_ARITH_TASK_ID,
    TOTAL_QUERY_ID as ARITH_TOTAL_QUERY_ID,
)
from trace_tasks.tasks.three_d.carousel.object_type_ordered_adjacent_pair_count import TASK_ID as OBJECT_TYPE_ORDERED_PAIR_TASK_ID
from trace_tasks.tasks.three_d.carousel.object_type_transfer_total_count import TASK_ID as OBJECT_TYPE_TRANSFER_TASK_ID
from trace_tasks.tasks.three_d.carousel.scoped_belt_color_count import TASK_ID as SCOPED_COLOR_TASK_ID
from trace_tasks.tasks.three_d.carousel.scoped_belt_object_type_count import TASK_ID as SCOPED_OBJECT_TYPE_TASK_ID
from trace_tasks.tasks.three_d.carousel.scoped_color_type_count import (
    TASK_ID as COLOR_TYPE_TASK_ID,
)
from trace_tasks.tasks.three_d.carousel.shared.state import CONVEYOR_OBJECT_SHAPE_TYPES
from trace_tasks.tasks.three_d.shared.object_confusions import confusable_shape_names
from trace_tasks.tasks.three_d.shared.semantic_colors import confusable_color_names
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


OBJECT_TYPE_BELT_COUNT_INTERNAL_QUERY_ID = "object_type_belt_count"
COLOR_BELT_COUNT_INTERNAL_QUERY_ID = "color_belt_count"
COLOR_TYPE_BELT_COUNT_INTERNAL_QUERY_ID = "color_type_belt_count"
BELT_TOTAL_INTERNAL_QUERY_ID = "belt_total_count"
OBJECT_COUNT_SUM_INTERNAL_QUERY_ID = "object_count_sum"
OBJECT_COUNT_DIFFERENCE_INTERNAL_QUERY_ID = "object_count_difference"
OBJECT_ORDERED_PAIR_INTERNAL_QUERY_ID = "object_ordered_pair_count"
COLOR_ORDERED_PAIR_INTERNAL_QUERY_ID = "color_ordered_pair_count"
OBJECT_TRANSFER_INTERNAL_QUERY_ID = "object_transfer_total_count"
COLOR_TRANSFER_INTERNAL_QUERY_ID = "color_transfer_total_count"
BETWEEN_OBJECT_ANCHORS_INTERNAL_QUERY_ID = "between_object_anchors_count"


def _rounded_segment_for_pair(render_map: dict, pair: list[str]) -> list[list[float]]:
    return [
        [round(float(value), 3) for value in render_map["object_centers_px"][str(pair[0])]],
        [round(float(value), 3) for value in render_map["object_centers_px"][str(pair[1])]],
    ]


def test_carousel_object_pool_excludes_cylinder_confusers() -> None:
    assert "cylinder" in CONVEYOR_OBJECT_SHAPE_TYPES
    assert "drum" not in CONVEYOR_OBJECT_SHAPE_TYPES
    assert "pencil" not in CONVEYOR_OBJECT_SHAPE_TYPES
    assert "ruler" not in CONVEYOR_OBJECT_SHAPE_TYPES


def _assert_count_output(output, *, expected_query_id: str, expected_internal_query_id: str | None = None) -> None:
    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    target_ids = [str(object_id) for object_id in trace["target_object_ids"]]

    assert output.scene_id == "carousel"
    assert output.query_id == expected_query_id
    if expected_internal_query_id is not None:
        assert output.trace_payload["query_spec"]["params"]["internal_query_id"] == expected_internal_query_id
    assert output.answer_gt.type == "integer"
    assert output.annotation_gt.type == "bbox_set"
    assert int(output.answer_gt.value) == len(target_ids)
    assert len(output.annotation_gt.value) == len(target_ids)
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_ids]
    assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
    assert trace["layout_family"] == "elliptical_carousel"
    assert trace["target_belt_key"] in {"inner", "outer"}
    assert trace["target_belt_label"] in {"INNER", "OUTER"}
    assert trace["target_belt_object_ids"]
    assert set(render_map["belt_bboxes_px"]) == {"inner", "outer"}
    assert "{target_" not in output.prompt
    assert "unlettered" not in output.prompt.lower()
    assert "segment" not in output.prompt.lower()
    assert_three_d_canvas_contract(output)

    image_w, image_h = output.image.size
    for bbox in output.annotation_gt.value:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 < x1 <= float(image_w)
        assert 0.0 <= y0 < y1 <= float(image_h)
        assert min(x1 - x0, y1 - y0) >= 24.0
    if trace["predicate_kind"] == "object_type":
        same_belt_distractors = [
            spec
            for spec in trace["object_specs"]
            if str(spec["belt_key"]) == str(trace["target_belt_key"]) and not bool(spec["matches_query"])
        ]
        assert same_belt_distractors
        assert any(str(spec["shape_type"]) != str(trace["target_shape_type"]) for spec in same_belt_distractors)
    if trace["predicate_kind"] == "color":
        same_belt_distractors = [
            spec
            for spec in trace["object_specs"]
            if str(spec["belt_key"]) == str(trace["target_belt_key"]) and not bool(spec["matches_query"])
        ]
        assert same_belt_distractors
        assert any(str(spec["color_name"]) != str(trace["target_color_name"]) for spec in same_belt_distractors)
    assert max(int(value) for value in trace["belt_counts"].values()) <= 12
    assert int(trace["belt_counts"].get("inner", 0)) <= 8
    assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_scoped_belt_count_query_ids() -> None:
    cases = (
        (SCOPED_OBJECT_TYPE_TASK_ID, OBJECT_TYPE_BELT_COUNT_INTERNAL_QUERY_ID, 2026062401),
        (SCOPED_COLOR_TASK_ID, COLOR_BELT_COUNT_INTERNAL_QUERY_ID, 2026062402),
    )
    for task_id, internal_query_id, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=120,
        )
        _assert_count_output(output, expected_query_id=SINGLE_QUERY_ID, expected_internal_query_id=internal_query_id)
        assert 0 <= int(output.answer_gt.value) <= 5


def test_carousel_scoped_belt_count_supports_zero_and_five() -> None:
    cases = (
        (SCOPED_OBJECT_TYPE_TASK_ID, OBJECT_TYPE_BELT_COUNT_INTERNAL_QUERY_ID, "inner", 0, 2026062511),
        (SCOPED_OBJECT_TYPE_TASK_ID, OBJECT_TYPE_BELT_COUNT_INTERNAL_QUERY_ID, "outer", 5, 2026062512),
        (SCOPED_COLOR_TASK_ID, COLOR_BELT_COUNT_INTERNAL_QUERY_ID, "inner", 0, 2026062513),
        (SCOPED_COLOR_TASK_ID, COLOR_BELT_COUNT_INTERNAL_QUERY_ID, "outer", 5, 2026062514),
    )
    for task_id, internal_query_id, belt_key, target_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "target_belt_key": belt_key,
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=120,
        )
        _assert_count_output(output, expected_query_id=SINGLE_QUERY_ID, expected_internal_query_id=internal_query_id)
        assert int(output.answer_gt.value) == int(target_count)
        if int(target_count) == 0:
            assert output.annotation_gt.value == []


def test_carousel_scoped_color_type_count_uses_conjunction_distractors() -> None:
    task = create_task(COLOR_TYPE_TASK_ID)
    cases = (
        ("inner", 0, 2026062601),
        ("outer", 5, 2026062602),
    )
    for belt_key, target_count, seed in cases:
        output = task.generate(
            seed,
            params={
                "target_belt_key": belt_key,
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=120,
        )
        _assert_count_output(output, expected_query_id=SINGLE_QUERY_ID)
        trace = output.trace_payload["execution_trace"]
        target_ids = [str(object_id) for object_id in trace["target_object_ids"]]
        target_shape = str(trace["target_shape_type"])
        target_color = str(trace["target_color_name"])
        target_belt = str(trace["target_belt_key"])
        confusable_shapes = set(confusable_shape_names(target_shape))

        assert trace["predicate_kind"] == "color_type"
        assert int(output.answer_gt.value) == int(target_count)
        assert len(target_ids) == int(target_count)
        assert output.trace_payload["query_spec"]["internal_query_id"] == SINGLE_QUERY_ID
        for spec in trace["object_specs"]:
            is_target = str(spec["object_id"]) in set(target_ids)
            if is_target:
                assert str(spec["belt_key"]) == target_belt
                assert str(spec["shape_type"]) == target_shape
                assert str(spec["color_name"]) == target_color
        same_belt_roles = {
            str(spec["count_role"])
            for spec in trace["object_specs"]
            if str(spec["belt_key"]) == target_belt and not bool(spec["matches_query"])
        }
        assert all(
            str(spec["shape_type"]) not in confusable_shapes
            for spec in trace["object_specs"]
            if str(spec["shape_type"]) != target_shape
        )
        assert "same_belt_same_color_wrong_type" in same_belt_roles
        assert "same_belt_same_type_wrong_color" in same_belt_roles
        assert any(
            str(spec["belt_key"]) != target_belt
            and str(spec["shape_type"]) == target_shape
            and str(spec["color_name"]) == target_color
            for spec in trace["object_specs"]
        )
        if int(target_count) == 0:
            assert output.annotation_gt.value == []


def test_carousel_color_type_distractors_avoid_visually_confusable_shapes() -> None:
    output = create_task(COLOR_TYPE_TASK_ID).generate(
        2026062621,
        params={
            "target_shape_type": "card",
            "target_color_name": "red",
            "target_count": 3,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=120,
    )
    trace = output.trace_payload["execution_trace"]
    confusable_shapes = set(confusable_shape_names("card"))

    assert trace["target_shape_type"] == "card"
    assert trace["target_color_name"] == "red"
    assert all(
        str(spec["shape_type"]) not in confusable_shapes
        for spec in trace["object_specs"]
        if str(spec["shape_type"]) != "card"
    )


def test_carousel_color_readout_excludes_target_confusable_colors() -> None:
    cases = (
        (SCOPED_COLOR_TASK_ID, {"target_color_name": "red"}, 2026062611),
        (COLOR_TYPE_TASK_ID, {"target_color_name": "red"}, 2026062612),
    )
    for task_id, params, seed in cases:
        output = create_task(task_id).generate(
            seed,
            params={**params, "post_image_noise_apply_prob": 0.0},
            max_attempts=120,
        )
        trace = output.trace_payload["execution_trace"]
        colors = {str(spec["color_name"]) for spec in trace["object_specs"]}
        assert str(trace["target_color_name"]) == "red"
        assert colors.isdisjoint(set(confusable_color_names("red")))


def test_carousel_belt_total_count_uses_belt_specific_support() -> None:
    task = create_task(TOTAL_TASK_ID)
    cases = (
        ("inner", 8, 2026062501),
        ("outer", 10, 2026062502),
    )
    for belt_key, target_count, seed in cases:
        output = task.generate(
            seed,
            params={
                "target_belt_key": belt_key,
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=120,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        target_ids = [str(object_id) for object_id in trace["target_object_ids"]]

        assert output.scene_id == "carousel"
        assert output.query_id == SINGLE_QUERY_ID
        assert output.trace_payload["query_spec"]["params"]["internal_query_id"] == BELT_TOTAL_INTERNAL_QUERY_ID
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set"
        assert trace["target_belt_key"] == belt_key
        assert trace["predicate_kind"] == "belt_total"
        assert int(output.answer_gt.value) == int(target_count)
        assert int(output.answer_gt.value) == len(target_ids)
        assert target_ids == trace["target_belt_object_ids"]
        assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_ids]
        assert len({str(spec["shape_type"]) for spec in trace["object_specs"]}) == 1
        assert len({str(spec["color_name"]) for spec in trace["object_specs"]}) >= 2
        assert "{target_" not in output.prompt
        assert "color" not in output.prompt.lower()
        assert "type" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)

        image_w, image_h = output.image.size
        for bbox in output.annotation_gt.value:
            x0, y0, x1, y1 = [float(value) for value in bbox]
            assert 0.0 <= x0 < x1 <= float(image_w)
            assert 0.0 <= y0 < y1 <= float(image_h)
            assert min(x1 - x0, y1 - y0) >= 24.0


def test_carousel_belt_count_arithmetic_query_ids() -> None:
    cases = (
        (OBJECT_TYPE_ARITH_TASK_ID, ARITH_TOTAL_QUERY_ID, OBJECT_COUNT_SUM_INTERNAL_QUERY_ID, 2026062703),
        (OBJECT_TYPE_ARITH_TASK_ID, ARITH_DIFFERENCE_QUERY_ID, OBJECT_COUNT_DIFFERENCE_INTERNAL_QUERY_ID, 2026062704),
    )
    for task_id, query_id, internal_query_id, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={"query_id": query_id, "post_image_noise_apply_prob": 0.0},
            max_attempts=160,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        operands = {str(key): int(value) for key, value in trace["operand_counts_by_scope"].items()}

        assert output.scene_id == "carousel"
        assert output.query_id == query_id
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set_map"
        assert set(output.annotation_gt.value) == {"inner_objects", "outer_objects"}
        assert output.trace_payload["projected_annotation"]["bbox_set_map"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_bbox_set_map"] == output.annotation_gt.value
        assert "{target_" not in output.prompt
        assert "unlettered" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)

        if trace["arithmetic_operation"] == "sum":
            assert int(output.answer_gt.value) == operands["inner_objects"] + operands["outer_objects"]
            assert 1 <= int(output.answer_gt.value) <= 12
        else:
            assert int(output.answer_gt.value) == abs(operands["inner_objects"] - operands["outer_objects"])
            assert 1 <= int(output.answer_gt.value) <= 5
        for key, ids in trace["target_object_ids_by_annotation_key"].items():
            expected = [render_map["object_bboxes_px"][str(object_id)] for object_id in ids]
            assert output.annotation_gt.value[str(key)] == expected
            assert len(expected) == operands[str(key)]
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12
        assert trace["predicate_kind"] == "object_type_count_arithmetic"
        assert trace["target_object_plural"]


def test_carousel_adjacent_pair_count_query_ids() -> None:
    cases = (
        (COLOR_ORDERED_PAIR_TASK_ID, COLOR_ORDERED_PAIR_INTERNAL_QUERY_ID, 2026062803),
        (OBJECT_TYPE_ORDERED_PAIR_TASK_ID, OBJECT_ORDERED_PAIR_INTERNAL_QUERY_ID, 2026062804),
    )
    for task_id, internal_query_id, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        object_by_id = {str(spec["object_id"]): spec for spec in trace["object_specs"]}
        pair_ids = [[str(pair[0]), str(pair[1])] for pair in trace["target_pair_object_id_pairs"]]
        target_belt = str(trace["target_belt_key"])
        belt_sequence = [str(object_id) for object_id in trace["object_sequences_by_belt"][target_belt]]

        assert output.scene_id == "carousel"
        assert output.query_id == SINGLE_QUERY_ID
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "segment_set"
        assert 0 <= int(output.answer_gt.value) <= 4
        assert len(pair_ids) == int(output.answer_gt.value)
        assert output.annotation_gt.value == [_rounded_segment_for_pair(render_map, pair) for pair in pair_ids]
        assert output.trace_payload["projected_annotation"]["segment_set"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_segment_set"] == output.annotation_gt.value
        assert "{target_" not in output.prompt
        assert "unlettered" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)

        for first_id, second_id in pair_ids:
            first_index = belt_sequence.index(str(first_id))
            assert belt_sequence[(first_index + 1) % len(belt_sequence)] == str(second_id)
            assert str(object_by_id[first_id]["belt_key"]) == target_belt
            assert str(object_by_id[second_id]["belt_key"]) == target_belt
            if task_id == COLOR_ORDERED_PAIR_TASK_ID:
                assert trace["predicate_kind"] == "ordered_color_pair"
                assert str(object_by_id[first_id]["color_name"]) == str(trace["target_color_name"])
                assert str(object_by_id[second_id]["color_name"]) == str(trace["second_target_color_name"])
                assert trace["target_color_label"]
                assert trace["second_target_color_label"]
            else:
                assert trace["predicate_kind"] == "ordered_object_pair"
                assert str(object_by_id[first_id]["shape_type"]) == str(trace["target_shape_pair"][0])
                assert str(object_by_id[second_id]["shape_type"]) == str(trace["target_shape_pair"][1])
                assert trace["target_object_plural_pair"][0]
                assert trace["target_object_plural_pair"][1]
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_adjacent_pair_count_supports_zero_and_four() -> None:
    cases = (
        (COLOR_ORDERED_PAIR_TASK_ID, 0, "inner", 2026062813),
        (OBJECT_TYPE_ORDERED_PAIR_TASK_ID, 4, "outer", 2026062814),
    )
    for task_id, target_count, belt_key, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "target_belt_key": belt_key,
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]

        assert int(output.answer_gt.value) == int(target_count)
        assert len(output.annotation_gt.value) == int(target_count)
        assert len(trace["target_pair_object_id_pairs"]) == int(target_count)
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_between_marked_items_count_query_ids() -> None:
    cases = (
        (BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID, BETWEEN_OBJECT_ANCHORS_INTERNAL_QUERY_ID, 2026062836),
    )
    for task_id, internal_query_id, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        target_ids = [str(object_id) for object_id in trace["target_object_ids"]]
        target_belt = str(trace["target_belt_key"])
        belt_sequence = [str(object_id) for object_id in trace["object_sequences_by_belt"][target_belt]]
        start_index = int(trace["start_anchor_index"])
        end_index = int(trace["end_anchor_index"])
        expected_between_ids = [
            belt_sequence[(start_index + offset) % len(belt_sequence)]
            for offset in range(1, int(output.answer_gt.value) + 1)
        ]

        assert output.scene_id == "carousel"
        assert output.query_id == SINGLE_QUERY_ID
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set"
        assert 1 <= int(output.answer_gt.value) <= 5
        assert int(output.answer_gt.value) == len(target_ids)
        assert target_ids == expected_between_ids
        assert target_ids == [str(object_id) for object_id in trace["between_object_ids"]]
        assert trace["marked_anchor_object_ids"] == [trace["start_anchor_object_id"], trace["end_anchor_object_id"]]
        assert trace["start_anchor_object_id"] == belt_sequence[start_index]
        assert trace["end_anchor_object_id"] == belt_sequence[end_index]
        assert end_index == (start_index + int(output.answer_gt.value) + 1) % len(belt_sequence)
        assert trace["start_anchor_object_id"] not in target_ids
        assert trace["end_anchor_object_id"] not in target_ids
        assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_ids]
        assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
        assert set(render_map["marked_anchor_object_bboxes_px"]) == {
            str(trace["start_anchor_object_id"]),
            str(trace["end_anchor_object_id"]),
        }
        assert "{target_" not in output.prompt
        assert "unlettered" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)
        assert trace["predicate_kind"] == "between_object_anchors"
        assert trace["target_object_name_pair"][0]
        assert trace["target_object_name_pair"][1]
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_between_marked_items_count_supports_one_and_five() -> None:
    cases = (
        (BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID, "inner", 1, 2026062837),
        (BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID, "outer", 5, 2026062838),
    )
    for task_id, belt_key, target_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "target_belt_key": belt_key,
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        belt_sequence = [str(object_id) for object_id in trace["object_sequences_by_belt"][str(trace["target_belt_key"])]]

        assert int(output.answer_gt.value) == int(target_count)
        assert len(output.annotation_gt.value) == int(target_count)
        assert len(trace["between_object_ids"]) == int(target_count)
        assert int(trace["end_anchor_index"]) == (int(trace["start_anchor_index"]) + int(target_count) + 1) % len(belt_sequence)
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_scope_count_after_transfer_query_ids() -> None:
    cases = (
        (COLOR_TRANSFER_TASK_ID, COLOR_TRANSFER_INTERNAL_QUERY_ID, 2026062825),
        (OBJECT_TYPE_TRANSFER_TASK_ID, OBJECT_TRANSFER_INTERNAL_QUERY_ID, 2026062826),
    )
    for task_id, internal_query_id, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        operands = {str(key): int(value) for key, value in trace["operand_counts_by_scope"].items()}
        target_ids_by_key = {
            str(key): [str(object_id) for object_id in object_ids]
            for key, object_ids in trace["target_object_ids_by_annotation_key"].items()
        }
        object_by_id = {str(spec["object_id"]): spec for spec in trace["object_specs"]}
        destination_cap = 8 if str(trace["destination_belt_key"]) == "inner" else 12

        assert output.scene_id == "carousel"
        assert output.query_id == SINGLE_QUERY_ID
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set_map"
        assert set(output.annotation_gt.value) == {"source_moved_objects", "destination_existing_objects"}
        assert output.trace_payload["projected_annotation"]["bbox_set_map"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_bbox_set_map"] == output.annotation_gt.value
        assert int(output.answer_gt.value) == operands["source_moved_objects"] + operands["destination_existing_objects"]
        assert 1 <= operands["source_moved_objects"] <= 4
        assert 1 <= operands["destination_existing_objects"] <= destination_cap
        assert 2 <= int(output.answer_gt.value) <= destination_cap
        assert str(trace["source_belt_key"]) != str(trace["destination_belt_key"])
        assert "{target_" not in output.prompt
        assert "unlettered" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)

        for key, ids in target_ids_by_key.items():
            expected = [render_map["object_bboxes_px"][str(object_id)] for object_id in ids]
            assert output.annotation_gt.value[str(key)] == expected
            assert len(expected) == operands[str(key)]
        for object_id in target_ids_by_key["source_moved_objects"]:
            spec = object_by_id[str(object_id)]
            assert str(spec["belt_key"]) == str(trace["source_belt_key"])
            if task_id == COLOR_TRANSFER_TASK_ID:
                assert trace["predicate_kind"] == "color_transfer"
                assert str(spec["color_name"]) == str(trace["target_color_name"])
            else:
                assert trace["predicate_kind"] == "object_type_transfer"
                assert str(spec["shape_type"]) == str(trace["target_shape_type"])
        for object_id in target_ids_by_key["destination_existing_objects"]:
            assert str(object_by_id[str(object_id)]["belt_key"]) == str(trace["destination_belt_key"])
        assert target_ids_by_key["destination_existing_objects"] == trace["target_belt_object_ids"]
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_scope_count_after_transfer_supports_destination_caps() -> None:
    cases = (
        (COLOR_TRANSFER_TASK_ID, "outer", "inner", 8, 4, 4, 2026062827),
        (OBJECT_TYPE_TRANSFER_TASK_ID, "inner", "outer", 12, 4, 8, 2026062828),
    )
    for task_id, source_belt, destination_belt, answer_value, moved_count, destination_existing_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "source_belt_key": source_belt,
                "destination_belt_key": destination_belt,
                "answer_value": answer_value,
                "moved_count": moved_count,
                "destination_existing_count": destination_existing_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]

        assert int(output.answer_gt.value) == int(answer_value)
        assert str(trace["source_belt_key"]) == str(source_belt)
        assert str(trace["destination_belt_key"]) == str(destination_belt)
        assert int(trace["moved_count"]) == int(moved_count)
        assert int(trace["destination_existing_count"]) == int(destination_existing_count)
        assert len(output.annotation_gt.value["source_moved_objects"]) == int(moved_count)
        assert len(output.annotation_gt.value["destination_existing_objects"]) == int(destination_existing_count)
        assert int(trace["belt_counts"].get("inner", 0)) <= 8
        assert int(trace["belt_counts"].get("outer", 0)) <= 12


def test_carousel_renderer_has_no_unqueried_gate_decoration() -> None:
    source = Path("src/trace_tasks/tasks/three_d/carousel/shared/rendering.py").read_text()

    assert "inspection_gate" not in source
    assert "three_d_conveyor_inspection_gate" not in source
