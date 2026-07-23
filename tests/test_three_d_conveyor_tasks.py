"""Tests for synthetic 3D straight conveyor tasks."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import create_task
from trace_tasks.tasks.three_d.conveyor.belt_total_object_count import (
    QUERY_ID as TOTAL_QUERY_ID,
    TASK_ID as TOTAL_TASK_ID,
)
from trace_tasks.tasks.three_d.conveyor.between_object_type_anchors_count import TASK_ID as BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID
from trace_tasks.tasks.three_d.conveyor.color_ordered_adjacent_pair_count import TASK_ID as COLOR_ORDERED_PAIR_TASK_ID
from trace_tasks.tasks.three_d.conveyor.color_transfer_total_count import TASK_ID as COLOR_TRANSFER_TASK_ID
from trace_tasks.tasks.three_d.conveyor.lane_object_type_count_arithmetic_value import (
    DIFFERENCE_QUERY_ID as ARITH_DIFFERENCE_QUERY_ID,
    TASK_ID as OBJECT_TYPE_ARITH_TASK_ID,
    TOTAL_QUERY_ID as ARITH_TOTAL_QUERY_ID,
)
from trace_tasks.tasks.three_d.conveyor.object_type_ordered_adjacent_pair_count import TASK_ID as OBJECT_TYPE_ORDERED_PAIR_TASK_ID
from trace_tasks.tasks.three_d.conveyor.object_type_transfer_total_count import TASK_ID as OBJECT_TYPE_TRANSFER_TASK_ID
from trace_tasks.tasks.three_d.conveyor.scoped_belt_color_count import TASK_ID as SCOPED_COLOR_TASK_ID
from trace_tasks.tasks.three_d.conveyor.scoped_belt_object_type_count import TASK_ID as SCOPED_OBJECT_TYPE_TASK_ID
from trace_tasks.tasks.three_d.conveyor.scoped_color_type_count import (
    TASK_ID as COLOR_TYPE_TASK_ID,
)
from trace_tasks.tasks.three_d.conveyor.shared.state import CONVEYOR_OBJECT_SHAPE_TYPES
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


def test_conveyor_object_pool_excludes_cylinder_confusers() -> None:
    assert "cylinder" in CONVEYOR_OBJECT_SHAPE_TYPES
    assert "drum" not in CONVEYOR_OBJECT_SHAPE_TYPES
    assert "pencil" not in CONVEYOR_OBJECT_SHAPE_TYPES
    assert "ruler" not in CONVEYOR_OBJECT_SHAPE_TYPES


def _assert_count_output(output) -> None:
    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    target_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    predicate_kind = str(trace["predicate_kind"])
    expected_query_id_by_predicate = {
        "belt_total": TOTAL_QUERY_ID,
        "object_type": SINGLE_QUERY_ID,
        "color": SINGLE_QUERY_ID,
        "color_type": SINGLE_QUERY_ID,
    }
    expected_internal_query_id_by_predicate = {
        "belt_total": BELT_TOTAL_INTERNAL_QUERY_ID,
        "object_type": OBJECT_TYPE_BELT_COUNT_INTERNAL_QUERY_ID,
        "color": COLOR_BELT_COUNT_INTERNAL_QUERY_ID,
        "color_type": COLOR_TYPE_BELT_COUNT_INTERNAL_QUERY_ID,
    }

    assert output.scene_id == "conveyor"
    assert output.query_id == expected_query_id_by_predicate[predicate_kind]
    assert output.trace_payload["query_spec"]["params"]["internal_query_id"] == expected_internal_query_id_by_predicate[predicate_kind]
    assert output.answer_gt.type == "integer"
    assert output.annotation_gt.type == "bbox_set"
    assert int(output.answer_gt.value) == len(target_ids)
    assert len(output.annotation_gt.value) == len(target_ids)
    assert output.annotation_gt.value == [render_map["object_bboxes_px"][object_id] for object_id in target_ids]
    assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox_set"] == output.annotation_gt.value
    assert trace["layout_family"] == "straight_parallel_conveyors"
    assert trace["layout_orientation"] in {"horizontal_lanes", "vertical_lanes"}
    assert trace["target_lane_key"] in {"top", "middle", "bottom", "left", "right"}
    assert trace["target_lane_label"] in {"TOP", "MIDDLE", "BOTTOM", "LEFT", "RIGHT"}
    if predicate_kind == "belt_total":
        assert target_ids == trace["target_lane_object_ids"]
    else:
        assert set(target_ids).issubset(set(str(object_id) for object_id in trace["target_lane_object_ids"]))
    assert set(render_map["belt_bboxes_px"]) in (
        {"top", "middle", "bottom"},
        {"left", "middle", "right"},
    )
    if predicate_kind == "belt_total":
        assert len({str(spec["shape_type"]) for spec in trace["object_specs"]}) == 1
    assert trace["target_shape_type"] not in {"pencil", "ruler"}
    assert len({str(spec["color_name"]) for spec in trace["object_specs"]}) >= 2
    assert "{target_" not in output.prompt
    if predicate_kind == "belt_total":
        assert "color" not in output.prompt.lower()
        assert "type" not in output.prompt.lower()
    assert "unlettered" not in output.prompt.lower()
    assert_three_d_canvas_contract(output)

    image_w, image_h = output.image.size
    for bbox in output.annotation_gt.value:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 < x1 <= float(image_w)
        assert 0.0 <= y0 < y1 <= float(image_h)
        assert min(x1 - x0, y1 - y0) >= 24.0
    conveyor_bbox = [float(value) for value in render_map["conveyor_bbox_px"]]
    if trace["layout_orientation"] == "horizontal_lanes":
        assert (conveyor_bbox[2] - conveyor_bbox[0]) / float(image_w) >= 0.78
    else:
        assert (conveyor_bbox[3] - conveyor_bbox[1]) / float(image_h) >= 0.62
    if predicate_kind == "object_type":
        same_lane_distractors = [
            spec
            for spec in trace["object_specs"]
            if str(spec["lane_key"]) == str(trace["target_lane_key"]) and not bool(spec["matches_query"])
        ]
        assert same_lane_distractors
        assert any(str(spec["shape_type"]) != str(trace["target_shape_type"]) for spec in same_lane_distractors)
    if predicate_kind == "color":
        same_lane_distractors = [
            spec
            for spec in trace["object_specs"]
            if str(spec["lane_key"]) == str(trace["target_lane_key"]) and not bool(spec["matches_query"])
        ]
        assert same_lane_distractors
        assert any(str(spec["color_name"]) != str(trace["target_color_name"]) for spec in same_lane_distractors)
    assert max(int(value) for value in trace["lane_counts"].values()) <= 8


def test_conveyor_belt_total_count_uses_lane_positions() -> None:
    task = create_task(TOTAL_TASK_ID)
    cases = (
        ({"canvas_preset": "landscape", "target_lane_key": "top", "target_count": 8}, 2026062503),
        ({"canvas_preset": "portrait", "target_lane_key": "left", "target_count": 8}, 2026062504),
        ({"canvas_preset": "square", "layout_orientation": "horizontal_lanes", "target_lane_key": "bottom", "target_count": 5}, 2026062505),
        ({"canvas_preset": "square", "layout_orientation": "vertical_lanes", "target_lane_key": "right", "target_count": 6}, 2026062506),
    )
    for params, seed in cases:
        output = task.generate(
            seed,
            params={**params, "post_image_noise_apply_prob": 0.0},
            max_attempts=120,
        )
        _assert_count_output(output)
        assert int(output.answer_gt.value) == int(params["target_count"])
        trace = output.trace_payload["execution_trace"]
        if params["canvas_preset"] == "landscape":
            assert trace["layout_orientation"] == "horizontal_lanes"
        if params["canvas_preset"] == "portrait":
            assert trace["layout_orientation"] == "vertical_lanes"


def test_conveyor_scoped_belt_count_query_ids() -> None:
    cases = (
        (SCOPED_OBJECT_TYPE_TASK_ID, OBJECT_TYPE_BELT_COUNT_INTERNAL_QUERY_ID, 2026062581),
        (SCOPED_COLOR_TASK_ID, COLOR_BELT_COUNT_INTERNAL_QUERY_ID, 2026062582),
    )
    for task_id, internal_query_id, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=120,
        )
        _assert_count_output(output)
        assert output.query_id == SINGLE_QUERY_ID
        assert output.trace_payload["execution_trace"]["internal_query_id"] == internal_query_id
        assert 0 <= int(output.answer_gt.value) <= 5


def test_conveyor_scoped_belt_count_supports_zero_and_five() -> None:
    cases = (
        (SCOPED_OBJECT_TYPE_TASK_ID, "landscape", "top", 0, 2026062583),
        (SCOPED_OBJECT_TYPE_TASK_ID, "portrait", "left", 5, 2026062585),
        (SCOPED_COLOR_TASK_ID, "landscape", "middle", 0, 2026062585),
        (SCOPED_COLOR_TASK_ID, "portrait", "right", 5, 2026062586),
    )
    for task_id, canvas_preset, lane_key, target_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "canvas_preset": canvas_preset,
                "target_lane_key": lane_key,
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=120,
        )
        _assert_count_output(output)
        assert output.query_id == SINGLE_QUERY_ID
        assert int(output.answer_gt.value) == int(target_count)
        if int(target_count) == 0:
            assert output.annotation_gt.value == []


def test_conveyor_scoped_color_type_count_uses_conjunction_distractors() -> None:
    task = create_task(COLOR_TYPE_TASK_ID)
    cases = (
        ({"canvas_preset": "landscape", "target_lane_key": "top", "target_count": 0}, 2026062603),
        ({"canvas_preset": "portrait", "target_lane_key": "left", "target_count": 5}, 2026062604),
    )
    for params, seed in cases:
        output = task.generate(
            seed,
            params={**params, "post_image_noise_apply_prob": 0.0},
            max_attempts=120,
        )
        _assert_count_output(output)
        trace = output.trace_payload["execution_trace"]
        target_ids = [str(object_id) for object_id in trace["target_object_ids"]]
        target_shape = str(trace["target_shape_type"])
        target_color = str(trace["target_color_name"])
        target_lane = str(trace["target_lane_key"])
        confusable_shapes = set(confusable_shape_names(target_shape))

        assert trace["predicate_kind"] == "color_type"
        assert int(output.answer_gt.value) == int(params["target_count"])
        assert len(target_ids) == int(params["target_count"])
        assert output.trace_payload["query_spec"]["params"]["internal_query_id"] == COLOR_TYPE_BELT_COUNT_INTERNAL_QUERY_ID
        for spec in trace["object_specs"]:
            is_target = str(spec["object_id"]) in set(target_ids)
            if is_target:
                assert str(spec["lane_key"]) == target_lane
                assert str(spec["shape_type"]) == target_shape
                assert str(spec["color_name"]) == target_color
        same_lane_roles = {
            str(spec["count_role"])
            for spec in trace["object_specs"]
            if str(spec["lane_key"]) == target_lane and not bool(spec["matches_query"])
        }
        assert all(
            str(spec["shape_type"]) not in confusable_shapes
            for spec in trace["object_specs"]
            if str(spec["shape_type"]) != target_shape
        )
        assert "same_belt_same_color_wrong_type" in same_lane_roles
        assert "same_belt_same_type_wrong_color" in same_lane_roles
        assert any(
            str(spec["lane_key"]) != target_lane
            and str(spec["shape_type"]) == target_shape
            and str(spec["color_name"]) == target_color
            for spec in trace["object_specs"]
        )
        if int(params["target_count"]) == 0:
            assert output.annotation_gt.value == []


def test_conveyor_color_type_distractors_avoid_visually_confusable_shapes() -> None:
    output = create_task(COLOR_TYPE_TASK_ID).generate(
        2026062622,
        params={
            "canvas_preset": "landscape",
            "target_lane_key": "top",
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


def test_conveyor_color_readout_excludes_target_confusable_colors() -> None:
    cases = (
        (SCOPED_COLOR_TASK_ID, {"target_color_name": "red"}, 2026062613),
        (COLOR_TYPE_TASK_ID, {"target_color_name": "red"}, 2026062614),
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


def test_conveyor_sampled_canvas_orientation_follows_long_axis() -> None:
    task = create_task(TOTAL_TASK_ID)
    seen_presets = set()
    for seed in range(2026062510, 2026062570):
        output = task.generate(
            seed,
            params={"post_image_noise_apply_prob": 0.0},
            max_attempts=120,
        )
        _assert_count_output(output)
        render_spec = output.trace_payload["render_spec"]
        trace = output.trace_payload["execution_trace"]
        width = int(render_spec["scene_canvas_width"])
        height = int(render_spec["scene_canvas_height"])
        seen_presets.add(str(render_spec["scene_canvas_preset"]))
        if width > height:
            assert trace["layout_orientation"] == "horizontal_lanes"
        elif height > width:
            assert trace["layout_orientation"] == "vertical_lanes"
        else:
            assert trace["layout_orientation"] in {"horizontal_lanes", "vertical_lanes"}
        if {"landscape", "portrait", "square"}.issubset(seen_presets):
            break
    assert {"landscape", "portrait", "square"}.issubset(seen_presets)


def test_conveyor_non_square_canvas_ignores_conflicting_layout_orientation() -> None:
    task = create_task(TOTAL_TASK_ID)
    cases = (
        (
            {
                "canvas_preset": "portrait",
                "layout_orientation": "horizontal_lanes",
                "target_lane_key": "left",
                "post_image_noise_apply_prob": 0.0,
            },
            "vertical_lanes",
        ),
        (
            {
                "canvas_preset": "landscape",
                "layout_orientation": "vertical_lanes",
                "target_lane_key": "top",
                "post_image_noise_apply_prob": 0.0,
            },
            "horizontal_lanes",
        ),
    )
    for params, expected_orientation in cases:
        output = task.generate(
            2026062571,
            params=dict(params),
            max_attempts=120,
        )
        _assert_count_output(output)
        assert output.trace_payload["execution_trace"]["layout_orientation"] == expected_orientation


def test_conveyor_lane_count_arithmetic_query_ids() -> None:
    cases = (
        (OBJECT_TYPE_ARITH_TASK_ID, ARITH_TOTAL_QUERY_ID, OBJECT_COUNT_SUM_INTERNAL_QUERY_ID, {"canvas_preset": "landscape"}, 2026062707),
        (OBJECT_TYPE_ARITH_TASK_ID, ARITH_DIFFERENCE_QUERY_ID, OBJECT_COUNT_DIFFERENCE_INTERNAL_QUERY_ID, {"canvas_preset": "portrait"}, 2026062708),
    )
    for task_id, query_id, internal_query_id, params, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={**params, "query_id": query_id, "post_image_noise_apply_prob": 0.0},
            max_attempts=160,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        operands = {str(key): int(value) for key, value in trace["operand_counts_by_scope"].items()}
        expected_keys = {f"{lane_key}_objects" for lane_key in trace["scope_keys"]}

        assert output.scene_id == "conveyor"
        assert output.query_id == query_id
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set_map"
        assert set(output.annotation_gt.value) == expected_keys
        assert len(expected_keys) == 2
        assert output.trace_payload["projected_annotation"]["bbox_set_map"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_bbox_set_map"] == output.annotation_gt.value
        assert "{target_" not in output.prompt
        assert "unlettered" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)

        if trace["arithmetic_operation"] == "sum":
            assert int(output.answer_gt.value) == sum(operands.values())
            assert 1 <= int(output.answer_gt.value) <= 12
        else:
            operand_values = list(operands.values())
            assert int(output.answer_gt.value) == abs(int(operand_values[0]) - int(operand_values[1]))
            assert 1 <= int(output.answer_gt.value) <= 5
        for key, ids in trace["target_object_ids_by_annotation_key"].items():
            expected = [render_map["object_bboxes_px"][str(object_id)] for object_id in ids]
            assert output.annotation_gt.value[str(key)] == expected
            assert len(expected) == operands[str(key)]
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8
        assert trace["predicate_kind"] == "object_type_count_arithmetic"
        assert trace["target_object_plural"]


def test_conveyor_adjacent_pair_count_query_ids() -> None:
    cases = (
        (COLOR_ORDERED_PAIR_TASK_ID, COLOR_ORDERED_PAIR_INTERNAL_QUERY_ID, {"canvas_preset": "landscape"}, 2026062801),
        (OBJECT_TYPE_ORDERED_PAIR_TASK_ID, OBJECT_ORDERED_PAIR_INTERNAL_QUERY_ID, {"canvas_preset": "portrait"}, 2026062802),
    )
    for task_id, internal_query_id, params, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={**params, "post_image_noise_apply_prob": 0.0},
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        object_by_id = {str(spec["object_id"]): spec for spec in trace["object_specs"]}
        pair_ids = [[str(pair[0]), str(pair[1])] for pair in trace["target_pair_object_id_pairs"]]
        target_lane = str(trace["target_lane_key"])
        lane_sequence = [str(object_id) for object_id in trace["object_sequences_by_lane"][target_lane]]

        assert output.scene_id == "conveyor"
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
            first_index = lane_sequence.index(str(first_id))
            assert first_index + 1 < len(lane_sequence)
            assert lane_sequence[first_index + 1] == str(second_id)
            assert str(object_by_id[first_id]["lane_key"]) == target_lane
            assert str(object_by_id[second_id]["lane_key"]) == target_lane
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
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8


def test_conveyor_adjacent_pair_count_supports_zero_and_four() -> None:
    cases = (
        (COLOR_ORDERED_PAIR_TASK_ID, 0, 2026062811),
        (OBJECT_TYPE_ORDERED_PAIR_TASK_ID, 4, 2026062812),
    )
    for task_id, target_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]

        assert int(output.answer_gt.value) == int(target_count)
        assert len(output.annotation_gt.value) == int(target_count)
        assert len(trace["target_pair_object_id_pairs"]) == int(target_count)
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8


def test_conveyor_between_marked_items_count_query_ids() -> None:
    cases = (
        (BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID, BETWEEN_OBJECT_ANCHORS_INTERNAL_QUERY_ID, {"canvas_preset": "portrait"}, 2026062832),
    )
    for task_id, internal_query_id, params, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={**params, "post_image_noise_apply_prob": 0.0},
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        target_ids = [str(object_id) for object_id in trace["target_object_ids"]]
        target_lane = str(trace["target_lane_key"])
        lane_sequence = [str(object_id) for object_id in trace["object_sequences_by_lane"][target_lane]]
        start_index = int(trace["start_anchor_index"])
        end_index = int(trace["end_anchor_index"])
        expected_between_ids = lane_sequence[start_index + 1 : end_index]

        assert output.scene_id == "conveyor"
        assert output.query_id == SINGLE_QUERY_ID
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set"
        assert 1 <= int(output.answer_gt.value) <= 5
        assert int(output.answer_gt.value) == len(target_ids)
        assert target_ids == expected_between_ids
        assert target_ids == [str(object_id) for object_id in trace["between_object_ids"]]
        assert trace["marked_anchor_object_ids"] == [trace["start_anchor_object_id"], trace["end_anchor_object_id"]]
        assert trace["start_anchor_object_id"] == lane_sequence[start_index]
        assert trace["end_anchor_object_id"] == lane_sequence[end_index]
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
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8


def test_conveyor_between_marked_items_count_supports_one_and_five() -> None:
    cases = (
        (BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID, 1, 2026062833),
        (BETWEEN_OBJECT_TYPE_ANCHORS_TASK_ID, 5, 2026062834),
    )
    for task_id, target_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "target_count": target_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]

        assert int(output.answer_gt.value) == int(target_count)
        assert len(output.annotation_gt.value) == int(target_count)
        assert len(trace["between_object_ids"]) == int(target_count)
        assert int(trace["end_anchor_index"]) - int(trace["start_anchor_index"]) - 1 == int(target_count)
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8


def test_conveyor_scope_count_after_transfer_query_ids() -> None:
    cases = (
        (COLOR_TRANSFER_TASK_ID, COLOR_TRANSFER_INTERNAL_QUERY_ID, {"canvas_preset": "landscape"}, 2026062821),
        (OBJECT_TYPE_TRANSFER_TASK_ID, OBJECT_TRANSFER_INTERNAL_QUERY_ID, {"canvas_preset": "portrait"}, 2026062822),
    )
    for task_id, internal_query_id, params, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={**params, "post_image_noise_apply_prob": 0.0},
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

        assert output.scene_id == "conveyor"
        assert output.query_id == SINGLE_QUERY_ID
        assert trace["internal_query_id"] == internal_query_id
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "bbox_set_map"
        assert set(output.annotation_gt.value) == {"source_moved_objects", "destination_existing_objects"}
        assert output.trace_payload["projected_annotation"]["bbox_set_map"] == output.annotation_gt.value
        assert output.trace_payload["projected_annotation"]["pixel_bbox_set_map"] == output.annotation_gt.value
        assert int(output.answer_gt.value) == operands["source_moved_objects"] + operands["destination_existing_objects"]
        assert 1 <= operands["source_moved_objects"] <= 4
        assert 1 <= operands["destination_existing_objects"] <= 8
        assert 2 <= int(output.answer_gt.value) <= 12
        assert str(trace["source_lane_key"]) != str(trace["destination_lane_key"])
        assert "{target_" not in output.prompt
        assert "unlettered" not in output.prompt.lower()
        assert_three_d_canvas_contract(output)

        for key, ids in target_ids_by_key.items():
            expected = [render_map["object_bboxes_px"][str(object_id)] for object_id in ids]
            assert output.annotation_gt.value[str(key)] == expected
            assert len(expected) == operands[str(key)]
        for object_id in target_ids_by_key["source_moved_objects"]:
            spec = object_by_id[str(object_id)]
            assert str(spec["lane_key"]) == str(trace["source_lane_key"])
            if task_id == COLOR_TRANSFER_TASK_ID:
                assert trace["predicate_kind"] == "color_transfer"
                assert str(spec["color_name"]) == str(trace["target_color_name"])
            else:
                assert trace["predicate_kind"] == "object_type_transfer"
                assert str(spec["shape_type"]) == str(trace["target_shape_type"])
        for object_id in target_ids_by_key["destination_existing_objects"]:
            assert str(object_by_id[str(object_id)]["lane_key"]) == str(trace["destination_lane_key"])
        assert target_ids_by_key["destination_existing_objects"] == trace["target_lane_object_ids"]
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8


def test_conveyor_scope_count_after_transfer_supports_answer_extremes() -> None:
    cases = (
        (COLOR_TRANSFER_TASK_ID, 2, 1, 1, 2026062823),
        (OBJECT_TYPE_TRANSFER_TASK_ID, 12, 4, 8, 2026062824),
    )
    for task_id, answer_value, moved_count, destination_existing_count, seed in cases:
        task = create_task(task_id)
        output = task.generate(
            seed,
            params={
                "answer_value": answer_value,
                "moved_count": moved_count,
                "destination_existing_count": destination_existing_count,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]

        assert int(output.answer_gt.value) == int(answer_value)
        assert int(trace["moved_count"]) == int(moved_count)
        assert int(trace["destination_existing_count"]) == int(destination_existing_count)
        assert len(output.annotation_gt.value["source_moved_objects"]) == int(moved_count)
        assert len(output.annotation_gt.value["destination_existing_objects"]) == int(destination_existing_count)
        assert max(int(value) for value in trace["lane_counts"].values()) <= 8
