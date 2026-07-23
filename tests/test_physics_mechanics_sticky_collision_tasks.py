"""Contract tests for physics mechanics sticky-collision tasks."""

from __future__ import annotations

import math
from collections import Counter

from trace_tasks.tasks.physics.collision.sticky_collision_direction_choice import (
    PhysicsCollisionStickyCollisionDirectionChoiceTask,
)
from trace_tasks.tasks.physics.collision.sticky_collision_speed_value import (
    PhysicsCollisionStickyCollisionSpeedValueTask,
)


def _assert_segment_in_bounds(out, segment) -> None:
    width, height = out.image.size
    assert len(segment) == 2
    for point in segment:
        assert len(point) == 2
        assert 0 <= point[0] <= width
        assert 0 <= point[1] <= height


def test_physics_mechanics_sticky_collision_direction_choice_contract() -> None:
    out = PhysicsCollisionStickyCollisionDirectionChoiceTask().generate(
        41001,
        params={
            "scene_variant": "wide_table",
            "target_answer": "D",
            "horizontal_mass": 2,
            "vertical_mass": 2,
            "horizontal_speed": 8,
            "vertical_speed": 6,
            "horizontal_direction": "right",
            "vertical_direction": "up",
        },
        max_attempts=30,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scenario = execution["scenario"]


    assert out.answer_gt.type == "option_letter"

    assert out.answer_gt.value == "D"

    assert out.annotation_gt.type == "segment_set"

    assert out.annotation_gt.value == [
        trace["render_map"]["horizontal_motion_arrow_segment_px"],
        trace["render_map"]["vertical_motion_arrow_segment_px"],
    ]
    for segment in out.annotation_gt.value:
        _assert_segment_in_bounds(out, segment)

    assert out.query_id == "single"
    assert trace["query_spec"]["query_id"] == "single"

    assert trace["query_spec"]["params"]["operation_kind"] == "direction_choice"

    assert execution["operation_kind"] == "direction_choice"

    assert scenario["correct_option_letter"] == "D"
    assert scenario["option_letters"] == ["A", "B", "C", "D"]
    assert set(trace["render_map"]["option_bboxes_px"]) == {"A", "B", "C", "D"}
    assert set(trace["render_map"]["option_angles_degrees"]) == {"A", "B", "C", "D"}

    assert execution["annotation_entity_ids"] == [
        "horizontal_motion_arrow",
        "vertical_motion_arrow",
    ]
    assert execution["annotation_key_by_entity_id"] == {}
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_segment_set"] == out.annotation_gt.value

    assert "resultant_arrow_bbox_px" not in trace["render_map"]

    assert int(scenario["final_vx"]) == 4

    assert int(scenario["final_vy"]) == 3

    assert math.isclose(float(scenario["direction_angle_degrees"]), math.degrees(math.atan2(3, 4)), abs_tol=0.01)


def test_physics_mechanics_sticky_collision_speed_contract() -> None:
    params = {
        "scene_variant": "gridded_table",
        "horizontal_mass": 2,
        "vertical_mass": 2,
        "horizontal_speed": 8,
        "vertical_speed": 6,
        "horizontal_direction": "left",
        "vertical_direction": "down",
        "correct_option_letter": "F",
        "target_answer": 5.0,
    }
    speed = PhysicsCollisionStickyCollisionSpeedValueTask().generate(
        41011,
        params=params,
        max_attempts=30,
    )

    assert speed.answer_gt.type == "number"

    assert float(speed.answer_gt.value) == 5.0

    assert speed.query_id == "single"

    assert speed.annotation_gt.type == "segment_set"
    assert speed.annotation_gt.value == [
        speed.trace_payload["render_map"]["horizontal_motion_arrow_segment_px"],
        speed.trace_payload["render_map"]["vertical_motion_arrow_segment_px"],
    ]
    for segment in speed.annotation_gt.value:
        _assert_segment_in_bounds(speed, segment)

    assert speed.trace_payload["execution_trace"]["annotation_entity_ids"] == [
        "horizontal_motion_arrow",
        "vertical_motion_arrow",
    ]
    assert speed.trace_payload["execution_trace"]["annotation_key_by_entity_id"] == {}
    assert speed.trace_payload["projected_annotation"]["segment_set"] == speed.annotation_gt.value
    assert speed.trace_payload["render_map"]["show_candidate_options"] is False
    assert speed.trace_payload["render_map"]["option_bboxes_px"] == {}

    scenario = speed.trace_payload["execution_trace"]["scenario"]
    assert int(scenario["final_vx"]) == -4

    assert int(scenario["final_vy"]) == -3

    assert float(scenario["final_speed_rounded"]) == 5.0


def test_physics_mechanics_sticky_collision_is_deterministic() -> None:
    params = {
        "scene_variant": "compact_table",
        "target_answer": 5.0,
        "correct_option_letter": "C",
        "accent_color_name": "cyan",
    }
    task = PhysicsCollisionStickyCollisionSpeedValueTask()
    out_a = task.generate(41021, params=params, max_attempts=60)
    out_b = task.generate(41021, params=params, max_attempts=60)


    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()

    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()

    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]

    assert out_a.prompt == out_b.prompt

    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_mechanics_sticky_collision_sampling_covers_answers() -> None:
    direction_letters: Counter[str] = Counter()
    speed_values: set[float] = set()
    for sampling_index in range(96):
        direction = PhysicsCollisionStickyCollisionDirectionChoiceTask().generate(
            41100 + sampling_index,
            params={},
            max_attempts=80,
        )
        speed = PhysicsCollisionStickyCollisionSpeedValueTask().generate(
            41200 + sampling_index,
            params={},
            max_attempts=80,
        )
        direction_letters[str(direction.answer_gt.value)] += 1
        speed_values.add(float(speed.answer_gt.value))


    assert set(direction_letters) == {"A", "B", "C", "D"}

    assert speed_values == {
        1.4,
        2.2,
        2.8,
        3.2,
        3.6,
        4.1,
        4.2,
        4.5,
        5.0,
        5.1,
        5.4,
        5.7,
        5.8,
        6.1,
        6.3,
        6.4,
        6.7,
        7.1,
        7.2,
        7.8,
        8.5,
    }
