"""Contract tests for physics electrostatic-field tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.tasks.physics.electrostatic_field.field_direction_choice import (
    PhysicsElectrostaticFieldDirectionChoiceTask,
)
from trace_tasks.tasks.physics.electrostatic_field.potential_value import (
    PhysicsElectrostaticFieldPotentialValueTask,
)
from trace_tasks.tasks.physics.electrostatic_field.zero_field_point_label import (
    PhysicsElectrostaticFieldZeroFieldPointLabelTask,
)


def _bbox_overlaps(left, right) -> bool:
    return not (
        float(left[2]) <= float(right[0])
        or float(left[0]) >= float(right[2])
        or float(left[3]) <= float(right[1])
        or float(left[1]) >= float(right[3])
    )


def test_physics_electrostatic_field_direction_choice_contract() -> None:
    out = PhysicsElectrostaticFieldDirectionChoiceTask().generate(
        81001,
        params={
            "scene_variant": "clean_grid",
            "direction_mode": "force_on_negative_charge",
            "target_direction": "northwest",
            "correct_option_letter": "C",
            "accent_color_name": "blue",
        },
        max_attempts=30,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scenario = execution["direction_scenario"]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"Q1", "Q2", "Q3", "P"}
    assert all(len(point) == 2 for point in out.annotation_gt.value.values())
    assert out.scene_id == "electrostatic_field"
    assert out.query_id == "single"
    assert trace["query_spec"]["query_id"] == "single"
    assert execution["direction_mode"] == "force_on_negative_charge"
    assert scenario["requested_direction"] == "northwest"
    assert scenario["option_directions"]["C"] == "northwest"
    assert execution["annotation_entity_ids"] == [
        "charge_main",
        "charge_cancel_a",
        "charge_cancel_b",
        "query_point",
    ]
    assert execution["annotation_key_by_entity_id"] == {
        "charge_main": "Q1",
        "charge_cancel_a": "Q2",
        "charge_cancel_b": "Q3",
        "query_point": "P",
    }
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["render_map"]["annotation_keyed_points_px"] == out.annotation_gt.value
    assert trace["render_spec"]["technical_diagram_style"]["kind"] == "technical_diagram_style"
    assert trace["render_spec"]["technical_diagram_style"]["protected_colors_rgb"]
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_electrostatics_diagram_offset"

    entities = {str(entity["entity_id"]): entity for entity in trace["scene_ir"]["entities"]}
    charge_entity = entities["charge_main"]
    charge_bbox = charge_entity["bbox_px"]
    marker_bbox = charge_entity["meta"]["charge_marker_bbox_px"]
    id_label_bbox = charge_entity["meta"]["charge_id_label_bbox_px"]
    label_bbox = charge_entity["meta"]["charge_label_bbox_px"]
    assert charge_bbox[0] <= marker_bbox[0] <= marker_bbox[2] <= charge_bbox[2]
    assert charge_bbox[1] <= marker_bbox[1] <= marker_bbox[3] <= charge_bbox[3]
    assert charge_bbox[0] <= id_label_bbox[0] <= id_label_bbox[2] <= charge_bbox[2]
    assert charge_bbox[1] <= id_label_bbox[1] <= id_label_bbox[3] <= charge_bbox[3]
    assert charge_bbox[0] <= label_bbox[0] <= label_bbox[2] <= charge_bbox[2]
    assert charge_bbox[1] <= label_bbox[1] <= label_bbox[3] <= charge_bbox[3]
    assert charge_entity["meta"]["display_label"] == "Q1"
    assert str(charge_entity["meta"]["charge_label_text"]).startswith("Q1=")
    assert "candidate arrow" in out.prompt


def test_physics_electrostatic_field_zero_field_point_label_contract() -> None:
    out = PhysicsElectrostaticFieldZeroFieldPointLabelTask().generate(
        81002,
        params={
            "scene_variant": "paper_grid",
            "correct_option_letter": "E",
            "accent_color_name": "green",
        },
        max_attempts=30,
    )
    scenario = out.trace_payload["execution_trace"]["zero_field_scenario"]
    correct = [point for point in scenario["candidate_points"] if point["is_correct"]][0]
    field_x = 0.0
    field_y = 0.0
    for charge in scenario["charges"]:
        dx = int(correct["x"]) - int(charge["x"])
        dy = int(correct["y"]) - int(charge["y"])
        distance = float((dx * dx + dy * dy) ** 0.5)
        field_x += float(charge["charge_value"]) * float(dx) / float(distance**3)
        field_y += float(charge["charge_value"]) * float(dy) / float(distance**3)
    charge_values = [int(charge["charge_value"]) for charge in scenario["charges"]]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "E"
    assert out.query_id == "single"
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"Q1", "Q2", "zero_point"}
    assert all(len(point) == 2 for point in out.annotation_gt.value.values())
    assert scenario["correct_option_letter"] == "E"
    assert correct["option_letter"] == "E"
    assert len(set(abs(value) for value in charge_values)) == 2
    assert charge_values[0] * charge_values[1] > 0
    assert abs(field_x) < 1e-9
    assert abs(field_y) < 1e-9
    expected_ids = [str(charge["charge_id"]) for charge in scenario["charges"]] + ["candidate_E"]
    assert out.trace_payload["execution_trace"]["annotation_entity_ids"] == expected_ids
    assert out.trace_payload["execution_trace"]["annotation_key_by_entity_id"] == {
        expected_ids[0]: "Q1",
        expected_ids[1]: "Q2",
        "candidate_E": "zero_point",
    }


def test_physics_electrostatic_field_zero_field_candidate_labels_avoid_charge_labels() -> None:
    out = PhysicsElectrostaticFieldZeroFieldPointLabelTask().generate(
        7304052978720082,
        params={
            "scene_variant": "dense_grid",
            "correct_option_letter": "E",
            "accent_color_name": "yellow",
            "post_image_noise": {"enabled": False},
        },
        max_attempts=30,
    )
    render_map = out.trace_payload["render_map"]

    assert set(render_map["candidate_label_bboxes_px"]) == {"A", "B", "C", "D", "E", "F"}
    assert len(render_map["charge_label_bboxes_px"]) == 2
    for candidate_bbox in render_map["candidate_label_bboxes_px"].values():
        for charge_label_bbox in render_map["charge_label_bboxes_px"]:
            assert not _bbox_overlaps(candidate_bbox, charge_label_bbox)


def test_physics_electrostatic_field_potential_value_contract() -> None:
    out = PhysicsElectrostaticFieldPotentialValueTask().generate(
        81003,
        params={
            "scene_variant": "dense_grid",
            "target_answer": 4,
            "potential_contributions": [1, 2, 1],
            "accent_color_name": "cyan",
        },
        max_attempts=30,
    )
    scenario = out.trace_payload["execution_trace"]["potential_scenario"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 4
    assert out.query_id == "single"
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"Q1", "Q2", "Q3", "P"}
    assert all(len(point) == 2 for point in out.annotation_gt.value.values())
    assert [charge["potential_contribution"] for charge in scenario["charges"]] == [1, 2, 1]
    assert len(scenario["charges"]) == 3
    assert sum(int(charge["charge_value"]) // int(charge["distance_units"]) for charge in scenario["charges"]) == 4
    assert scenario["potential_value"] == 4
    expected_ids = [str(charge["charge_id"]) for charge in scenario["charges"]] + ["query_point"]
    assert out.trace_payload["execution_trace"]["annotation_entity_ids"] == expected_ids
    assert out.trace_payload["execution_trace"]["annotation_key_by_entity_id"] == {
        expected_ids[0]: "Q1",
        expected_ids[1]: "Q2",
        expected_ids[2]: "Q3",
        "query_point": "P",
    }
    assert out.trace_payload["projected_annotation"]["point_map"] == out.annotation_gt.value


def test_physics_electrostatic_field_tasks_are_deterministic() -> None:
    params = {
        "scene_variant": "paper_grid",
        "target_answer": -3,
        "accent_color_name": "orange",
    }
    task = PhysicsElectrostaticFieldPotentialValueTask()
    out_a = task.generate(81031, params=params, max_attempts=60)
    out_b = task.generate(81031, params=params, max_attempts=60)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_electrostatic_field_sampling_covers_internal_axes() -> None:
    direction_modes: Counter[str] = Counter()
    target_directions: Counter[str] = Counter()
    direction_letters: Counter[str] = Counter()
    zero_letters: Counter[str] = Counter()
    potential_answers: set[int] = set()

    for sampling_index in range(96):
        direction = PhysicsElectrostaticFieldDirectionChoiceTask().generate(
            81100 + sampling_index,
            params={},
            max_attempts=80,
        )
        zero = PhysicsElectrostaticFieldZeroFieldPointLabelTask().generate(
            81200 + sampling_index,
            params={},
            max_attempts=80,
        )
        potential = PhysicsElectrostaticFieldPotentialValueTask().generate(
            81300 + sampling_index,
            params={},
            max_attempts=80,
        )
        direction_execution = direction.trace_payload["execution_trace"]
        direction_modes[str(direction_execution["direction_mode"])] += 1
        target_directions[str(direction_execution["target_direction"])] += 1
        direction_letters[str(direction.answer_gt.value)] += 1
        zero_letters[str(zero.answer_gt.value)] += 1
        potential_answers.add(int(potential.answer_gt.value))

    assert set(direction_modes) == {
        "electric_field_direction",
        "force_on_positive_charge",
        "force_on_negative_charge",
    }
    assert set(target_directions) == {
        "east",
        "northeast",
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
    }
    assert set(direction_letters) == {"A", "B", "C", "D", "E", "F", "G", "H"}
    assert set(zero_letters) == {"A", "B", "C", "D", "E", "F"}
    assert len(potential_answers) >= 12


def test_physics_electrostatic_field_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/physics/electrostatic_field/physics_electrostatic_field_v1.json").read_text(encoding="utf-8")
    )

    assert len(bundle["templates"]["scene"]["electrostatic_field_map"]) == 5
    assert set(bundle["templates"]["query"]) == {
        "field_direction_choice",
        "zero_field_point_label",
        "potential_value",
    }
    assert len(bundle["templates"]["query"]["field_direction_choice"]) == 5
    assert len(bundle["templates"]["query"]["zero_field_point_label"]) == 5
    assert len(bundle["templates"]["query"]["potential_value"]) == 5
