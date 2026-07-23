"""Contract tests for magnetic-force scene tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.tasks.physics.magnetic_force.force_direction_choice import (
    PhysicsMagneticForceForceDirectionChoiceTask,
)


def _bbox_inside(inner: list[float], outer: list[float], *, margin_px: float = 0.0) -> bool:
    return (
        float(inner[0]) >= float(outer[0]) - float(margin_px)
        and float(inner[1]) >= float(outer[1]) - float(margin_px)
        and float(inner[2]) <= float(outer[2]) + float(margin_px)
        and float(inner[3]) <= float(outer[3]) + float(margin_px)
    )


def test_physics_magnetism_force_direction_choice_contract() -> None:
    out = PhysicsMagneticForceForceDirectionChoiceTask().generate(
        97001,
        params={
            "scene_variant": "clean_panel",
            "field_orientation": "out_of_page",
            "velocity_direction": "east",
            "charge_sign": 1,
            "correct_option_letter": "D",
            "accent_color_name": "blue",
        },
        max_attempts=30,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scenario = execution["direction_scenario"]


    assert out.answer_gt.type == "option_letter"

    assert out.answer_gt.value == "D"

    assert out.annotation_gt.type == "bbox_map"

    assert set(out.annotation_gt.value) == {"field_orientation", "charge", "velocity"}

    assert out.scene_id == "magnetic_force"

    assert out.query_id == "single"
    assert trace["query_spec"]["query_id"] == "single"

    assert trace["query_spec"]["params"]["internal_query_id"] == "force_direction_choice"

    assert execution["field_orientation"] == "out_of_page"

    assert execution["charge_sign"] == 1

    assert scenario["velocity_direction"] == "east"

    assert scenario["force_direction"] == "south"

    assert scenario["option_directions"]["D"] == "south"

    assert execution["annotation_entity_ids"] == ["field_orientation_label", "particle", "velocity_vector"]

    assert trace["projected_annotation"]["type"] == "bbox_map"

    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value

    assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value

    assert trace["render_map"]["correct_option_bbox_px"] not in out.annotation_gt.value.values()

    assert set(trace["render_map"]["option_cell_bboxes_px"]) == set("ABCDEFGH")

    assert set(trace["render_map"]["option_arrow_bboxes_px"]) == set("ABCDEFGH")

    for letter, arrow_bbox in trace["render_map"]["option_arrow_bboxes_px"].items():
        assert _bbox_inside(arrow_bbox, trace["render_map"]["option_cell_bboxes_px"][letter])

    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"

    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_magnetic_force_diagram_offset"


def test_physics_magnetism_tasks_are_deterministic() -> None:
    params = {
        "scene_variant": "lab_card",
        "velocity_direction": "north",
        "field_orientation": "into_page",
        "charge_sign": -1,
        "correct_option_letter": "F",
        "accent_color_name": "orange",
    }
    task = PhysicsMagneticForceForceDirectionChoiceTask()
    out_a = task.generate(97031, params=params, max_attempts=60)
    out_b = task.generate(97031, params=params, max_attempts=60)


    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()

    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()

    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]

    assert out_a.prompt == out_b.prompt

    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_magnetism_sampling_covers_internal_axes() -> None:
    field_orientations: Counter[str] = Counter()
    velocity_directions: Counter[str] = Counter()
    charge_signs: Counter[int] = Counter()
    direction_letters: Counter[str] = Counter()

    for sampling_index in range(96):
        direction = PhysicsMagneticForceForceDirectionChoiceTask().generate(
            97100 + sampling_index,
            params={},
            max_attempts=80,
        )
        direction_execution = direction.trace_payload["execution_trace"]
        field_orientations[str(direction_execution["field_orientation"])] += 1
        velocity_directions[str(direction_execution["velocity_direction"])] += 1
        charge_signs[int(direction_execution["charge_sign"])] += 1
        direction_letters[str(direction.answer_gt.value)] += 1


    assert set(field_orientations) == {"out_of_page", "into_page"}

    assert set(velocity_directions) == {
        "east",
        "northeast",
        "north",
        "northwest",
        "west",
        "southwest",
        "south",
        "southeast",
    }

    assert set(charge_signs) == {-1, 1}

    assert {"B", "C", "D", "E", "G", "H"}.issubset(set(direction_letters))


def test_physics_magnetism_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(Path("src/trace_tasks/resources/prompts/physics/magnetic_force/physics_magnetic_force_v1.json").read_text(encoding="utf-8"))

    assert len(bundle["templates"]["scene"]["magnetic_force_field"]) == 5

    assert set(bundle["templates"]["query"]) == {"single"}

    assert len(bundle["templates"]["query"]["single"]) == 5
