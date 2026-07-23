"""Contract tests for physics pressure-volume diagram tasks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from trace_tasks.tasks.physics.pv_diagram.pv_process_sign_choice import (
    PhysicsPVDiagramProcessSignChoiceTask,
)
from trace_tasks.tasks.physics.pv_diagram.pv_work_value import PhysicsPVDiagramWorkValueTask


def test_physics_pv_work_value_single_process_contract() -> None:
    out = PhysicsPVDiagramWorkValueTask().generate(
        71001,
        params={
            "scene_variant": "clean_grid",
            "work_mode": "single_process",
            "target_answer": 24,
            "pressure": 6,
            "volume_start": 2,
            "volume_end": 6,
            "accent_color_name": "blue",
        },
        max_attempts=30,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    scenario = execution["scenario"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == 24
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.scene_id == "pv_diagram"
    assert out.query_id == "single"
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["params"]["prompt_branch"] == "work_value"
    assert execution["query_id"] == "single"
    assert execution["prompt_branch"] == "work_value"
    assert execution["annotation_entity_ids"] == ["work_witness_region"]
    assert str(scenario["work_mode"]) == "single_process"
    assert int(scenario["pressure_kpa"]) == 6
    assert int(scenario["volume_start_l"]) == 2
    assert int(scenario["volume_end_l"]) == 6
    assert int(scenario["work_value"]) == 24
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert trace["render_map"]["annotation_bboxes_px"] == [out.annotation_gt.value]
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_pv_diagram_offset"


def test_physics_pv_work_value_cycle_contract() -> None:
    out = PhysicsPVDiagramWorkValueTask().generate(
        71011,
        params={
            "scene_variant": "bold_grid",
            "work_mode": "rectangular_cycle",
            "target_answer": -12,
            "pressure_low": 3,
            "pressure_high": 6,
            "volume_left": 2,
            "volume_right": 6,
            "cycle_direction": "counterclockwise",
            "accent_color_name": "green",
        },
        max_attempts=30,
    )
    scenario = out.trace_payload["execution_trace"]["scenario"]

    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == -12
    assert str(scenario["work_mode"]) == "rectangular_cycle"
    assert str(scenario["cycle_direction"]) == "counterclockwise"
    assert int(scenario["pressure_high_kpa"]) - int(scenario["pressure_low_kpa"]) == 3
    assert int(scenario["volume_right_l"]) - int(scenario["volume_left_l"]) == 4
    assert int(scenario["work_value"]) == -12
    assert out.trace_payload["execution_trace"]["annotation_entity_ids"] == ["work_witness_region"]
    assert out.annotation_gt.type == "bbox"


def test_physics_pv_process_sign_choice_contract() -> None:
    out = PhysicsPVDiagramProcessSignChoiceTask().generate(
        71021,
        params={
            "scene_variant": "paper_grid",
            "target_sign": "zero",
            "correct_option_letter": "D",
            "accent_color_name": "cyan",
        },
        max_attempts=30,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    candidates = execution["process_candidates"]

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.query_id == "single"
    assert trace["query_spec"]["params"]["prompt_branch"] == "process_sign_choice"
    assert execution["target_sign"] == "zero"
    assert execution["correct_option_letter"] == "D"
    assert execution["annotation_entity_ids"] == ["option_D_process"]
    assert sum(1 for candidate in candidates if candidate["sign"] == "zero") == 1
    assert [candidate for candidate in candidates if candidate["is_correct"]][0]["option_letter"] == "D"
    assert trace["render_map"]["option_signs"]["D"] == "zero"
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["render_map"]["annotation_bboxes_px"] == [out.annotation_gt.value]
    assert trace["render_spec"]["font"]["selection_policy"]["pool"] == "global_approved_font_pool"
    assert trace["render_spec"]["layout_placement"]["mode"] == "whole_pv_diagram_offset"


def test_physics_pv_tasks_are_deterministic() -> None:
    params = {
        "scene_variant": "paper_grid",
        "work_mode": "single_process",
        "target_answer": -18,
        "accent_color_name": "orange",
    }
    task = PhysicsPVDiagramWorkValueTask()
    out_a = task.generate(71031, params=params, max_attempts=60)
    out_b = task.generate(71031, params=params, max_attempts=60)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_pv_sampling_covers_internal_axes() -> None:
    work_modes: Counter[str] = Counter()
    work_answers: set[int] = set()
    target_signs: Counter[str] = Counter()
    option_letters: Counter[str] = Counter()
    for sampling_index in range(96):
        work = PhysicsPVDiagramWorkValueTask().generate(
            71100 + sampling_index,
            params={},
            max_attempts=80,
        )
        sign = PhysicsPVDiagramProcessSignChoiceTask().generate(
            71200 + sampling_index,
            params={},
            max_attempts=80,
        )
        work_execution = work.trace_payload["execution_trace"]
        sign_execution = sign.trace_payload["execution_trace"]
        work_modes[str(work_execution["work_mode"])] += 1
        work_answers.add(int(work.answer_gt.value))
        target_signs[str(sign_execution["target_sign"])] += 1
        option_letters[str(sign.answer_gt.value)] += 1

    assert set(work_modes) == {"single_process"}
    assert len(work_answers) >= 18
    assert set(target_signs) == {"positive", "negative", "zero"}
    assert set(option_letters) == {"A", "B", "C", "D", "E", "F", "G", "H"}


def test_physics_pv_prompt_bundle_supports_variants() -> None:
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/physics/pv_diagram/physics_pv_diagram_v1.json").read_text(encoding="utf-8")
    )

    assert bundle["schema_version"] == "v1"
    assert len(bundle["templates"]["scene"]["pressure_volume_diagram"]) == 5
    assert set(bundle["templates"]["query"]) == {"work_value", "process_sign_choice"}
    assert len(bundle["templates"]["query"]["work_value"]) == 5
    assert len(bundle["templates"]["query"]["process_sign_choice"]) == 5
    assert len(set(bundle["templates"]["output"]["answer_and_annotation"])) == 5
