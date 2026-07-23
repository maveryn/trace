"""Contract tests for physics thermal-mixing tasks."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.tasks.physics.thermal_mixing.final_temperature_value import (
    PhysicsThermalMixingFinalTemperatureValueTask,
)


def _assert_bbox_set_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox_set"
    for bbox in out.annotation_gt.value:
        assert 0 <= bbox[0] < bbox[2] <= width
        assert 0 <= bbox[1] < bbox[3] <= height


def test_thermal_mixing_final_temperature_contract_for_supported_cup_counts() -> None:
    task = PhysicsThermalMixingFinalTemperatureValueTask()

    for cup_count in [2, 3, 4]:
        out = task.generate(
            95100 + cup_count,
            params={"cup_count": cup_count, "target_answer": 45},
            max_attempts=10,
        )
        execution = out.trace_payload["execution_trace"]
        temperatures = [int(value) for value in execution["initial_temperatures_c"]]

        assert out.scene_id == "thermal_mixing"
        assert out.query_id == "single"
        assert execution["internal_query_id"] == "equal_amount_final_temperature"
        assert out.answer_gt.type == "integer"
        assert out.answer_gt.value == 45
        assert len(temperatures) == cup_count
        assert sum(temperatures) // len(temperatures) == out.answer_gt.value
        assert sum(temperatures) % len(temperatures) == 0
        assert len(out.annotation_gt.value) == cup_count
        _assert_bbox_set_in_bounds(out)
        assert out.trace_payload["projected_annotation"]["bboxes"] == out.annotation_gt.value
        assert out.trace_payload["render_map"]["annotation_source"] == "temperature_label_bboxes_px"
        assert out.trace_payload["render_map"]["temperature_label_bboxes_px"] == out.annotation_gt.value
        assert out.prompt_variants["answer_only"]
        assert out.prompt_variants["answer_and_annotation"]
        assert "temperature label" in out.prompt_variants["answer_and_annotation"]
        assert "initial cups and their visible temperature labels" not in out.prompt_variants["answer_and_annotation"]


def test_thermal_mixing_keeps_final_answer_hidden_in_image_metadata() -> None:
    out = PhysicsThermalMixingFinalTemperatureValueTask().generate(
        95131,
        params={"cup_count": 4, "target_answer": 50},
        max_attempts=10,
    )
    render_map = out.trace_payload["render_map"]

    assert render_map["final_temperature_c"] == 50
    assert render_map["initial_temperatures_c"] == out.trace_payload["execution_trace"]["initial_temperatures_c"]
    assert all("initial_cup_" in entity_id for entity_id in out.trace_payload["execution_trace"]["annotation_entity_ids"])
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 4
