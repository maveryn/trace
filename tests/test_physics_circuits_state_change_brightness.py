"""Contract tests for physics circuit state-change bulb brightness tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.physics.circuit_state_change.bulb_brightness_change_label import (
    PhysicsCircuitStateChangeBulbBrightnessLabelTask,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


QUERY_TO_CHANGE_CLASS = {
    "brightens_after_switch_change": "brightens",
    "dims_after_switch_change": "dims",
    "turns_on_after_switch_change": "turns_on",
    "turns_off_after_switch_change": "turns_off",
}


def _assert_bbox_map_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox_map"
    for bbox in out.annotation_gt.value.values():
        assert 0 <= bbox[0] < bbox[2] <= width
        assert 0 <= bbox[1] < bbox[3] <= height


@pytest.mark.parametrize(
    ("query_id", "switch_action"),
    (
        ("brightens_after_switch_change", "closes"),
        ("dims_after_switch_change", "opens"),
        ("turns_on_after_switch_change", "closes"),
        ("turns_off_after_switch_change", "opens"),
    ),
)
def test_physics_state_change_brightness_answer_matches_trace(query_id: str, switch_action: str) -> None:
    out = PhysicsCircuitStateChangeBulbBrightnessLabelTask().generate(
        28901,
        params={
            "query_id": query_id,
            "switch_action": switch_action,
            "target_label": "B2",
            "resistance_values": {
                "series_bulb": 5,
                "main_branch_bulb": 3,
                "switched_branch_bulb": 8,
                "reference_branch_bulb_1": 4,
                "reference_branch_bulb_2": 10,
            },
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    target_change = QUERY_TO_CHANGE_CLASS[query_id]
    matches = [spec for spec in execution["bulb_specs"] if str(spec["change_class"]) == target_change]

    assert out.scene_id == "circuit_state_change"
    assert out.query_id == query_id
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == "B2"
    assert len(matches) == 1
    assert str(matches[0]["label"]) == out.answer_gt.value
    assert execution["target_change_class"] == target_change
    assert execution["switch_action"] == switch_action
    assert set(out.annotation_gt.value) == {"changed_switch", "B1", "B2", "B3", "B4", "B5"}
    _assert_bbox_map_in_bounds(out)
    assert trace["render_map"]["correct_label"] == out.answer_gt.value
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["witness_symbolic"]["key_to_entity_id"] == {
        "changed_switch": "changed_switch",
        "B1": "B1",
        "B2": "B2",
        "B3": "B3",
        "B4": "B4",
        "B5": "B5",
    }


def test_physics_state_change_brightness_rejects_incompatible_switch_actions() -> None:
    task = PhysicsCircuitStateChangeBulbBrightnessLabelTask()
    with pytest.raises(ValueError, match="incompatible"):
        task.generate(
            28921,
            params={"query_id": "turns_on_after_switch_change", "switch_action": "opens"},
            max_attempts=20,
        )
    with pytest.raises(ValueError, match="incompatible"):
        task.generate(
            28922,
            params={"query_id": "turns_off_after_switch_change", "switch_action": "closes"},
            max_attempts=20,
        )


def test_physics_state_change_brightness_is_deterministic() -> None:
    params = {
        "query_id": "dims_after_switch_change",
        "switch_action": "closes",
        "target_label": "B1",
        "accent_color_name": "cyan",
        "resistance_values": {
            "series_bulb": 5,
            "main_branch_bulb": 3,
            "switched_branch_bulb": 8,
            "reference_branch_bulb_1": 4,
            "reference_branch_bulb_2": 10,
        },
    }
    task = PhysicsCircuitStateChangeBulbBrightnessLabelTask()
    out_a = task.generate(28931, params=params, max_attempts=20)
    out_b = task.generate(28931, params=params, max_attempts=20)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_state_change_brightness_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("physics", "circuit_state_change")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__circuit_state_change__bulb_brightness_change_label",
    )
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/physics/circuit_state_change/physics_circuit_state_change_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert list(generation["resistance_options"]) == [2, 3, 4, 5, 6, 8, 10, 12]
    assert int(rendering["canvas_width"]) == 1280
    assert str(prompt["bundle_id"]) == "physics_circuit_state_change_v1"
    assert str(prompt["task_key"]) == "bulb_brightness_change_query"
    assert "scene:circuit_state_change_diagram" in bundle["required_slots_by_key"]
    for query_id in QUERY_TO_CHANGE_CLASS:
        assert query_id in bundle["templates"]["query"]
        assert len(bundle["templates"]["query"][query_id]) == 5
        assert "B1 through B5" in str(bundle["static_slots_by_key"][f"query:{query_id}"]["annotation_hint"])
