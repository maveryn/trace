"""Contract tests for physics bulb-brightness circuit tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.physics.bulb_circuit.brightness_extremum_label import (
    PhysicsBulbCircuitBrightnessExtremumLabelTask,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def _assert_bbox_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox"
    bbox = out.annotation_gt.value
    assert 0 <= bbox[0] < bbox[2] <= width
    assert 0 <= bbox[1] < bbox[3] <= height


@pytest.mark.parametrize(
    ("scene_variant", "query_id", "resistance_values"),
    (
        ("series_unequal", "brightest_bulb_label", [2, 5, 9, 4, 7]),
        ("series_unequal", "dimmest_bulb_label", [2, 5, 9, 4, 7]),
        ("parallel_unequal", "brightest_bulb_label", [2, 5, 9, 4, 7]),
        ("parallel_unequal", "dimmest_bulb_label", [2, 5, 9, 4, 7]),
        ("mixed_branch", "brightest_bulb_label", [3, 8, 2, 5, 10]),
        ("mixed_branch", "dimmest_bulb_label", [3, 8, 2, 5, 10]),
    ),
)
def test_physics_bulb_brightness_answer_matches_computed_power(
    scene_variant: str,
    query_id: str,
    resistance_values: list[int],
) -> None:
    out = PhysicsBulbCircuitBrightnessExtremumLabelTask().generate(
        27001,
        params={
            "scene_variant": scene_variant,
            "query_id": query_id,
            "resistance_values": resistance_values,
            "target_label": "B2",
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    powers = {str(spec["label"]): float(spec["relative_power"]) for spec in execution["bulb_specs"]}
    expected = max(powers, key=powers.get) if query_id == "brightest_bulb_label" else min(powers, key=powers.get)

    assert out.scene_id == "bulb_circuit"
    assert out.query_id == query_id
    assert out.answer_gt.type == "string"
    assert out.answer_gt.value == expected
    assert out.answer_gt.value == "B2"
    assert out.annotation_gt.value == trace["render_map"]["bulb_bboxes"]["B2"]
    _assert_bbox_in_bounds(out)
    assert trace["render_map"]["correct_label"] == out.answer_gt.value
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["witness_symbolic"]["entity_id"] == "B2"


def test_physics_bulb_brightness_task_is_deterministic() -> None:
    params = {
        "scene_variant": "mixed_branch",
        "query_id": "dimmest_bulb_label",
        "resistance_values": [3, 8, 2, 5, 10],
        "target_label": "B1",
        "accent_color_name": "cyan",
    }
    task = PhysicsBulbCircuitBrightnessExtremumLabelTask()
    out_a = task.generate(27021, params=params, max_attempts=20)
    out_b = task.generate(27021, params=params, max_attempts=20)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_physics_bulb_brightness_rejects_ambiguous_resistances() -> None:
    with pytest.raises(ValueError, match="unique brightness order"):
        PhysicsBulbCircuitBrightnessExtremumLabelTask().generate(
            27031,
            params={
                "scene_variant": "series_unequal",
                "query_id": "brightest_bulb_label",
                "resistance_values": [4, 4, 8, 10, 12],
            },
            max_attempts=20,
        )


def test_physics_bulb_brightness_defaults_and_prompt_bundle() -> None:
    cfg = get_scene_defaults("physics", "bulb_circuit")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__bulb_circuit__brightness_extremum_label",
    )
    bundle = json.loads(
        Path("src/trace_tasks/resources/prompts/physics/bulb_circuit/physics_bulb_circuit_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert set(generation["scene_variant_weights"]) == {"series_unequal", "parallel_unequal", "mixed_branch"}
    assert list(generation["resistance_options"]) == [2, 3, 4, 5, 6, 8, 10, 12]
    assert int(rendering["bulb_label_font_size_px"]) == 20
    assert str(prompt["bundle_id"]) == "physics_bulb_circuit_v1"
    assert str(prompt["task_key"]) == "brightness_extremum_query"
    assert "brightest_bulb_label" in bundle["templates"]["query"]
    assert "dimmest_bulb_label" in bundle["templates"]["query"]
    assert len(bundle["templates"]["query"]["brightest_bulb_label"]) == 5
    assert len(bundle["templates"]["query"]["dimmest_bulb_label"]) == 5
