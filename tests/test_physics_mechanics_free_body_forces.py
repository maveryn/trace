"""Contract tests for physics mechanics free-body-force direction task."""

from __future__ import annotations

import json
from collections import Counter

import pytest
import yaml

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.physics.free_body_forces.net_force_direction_choice import (
    DIRECTION_NAMES,
    OPTION_LETTERS,
    PhysicsFreeBodyForcesNetForceDirectionChoiceTask,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def _assert_bbox_set_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox_set"
    for bbox in out.annotation_gt.value:
        assert 0 <= bbox[0] < bbox[2] <= width
        assert 0 <= bbox[1] < bbox[3] <= height


def test_free_body_forces_contract_and_trace() -> None:
    out = PhysicsFreeBodyForcesNetForceDirectionChoiceTask().generate(
        94001,
        params={
            "scene_variant": "gridded_table",
            "net_force_direction": "northeast",
            "correct_option_letter": "D",
            "force_specs": [
                {"direction": "east", "magnitude_n": 9},
                {"direction": "west", "magnitude_n": 3},
                {"direction": "north", "magnitude_n": 12},
                {"direction": "south", "magnitude_n": 6},
            ],
            "post_image_noise": {"enabled": False},
        },
        max_attempts=20,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "free_body_forces"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox_set"
    assert len(out.annotation_gt.value) == 4
    _assert_bbox_set_in_bounds(out)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox_set"] == out.annotation_gt.value
    assert execution["resultant_vector"] == [6, 6]
    assert execution["net_force_direction"] == "northeast"
    assert execution["option_directions"]["D"] == "northeast"
    assert set(execution["option_directions"]) == set(OPTION_LETTERS)
    assert set(execution["option_directions"].values()) == set(DIRECTION_NAMES)
    assert trace["render_map"]["option_bboxes_px"]["D"] not in out.annotation_gt.value
    assert out.prompt_variants["answer_only"]
    assert out.prompt_variants["answer_and_annotation"]


def test_free_body_forces_explicit_specs_must_match_direction() -> None:
    with pytest.raises(ValueError, match="force_specs do not match"):
        PhysicsFreeBodyForcesNetForceDirectionChoiceTask().generate(
            94011,
            params={
                "net_force_direction": "east",
                "force_specs": [
                    {"direction": "north", "magnitude_n": 5},
                    {"direction": "south", "magnitude_n": 2},
                ],
            },
            max_attempts=20,
        )


def test_free_body_forces_is_deterministic() -> None:
    params = {
        "scene_variant": "lab_card",
        "net_force_direction": "southwest",
        "correct_option_letter": "A",
        "accent_color_name": "cyan",
        "include_extra_canceling_pair": True,
        "post_image_noise": {"enabled": False},
    }
    task = PhysicsFreeBodyForcesNetForceDirectionChoiceTask()
    out_a = task.generate(94021, params=params, max_attempts=20)
    out_b = task.generate(94021, params=params, max_attempts=20)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_free_body_forces_sampling_covers_letters_and_directions() -> None:
    letters: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    task = PhysicsFreeBodyForcesNetForceDirectionChoiceTask()
    for index in range(128):
        out = task.generate(94100 + index, params={}, max_attempts=20)
        letters[str(out.answer_gt.value)] += 1
        directions[str(out.trace_payload["execution_trace"]["net_force_direction"])] += 1

    assert set(letters) == set(OPTION_LETTERS)
    assert set(directions) == set(DIRECTION_NAMES)


def test_free_body_forces_defaults_and_prompt_bundle() -> None:
    free_body_forces = get_scene_defaults("physics", "free_body_forces")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        free_body_forces,
        task_id="task_physics__free_body_forces__net_force_direction_choice",
    )

    assert "query_id_weights" not in generation
    assert set(generation["scene_variant_weights"]) == {"clean_table", "gridded_table", "lab_card"}
    assert set(generation["net_force_direction_weights"]) == set(DIRECTION_NAMES)
    assert set(generation["correct_option_letter_weights"]) == set(OPTION_LETTERS)
    assert int(rendering["canvas_width"]) == 1180
    assert str(prompt["bundle_id"]) == "physics_free_body_forces_v1"
    assert str(prompt["task_key"]) == "net_force_direction_choice_query"

    with open("src/trace_tasks/resources/configs/domains/physics/free_body_forces.yaml", "r", encoding="utf-8") as handle:
        yaml.safe_load(handle)
    with open("src/trace_tasks/resources/prompts/physics/free_body_forces/physics_free_body_forces_v1.json", "r", encoding="utf-8") as handle:
        prompt_bundle = json.load(handle)
    assert "single" in prompt_bundle["templates"]["query"]
