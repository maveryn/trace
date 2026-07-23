"""Contract tests for the lens-optics image-property task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.physics.lens_optics.image_property_choice import (
    CASE_TO_PROPERTY,
    OBJECT_POSITION_CASES,
    OPTION_LETTERS,
    PhysicsLensOpticsImagePropertyChoiceTask,
    TASK_ID,
)
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def _assert_bbox_map_in_bounds(out) -> None:
    width, height = out.image.size
    assert out.annotation_gt.type == "bbox_map"
    for bbox in out.annotation_gt.value.values():
        assert 0 <= bbox[0] < bbox[2] <= width
        assert 0 <= bbox[1] < bbox[3] <= height


def _bbox_overlaps(a, b) -> bool:
    return max(float(a[0]), float(b[0])) < min(float(a[2]), float(b[2])) and max(float(a[1]), float(b[1])) < min(float(a[3]), float(b[3]))


def test_lens_optics_task_is_registered_by_default() -> None:
    assert "task_physics__lens_optics__image_property_choice" in set(list_default_task_ids())


def test_lens_optics_all_object_positions_map_to_expected_property() -> None:
    task = PhysicsLensOpticsImagePropertyChoiceTask()
    for index, position_case in enumerate(OBJECT_POSITION_CASES):
        out = task.generate(
            96001 + index,
            params={
                "object_position_case": position_case,
                "correct_option_letter": "B",
                "post_image_noise": {"enabled": False},
            },
            max_attempts=20,
        )
        execution = out.trace_payload["execution_trace"]
        render_map = out.trace_payload["render_map"]

        assert out.scene_id == "lens_optics"
        assert out.query_id == "single"
        assert out.answer_gt.type == "option_letter"
        assert out.answer_gt.value == "B"
        assert out.trace_payload["query_spec"]["params"]["answer_support"] == list(OPTION_LETTERS)
        assert execution["lens_type"] == "converging"
        assert execution["internal_query_id"] == "converging_lens_image_property_choice"
        assert execution["object_position_case"] == position_case
        assert execution["option_map"][out.answer_gt.value] == CASE_TO_PROPERTY[position_case]
        assert render_map["image_property"] == CASE_TO_PROPERTY[position_case]
        assert render_map["correct_option_letter"] == out.answer_gt.value
        assert set(render_map["option_letter_bboxes_px"]) == set(OPTION_LETTERS)
        assert set(render_map["option_text_bboxes_px"]) == set(OPTION_LETTERS)
        for letter in OPTION_LETTERS:
            assert not _bbox_overlaps(render_map["option_letter_bboxes_px"][letter], render_map["option_text_bboxes_px"][letter])

        assert set(out.annotation_gt.value) == {"lens", "object_arrow", "focal_marks"}
        _assert_bbox_map_in_bounds(out)
        assert out.trace_payload["projected_annotation"]["bbox_map"] == out.annotation_gt.value
        assert out.trace_payload["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
        for option_bbox in render_map["option_bboxes_px"].values():
            assert option_bbox not in out.annotation_gt.value.values()
        assert "image_arrow_bbox_px" not in render_map
        assert out.prompt_variants["answer_only"]
        assert out.prompt_variants["answer_and_annotation"]


def test_lens_optics_generation_is_deterministic() -> None:
    task = PhysicsLensOpticsImagePropertyChoiceTask()
    params = {
        "object_position_case": "between_f_2f",
        "correct_option_letter": "D",
        "scene_variant": "paper_grid",
        "post_image_noise": {"enabled": False},
    }

    first = task.generate(96099, params=params, max_attempts=20)
    second = task.generate(96099, params=params, max_attempts=20)

    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_lens_optics_balanced_sampling_exposes_cases_and_letters() -> None:
    task = PhysicsLensOpticsImagePropertyChoiceTask()
    seen_cases = set()
    seen_letters = set()
    for seed in range(96120, 96144):
        out = task.generate(seed, params={"post_image_noise": {"enabled": False}}, max_attempts=20)
        execution = out.trace_payload["execution_trace"]
        seen_cases.add(str(execution["object_position_case"]))
        seen_letters.add(str(out.answer_gt.value))

    assert seen_cases == set(OBJECT_POSITION_CASES)
    assert seen_letters == set(OPTION_LETTERS)


def test_lens_optics_defaults_expose_prompt_and_rendering_contract() -> None:
    optics = get_scene_defaults("physics", "lens_optics")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        optics,
        task_id=TASK_ID,
    )

    assert set(generation["scene_variant_weights"]) == {"clean_axis", "paper_grid", "lab_card"}
    assert set(generation["object_position_case_weights"]) == set(OBJECT_POSITION_CASES)
    assert set(generation["correct_option_letter_weights"]) == set(OPTION_LETTERS)
    assert int(rendering["canvas_width"]) == 1120
    assert int(rendering["canvas_height"]) == 720
    assert str(prompt["task_key"]) == "lens_image_property_query"
    assert str(prompt["bundle_id"]) == "physics_lens_optics_v1"
