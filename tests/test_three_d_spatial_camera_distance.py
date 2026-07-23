"""Tests for the synthetic 3D object camera-distance option task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.three_d.object_scene.camera_distance_extremum_label import (
    CAMERA_YAW_BANDS_DEGREES,
    LARGE_CONTEXT_SHAPE_TYPES,
    MAX_ANSWER_CONTEXT_OVERLAP_FRACTION,
    NAMEABLE_CONTEXT_SHAPE_TYPES,
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
    OBJECT_NAME_BY_SHAPE_TYPE,
    SMALL_OBJECT_SHAPE_TYPES,
    UNRELIABLE_CAMERA_DISTANCE_ANSWER_SHAPES,
    _camera_yaw_band_for_instance,
    _max_context_overlap_fraction,
    _sample_camera,
)
from trace_tasks.tasks.registry import list_default_task_ids
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


TASK_ID = "task_three_d__object_scene__camera_distance_extremum_label"


@pytest.mark.parametrize("query_id", ["closest_to_camera", "farthest_from_camera"])
def test_camera_distance_extremum_answer_and_annotation(query_id: str) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260521,
        params={
            "query_id": query_id,
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=120,
    )

    point_specs = list(output.trace_payload["execution_trace"]["point_specs"])
    context_specs = list(output.trace_payload["execution_trace"]["context_object_specs"])
    object_specs = list(output.trace_payload["execution_trace"]["object_specs"])
    entities = list(output.trace_payload["scene_ir"]["entities"])
    sorted_specs = sorted(point_specs, key=lambda spec: (float(spec["camera_distance"]), str(spec["point_label"])))
    expected = sorted_specs[0] if query_id == "closest_to_camera" else sorted_specs[-1]
    expected_label = str(expected["point_label"])

    assert output.scene_id == "object_scene"
    assert output.query_id == query_id
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == expected_label
    assert len(point_specs) == 6
    assert len(context_specs) == 2
    assert len(object_specs) == 8
    assert {str(spec["shape_type"]) for spec in point_specs} <= set(SMALL_OBJECT_SHAPE_TYPES)
    assert {str(spec["shape_type"]) for spec in context_specs} <= set(LARGE_CONTEXT_SHAPE_TYPES)
    assert all(spec["is_answer_candidate"] for spec in point_specs)
    assert not any(spec["is_answer_candidate"] for spec in context_specs)
    assert all(spec["object_name"] == OBJECT_NAME_BY_SHAPE_TYPE[spec["shape_type"]] for spec in object_specs)
    assert all(spec["prompt_name"] == spec["object_name"] for spec in object_specs)
    assert {spec["object_name"] for spec in object_specs}.isdisjoint({"L-block", "T-block", "cross-block", "l-block", "t-block"})
    assert {
        spec["shape_type"] for spec in point_specs if bool(spec["nameable_for_prompt"])
    } <= set(NAMED_SMALL_OBJECT_SHAPE_TYPES)
    assert {
        spec["shape_type"] for spec in context_specs if bool(spec["nameable_for_prompt"])
    } <= set(NAMEABLE_CONTEXT_SHAPE_TYPES)
    assert any(entity["entity_id"] == "open_floor_stage" for entity in entities)
    stage_entity = next(entity for entity in entities if entity["entity_id"] == "open_floor_stage")
    assert stage_entity["attrs"]["full_bleed_floor"] is True
    assert stage_entity["attrs"]["grid_mode"] in {"screen_ray_floor_plane", "bounded_stage_fallback"}
    if stage_entity["attrs"]["grid_mode"] == "screen_ray_floor_plane":
        assert isinstance(stage_entity["attrs"]["grid_world_bbox"], list)
    else:
        assert stage_entity["attrs"]["grid_world_bbox"] is None
    image_width, _image_height = output.image.size
    panel_bbox = output.trace_payload["render_map"]["option_panel_bbox_px"]
    assert output.trace_payload["render_map"]["room_bbox_px"] == [0.0, 0.0, float(image_width), panel_bbox[1]]
    assert not any(entity["entity_id"] == "room_shell" for entity in entities)
    assert all(0.86 <= float(spec["dimension_scale"]) <= 1.16 for spec in point_specs)
    assert all(0.96 <= float(spec["dimension_scale"]) <= 1.20 for spec in context_specs)
    assert all(max(spec["dimensions_xyz"][:2]) < 0.85 for spec in point_specs)
    assert any(max(spec["dimensions_xyz"]) > 1.35 for spec in context_specs)
    expected_bbox = output.trace_payload["render_map"]["object_bboxes_px"][str(expected["object_id"])]
    expected_annotation_bbox = output.trace_payload["render_map"]["annotation_bboxes_px"][0]
    context_bboxes = list(output.trace_payload["render_map"]["context_object_bboxes_px"].values())
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_annotation_bbox
    assert str(expected["shape_type"]) not in set(UNRELIABLE_CAMERA_DISTANCE_ANSWER_SHAPES)
    assert _max_context_overlap_fraction(expected_bbox, context_bboxes) <= MAX_ANSWER_CONTEXT_OVERLAP_FRACTION
    assert output.trace_payload["projected_annotation"]["bbox"] == expected_annotation_bbox
    assert output.trace_payload["render_map"]["point_bboxes_px"][expected_label] == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        point_specs,
        answer_label=expected_label,
        answer_object_id=str(expected["object_id"]),
        expected_image_size=(1180, 1068),
    )
    assert set(output.trace_payload["render_map"]["context_object_bboxes_px"]) == {
        str(spec["object_id"]) for spec in context_specs
    }


def test_camera_distance_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id


def test_three_d_camera_sampler_uses_multiple_orbit_families() -> None:
    yaws = [
        float(_sample_camera(spawn_rng(20260521, f"three_d_camera_yaw_test.{index}")).yaw_degrees)
        for index in range(120)
    ]

    assert min(yaws) < -100.0
    assert max(yaws) > 100.0
    assert any(48.0 <= abs(yaw) <= 82.0 for yaw in yaws)
    assert any(20.0 <= abs(yaw) <= 42.0 for yaw in yaws)
    assert set(CAMERA_YAW_BANDS_DEGREES) == {
        (-145.0, -108.0),
        (-82.0, -48.0),
        (-42.0, -20.0),
        (20.0, 42.0),
        (48.0, 82.0),
        (108.0, 145.0),
    }
    sampled_bands = {
        _camera_yaw_band_for_instance(20260521 + index)
        for index in range(120)
    }
    assert sampled_bands == set(CAMERA_YAW_BANDS_DEGREES)
