"""Tests for the synthetic 3D object relation option task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.object_relation_label import (
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
    RELATION_CANDIDATE_SHAPE_TYPES,
    RELATION_CONTEXT_SHAPE_TYPES,
    RELATION_CANDIDATE_DIMENSION_SCALE,
    SUPPORTED_QUERY_IDS,
    UNDER_RELATION_CANDIDATE_DIMENSION_SCALE,
)
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


TASK_ID = "task_three_d__object_scene__object_relation_label"


def test_object_relation_candidate_pool_is_query_invariant() -> None:
    assert tuple(RELATION_CANDIDATE_SHAPE_TYPES) == tuple(NAMED_SMALL_OBJECT_SHAPE_TYPES)


def test_object_relation_context_pool_excludes_piano() -> None:
    assert "piano" not in set(RELATION_CONTEXT_SHAPE_TYPES)


@pytest.mark.parametrize("query_id", SUPPORTED_QUERY_IDS)
def test_object_relation_answer_and_annotation(query_id: str) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260521,
        params={
            "query_id": query_id,
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "context_object_count": 2,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=160,
    )

    trace = output.trace_payload["execution_trace"]
    point_specs = list(trace["point_specs"])
    context_specs = list(trace["context_object_specs"])
    relation_status = dict(trace["relation_status_by_label"])
    expected_labels = [str(label) for label, is_match in relation_status.items() if bool(is_match)]
    reference_id = str(trace["reference_object_id"])
    reference_spec = next(spec for spec in context_specs if str(spec["object_id"]) == reference_id)
    assert output.scene_id == "object_scene"
    assert output.query_id == query_id
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == expected_labels[0]
    assert len(expected_labels) == 1
    assert len(point_specs) == 6
    assert len(context_specs) == 2
    assert all(spec["is_answer_candidate"] for spec in point_specs)
    assert not any(spec["is_answer_candidate"] for spec in context_specs)
    assert not any(str(spec["shape_type"]) == "piano" for spec in context_specs)
    assert {str(spec["shape_type"]) for spec in point_specs} <= set(NAMED_SMALL_OBJECT_SHAPE_TYPES)
    assert not any(str(spec["shape_type"]) == "drum" for spec in point_specs)
    assert all(float(spec["dimension_scale"]) <= float(RELATION_CANDIDATE_DIMENSION_SCALE) * 1.17 for spec in point_specs)
    assert reference_spec["nameable_for_prompt"]
    answer_spec = next(spec for spec in point_specs if str(spec["point_label"]) == expected_labels[0])
    expected_bbox = output.trace_payload["render_map"]["object_bboxes_px"][str(answer_spec["object_id"])]
    expected_annotation_bbox = output.trace_payload["render_map"]["annotation_bboxes_px"][0]
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_annotation_bbox
    assert output.trace_payload["render_map"]["point_bboxes_px"][expected_labels[0]] == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        point_specs,
        answer_label=expected_labels[0],
        answer_object_id=str(answer_spec["object_id"]),
        expected_image_size=(1180, 1068),
    )
    assert any(entity["entity_id"] == "open_floor_stage" for entity in output.trace_payload["scene_ir"]["entities"])
    assert not any(entity["entity_id"] == "room_shell" for entity in output.trace_payload["scene_ir"]["entities"])

    if query_id == "on_top_of_prop":
        assert float(answer_spec["base_xyz"][2]) > float(reference_spec["base_xyz"][2]) + 0.85 * float(reference_spec["dimensions_xyz"][2])
    elif query_id == "under_prop":
        assert str(reference_spec["shape_type"]) in {"arch", "table"}
        assert str(answer_spec.get("visibility_role")) == "under_answer_opening"
        assert "render_order_bias" not in answer_spec
        assert float(answer_spec["dimension_scale"]) <= float(UNDER_RELATION_CANDIDATE_DIMENSION_SCALE) * 1.17
        assert abs(float(answer_spec["world_xyz"][0]) - float(reference_spec["world_xyz"][0])) < float(reference_spec["dimensions_xyz"][0]) * 0.4
        assert abs(float(answer_spec["world_xyz"][1]) - float(reference_spec["world_xyz"][1])) < float(reference_spec["dimensions_xyz"][1]) * 0.4
    else:
        assert str(reference_spec["shape_type"]) == "open_box"
        assert float(reference_spec["dimensions_xyz"][2]) < 1.00
        assert float(answer_spec["base_xyz"][2]) > float(reference_spec["base_xyz"][2])
        assert answer_spec["contained_by_object_id"] == reference_id
        assert float(answer_spec["render_order_bias"]) < 0.0
        assert abs(float(answer_spec["world_xyz"][0]) - float(reference_spec["world_xyz"][0])) < float(reference_spec["dimensions_xyz"][0]) * 0.3
        assert abs(float(answer_spec["world_xyz"][1]) - float(reference_spec["world_xyz"][1])) < float(reference_spec["dimensions_xyz"][1]) * 0.3


def test_object_relation_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
