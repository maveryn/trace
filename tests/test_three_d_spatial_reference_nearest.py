"""Tests for the synthetic 3D reference-nearest task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
import pytest

from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.reference_nearest_label import (
    EXCLUDED_REFERENCE_NEAREST_CANDIDATE_SHAPE_TYPES,
    REFERENCE_SHAPE_TYPES,
    SMALL_CANDIDATE_SHAPE_TYPES,
    TASK_ID,
)
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


@pytest.mark.parametrize("query_id", ["closest_to_reference", "farthest_from_reference"])
def test_reference_nearest_answer_and_annotation(query_id: str) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260521,
        params={
            "scene_variant": "floor_grid_room",
            "query_id": str(query_id),
            "point_count": 6,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=200,
    )

    trace = output.trace_payload["execution_trace"]
    point_specs = list(trace["point_specs"])
    context_specs = list(trace["context_object_specs"])
    screen_gaps_by_label = dict(trace["candidate_reference_screen_gaps_by_label"])
    sorted_labels = sorted(screen_gaps_by_label, key=lambda label: (float(screen_gaps_by_label[label]), str(label)))
    expected_label = str(sorted_labels[0] if str(query_id) == "closest_to_reference" else sorted_labels[-1])
    reference_id = str(trace["reference_object_id"])
    assert output.scene_id == "object_scene"
    assert output.query_id == str(query_id)
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == expected_label
    assert len(point_specs) == 6
    assert len(context_specs) == 1
    assert all(spec["is_answer_candidate"] for spec in point_specs)
    assert not context_specs[0]["is_answer_candidate"]
    assert reference_id == str(context_specs[0]["object_id"])
    assert reference_id not in {str(spec["object_id"]) for spec in point_specs}
    assert context_specs[0]["nameable_for_prompt"]
    assert str(context_specs[0]["prompt_name"]) == str(trace["reference_object_name"])
    assert str(context_specs[0]["prompt_name"]) not in {str(spec["prompt_name"]) for spec in point_specs}
    assert str(context_specs[0]["shape_type"]) in set(REFERENCE_SHAPE_TYPES)
    assert all(str(spec["shape_type"]) in set(SMALL_CANDIDATE_SHAPE_TYPES) for spec in point_specs)
    assert not {str(spec["shape_type"]) for spec in point_specs} & set(EXCLUDED_REFERENCE_NEAREST_CANDIDATE_SHAPE_TYPES)
    assert trace["large_candidate_count"] == 0
    answer_spec = next(spec for spec in point_specs if str(spec["point_label"]) == expected_label)
    expected_bbox = output.trace_payload["render_map"]["object_bboxes_px"][str(answer_spec["object_id"])]
    expected_annotation_bbox = output.trace_payload["render_map"]["annotation_bboxes_px"][0]
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_annotation_bbox
    assert output.trace_payload["render_map"]["point_bboxes_px"][expected_label] == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        point_specs,
        answer_label=expected_label,
        answer_object_id=str(answer_spec["object_id"]),
        expected_image_size=(1180, 1068),
    )
    screen_order = list(trace["solver_trace"]["reference_screen_distance_order"])
    assert (screen_order[0] if str(query_id) == "closest_to_reference" else screen_order[-1]) == expected_label
    assert trace["solver_trace"]["reference_excluded_from_options"] is True
    assert trace["solver_trace"]["sort_key"] == "projected_screen_center_distance_to_reference"
    assert float(trace["solver_trace"]["reference_nearest_margin"]) >= 16.0
    assert float(trace["solver_trace"]["reference_nearest_screen_margin_px"]) >= 16.0
    for spec in point_specs:
        assert float(spec["camera_distance"]) <= float(context_specs[0]["camera_distance"]) + 0.04


def test_reference_nearest_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
