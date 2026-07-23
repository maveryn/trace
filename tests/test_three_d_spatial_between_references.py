"""Tests for the synthetic 3D between-references task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.between_references_label import TASK_ID
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


def test_between_references_answer_and_annotation() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260521,
        params={
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "context_object_count": 2,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=220,
    )

    trace = output.trace_payload["execution_trace"]
    point_specs = list(trace["point_specs"])
    context_specs = list(trace["context_object_specs"])
    between_status = dict(trace["candidate_between_status_by_label"])
    expected_labels = [str(label) for label, is_between in between_status.items() if bool(is_between)]
    reference_ids = {str(item) for item in trace["reference_object_ids"]}
    reference_names = [str(item) for item in trace["reference_object_names"]]
    assert output.scene_id == "object_scene"
    assert output.query_id == "single"
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == expected_labels[0]
    assert len(expected_labels) == 1
    assert len(point_specs) == 6
    assert len(context_specs) == 2
    assert len(set(reference_names)) == 2
    assert all(spec["is_answer_candidate"] for spec in point_specs)
    assert all(not spec["is_answer_candidate"] for spec in context_specs)
    assert reference_ids == {str(spec["object_id"]) for spec in context_specs}
    assert not reference_ids.intersection({str(spec["object_id"]) for spec in point_specs})
    assert all(spec["nameable_for_prompt"] for spec in context_specs)
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

    metrics = dict(trace["candidate_between_metrics_by_label"][expected_labels[0]])
    assert 0.30 <= float(metrics["t"]) <= 0.70
    assert float(metrics["lateral_distance"]) <= float(
        trace["candidate_between_lateral_threshold_by_label"][expected_labels[0]]
    )
    assert trace["solver_trace"]["between_reference_labels"] == expected_labels
    assert trace["solver_trace"]["unique_between_answer"] is True


def test_between_references_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
