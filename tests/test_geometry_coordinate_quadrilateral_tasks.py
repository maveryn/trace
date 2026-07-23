"""Contract tests for coordinate-quadrilateral geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks import create_task
from trace_tasks.tasks.geometry.coordinate_plane.quadrilateral_completion_label import (
    COMPLETION_QUERY_IDS,
    COMPLETION_SCENE_ID,
    COMPLETION_TASK_ID,
    _is_ambiguous_for_prompt,
)
from trace_tasks.tasks.geometry.coordinate_panels.quadrilateral_shape_match_label import (
    PANEL_SCENE_ID,
    SHAPE_MATCH_QUERY_IDS,
    SHAPE_MATCH_TASK_ID,
)
from trace_tasks.tasks.geometry.coordinate_panels.point_set_transform_match_label import (
    POINT_SET_TRANSFORM_QUERY_IDS,
    POINT_SET_TRANSFORM_TASK_ID,
)
from trace_tasks.tasks.geometry.coordinate_panels.segment_relation_match_label import (
    SEGMENT_RELATION_QUERY_IDS,
    SEGMENT_RELATION_TASK_ID as PANEL_SEGMENT_RELATION_TASK_ID,
)
from trace_tasks.tasks.geometry.coordinate_panels.shared.construction import is_ambiguous_for_prompt as panel_is_ambiguous_for_prompt


@pytest.mark.parametrize("query_id", COMPLETION_QUERY_IDS)
def test_quadrilateral_completion_has_unique_candidate_answer(query_id: str) -> None:
    task = create_task(COMPLETION_TASK_ID)
    out = task.generate(77401, params={"query_id": query_id, "winner_label": "C"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    candidates = execution["candidate_points_by_label"]

    assert out.scene_id == COMPLETION_SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == candidates["C"]["point_px"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert len(execution["known_points_graph"]) == 3
    assert 4 <= len(candidates) <= 6
    assert "C" in candidates

    target_kind = execution["target_kind"]
    matching_labels = [
        label
        for label, payload in candidates.items()
        if _is_ambiguous_for_prompt(payload["classification_with_known_points"], target_kind)
    ]
    assert matching_labels == ["C"]
    assert candidates["C"]["classification_with_known_points"] == target_kind


@pytest.mark.parametrize("query_id", SHAPE_MATCH_QUERY_IDS)
def test_quadrilateral_panel_match_has_unique_panel_answer(query_id: str) -> None:
    task = create_task(SHAPE_MATCH_TASK_ID)
    out = task.generate(77411, params={"query_id": query_id, "winner_label": "D"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    panels = execution["panels_by_label"]

    assert out.scene_id == PANEL_SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "point_set"
    assert out.annotation_gt.value == panels["D"]["points_px"]
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert len(panels) == 6
    assert "D" in panels
    assert all(len(panel["points_graph"]) == 4 for panel in panels.values())
    assert all(len(panel["points_px"]) == 4 for panel in panels.values())

    target_kind = execution["target_kind"]
    matching_labels = [
        label
        for label, payload in panels.items()
        if panel_is_ambiguous_for_prompt(payload["classified_kind"], target_kind)
    ]
    assert matching_labels == ["D"]
    assert panels["D"]["classified_kind"] == target_kind


@pytest.mark.parametrize("query_id", SEGMENT_RELATION_QUERY_IDS)
def test_segment_relation_panel_match_has_unique_panel_answer(query_id: str) -> None:
    task = create_task(PANEL_SEGMENT_RELATION_TASK_ID)
    out = task.generate(77431, params={"query_id": query_id, "winner_label": "E"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    panels = execution["panels_by_label"]

    assert out.scene_id == PANEL_SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "E"
    assert out.annotation_gt.type == "segment_set"
    assert out.annotation_gt.value == panels["E"]["segments_px"]
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_segment_set"] == out.annotation_gt.value
    assert len(panels) == 6
    assert all(len(panel["segments_graph"]) == 2 for panel in panels.values())
    assert all(len(panel["segments_px"]) == 2 for panel in panels.values())

    relation_kind = execution["relation_kind"]
    matching_labels = [
        label
        for label, payload in panels.items()
        if bool(payload["relation_flags"][relation_kind])
    ]
    assert matching_labels == ["E"]


@pytest.mark.parametrize("query_id", POINT_SET_TRANSFORM_QUERY_IDS)
def test_point_set_transform_panel_match_has_unique_panel_answer(query_id: str) -> None:
    task = create_task(POINT_SET_TRANSFORM_TASK_ID)
    out = task.generate(77441, params={"query_id": query_id, "winner_label": "F"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    panels = execution["panels_by_label"]

    assert out.scene_id == PANEL_SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "F"
    assert out.annotation_gt.type == "point_set"
    assert out.annotation_gt.value == [
        *panels["F"]["source_points_px"],
        *panels["F"]["candidate_points_px"],
    ]
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert len(panels) == 6
    assert all(len(panel["source_points_graph"]) == 3 for panel in panels.values())
    assert all(len(panel["candidate_points_graph"]) == 3 for panel in panels.values())

    transform_kind = execution["transform_kind"]
    matching_labels = [
        label
        for label, payload in panels.items()
        if bool(payload["transform_flags"][transform_kind])
    ]
    assert matching_labels == ["F"]


@pytest.mark.parametrize(
    ("task_id", "params"),
    (
        (COMPLETION_TASK_ID, {"query_id": "square_completion_label", "winner_label": "B"}),
        (SHAPE_MATCH_TASK_ID, {"query_id": "rectangle_shape_match_label", "winner_label": "E"}),
        (PANEL_SEGMENT_RELATION_TASK_ID, {"query_id": "parallel_segments_match_label", "winner_label": "E"}),
        (POINT_SET_TRANSFORM_TASK_ID, {"query_id": "translation_match_label", "winner_label": "F"}),
    ),
)
def test_quadrilateral_coordinate_tasks_are_deterministic(task_id: str, params: dict[str, str]) -> None:
    task = create_task(task_id)
    out_a = task.generate(77421, params=dict(params), max_attempts=50)
    out_b = task.generate(77421, params=dict(params), max_attempts=50)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
