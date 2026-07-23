"""Contract tests for coordinate locus geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks import create_task
from trace_tasks.tasks.geometry.coordinate_plane.locus_point_label import (
    PANEL_QUERY_IDS,
    PANEL_TASK_ID,
    POINT_QUERY_IDS,
    POINT_TASK_ID,
    SCENE_ID,
)


@pytest.mark.parametrize("query_id", POINT_QUERY_IDS)
def test_locus_point_task_has_unique_region_member(query_id: str) -> None:
    task = create_task(POINT_TASK_ID)
    out = task.generate(78101, params={"query_id": query_id, "winner_label": "C"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    candidates = execution["candidate_points_by_label"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == candidates["C"]["point_px"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert len(candidates) == 6
    assert candidates["C"]["inside_region"] is True
    assert sum(1 for payload in candidates.values() if payload["inside_region"]) == 1


@pytest.mark.parametrize("query_id", PANEL_QUERY_IDS)
def test_locus_panel_task_has_unique_matching_panel(query_id: str) -> None:
    task = create_task(PANEL_TASK_ID)
    out = task.generate(78111, params={"query_id": query_id, "winner_label": "D"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    panels = execution["panels_by_label"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "bbox"
    assert out.annotation_gt.value == panels["D"]["panel_bbox"]
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert len(panels) == 6
    assert panels["D"]["is_answer"] is True
    assert sum(1 for payload in panels.values() if payload["is_answer"]) == 1
    assert execution["condition_text"] == panels["D"]["region"]["condition_text"]


@pytest.mark.parametrize(
    ("task_id", "query_id"),
    (
        (POINT_TASK_ID, "annulus_region_point"),
        (PANEL_TASK_ID, "two_inequality_panel_match"),
    ),
)
def test_coordinate_locus_tasks_are_deterministic(task_id: str, query_id: str) -> None:
    task = create_task(task_id)
    params = {"query_id": query_id, "winner_label": "B"}
    out_a = task.generate(78121, params=dict(params), max_attempts=50)
    out_b = task.generate(78121, params=dict(params), max_attempts=50)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
