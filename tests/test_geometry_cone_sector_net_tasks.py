"""Contracts for cone sector-net geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.cone_net.base_radius_from_sector_angle import (
    SCENE_ID,
    GeometryConeNetBaseRadiusFromSectorAngleTask,
)
from trace_tasks.tasks.geometry.cone_net.height_from_sector_angle import GeometryConeNetHeightFromSectorAngleTask

TASK_CLASSES = (
    GeometryConeNetBaseRadiusFromSectorAngleTask,
    GeometryConeNetHeightFromSectorAngleTask,
)

QUERY_IDS_BY_TASK = {
    GeometryConeNetBaseRadiusFromSectorAngleTask: ("single",),
    GeometryConeNetHeightFromSectorAngleTask: ("single",),
}

INTERNAL_QUERY_ID_BY_TASK = {
    GeometryConeNetBaseRadiusFromSectorAngleTask: "base_radius_from_sector_angle",
    GeometryConeNetHeightFromSectorAngleTask: "height_from_sector_angle",
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cone_sector_net_task_emits_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(59001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) in (
        {"S", "P", "Q", "C", "R"},
        {"S", "P", "Q", "C", "A"},
    )
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["execution_trace"]["internal_query_id"] == INTERNAL_QUERY_ID_BY_TASK[task_cls]
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["slant_height"] > 0
    assert 0 < trace["execution_trace"]["theta_degrees"] < 360

    slant_height = float(trace["execution_trace"]["slant_height"])
    theta_degrees = float(trace["execution_trace"]["theta_degrees"])
    base_radius = theta_degrees * slant_height / 360.0
    cone_height = (slant_height**2 - base_radius**2) ** 0.5
    assert trace["execution_trace"]["base_radius"] == pytest.approx(
        round(base_radius, 1)
    )
    assert trace["execution_trace"]["cone_height"] == pytest.approx(
        round(cone_height, 1)
    )
    if task_cls is GeometryConeNetBaseRadiusFromSectorAngleTask:
        assert out.answer_gt.value == pytest.approx(round(base_radius, 1))
    else:
        assert out.answer_gt.value == pytest.approx(round(cone_height, 1))


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cone_sector_net_task_is_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(59011, params=params, max_attempts=20)
    out_b = task.generate(59011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cone_sector_net_task_supports_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            59021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "number"
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cone_sector_net_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            59041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


def test_cone_sector_net_annotation_uses_labeled_construction_points_not_labels() -> None:
    expected_keys_by_task = {
        GeometryConeNetBaseRadiusFromSectorAngleTask: {"S", "P", "Q", "C", "R"},
        GeometryConeNetHeightFromSectorAngleTask: {"S", "P", "Q", "C", "A"},
    }
    for index, task_cls in enumerate(
        (
            GeometryConeNetBaseRadiusFromSectorAngleTask,
            GeometryConeNetHeightFromSectorAngleTask,
        )
    ):
        task = task_cls()
        out = task.generate(
            59061 + index,
            params={"query_id": "single"},
            max_attempts=20,
        )
        expected_keys = expected_keys_by_task[task_cls]
        assert out.annotation_gt.type == "point_map"
        assert set(out.annotation_gt.value) == expected_keys
        assert set(out.trace_payload["execution_trace"]["annotation_roles"]) == expected_keys
        assert all(
            "label" not in str(role)
            for role in out.trace_payload["execution_trace"]["annotation_roles"]
        )
        assert "label_bboxes" in out.trace_payload["render_map"]
        assert "point_label_bboxes" in out.trace_payload["render_map"]


def test_cone_sector_net_tasks_reject_unknown_query_id() -> None:
    task = GeometryConeNetBaseRadiusFromSectorAngleTask()
    with pytest.raises(ValueError):
        task.generate(59031, params={"query_id": "not_a_query"}, max_attempts=20)
