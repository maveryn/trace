"""Contracts for Pythagorean square-dissection geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.pythagorean_dissection.pythagorean_square_area_value import (
    SCENE_ID,
    GeometryPythagoreanSquareAreaValueTask,
)

TASK_CLASSES = (GeometryPythagoreanSquareAreaValueTask,)

QUERY_IDS_BY_TASK = {
    GeometryPythagoreanSquareAreaValueTask: (
        "single",
    ),
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_pythagorean_square_dissection_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(58001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"E", "F", "G", "H"}
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert "square EFGH" in out.prompt
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_map"] == out.annotation_gt.value
    displayed_labels = trace["render_map"]["displayed_segment_labels"]

    vertical_leg = trace["execution_trace"]["leg_a"]
    horizontal_leg = trace["execution_trace"]["leg_b"]
    outer_side = trace["execution_trace"]["outer_square_side"]
    assert displayed_labels["outer"].endswith(f"={outer_side}")
    assert displayed_labels["leg_a"].endswith(f"={vertical_leg}")
    assert displayed_labels["leg_b"].endswith(f"={horizontal_leg}")
    assert not any("leg=" in label for label in displayed_labels.values())
    corner_area = trace["execution_trace"]["corner_triangle_area_each"]
    central_area = outer_side**2 - (4.0 * corner_area)
    assert out.query_id == "single"
    assert outer_side == vertical_leg + horizontal_leg
    assert out.answer_gt.value == int(vertical_leg**2 + horizontal_leg**2)
    assert out.answer_gt.value == pytest.approx(float(central_area))
    assert trace["execution_trace"]["vertical_square_area"] == vertical_leg**2
    assert trace["execution_trace"]["horizontal_square_area"] == horizontal_leg**2
    assert trace["execution_trace"]["central_square_area"] == int(out.answer_gt.value)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_pythagorean_square_dissection_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(58011, params=params, max_attempts=20)
    out_b = task.generate(58011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_pythagorean_square_dissection_tasks_support_every_explicit_query(
    task_cls,
) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            58021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_pythagorean_square_dissection_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            58041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        for point in out.annotation_gt.value.values():
            assert len(point) == 2
            assert 0.0 <= float(point[0]) <= float(width)
            assert 0.0 <= float(point[1]) <= float(height)


def test_pythagorean_square_dissection_tasks_reject_unknown_query_id() -> None:
    task = GeometryPythagoreanSquareAreaValueTask()
    with pytest.raises(ValueError):
        task.generate(58031, params={"query_id": "not_a_query"}, max_attempts=20)


def test_pythagorean_square_dissection_target_orientation_varies() -> None:
    task = GeometryPythagoreanSquareAreaValueTask()
    orientations = set()
    annotation_centers = set()
    for index in range(16):
        out = task.generate(
            58101 + index,
            params={},
            max_attempts=20,
        )
        trace = out.trace_payload
        orientations.add(trace["render_spec"]["orientation"])
        e_x, e_y = trace["render_map"]["annotation_points"]["E"]
        annotation_centers.add((round(e_x / 20.0), round(e_y / 20.0)))

    assert len(orientations) == 4
    assert len(annotation_centers) >= 3
