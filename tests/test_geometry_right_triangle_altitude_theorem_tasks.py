from __future__ import annotations

import math

import pytest

import trace_tasks.tasks  # noqa: F401
from trace_tasks.tasks.geometry.triangle_relations.shared.construction import (
    altitude_from_two_projections_cases,
    leg_from_projection_cases,
    projection_from_altitude_cases,
    projection_from_leg_cases,
)
from trace_tasks.tasks.registry import create_task

TASK_QUERIES = {
    "task_geometry__triangle_relations__altitude_to_hypotenuse_value": (
        "altitude_from_split_hypotenuse",
        "missing_projection_from_altitude",
    ),
    "task_geometry__triangle_relations__leg_projection_length_value": (
        "leg_from_hypotenuse_projection",
        "projection_from_leg_and_hypotenuse",
    ),
}


def _generate(task_id: str, query_id: str, seed: int = 20260605):
    task = create_task(task_id)
    return task.generate(seed, params={"query_id": query_id}, max_attempts=3)


def test_triangle_relations_altitude_tasks_are_registered() -> None:
    for task_id in TASK_QUERIES:
        assert create_task(task_id).task_id == task_id


def test_triangle_relations_altitude_queries_emit_keyed_point_annotation() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for index, query_id in enumerate(query_ids):
            output = _generate(task_id, query_id, seed=20260605 + index)
            assert output.scene_id == "triangle_relations"
            assert output.query_id == query_id
            assert output.answer_gt.type == "integer"
            assert isinstance(output.answer_gt.value, int)
            assert output.annotation_gt.type == "point_map"
            assert isinstance(output.annotation_gt.value, dict)
            assert output.annotation_gt.value
            width, height = output.image.size
            for point in output.annotation_gt.value.values():
                assert isinstance(point, list)
                assert len(point) == 2
                assert 0.0 <= float(point[0]) <= float(width)
                assert 0.0 <= float(point[1]) <= float(height)
            trace = output.trace_payload
            assert trace["execution_trace"]["query_id"] == query_id
            assert trace["execution_trace"]["answer"] == output.answer_gt.value
            assert trace["projected_annotation"]["type"] == "point_map"
            assert (
                trace["projected_annotation"]["point_map"] == output.annotation_gt.value
            )
            assert (
                trace["projected_annotation"]["pixel_point_map"]
                == output.annotation_gt.value
            )
            assert "task_variant" not in trace["query_spec"]["params"]
            assert "query_variant" not in trace["query_spec"]["params"]


def test_triangle_relations_altitude_measurements_match_trace_values() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for query_id in query_ids:
            output = _generate(task_id, query_id, seed=20260617)
            trace = output.trace_payload["execution_trace"]
            left_projection = int(trace["left_projection"])
            right_projection = int(trace["right_projection"])
            altitude = int(trace["altitude"])
            hypotenuse = int(trace["hypotenuse"])
            assert left_projection + right_projection == hypotenuse
            assert altitude * altitude == left_projection * right_projection
            if trace["left_leg"] is not None:
                assert int(trace["left_leg"]) ** 2 == hypotenuse * left_projection
            if trace["right_leg"] is not None:
                assert int(trace["right_leg"]) ** 2 == hypotenuse * right_projection
            target_role = str(trace["target_role"])
            if target_role == "altitude":
                assert output.answer_gt.value == altitude
            elif target_role == "left_projection":
                assert output.answer_gt.value == left_projection
            elif target_role == "right_projection":
                assert output.answer_gt.value == right_projection
            elif target_role == "left_leg":
                assert output.answer_gt.value == int(trace["left_leg"])
            elif target_role == "right_leg":
                assert output.answer_gt.value == int(trace["right_leg"])
            else:
                raise AssertionError(f"unexpected target_role={target_role}")


def _angle_degrees(
    points: dict[str, tuple[float, float]] | dict[str, list[float]],
    *,
    vertex: str,
    arm_a: str,
    arm_b: str,
) -> float:
    v = points[vertex]
    a = points[arm_a]
    b = points[arm_b]
    va = (float(a[0]) - float(v[0]), float(a[1]) - float(v[1]))
    vb = (float(b[0]) - float(v[0]), float(b[1]) - float(v[1]))
    dot = va[0] * vb[0] + va[1] * vb[1]
    denom = math.hypot(*va) * math.hypot(*vb)
    assert denom > 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / denom))))


def test_triangle_relations_altitude_right_angle_markers_match_geometry() -> None:
    for cases in (
        altitude_from_two_projections_cases(),
        projection_from_altitude_cases(),
        leg_from_projection_cases(),
        projection_from_leg_cases(),
    ):
        for case in cases[:80]:
            for marker in case.right_angles:
                angle = _angle_degrees(
                    case.vertices,
                    vertex=marker.vertex,
                    arm_a=marker.arm_a,
                    arm_b=marker.arm_b,
                )
                assert angle == pytest.approx(90.0, abs=1e-6)


def test_triangle_relations_altitude_rendered_right_angles_survive_rotation() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for index, query_id in enumerate(query_ids):
            task = create_task(task_id)
            output = task.generate(
                91031 + index,
                params={"query_id": query_id, "scene_rotation_degrees": 37},
                max_attempts=3,
            )
            vertices = output.trace_payload["render_map"]["vertices"]
            assert _angle_degrees(
                vertices, vertex="A", arm_a="B", arm_b="C"
            ) == pytest.approx(90.0, abs=1e-3)
            assert _angle_degrees(
                vertices, vertex="D", arm_a="A", arm_b="C"
            ) == pytest.approx(90.0, abs=1e-3)


def test_triangle_relations_altitude_generation_is_deterministic() -> None:
    task_id = "task_geometry__triangle_relations__leg_projection_length_value"
    query_id = "projection_from_leg_and_hypotenuse"
    first = _generate(task_id, query_id, seed=817)
    second = _generate(task_id, query_id, seed=817)
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert (
        first.trace_payload["execution_trace"]
        == second.trace_payload["execution_trace"]
    )
