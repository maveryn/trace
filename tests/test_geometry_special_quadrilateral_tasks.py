from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.tasks.registry import create_task


TASK_QUERIES = {
    "task_geometry__special_quadrilateral__algebraic_angle_value": (
        "parallelogram_opposite_angle_expression",
        "parallelogram_consecutive_angle_expression",
        "rhombus_diagonal_half_angle_expression",
        "kite_opposite_angle_expression",
    ),
    "task_geometry__special_quadrilateral__segment_length_value": (
        "parallelogram_opposite_side_expression",
        "rhombus_all_sides_expression",
        "kite_adjacent_equal_side_expression",
        "parallelogram_diagonal_bisection_expression",
    ),
}


def _generate(task_id: str, query_id: str, seed: int = 20260605):
    task = create_task(task_id)
    return task.generate(seed, params={"query_id": query_id}, max_attempts=3)


def test_special_quadrilateral_tasks_are_registered() -> None:
    for task_id in TASK_QUERIES:
        assert create_task(task_id).task_id == task_id
    try:
        create_task("task_geometry__special_quadrilateral__diagonal_angle_value")
    except KeyError:
        pass
    else:
        raise AssertionError("retired diagonal-angle task is still registered")


def test_special_quadrilateral_queries_emit_point_map_annotation() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for index, query_id in enumerate(query_ids):
            output = _generate(task_id, query_id, seed=20260605 + index)
            assert output.scene_id == "special_quadrilateral"
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
            assert trace["projected_annotation"]["point_map"] == output.annotation_gt.value
            assert set(output.annotation_gt.value).issuperset({"A", "B", "C", "D"})
            assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
            assert trace["query_spec"]["prompt_variant"]["prompt_bundle_id"] == "geometry_special_quadrilateral_v1"
            assert "task_variant" not in trace["query_spec"]["params"]


def test_special_quadrilateral_expression_answers_match_trace_values() -> None:
    for task_id in (
        "task_geometry__special_quadrilateral__algebraic_angle_value",
        "task_geometry__special_quadrilateral__segment_length_value",
    ):
        for query_id in TASK_QUERIES[task_id]:
            output = _generate(task_id, query_id)
            trace = output.trace_payload["execution_trace"]
            target_expression = trace["target_expression"]
            assert int(target_expression["value"]) == output.answer_gt.value
            assert int(trace["x_value"]) > 0


def test_special_quadrilateral_generation_is_deterministic() -> None:
    task_id = "task_geometry__special_quadrilateral__segment_length_value"
    query_id = "parallelogram_diagonal_bisection_expression"
    first = _generate(task_id, query_id, seed=817)
    second = _generate(task_id, query_id, seed=817)
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_special_quadrilateral_rejects_unsupported_query() -> None:
    task = create_task("task_geometry__special_quadrilateral__algebraic_angle_value")
    try:
        task.generate(20260605, params={"query_id": "__bad__"}, max_attempts=3)
    except ValueError as exc:
        assert "query_id" in str(exc)
    else:
        raise AssertionError("unsupported query id was accepted")
