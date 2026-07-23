"""Contracts for refactored analytical measurement geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.angle_relations.algebraic_angle_value import GeometryAngleRelationsAlgebraicAngleValueTask
from trace_tasks.tasks.geometry.angle_relations.parallel_supplement_angle import (
    GeometryAngleRelationsParallelSupplementAngleTask,
)
from trace_tasks.tasks.geometry.angle_relations.triangle_exterior_angle import (
    GeometryAngleRelationsTriangleExteriorAngleTask,
)
from trace_tasks.tasks.geometry.composite_shape.house_outline_perimeter import GeometryMeasurementCompositePerimeterValueTask
from trace_tasks.tasks.geometry.composite_shape.l_profile_area import GeometryLProfileAreaTask
from trace_tasks.tasks.geometry.composite_shape.rectangle_triangle_cutout_area import GeometryRectangleTriangleCutoutAreaTask
from trace_tasks.tasks.geometry.composite_shape.tabbed_rectilinear_perimeter import GeometryCompositeShapeTabbedRectilinearPerimeterTask
from trace_tasks.tasks.geometry.triangle_relations.angle_bisector_segment_value import GeometryAngleBisectorSegmentValueTask
from trace_tasks.tasks.geometry.triangle_relations.centroid_median_segment_value import GeometryCentroidMedianSegmentValueTask
from trace_tasks.tasks.geometry.triangle_relations.parallel_section_segment_value import GeometryTriangleRelationsParallelSectionSegmentValueTask
from trace_tasks.tasks.geometry.triangle_relations.pythagorean_length_value_chained_rectangle_diagonal_length import GeometryPythagoreanLengthChainedRectangleDiagonalTask
from trace_tasks.tasks.geometry.triangle_relations.pythagorean_length_value_rectangle_triangle_shared_height_length import GeometryPythagoreanLengthRectangleTriangleSharedHeightTask
from trace_tasks.tasks.geometry.triangle_relations.similar_triangles_side_length import GeometryTriangleRelationsSimilarTrianglesSideLengthTask


TASK_CLASSES = (
    GeometryAngleRelationsParallelSupplementAngleTask,
    GeometryAngleRelationsTriangleExteriorAngleTask,
    GeometryAngleRelationsAlgebraicAngleValueTask,
    GeometryTriangleRelationsParallelSectionSegmentValueTask,
    GeometryTriangleRelationsSimilarTrianglesSideLengthTask,
    GeometryPythagoreanLengthChainedRectangleDiagonalTask,
    GeometryPythagoreanLengthRectangleTriangleSharedHeightTask,
    GeometryAngleBisectorSegmentValueTask,
    GeometryCentroidMedianSegmentValueTask,
    GeometryRectangleTriangleCutoutAreaTask,
    GeometryLProfileAreaTask,
    GeometryMeasurementCompositePerimeterValueTask,
    GeometryCompositeShapeTabbedRectilinearPerimeterTask,
)


def _scene_point_lookup(trace_payload) -> dict[str, list[float]]:
    points: dict[str, list[float]] = {}
    for entity in trace_payload["scene_ir"]["entities"]:
        entity_points = entity.get("points")
        if isinstance(entity_points, dict):
            for label, point in entity_points.items():
                points[str(label)] = [float(point[0]), float(point[1])]
    return points


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_composite_measurement_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(44001, params={}, max_attempts=20)
    scene_id = str(out.scene_id)

    assert out.query_id
    assert out.answer_gt.type == "integer"
    if scene_id == "angle_relations":
        assert out.annotation_gt.type == "point_map"
        assert 2 <= len(out.annotation_gt.value) <= 4
    elif scene_id == "composite_shape":
        assert out.annotation_gt.type == "point_map"
        assert 3 <= len(out.annotation_gt.value) <= 8
    elif scene_id == "triangle_relations":
        assert out.query_id == "single"
        assert out.annotation_gt.type == "segment"
        assert len(out.annotation_gt.value) == 2
    else:
        assert out.annotation_gt.type == "bbox_set"
        assert 2 <= len(out.annotation_gt.value) <= 6
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == scene_id
    assert trace["scene_ir"]["scene_id"] == scene_id
    assert trace["witness_symbolic"]["scene_id"] == scene_id
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == out.annotation_gt.type


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_composite_measurement_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(44011, params=params, max_attempts=20)
    out_b = task.generate(44011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]


def test_composite_measurement_tasks_support_explicit_query_selection() -> None:
    task = GeometryPythagoreanLengthRectangleTriangleSharedHeightTask()
    out = task.generate(
        44021,
        params={"query_id": "single", "target_answer": 37},
        max_attempts=20,
    )

    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["internal_case_kind"] == "rectangle_triangle_shared_height"
    assert out.answer_gt.value == 37
    assert out.trace_payload["query_spec"]["params"]["query_id_probabilities"] == {
        "single": 1.0
    }


def test_parallel_section_scale_task_supports_every_query() -> None:
    tasks = (
        (GeometryTriangleRelationsSimilarTrianglesSideLengthTask(), "nested_similarity_side"),
        (
            GeometryTriangleRelationsParallelSectionSegmentValueTask(),
            {"parallel_base_scale", "parallel_cross_section"},
        ),
    )
    for task, internal_case_kind in tasks:
        out = task.generate(44025, params={"query_id": "single", "target_answer": 10}, max_attempts=20)
        assert out.scene_id == "triangle_relations"
        assert out.query_id == "single"
        observed_case_kind = out.trace_payload["query_spec"]["params"]["internal_case_kind"]
        if isinstance(internal_case_kind, set):
            assert observed_case_kind in internal_case_kind
        else:
            assert observed_case_kind == internal_case_kind
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "segment"


def test_algebraic_triangle_cases_keep_annotation_inside_canvas() -> None:
    task = GeometryAngleRelationsAlgebraicAngleValueTask()
    for query_id in task.supported_query_ids:
        for case_index in range(6):
            out = task.generate(
                44041 + case_index,
                params={"query_id": query_id, "case_index": case_index},
                max_attempts=20,
            )
            width, height = out.image.size
            assert out.annotation_gt.type == "point_map"
            for x, y in out.annotation_gt.value.values():
                assert 0.0 <= x <= float(width)
                assert 0.0 <= y <= float(height)


@pytest.mark.parametrize(
    "task_cls",
    (
        GeometryAngleRelationsParallelSupplementAngleTask,
        GeometryAngleRelationsTriangleExteriorAngleTask,
        GeometryAngleRelationsAlgebraicAngleValueTask,
    ),
)
def test_angle_relations_public_annotation_uses_angle_primitives_not_label_boxes(task_cls) -> None:
    task = task_cls()
    for query_id in task.supported_query_ids:
        out = task.generate(44061, params={"query_id": query_id, "case_index": 0}, max_attempts=20)
        assert out.annotation_gt.type == "point_map"
        assert out.trace_payload["projected_annotation"]["type"] == "point_map"
        assert set(out.annotation_gt.value) == set(out.trace_payload["execution_trace"]["annotation_roles"])
        roles = out.trace_payload["execution_trace"]["annotation_roles"]
        assert all("label" not in str(role) for role in roles)
        if task_cls is GeometryAngleRelationsAlgebraicAngleValueTask:
            assert set(roles) == {"A", "B", "C", "D"}
        else:
            assert all(str(role).isupper() and len(str(role)) == 3 for role in roles)
        assert "angle_label_bboxes" in out.trace_payload["render_map"]
        width, height = out.image.size
        scene_points = _scene_point_lookup(out.trace_payload)
        for point in out.annotation_gt.value.values():
            assert len(point) == 2
            assert 0.0 <= float(point[0]) <= float(width)
            assert 0.0 <= float(point[1]) <= float(height)
        for key, point in out.annotation_gt.value.items():
            if str(key).isupper() and len(str(key)) == 3:
                expected_vertex = scene_points[str(key)[1]]
                assert [float(point[0]), float(point[1])] == pytest.approx(expected_vertex, abs=1e-3)
            elif task_cls is GeometryAngleRelationsAlgebraicAngleValueTask:
                expected_point = scene_points[str(key)]
                assert [float(point[0]), float(point[1])] == pytest.approx(expected_point, abs=1e-3)


def test_algebraic_angle_uses_varied_expression_forms() -> None:
    task = GeometryAngleRelationsAlgebraicAngleValueTask()
    expressions: set[str] = set()
    for query_id in task.supported_query_ids:
        for case_index in range(6):
            out = task.generate(
                44101 + case_index,
                params={"query_id": query_id, "case_index": case_index},
                max_attempts=20,
            )
            trace = out.trace_payload["execution_trace"]
            expressions.add(trace["expression_angle_ABC"])
            expressions.add(trace["expression_exterior_BCD"])

    assert any(expr.startswith("x+") for expr in expressions)
    assert any(expr.startswith("2x") for expr in expressions)
    assert any(expr.startswith("3x") for expr in expressions)
    assert any(expr.startswith("4x") for expr in expressions)


def test_composite_measurement_tasks_reject_unknown_query_id() -> None:
    task = GeometryAngleRelationsParallelSupplementAngleTask()
    with pytest.raises(ValueError):
        task.generate(44031, params={"query_id": "not_a_query"}, max_attempts=20)


@pytest.mark.parametrize(
    "task_cls, expected_keys_by_query",
    (
        (
            GeometryRectangleTriangleCutoutAreaTask,
            {
                "single": {"A", "B", "C", "D", "E", "F"},
            },
        ),
        (GeometryLProfileAreaTask, {"single": {"A", "B", "C", "D", "E", "F"}}),
        (GeometryMeasurementCompositePerimeterValueTask, {"single": {"A", "B", "C", "D", "E"}}),
        (
            GeometryCompositeShapeTabbedRectilinearPerimeterTask,
            {"single": {"A", "B", "C", "D", "E", "F", "G", "H"}},
        ),
    ),
)
def test_rectilinear_composite_public_annotation_uses_labeled_points(task_cls, expected_keys_by_query) -> None:
    task = task_cls()
    for query_id in tuple(task.supported_query_ids):
        out = task.generate(44121, params={"query_id": query_id, "case_index": 0}, max_attempts=20)
        assert out.annotation_gt.type == "point_map"
        assert set(out.annotation_gt.value) == expected_keys_by_query[str(query_id)]
        assert out.trace_payload["projected_annotation"]["type"] == "point_map"
        assert out.trace_payload["projected_annotation"]["point_map"] == out.annotation_gt.value
        assert out.trace_payload["projected_annotation"]["pixel_point_map"] == out.annotation_gt.value
        assert set(out.annotation_gt.value) == set(out.trace_payload["execution_trace"]["annotation_roles"])
        assert all("label" not in str(role) for role in out.trace_payload["execution_trace"]["annotation_roles"])
        assert "measurement_label_bboxes" in out.trace_payload["render_map"]
        assert set(out.trace_payload["render_map"]["point_label_bboxes"]) == set(out.annotation_gt.value)


@pytest.mark.parametrize(
    "task_cls, query_id, expected_count",
    (
        (GeometryRectangleTriangleCutoutAreaTask, "single", 5),
        (GeometryLProfileAreaTask, "single", 6),
        (GeometryMeasurementCompositePerimeterValueTask, "single", 6),
        (GeometryCompositeShapeTabbedRectilinearPerimeterTask, "single", 8),
    ),
)
def test_rectilinear_composite_visual_notation_is_render_metadata(task_cls, query_id, expected_count) -> None:
    task = task_cls()
    out = task.generate(
        44131,
        params={"query_id": query_id, "case_index": 0, "scene_rotation_degrees": 35},
        max_attempts=20,
    )
    notation = out.trace_payload["render_map"]["visual_notation_bboxes"]

    assert len(notation) == expected_count
    assert set(notation).isdisjoint(set(out.annotation_gt.value))
    for x0, y0, x1, y1 in notation.values():
        assert 0.0 <= x0 < x1 <= float(out.image.size[0])
        assert 0.0 <= y0 < y1 <= float(out.image.size[1])
