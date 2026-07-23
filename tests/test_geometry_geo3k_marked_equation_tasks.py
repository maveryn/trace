from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.tasks.registry import create_task


TASK_QUERIES = {
    "task_geometry__similar_figure_measure_transfer__variable_value": (
        "single",
    ),
}

SIMILAR_CONSTRUCTION_FAMILIES = {
    "task_geometry__similar_figure_measure_transfer__variable_value": (
        "triangle_ratio",
        "polygon_ratio",
        "two_expression_ratio",
    ),
}

def _generate(task_id: str, query_id: str, seed: int = 20260607, **extra_params):
    task = create_task(task_id)
    return task.generate(seed, params={"query_id": query_id, **extra_params}, max_attempts=3)


def test_geo3k_marked_equation_tasks_are_registered() -> None:
    for task_id in TASK_QUERIES:
        assert create_task(task_id).task_id == task_id


def test_geo3k_marked_equation_queries_emit_keyed_point_annotation() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for index, query_id in enumerate(query_ids):
            output = _generate(task_id, query_id, seed=20260607 + index)
            assert output.query_id == query_id
            assert output.answer_gt.type == "number"
            assert isinstance(output.answer_gt.value, (int, float))
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
            assert trace["projected_annotation"]["pixel_point_map"] == output.annotation_gt.value
            assert "task_variant" not in trace["query_spec"]["params"]
            assert "query_variant" not in trace["query_spec"]["params"]


def test_geo3k_marked_equation_queries_use_expected_scene_ids() -> None:
    assert _generate(
        "task_geometry__similar_figure_measure_transfer__variable_value",
        "single",
    ).scene_id == "similar_figure_measure_transfer"


def test_similar_figure_equation_construction_families_are_trace_metadata() -> None:
    for task_id, families in SIMILAR_CONSTRUCTION_FAMILIES.items():
        for index, family in enumerate(families):
            output = _generate(
                task_id,
                "single",
                seed=20260616 + index,
                construction_family=family,
            )
            assert output.query_id == "single"
            trace = output.trace_payload
            assert trace["execution_trace"]["construction_family"] == family
            assert trace["query_spec"]["params"]["construction_family"] == family
            assert trace["query_spec"]["params"]["query_id"] == "single"
