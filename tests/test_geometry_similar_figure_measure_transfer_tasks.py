from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.tasks.registry import create_task


TASK_QUERIES = {
    "task_geometry__similar_figure_measure_transfer__corresponding_side_value": (
        "single",
    ),
    "task_geometry__similar_figure_measure_transfer__area_scale_side_length_value": (
        "side_length_from_area_pair",
        "side_length_from_area_ratio",
    ),
}

CONSTRUCTION_FAMILIES = {
    "task_geometry__similar_figure_measure_transfer__corresponding_side_value": (
        "direct_side_transfer",
        "two_pair_side_transfer",
        "nested_side_transfer",
    ),
    "task_geometry__similar_figure_measure_transfer__area_scale_side_length_value": (
        "area_pair_labels",
        "area_known_side_nested",
    ),
}


def _generate(task_id: str, query_id: str, seed: int = 20260605, **extra_params):
    task = create_task(task_id)
    return task.generate(seed, params={"query_id": query_id, **extra_params}, max_attempts=3)


def test_similar_figure_measure_transfer_tasks_are_registered() -> None:
    for task_id in TASK_QUERIES:
        assert create_task(task_id).task_id == task_id


def test_similar_figure_measure_transfer_queries_emit_point_map_annotation() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for index, query_id in enumerate(query_ids):
            output = _generate(task_id, query_id, seed=20260605 + index)
            assert output.scene_id == "similar_figure_measure_transfer"
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
            assert trace["projected_annotation"]["pixel_point_map"] == output.annotation_gt.value
            assert "task_variant" not in trace["query_spec"]["params"]
            assert "query_variant" not in trace["query_spec"]["params"]


def test_similar_figure_measure_transfer_measurements_match_trace_values() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for query_id in query_ids:
            output = _generate(task_id, query_id, seed=20260617)
            trace = output.trace_payload["execution_trace"]
            scale_factor = int(trace["scale_factor"])
            assert output.answer_gt.value == int(trace["target_target_side_value"])
            source_side = trace["source_target_side_value"]
            target_side = trace["target_target_side_value"]
            if source_side is not None and target_side is not None:
                assert int(target_side) == int(source_side) * scale_factor
            source_area = trace["source_area"]
            target_area = trace["target_area"]
            if source_area is not None and target_area is not None:
                assert int(target_area) == int(source_area) * scale_factor * scale_factor


def test_similar_figure_measure_transfer_generation_is_deterministic() -> None:
    task_id = "task_geometry__similar_figure_measure_transfer__area_scale_side_length_value"
    query_id = "side_length_from_area_pair"
    first = _generate(task_id, query_id, seed=817)
    second = _generate(task_id, query_id, seed=817)
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_similar_figure_measure_transfer_construction_families_are_trace_metadata() -> None:
    for task_id, families in CONSTRUCTION_FAMILIES.items():
        for index, family in enumerate(families):
            output = _generate(task_id, TASK_QUERIES[task_id][0], seed=20260630 + index, construction_family=family)
            assert output.trace_payload["execution_trace"]["construction_family"] == family
            assert output.trace_payload["query_spec"]["params"]["construction_family"] == family
