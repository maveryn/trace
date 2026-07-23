from __future__ import annotations

import math

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import create_task


TASK_QUERIES = {
    "task_geometry__regular_polygon_decomposition__marked_piece_area_value": (
        SINGLE_QUERY_ID,
    ),
    "task_geometry__regular_polygon_decomposition__wedge_area_from_side_apothem_value": (
        SINGLE_QUERY_ID,
    ),
    "task_geometry__regular_polygon_decomposition__central_angle_value": (
        SINGLE_QUERY_ID,
    ),
    "task_geometry__regular_polygon_decomposition__perimeter_value": (
        SINGLE_QUERY_ID,
    ),
    "task_geometry__regular_polygon_decomposition__side_length_value": (
        "side_length_from_perimeter",
        "side_length_from_total_area_and_apothem",
        "side_length_from_wedge_area_and_apothem",
    ),
}


def _generate(task_id: str, query_id: str, seed: int = 20260605):
    task = create_task(task_id)
    return task.generate(seed, params={"query_id": query_id}, max_attempts=3)


def _bboxes_overlap(box_a, box_b, *, margin: float = 0.0) -> bool:
    return not (
        float(box_a[2]) + float(margin) <= float(box_b[0])
        or float(box_b[2]) + float(margin) <= float(box_a[0])
        or float(box_a[3]) + float(margin) <= float(box_b[1])
        or float(box_b[3]) + float(margin) <= float(box_a[1])
    )


def test_regular_polygon_decomposition_tasks_are_registered() -> None:
    for task_id in TASK_QUERIES:
        assert create_task(task_id).task_id == task_id


def test_regular_polygon_decomposition_queries_emit_keyed_point_annotation() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for index, query_id in enumerate(query_ids):
            output = _generate(task_id, query_id, seed=20260605 + index)
            assert output.scene_id == "regular_polygon_decomposition"
            assert output.query_id == query_id
            if task_id.endswith("__central_angle_value") or task_id.endswith("__perimeter_value") or task_id.endswith("__side_length_value"):
                assert output.answer_gt.type == "integer"
                assert isinstance(output.answer_gt.value, int)
            else:
                assert output.answer_gt.type == "number"
                assert isinstance(output.answer_gt.value, float)
            assert output.annotation_gt.type == "point_map"
            assert isinstance(output.annotation_gt.value, dict)
            assert output.annotation_gt.value
            if task_id.endswith("__marked_piece_area_value"):
                assert set(output.annotation_gt.value) == {"O", "A", "B"}
            if task_id.endswith("__wedge_area_from_side_apothem_value"):
                assert set(output.annotation_gt.value) == {"O", "A", "B", "M"}
            if task_id.endswith("__perimeter_value"):
                assert set(output.annotation_gt.value) == {"O", "M"}
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


def test_regular_polygon_decomposition_measurements_match_trace_values() -> None:
    for task_id, query_ids in TASK_QUERIES.items():
        for query_id in query_ids:
            output = _generate(task_id, query_id, seed=20260617)
            trace = output.trace_payload["execution_trace"]
            n_sides = int(trace["n_sides"])
            wedge_count = int(trace["wedge_count"])
            assert n_sides >= 5
            assert int(trace["central_angle_degrees"]) == int(round(360.0 / float(n_sides)))
            if task_id.endswith("__marked_piece_area_value"):
                assert 1 <= wedge_count <= min(4, n_sides // 2)
                assert math.isclose(float(output.answer_gt.value), float(trace["wedge_area"]) * float(wedge_count))
            elif task_id.endswith("__wedge_area_from_side_apothem_value"):
                assert wedge_count == 1
                assert output.answer_gt.value == round(float(trace["side_length"]) * float(trace["apothem"]) / 2.0 + 1e-9, 1)
            elif task_id.endswith("__central_angle_value"):
                assert 1 <= wedge_count <= min(4, n_sides // 2)
                assert output.answer_gt.value == int(wedge_count * int(trace["central_angle_degrees"]))
            elif task_id.endswith("__perimeter_value"):
                assert wedge_count == 1
                assert output.answer_gt.value == int(round((2.0 * float(trace["total_area"])) / float(trace["apothem"])))
            elif query_id == "side_length_from_perimeter":
                assert output.answer_gt.value == int(round(float(trace["perimeter"]) / float(n_sides)))
            elif query_id == "side_length_from_total_area_and_apothem":
                assert output.answer_gt.value == int(round((2.0 * float(trace["total_area"])) / (float(n_sides) * float(trace["apothem"]))))
            elif query_id == "side_length_from_wedge_area_and_apothem":
                assert output.answer_gt.value == int(round((2.0 * float(trace["wedge_area"])) / float(trace["apothem"])))
            else:
                raise AssertionError(f"unexpected query_id={query_id}")


def test_regular_polygon_decomposition_generation_is_deterministic() -> None:
    task_id = "task_geometry__regular_polygon_decomposition__wedge_area_from_side_apothem_value"
    query_id = SINGLE_QUERY_ID
    first = _generate(task_id, query_id, seed=817)
    second = _generate(task_id, query_id, seed=817)
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_regular_polygon_midpoint_label_does_not_overlap_side_length_label() -> None:
    task_queries = (
        (
            "task_geometry__regular_polygon_decomposition__side_length_value",
            "side_length_from_total_area_and_apothem",
            "target_side_label",
        ),
        (
            "task_geometry__regular_polygon_decomposition__side_length_value",
            "side_length_from_wedge_area_and_apothem",
            "target_side_label",
        ),
        (
            "task_geometry__regular_polygon_decomposition__wedge_area_from_side_apothem_value",
            SINGLE_QUERY_ID,
            "side_length_label",
        ),
    )
    for task_id, query_id, side_label_key in task_queries:
        for seed in range(20260605, 20260625):
            output = _generate(task_id, query_id, seed=seed)
            readout_bboxes = output.trace_payload["render_map"]["readout_bboxes"]
            assert "label_M" in readout_bboxes
            assert side_label_key in readout_bboxes
            assert not _bboxes_overlap(readout_bboxes["label_M"], readout_bboxes[side_label_key], margin=2.0)
