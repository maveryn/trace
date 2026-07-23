from __future__ import annotations

from trace_tasks.tasks.registry import create_task

ANGLE_KEYS = {"angle_vertex", "protractor_reading_tick"}

TASK_EXPECTATIONS = {
    "task_geometry__measuring_tools__ruler_length_value": {
        "keys": {"measure_start", "measure_end"},
        "tool_kind": "ruler",
        "measurement_kind": "ruler_length_reading",
        "annotation_type": "segment",
    },
    "task_geometry__measuring_tools__protractor_angle_value": {
        "keys": ANGLE_KEYS,
        "tool_kind": "protractor",
        "measurement_kind": "protractor_angle_reading",
        "annotation_type": "point_map",
    },
}


def test_measuring_tools_tasks_use_single_query_integer_answers_and_annotations() -> None:
    for index, (task_id, expected) in enumerate(TASK_EXPECTATIONS.items()):
        out = create_task(task_id).generate(
            instance_seed=2026062300 + index,
            params={},
            max_attempts=50,
        )

        assert out.query_id == "single"
        assert out.answer_gt.type == "integer"
        assert isinstance(out.answer_gt.value, int)
        assert out.annotation_gt.type == expected["annotation_type"]

        projected = out.trace_payload["projected_annotation"]
        if expected["annotation_type"] == "point_map":
            assert set(out.annotation_gt.value) == expected["keys"]
            assert projected["type"] == "point_map"
            assert projected["point_map"] == out.annotation_gt.value
            assert projected["pixel_point_map"] == out.annotation_gt.value
        else:
            assert projected["type"] == "segment"
            assert projected["segment"] == out.annotation_gt.value
            assert projected["pixel_segment"] == out.annotation_gt.value
            measurement_points = out.trace_payload["render_map"]["measurement_points"]
            assert out.annotation_gt.value == [
                measurement_points["measure_start"],
                measurement_points["measure_end"],
            ]

        query_spec = out.trace_payload["query_spec"]
        assert query_spec["query_id"] == "single"
        assert query_spec["params"]["query_id"] == "single"
        assert query_spec["params"]["tool_kind"] == expected["tool_kind"]
        assert query_spec["params"]["measurement_kind"] == expected["measurement_kind"]
        assert query_spec["prompt_variant"]["prompt_schema_version"] == "v1"


def test_measuring_tools_tasks_validate_query_id_params() -> None:
    task = create_task("task_geometry__measuring_tools__ruler_length_value")
    assert (
        task.generate(instance_seed=17, params={"query_id": "single"}, max_attempts=50).query_id
        == "single"
    )
    assert (
        task.generate(
            instance_seed=17,
            params={"query_variant": "single"},
            max_attempts=50,
        ).query_id
        == "single"
    )

    try:
        task.generate(instance_seed=17, params={"query_id": "unsupported"}, max_attempts=50)
    except ValueError as exc:
        assert "query_id" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("unsupported query_id should fail")


def test_measuring_tools_merged_tasks_accept_internal_shape_kind_replay() -> None:
    cases = {
        "task_geometry__measuring_tools__ruler_length_value": (
            "circle",
            "triangle",
            "parallelogram",
            "trapezoid",
        ),
        "task_geometry__measuring_tools__protractor_angle_value": ("triangle", "quadrilateral"),
    }
    for task_id, shape_kinds in cases.items():
        task = create_task(task_id)
        for shape_kind in shape_kinds:
            out = task.generate(
                instance_seed=2026062700,
                params={"shape_kind": shape_kind},
                max_attempts=50,
            )
            assert out.trace_payload["render_map"]["shape_kind"] == shape_kind
