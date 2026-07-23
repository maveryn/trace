"""Contract tests for geometry function-graph count tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.function_graph.average_rate_value import GeometryGraphingAverageRateValueTask
from trace_tasks.tasks.geometry.function_graph.extremum_count_local_extremum_count import (
    GeometryGraphingLocalExtremumCountTask,
)
from trace_tasks.tasks.geometry.function_graph.extremum_count_turning_point_count import (
    GeometryGraphingTurningPointCountTask,
)


@pytest.mark.parametrize(
    ("task", "params", "expected_query", "expected_answer"),
    (
        (
            GeometryGraphingTurningPointCountTask(),
            {"scene_variant": "piecewise_linear", "query_id": "single", "target_count": 6},
            "single",
            6,
        ),
        (
            GeometryGraphingLocalExtremumCountTask(),
            {"scene_variant": "sinusoid", "query_id": "minimum", "target_count": 2},
            "minimum",
            2,
        ),
        (
            GeometryGraphingLocalExtremumCountTask(),
            {"scene_variant": "piecewise_linear", "query_id": "maximum", "target_count": 6},
            "maximum",
            6,
        ),
    ),
)
def test_geometry_graphing_count_tasks_emit_expected_contract(
    task,
    params: dict[str, int | str],
    expected_query: str,
    expected_answer: int,
) -> None:
    out = task.generate(23401, params=params, max_attempts=40)
    assert out.answer_gt.type == "integer"
    assert int(out.answer_gt.value) == int(expected_answer)
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == int(expected_answer)
    assert out.trace_payload["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert out.query_id == expected_query
    assert out.trace_payload["query_spec"]["query_id"] == expected_query
    assert out.trace_payload["query_spec"]["params"]["query_id"] == expected_query
    assert out.trace_payload["execution_trace"]["target_count"] == int(expected_answer)
    assert out.trace_payload["query_spec"]["params"]["scene_variant"] == params["scene_variant"]


@pytest.mark.parametrize(
    ("task", "params"),
    (
        (GeometryGraphingTurningPointCountTask(), {"scene_variant": "cubic", "query_id": "single", "target_count": 3}),
        (GeometryGraphingLocalExtremumCountTask(), {"scene_variant": "absolute_value", "query_id": "minimum", "target_count": 2}),
    ),
)
def test_geometry_graphing_count_tasks_reject_incompatible_scene_pairs(task, params: dict[str, int | str]) -> None:
    with pytest.raises(ValueError):
        task.generate(23411, params=params, max_attempts=20)


@pytest.mark.parametrize(
    ("task", "source_query_id"),
    (
        (GeometryGraphingTurningPointCountTask(), "turning_point_count"),
        (GeometryGraphingLocalExtremumCountTask(), "local_extremum_count"),
        (GeometryGraphingLocalExtremumCountTask(), "local_minima_count"),
    ),
)
def test_geometry_graphing_count_tasks_reject_source_query_ids(task, source_query_id: str) -> None:
    with pytest.raises(ValueError):
        task.generate(23414, params={"query_id": source_query_id}, max_attempts=20)


def test_geometry_graphing_average_rate_does_not_draw_secant_helper_line() -> None:
    out = GeometryGraphingAverageRateValueTask().generate(23415, params={}, max_attempts=20)
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"A", "B"}
    assert "secant_segment_graph" not in out.trace_payload["render_map"]
    assert "secant_segment_pixel" not in out.trace_payload["render_map"]
