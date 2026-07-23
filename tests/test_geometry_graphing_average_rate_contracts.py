"""Contract tests for geometry function-graph average-rate task."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.function_graph.average_rate_value import GeometryGraphingAverageRateValueTask


def test_geometry_graphing_average_rate_emits_expected_contract() -> None:
    out = GeometryGraphingAverageRateValueTask().generate(
        24001,
        params={"target_rate": 1.5},
        max_attempts=20,
    )

    assert out.answer_gt.type == "number"
    assert float(out.answer_gt.value) == 1.5
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"A", "B"}
    assert out.trace_payload["projected_annotation"]["type"] == "point_map"
    assert out.trace_payload["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["query_id"] == "single"
    assert out.trace_payload["query_spec"]["params"]["query_id"] == "single"
    assert out.trace_payload["query_spec"]["params"]["target_rate"] == 1.5
    assert out.trace_payload["execution_trace"]["average_rate"] == 1.5
    assert out.trace_payload["execution_trace"]["delta_y"] / out.trace_payload["execution_trace"]["delta_x"] == 1.5


def test_geometry_graphing_average_rate_rejects_unknown_query_id() -> None:
    task = GeometryGraphingAverageRateValueTask()
    with pytest.raises(ValueError, match="unsupported query_id"):
        task.generate(
            24002,
            params={"query_id": "instantaneous_slope_value"},
            max_attempts=20,
        )
