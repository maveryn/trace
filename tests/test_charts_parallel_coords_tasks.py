from __future__ import annotations

from typing import Any

import pytest

from trace_tasks.tasks.registry import create_task


TASK_QUERIES: dict[str, tuple[str, ...]] = {
    "task_charts__parallel_coords__all_crossings_between_adjacent_axes": ("single",),
    "task_charts__parallel_coords__axis_condition_count": (
        "above_on_both_axes",
        "below_on_both_axes",
        "above_on_one_below_on_other",
    ),
    "task_charts__parallel_coords__axis_delta_extremum_label": (
        "largest_increase_between_axes",
        "largest_decrease_between_axes",
        "largest_absolute_change_between_axes",
    ),
}


def _point_in_canvas(point: list[float], *, width: int, height: int) -> bool:
    return len(point) == 2 and 0 <= float(point[0]) <= width and 0 <= float(point[1]) <= height


def _segment_in_canvas(segment: list[list[float]], *, width: int, height: int) -> bool:
    return len(segment) == 2 and all(_point_in_canvas(list(point), width=width, height=height) for point in segment)


@pytest.mark.parametrize(
    ("task_id", "query_id"),
    [(task_id, query_id) for task_id, queries in TASK_QUERIES.items() for query_id in queries],
)
def test_parallel_coords_task_query_and_annotation_contract(task_id: str, query_id: str) -> None:
    task = create_task(task_id)
    output = task.generate(2026062001, params={"query_id": query_id}, max_attempts=512)

    assert output.scene_id == "parallel_coords"
    assert output.query_id == query_id
    assert output.answer_gt is not None
    assert output.annotation_gt is not None

    trace = output.trace_payload
    assert isinstance(trace, dict)
    query_spec = trace["query_spec"]
    assert query_spec["query_id"] == query_id
    assert query_spec["params"]["query_id"] == query_id

    render_spec = trace["render_spec"]
    width = int(render_spec["canvas_width"])
    height = int(render_spec["canvas_height"])
    projected = trace["projected_annotation"]

    if task_id.endswith("__axis_delta_extremum_label"):
        assert output.answer_gt.type == "string"
        assert output.annotation_gt.type == "segment"
        segment = output.annotation_gt.value
        assert isinstance(segment, list)
        assert _segment_in_canvas([list(point) for point in segment], width=width, height=height)
        assert projected["type"] == "segment"
        assert projected["segment"] == segment
        assert projected["pixel_segment"] == segment
    elif task_id.endswith("__axis_condition_count"):
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "segment_set"
        segments = output.annotation_gt.value
        assert isinstance(segments, list)
        assert len(segments) == int(output.answer_gt.value)
        assert all(_segment_in_canvas([list(point) for point in segment], width=width, height=height) for segment in segments)
        assert projected["type"] == "segment_set"
        assert projected["segment_set"] == segments
        assert projected["pixel_segment_set"] == segments
    else:
        assert task_id.endswith("__all_crossings_between_adjacent_axes")
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "point_set"
        points = output.annotation_gt.value
        assert isinstance(points, list)
        assert len(points) == int(output.answer_gt.value)
        assert all(_point_in_canvas(point, width=width, height=height) for point in points)
        assert projected["type"] == "point_set"
        assert projected["pixel_point_set"] == points


def test_parallel_coords_generation_is_deterministic_for_same_seed_and_query() -> None:
    task = create_task("task_charts__parallel_coords__axis_delta_extremum_label")
    params: dict[str, Any] = {"query_id": "largest_absolute_change_between_axes"}

    first = task.generate(2026062099, params=params, max_attempts=512)
    second = task.generate(2026062099, params=params, max_attempts=512)

    assert first.query_id == second.query_id
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.prompt == second.prompt
    assert first.trace_payload["query_spec"] == second.trace_payload["query_spec"]
