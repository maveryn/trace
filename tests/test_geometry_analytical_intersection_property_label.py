"""Contract tests for geometry intersection-property panel label task."""

from __future__ import annotations

from collections import Counter

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.geometry.function_panels.intersection_property_label import (
    SUPPORTED_QUERY_IDS,
    TASK_ID,
    GeometryFunctionPanelsIntersectionPropertyLabelTask,
)


def _matching_labels(branch: str, panels_by_label: dict[str, dict]) -> list[str]:
    if branch == "line_circle_tangent_label":
        return [
            label
            for label, panel in panels_by_label.items()
            if panel["pair_kind"] == "line_circle" and panel["relation_class"] == "tangent"
        ]
    if branch == "line_circle_two_intersections_label":
        return [
            label
            for label, panel in panels_by_label.items()
            if panel["pair_kind"] == "line_circle" and int(panel["intersection_count"]) == 2
        ]
    if branch == "circle_circle_two_intersections_label":
        return [
            label
            for label, panel in panels_by_label.items()
            if panel["pair_kind"] == "circle_circle" and int(panel["intersection_count"]) == 2
        ]
    raise AssertionError(f"unsupported branch in test: {branch}")


@pytest.mark.parametrize("query_id", SUPPORTED_QUERY_IDS)
def test_geometry_intersection_property_label_contract(query_id: str) -> None:
    task = GeometryFunctionPanelsIntersectionPropertyLabelTask()
    out = task.generate(
        hash64(94210, TASK_ID, 3),
        params={"query_id": query_id, "winner_label": "C"},
        max_attempts=10,
    )

    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox_set_map"
    assert set(out.annotation_gt.value) == {"selected_panel", "intersection_points"}
    assert len(out.annotation_gt.value["selected_panel"]) == 1
    assert len(out.annotation_gt.value["intersection_points"]) >= 1
    assert out.trace_payload["projected_annotation"]["bbox_set_map"] == out.annotation_gt.value
    assert out.trace_payload["query_spec"]["template_id"] == "geometry_analytical_intersection_property_v1"
    assert out.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert out.image.size == (1024, 1024)

    panels = out.trace_payload["execution_trace"]["panels_by_label"]
    assert 4 <= len(panels) <= 9
    assert "C" in panels
    assert set(panels).issubset({"A", "B", "C", "D", "E", "F", "G", "H", "I"})
    assert _matching_labels(query_id, panels) == ["C"]

    c_panel_bbox = out.trace_payload["projected_annotation"]["panel_bbox_by_label"]["C"]
    c_point_bboxes = out.trace_payload["projected_annotation"]["intersection_point_bboxes_by_label"]["C"]
    assert out.annotation_gt.value == {"selected_panel": [c_panel_bbox], "intersection_points": c_point_bboxes}


def test_geometry_intersection_property_label_balances_variants_and_answers() -> None:
    task = GeometryFunctionPanelsIntersectionPropertyLabelTask()
    per_branch_labels = {branch: Counter() for branch in SUPPORTED_QUERY_IDS}

    for index in range(99):
        out = task.generate(hash64(94220, TASK_ID, index), params={}, max_attempts=10)
        per_branch_labels[str(out.query_id)][str(out.answer_gt.value)] += 1

    branch_counts = {branch: sum(counter.values()) for branch, counter in per_branch_labels.items()}
    assert set(branch_counts) == set(SUPPORTED_QUERY_IDS)
    assert all(20 <= count <= 45 for count in branch_counts.values())
    for counts in per_branch_labels.values():
        assert set(counts.keys()).issubset({"A", "B", "C", "D", "E", "F", "G", "H", "I"})
        assert len(counts) >= 6
        assert max(counts.values()) <= 18


def test_geometry_intersection_property_label_randomizes_object_colors() -> None:
    task = GeometryFunctionPanelsIntersectionPropertyLabelTask()
    color_orders = set()

    for index in range(18):
        out = task.generate(hash64(94230, TASK_ID, index), params={}, max_attempts=10)
        colors = out.trace_payload["render_spec"]["object_colors"]
        color_orders.add(tuple(tuple(int(channel) for channel in color) for color in colors))
        assert len(colors) >= 2
        assert all(len(color) == 3 for color in colors)
        assert out.trace_payload["render_spec"]["object_color_selection"]["palette_index"] is not None

    assert len(color_orders) >= 6
