"""Behavior tests for annotated-series chart tasks."""

from __future__ import annotations

from trace_tasks.tasks.charts.annotated_series.callout_endpoint_change_value import (
    ChartsAnnotatedSeriesCalloutEndpointChangeValueTask,
)


def _assert_point_map_contract(out: object) -> None:
    trace = out.trace_payload
    annotation_points = {str(key): list(point) for key, point in out.annotation_gt.value.items()}
    assert out.annotation_gt.type == "point_map"
    assert trace["projected_annotation"] == {
        "type": "point_map",
        "point_map": annotation_points,
        "pixel_point_map": annotation_points,
    }
    assert trace["witness_symbolic"]["type"] == "object_key_map"
    assert set(trace["witness_symbolic"]["keys"]) == {"callout_mark", "endpoint_mark"}
    assert "annotation_bboxes" in trace["render_map"]
    assert "annotation_bboxes" in trace["render_spec"]
    assert set(annotation_points) == {"callout_mark", "endpoint_mark"}
    assert all(len(point) == 2 for point in annotation_points.values())
    answer_and_annotation_prompt = str(out.prompt_variants["answer_and_annotation"])
    assert "callout_mark" in answer_and_annotation_prompt
    assert "endpoint_mark" in answer_and_annotation_prompt


def _bbox_overlap_area(box_a: list[float], box_b: list[float]) -> float:
    width = min(float(box_a[2]), float(box_b[2])) - max(float(box_a[0]), float(box_b[0]))
    height = min(float(box_a[3]), float(box_b[3])) - max(float(box_a[1]), float(box_b[1]))
    return max(0.0, width) * max(0.0, height)


def test_annotated_series_callout_change_uses_anchor_and_endpoint_points() -> None:
    task = ChartsAnnotatedSeriesCalloutEndpointChangeValueTask()
    out = task.generate(24012, params={"scene_variant": "line"}, max_attempts=10)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    anchor_label = str(execution["anchor_label"])
    endpoint_label = str(execution["endpoint_label"])

    _assert_point_map_contract(out)
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.value == {
        "callout_mark": trace["render_map"]["mark_center_by_label"][anchor_label],
        "endpoint_mark": trace["render_map"]["mark_center_by_label"][endpoint_label],
    }


def test_annotated_series_callout_box_does_not_cover_critical_endpoint_marks() -> None:
    task = ChartsAnnotatedSeriesCalloutEndpointChangeValueTask()
    cases = (
        (
            347784840739934,
            {
                "scene_variant": "lollipop",
                "endpoint_side": "last",
                "mark_count_min": 12,
                "mark_count_max": 12,
            },
        ),
        (
            762788342831494,
            {
                "scene_variant": "line",
                "endpoint_side": "first",
                "mark_count_min": 13,
                "mark_count_max": 13,
            },
        ),
    )
    for instance_seed, params in cases:
        out = task.generate(instance_seed, params=params, max_attempts=10)
        trace = out.trace_payload
        render_map = trace["render_map"]
        execution = trace["execution_trace"]
        callout_box = render_map["annotation_bboxes"]["annotation_callout"]
        for label in (execution["anchor_label"], execution["endpoint_label"]):
            mark_box = render_map["mark_bbox_by_label"][str(label)]
            assert _bbox_overlap_area(callout_box, mark_box) == 0.0
