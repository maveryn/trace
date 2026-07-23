"""Contract smoke tests for area chart panel tasks."""

from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY


AREA_TASKS = {
    "task_charts__area__interval_area_value": {"single"},
    "task_charts__area__stacked_band_interval_sum_value": {"single"},
    "task_charts__area__stacked_band_dominance_label": {"single"},
}


def test_area_tasks_registered() -> None:
    assert set(AREA_TASKS).issubset(set(TASK_REGISTRY))


def test_area_tasks_generate_default_query_outputs() -> None:
    for seed_index, (task_id, allowed_query_ids) in enumerate(sorted(AREA_TASKS.items())):
        task = TASK_REGISTRY[task_id]()
        output = task.generate(
            74_000 + seed_index,
            params={},
            max_attempts=80,
        )
        assert output.scene_id == "area"
        allowed = {str(allowed_query_ids)} if isinstance(allowed_query_ids, str) else set(allowed_query_ids)
        assert output.query_id in allowed
        assert output.trace_payload["query_spec"]["params"]["query_id"] == output.query_id
        assert output.answer_gt.type in {"integer", "string"}
        assert output.annotation_gt.type == "point_set"
        assert output.annotation_gt.value
        projected = output.trace_payload["projected_annotation"]
        assert projected["type"] == "point_set"
        assert projected["point_set"] == output.annotation_gt.value
        assert projected["pixel_point_set"] == output.annotation_gt.value
        assert "bbox_set" not in projected
        render_spec = output.trace_payload["render_spec"]
        assert render_spec["font_assets"]["chart_font_family"]
        assert render_spec["font_assets"]["implicit_readout_font_family"]["font_family"]
        assert render_spec["text_style"]["font_asset_version"]
        assert render_spec["text_style"]["chart_font_family"]
        assert render_spec["information_scene_style"]["kind"] == "information_scene_style"


def test_area_interval_annotation_uses_curve_marker_not_value_label() -> None:
    output = TASK_REGISTRY["task_charts__area__interval_area_value"]().generate(
        74_200,
        params={},
        max_attempts=80,
    )

    point_traces = output.trace_payload["render_map"]["point_traces"]
    queried_traces = [trace for trace in point_traces if bool(trace["queried"])]
    assert output.annotation_gt.value == [trace["mark_center_px"] for trace in queried_traces]
    assert queried_traces
    assert all(trace["mark_center_px"] != trace["value_center_px"] for trace in queried_traces)


def test_area_numeric_tasks_generate_default_query_outputs() -> None:
    task_ids = (
        "task_charts__area__interval_area_value",
        "task_charts__area__stacked_band_interval_sum_value",
    )
    for seed_index, task_id in enumerate(sorted(task_ids)):
        task = TASK_REGISTRY[task_id]()
        output = task.generate(
            74_230 + seed_index,
            params={},
            max_attempts=80,
        )
        assert output.scene_id == "area"
        assert output.query_id == "single"
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "point_set"
        assert output.annotation_gt.value


def test_area_stacked_dominance_annotation_only_uses_winning_category() -> None:
    output = TASK_REGISTRY["task_charts__area__stacked_band_dominance_label"]().generate(
        74_250,
        params={},
        max_attempts=80,
    )

    execution = output.trace_payload["execution_trace"]
    answer = str(output.answer_gt.value)
    start_index = int(execution["start_index"])
    end_index = int(execution["end_index"])
    annotation_pairs = execution["annotation_pairs"]

    assert annotation_pairs
    assert len(annotation_pairs) == end_index - start_index + 1
    assert all(str(series_label) == answer for series_label, _ in annotation_pairs)

    point_traces = output.trace_payload["render_map"]["point_traces"]
    queried_traces = [trace for trace in point_traces if bool(trace["queried"])]
    assert output.annotation_gt.value == [trace["mark_center_px"] for trace in queried_traces]
    assert all(str(trace["series_label"]) == answer for trace in queried_traces)
