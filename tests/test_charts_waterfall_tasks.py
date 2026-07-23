"""Contract smoke tests for waterfall chart tasks."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY, ensure_scene_tasks_registered
from trace_tasks.tasks.charts.waterfall.running_total_extremum_value import (
    MAXIMUM_QUERY_ID,
    MINIMUM_QUERY_ID,
)


WATERFALL_TASKS = {
    "task_charts__waterfall__running_total_value": {SINGLE_QUERY_ID},
    "task_charts__waterfall__running_total_extremum_value": {
        MAXIMUM_QUERY_ID,
        MINIMUM_QUERY_ID,
    },
    "task_charts__waterfall__remove_step_final_total": {SINGLE_QUERY_ID},
    "task_charts__waterfall__reverse_step_final_total": {SINGLE_QUERY_ID},
}


def _steps_by_id(output):
    return {
        str(step["step_id"]): dict(step)
        for step in output.trace_payload["execution_trace"]["steps"]
    }


def _round_bbox(bbox):
    return [round(float(value), 3) for value in bbox]


def _task(task_id: str):
    ensure_scene_tasks_registered("charts", "waterfall")
    return TASK_REGISTRY[str(task_id)]()


def test_waterfall_tasks_registered() -> None:
    ensure_scene_tasks_registered("charts", "waterfall")
    for task_id in WATERFALL_TASKS:
        assert task_id in TASK_REGISTRY


def test_waterfall_tasks_generate_default_query_outputs() -> None:
    for seed_index, (task_id, allowed_query_ids) in enumerate(sorted(WATERFALL_TASKS.items())):
        output = _task(task_id).generate(
            103_000 + seed_index,
            params={},
            max_attempts=100,
        )
        assert output.scene_id == "waterfall"
        assert output.query_id in allowed_query_ids
        assert output.trace_payload["query_spec"]["params"]["query_id"] == output.query_id
        assert output.annotation_gt.value is not None
        assert output.trace_payload["projected_annotation"]["type"] == output.annotation_gt.type
        if output.annotation_gt.type == "bbox_set_map":
            assert output.trace_payload["projected_annotation"]["bbox_set_map"] == output.annotation_gt.value
        elif output.annotation_gt.type == "bbox_map":
            assert output.trace_payload["projected_annotation"]["bbox_map"] == output.annotation_gt.value
        elif output.annotation_gt.type == "bbox_set":
            assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value
        elif output.annotation_gt.type == "bbox":
            assert output.trace_payload["projected_annotation"]["bbox"] == output.annotation_gt.value
        else:  # pragma: no cover - defensive contract guard
            raise AssertionError(f"unexpected waterfall annotation type: {output.annotation_gt.type}")
        assert str(output.trace_payload["render_spec"]["font_assets"]["chart_font_family"]).strip()
        assert output.trace_payload["render_spec"]["background_style"]
        assert output.trace_payload["render_map"]["bar_bboxes_px"]


def test_waterfall_tasks_generate_each_query_branch_and_answer_contract() -> None:
    seed_index = 0
    for task_id, allowed_query_ids in sorted(WATERFALL_TASKS.items()):
        for query_id in sorted(allowed_query_ids):
            output = _task(task_id).generate(
                104_000 + seed_index,
                params={"query_id": query_id},
                max_attempts=100,
            )
            assert output.scene_id == "waterfall"
            assert output.query_id == query_id
            execution = output.trace_payload["execution_trace"]
            steps = _steps_by_id(output)

            if task_id.endswith("__running_total_value"):
                target = steps[str(execution["target_step_id"])]
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == int(target["running_after"])
                assert output.annotation_gt.type == "bbox_set"
                expected_ids = ["start"] + [
                    str(step["step_id"])
                    for step in execution["steps"][: int(execution["target_step_index"]) + 1]
                ]
                expected_boxes = [
                    _round_bbox(output.trace_payload["render_map"]["bar_bboxes_px"][bar_id])
                    for bar_id in expected_ids
                ]
                assert output.annotation_gt.value == expected_boxes
            elif task_id.endswith("__running_total_extremum_value"):
                candidates = list(execution["candidate_running_totals"])
                values = {
                    str(candidate["bar_id"]): int(candidate["running_total"])
                    for candidate in candidates
                }
                expected_id = (
                    max(values, key=values.get)
                    if str(query_id) == MAXIMUM_QUERY_ID
                    else min(values, key=values.get)
                )
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == int(values[str(expected_id)])
                assert output.annotation_gt.type == "bbox"
                assert str(execution["answer_bar_id"]) == str(expected_id)
                assert int(execution["answer_running_total"]) == int(values[str(expected_id)])
                assert output.annotation_gt.value == _round_bbox(
                    output.trace_payload["render_map"]["bar_bboxes_px"][str(expected_id)]
                )
            elif task_id.endswith("__remove_step_final_total"):
                target = steps[str(execution["target_step_id"])]
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == int(execution["final_value"]) - int(target["delta"])
                assert output.annotation_gt.type == "bbox_map"
                assert set(output.annotation_gt.value) == {
                    "final_total_bar",
                    "target_contribution_bar",
                }
            elif task_id.endswith("__reverse_step_final_total"):
                target = steps[str(execution["target_step_id"])]
                assert output.answer_gt.type == "integer"
                assert output.answer_gt.value == int(execution["final_value"]) - (2 * int(target["delta"]))
                assert output.annotation_gt.type == "bbox_map"
                assert set(output.annotation_gt.value) == {
                    "final_total_bar",
                    "target_contribution_bar",
                }

            seed_index += 1


def test_waterfall_running_total_uses_late_nonfinal_targets() -> None:
    observed_step_counts = set()
    observed_target_indices = set()

    for offset in range(12):
        output = _task("task_charts__waterfall__running_total_value").generate(
            2026052300 + offset,
            params={},
            max_attempts=100,
        )
        execution = output.trace_payload["execution_trace"]
        step_count = int(execution["step_count"])
        target_index = int(execution["target_step_index"])
        observed_step_counts.add(step_count)
        observed_target_indices.add(target_index)
        assert 8 <= step_count <= 10
        assert 4 <= target_index <= step_count - 2

    assert observed_step_counts
    assert observed_target_indices
