"""Regression tests for combo chart panel tasks."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.registry import create_task, list_default_task_ids


COMBO_TASK_IDS = (
    "task_charts__combo_mark__cross_mark_difference_value",
    "task_charts__combo_mark__conditioned_line_extremum_label",
    "task_charts__combo_mark__conditioned_primary_extremum_label",
    "task_charts__combo_mark__dual_threshold_condition_count",
    "task_charts__combo_mark__interval_threshold_condition_count",
    "task_charts__combo_mark__absolute_gap_extremum_label",
    "task_charts__combo_mark__directional_gap_extremum_label",
    "task_charts__combo_mark__series_threshold_crossing_label",
)


def test_combo_tasks_are_default_and_taxonomy_aligned() -> None:
    default_task_ids = set(list_default_task_ids())
    for task_id in COMBO_TASK_IDS:
        assert task_id in default_task_ids
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "charts"
        assert taxonomy.scene_id == "combo_mark"
        assert not taxonomy.source_scene_id


def test_combo_tasks_generate_default_public_variant() -> None:
    for offset, task_id in enumerate(COMBO_TASK_IDS):
        output = create_task(task_id).generate(
            2026052200 + offset,
            params={},
            max_attempts=160,
        )
        assert output.scene_id == "combo_mark"
        assert output.query_id
        assert output.answer_gt.value is not None
        assert output.annotation_gt.type in {"point_map", "point", "segment_set"}
        assert output.annotation_gt.value
        projected_annotation = output.trace_payload["projected_annotation"]
        assert projected_annotation["type"] == output.annotation_gt.type
        if output.annotation_gt.type == "point":
            assert projected_annotation["point"] == output.annotation_gt.value
        elif output.annotation_gt.type == "point_map":
            assert set(output.annotation_gt.value) in (
                {"primary_mark", "line_mark"},
                {"answer_mark"},
            )
            assert projected_annotation["point_map"] == output.annotation_gt.value
        else:
            assert projected_annotation["segment_set"] == output.annotation_gt.value
            assert all(len(pair) == 2 for pair in output.annotation_gt.value)
            assert all(len(point) == 2 for pair in output.annotation_gt.value for point in pair)
        assert output.image.size[0] > 0
        assert output.image.size[1] > 0


def test_combo_label_answer_tasks_use_fixed_paired_mark_annotation() -> None:
    for offset, task_id in enumerate(
        (
            "task_charts__combo_mark__conditioned_line_extremum_label",
            "task_charts__combo_mark__conditioned_primary_extremum_label",
            "task_charts__combo_mark__absolute_gap_extremum_label",
            "task_charts__combo_mark__directional_gap_extremum_label",
        )
    ):
        output = create_task(task_id).generate(
            2026052810 + offset,
            params={},
            max_attempts=200,
        )
        assert output.answer_gt.type == "string"
        assert output.annotation_gt.type == "point_map"
        assert set(output.annotation_gt.value) == {"primary_mark", "line_mark"}
        assert output.trace_payload["projected_annotation"]["point_map"] == output.annotation_gt.value


def test_combo_cross_mark_difference_uses_calibrated_signed_queries() -> None:
    supported_queries = set()
    supported_scenes = set()
    label_counts = set()
    for offset in range(16):
        output = create_task("task_charts__combo_mark__cross_mark_difference_value").generate(
            2026052300 + offset,
            params={},
            max_attempts=160,
        )
        supported_queries.add(str(output.query_id))
        execution_trace = output.trace_payload["execution_trace"]
        supported_scenes.add(str(execution_trace["scene_variant"]))
        label_counts.add(int(execution_trace["label_count"]))

    assert supported_queries <= {
        "primary_minus_line_at_label",
        "line_minus_primary_at_label",
    }
    assert supported_scenes <= {
        "bar_line_shared_axis",
        "stacked_bar_line",
    }
    assert min(label_counts) >= 9
    assert max(label_counts) <= 12


def test_combo_series_threshold_crossing_label_matches_contract() -> None:
    task_id = "task_charts__combo_mark__series_threshold_crossing_label"
    query_ids = (
        "primary_first_above_threshold_label",
        "primary_first_below_threshold_label",
        "line_first_above_threshold_label",
        "line_first_below_threshold_label",
    )
    for offset, query_id in enumerate(query_ids):
        output = create_task(task_id).generate(
            2026060400 + offset,
            params={"query_id": query_id},
            max_attempts=160,
        )
        execution_trace = output.trace_payload["execution_trace"]
        labels = [str(label) for label in execution_trace["labels"]]
        target_role = str(execution_trace["target_series_role"])
        target_values = [
            int(value)
            for value in (
                execution_trace["primary_values"]
                if target_role == "primary"
                else execution_trace["line_values"]
            )
        ]
        threshold = int(execution_trace["threshold_value"])
        answer_index = int(execution_trace["answer_index"])
        answer_label = str(labels[int(answer_index)])
        above = str(execution_trace["crossing_direction"]) == "above"

        assert output.answer_gt.type == "string"
        assert str(output.answer_gt.value) == answer_label
        assert output.annotation_gt.type == "point"
        assert 2 <= int(execution_trace["crossing_index"]) <= min(8, len(labels) - 2)
        assert str(execution_trace["query_id"]) == str(query_id)
        assert f'"{execution_trace["target_series_name"]}"' in str(output.prompt)
        assert str(threshold) in str(output.prompt)

        for index, value in enumerate(target_values):
            satisfies = int(value) > int(threshold) if above else int(value) < int(threshold)
            if int(index) < int(answer_index):
                assert not satisfies
            if int(index) == int(answer_index):
                assert satisfies

        render_map = output.trace_payload["render_map"]
        target_points = render_map["primary_points_px"] if target_role == "primary" else render_map["line_points_px"]
        expected_point = [round(float(value), 3) for value in target_points[int(answer_index)]]
        assert output.annotation_gt.value == expected_point
        assert output.trace_payload["projected_annotation"]["point"] == expected_point


def test_combo_count_tasks_use_segment_set_annotation() -> None:
    for offset, task_id in enumerate(
        (
            "task_charts__combo_mark__dual_threshold_condition_count",
            "task_charts__combo_mark__interval_threshold_condition_count",
        )
    ):
        output = create_task(task_id).generate(
            2026060600 + offset,
            params={},
            max_attempts=200,
        )
        assert output.answer_gt.type == "integer"
        assert output.annotation_gt.type == "segment_set"
        assert len(output.annotation_gt.value) == int(output.answer_gt.value)
        assert output.trace_payload["projected_annotation"]["segment_set"] == output.annotation_gt.value
        for pair in output.annotation_gt.value:
            assert len(pair) == 2
            assert all(len(point) == 2 for point in pair)
