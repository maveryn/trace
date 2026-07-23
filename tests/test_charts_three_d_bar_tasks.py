"""Contract smoke tests for 3D bar-grid chart tasks."""

from __future__ import annotations

from trace_tasks.tasks.registry import create_task


THREE_D_BAR_TASKS = {
    "task_charts__bar_3d__series_category_scope_total_value": {"series_total_value", "series_interval_total_value"},
    "task_charts__bar_3d__category_total_value": {"single"},
    "task_charts__bar_3d__series_total_gap_value": {"single"},
    "task_charts__bar_3d__category_total_gap_value": {"single"},
    "task_charts__bar_3d__category_extremum_gap_value": {"single"},
    "task_charts__bar_3d__series_threshold_count": {"single"},
    "task_charts__bar_3d__category_threshold_count": {"single"},
    "task_charts__bar_3d__pairwise_comparison_count": {
        "single",
    },
}

EXPECTED_ANNOTATION_TYPES = {
    "task_charts__bar_3d__category_extremum_gap_value": "point_map",
    "task_charts__bar_3d__category_total_gap_value": "point_set_map",
    "task_charts__bar_3d__pairwise_comparison_count": "segment_set",
    "task_charts__bar_3d__series_total_gap_value": "point_set_map",
}


def test_three_d_bar_tasks_registered() -> None:
    for task_id in THREE_D_BAR_TASKS:
        assert create_task(task_id).task_id == task_id


def test_three_d_bar_tasks_generate_default_query_outputs() -> None:
    for seed_index, (task_id, allowed_query_ids) in enumerate(sorted(THREE_D_BAR_TASKS.items())):
        task = create_task(task_id)
        output = task.generate(
            91_000 + seed_index,
            params={},
            max_attempts=100,
        )
        assert output.scene_id == "bar_3d"
        assert output.query_id in allowed_query_ids
        assert output.trace_payload["query_spec"]["params"]["query_id"] == output.query_id
        assert output.answer_gt.type == "integer"
        expected_annotation_type = EXPECTED_ANNOTATION_TYPES.get(task_id, "point_set")
        assert output.annotation_gt.type == expected_annotation_type
        assert output.annotation_gt.value
        projected = output.trace_payload["projected_annotation"]
        assert projected["type"] == expected_annotation_type
        if expected_annotation_type == "point_set":
            assert projected["point_set"] == output.annotation_gt.value
            assert projected["pixel_point_set"] == output.annotation_gt.value
        elif expected_annotation_type == "segment_set":
            assert projected["segment_set"] == output.annotation_gt.value
            assert projected["pixel_segment_set"] == output.annotation_gt.value
        elif expected_annotation_type == "point_map":
            assert projected["point_map"] == output.annotation_gt.value
            assert projected["pixel_point_map"] == output.annotation_gt.value
        else:
            assert expected_annotation_type == "point_set_map"
            assert projected["point_set_map"] == output.annotation_gt.value
            assert projected["pixel_point_set_map"] == output.annotation_gt.value
        assert output.trace_payload["render_spec"]["font_assets"]["chart_font_family"]
        assert output.trace_payload["render_map"]["bar_traces"]


def test_three_d_bar_tasks_generate_each_query_branch() -> None:
    seed_index = 0
    for task_id, allowed_query_ids in sorted(THREE_D_BAR_TASKS.items()):
        task = create_task(task_id)
        for query_id in sorted(allowed_query_ids):
            output = task.generate(
                92_000 + seed_index,
                params={"query_id": query_id},
                max_attempts=100,
            )
            assert output.scene_id == "bar_3d"
            assert output.query_id == query_id
            assert output.trace_payload["query_spec"]["params"]["query_id"] == query_id
            assert output.answer_gt.type == "integer"
            assert output.annotation_gt.value
            seed_index += 1


def test_three_d_bar_axis_aggregate_uses_calibrated_grid_size() -> None:
    task = create_task("task_charts__bar_3d__series_category_scope_total_value")
    output = task.generate(
        92_500,
        params={"query_id": "series_interval_total_value"},
        max_attempts=100,
    )
    execution = output.trace_payload["execution_trace"]
    assert list(execution["category_count_range"]) == [3, 6]
    assert list(execution["series_count_range"]) == [3, 6]
    assert 3 <= int(execution["category_count"]) <= 6
    assert 3 <= int(execution["series_count"]) <= 6
    assert int(execution["max_bar_count"]) == 24
    assert int(execution["category_count"]) * int(execution["series_count"]) <= 24
    interval_range = list(execution["interval_category_count_range"])
    assert interval_range[0] == 3
    assert 3 <= interval_range[1] <= 4
    assert interval_range[0] <= int(execution["interval_category_count"]) <= interval_range[1]


def test_three_d_bar_condition_tasks_avoid_too_small_or_crowded_grids() -> None:
    for task_id, query_id, expected_category_range, expected_series_range in (
        ("task_charts__bar_3d__series_threshold_count", "single", [6, 6], [3, 4]),
        ("task_charts__bar_3d__category_threshold_count", "single", [3, 4], [6, 6]),
        ("task_charts__bar_3d__pairwise_comparison_count", "single", [4, 6], [4, 6]),
    ):
        output = create_task(task_id).generate(
            92_700,
            params={"query_id": query_id},
            max_attempts=100,
        )
        execution = output.trace_payload["execution_trace"]
        assert list(execution["category_count_range"]) == expected_category_range
        assert list(execution["series_count_range"]) == expected_series_range
        assert int(expected_category_range[0]) <= int(execution["category_count"]) <= int(expected_category_range[1])
        assert int(expected_series_range[0]) <= int(execution["series_count"]) <= int(expected_series_range[1])
        assert int(execution["max_bar_count"]) == 24
        assert int(execution["category_count"]) * int(execution["series_count"]) <= 24
        if task_id in {"task_charts__bar_3d__series_threshold_count", "task_charts__bar_3d__category_threshold_count"}:
            assert int(execution["target_count"]) == int(output.answer_gt.value)
            assert output.annotation_gt.type == "point_set"
            assert len(output.annotation_gt.value) == int(output.answer_gt.value)
            assert len(execution["annotation_bar_ids"]) == int(output.answer_gt.value)
            assert execution["matched_bar_ids"] == execution["annotation_bar_ids"]
        elif task_id == "task_charts__bar_3d__pairwise_comparison_count":
            assert int(execution["target_count"]) == int(output.answer_gt.value)
            assert output.annotation_gt.type == "segment_set"
            assert len(output.annotation_gt.value) == int(output.answer_gt.value)
            assert len(execution["annotation_bar_id_pairs"]) == int(output.answer_gt.value)


def test_three_d_bar_gap_annotations_are_keyed_by_visible_operands() -> None:
    category_gap = create_task("task_charts__bar_3d__category_total_gap_value").generate(
        92_810,
        params={"query_id": "single"},
        max_attempts=100,
    )
    category_execution = category_gap.trace_payload["execution_trace"]
    assert category_gap.annotation_gt.type == "point_set_map"
    assert set(category_gap.annotation_gt.value) == {
        category_execution["category_label_a"],
        category_execution["category_label_b"],
    }
    assert all(len(points) == int(category_execution["series_count"]) for points in category_gap.annotation_gt.value.values())

    series_gap = create_task("task_charts__bar_3d__series_total_gap_value").generate(
        92_811,
        params={"query_id": "single"},
        max_attempts=100,
    )
    series_execution = series_gap.trace_payload["execution_trace"]
    assert series_gap.annotation_gt.type == "point_set_map"
    assert set(series_gap.annotation_gt.value) == {
        series_execution["series_label_a"],
        series_execution["series_label_b"],
    }
    assert all(len(points) == int(series_execution["category_count"]) for points in series_gap.annotation_gt.value.values())


def test_three_d_bar_category_extremum_annotation_uses_highest_lowest_keys() -> None:
    output = create_task("task_charts__bar_3d__category_extremum_gap_value").generate(
        92_820,
        params={"query_id": "single"},
        max_attempts=100,
    )
    execution = output.trace_payload["execution_trace"]
    assert output.annotation_gt.type == "point_map"
    assert set(output.annotation_gt.value) == {"highest", "lowest"}
    assert set(execution["annotation_bar_id_groups"]) == {"highest", "lowest"}
    assert len(execution["annotation_bar_ids"]) == 2
    assert int(output.answer_gt.value) == int(execution["max_value"]) - int(execution["min_value"])


def test_three_d_bar_sampling_never_exceeds_max_bar_count() -> None:
    for seed in range(93_000, 93_040):
        for task_id, query_ids in THREE_D_BAR_TASKS.items():
            output = create_task(task_id).generate(
                seed,
                params={"query_id": sorted(query_ids)[0]},
                max_attempts=100,
            )
            execution = output.trace_payload["execution_trace"]
            assert int(execution["category_count"]) * int(execution["series_count"]) <= 24
